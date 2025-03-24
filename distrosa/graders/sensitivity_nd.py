#!/usr/bin/env python3
# vim:fenc=utf-8

"""Implementations of a calculator for N-D sensitivity.
"""
from typing import Callable
from typing import Sequence
from torch import Tensor
import torch
from .._misc import minterp as _minterp
from .._misc import getcdf as _getcdf
from . import SensitivityBase


class SensitivityND(SensitivityBase):
    """Sensitivity/gradient calculator for a N-D distribution.

    Arguments
    ---------
    npars : int
        Number of parameters in the PDF.

    gridlines : Sequence[Tensor]
        Vertices along a 1D gridline in all dimensions. `len(gridlines)` must be equal
        to the number of dimensions. And `len(gridlines[i])` is the number of vertices
        in the i-th dimension. Vertices in each dimension must be in ascending order.
        Note for this 1D class, `gridlines` must be a list of only one Tensor in it.

    eps : float | Tensor
        Finite difference step size(s). If a scalar, it is used for all parameters.
        Otherwise, it must have the same length as `params`.

    pdf : None or Callable, (x: Tensor, params: Tensor) -> pdfvals: Tensor
        Parametric probability density function (PDF). Potentially unnormalized.
        Broadcast should be supported for arbitrary shapes of `x`. Except for 1D,
        the function should expect x.shape[-1] to be the dimensionality and should
        return a Tensor with a shape of x.shape[:-1]. And for 1D, the function
        should always return a Tensor with a shape of x.shape. If `pdf` is `None`,
        users should later register it with `.register(...)`. Default is `None`.

    Notes
    -----
    * All init inputs are hard copied.
    * To make code more readable, not much sanity checks are done.
    """

    def __init__(
        self,
        npars: int,
        gridlines: Sequence[Tensor],
        eps: float | Tensor,
        pdf: None | Callable[[Tensor, Tensor], Tensor] = None,
    ):

        super().__init__(npars, gridlines, eps, pdf)

        # type hints to make static type checkers happy
        self.epsx: Tensor
        self.epsx2: Tensor

        # finite-difference step size in spatial dimensions
        self.register_buffer("epsx", torch.zeros(self.ndim))
        self.register_buffer("epsx2", torch.zeros(self.ndim))

        # use 0.1 of the smallest cell size as the default epsx
        self.epsx[:] = torch.tensor([_.min()/10.0 for _ in self.dx])
        self.epsx2[:] = self.epsx * 2.0

        self.requires_grad_(False)

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        # docstring is inherited from SensitivityBase.forward

        # calculate the chunk size
        nelms_per_1g = 134217728  # number of doubles per 1 GB
        bsize = nelms_per_1g // max(self.nverts) // self.ndim  # chunck size

        # easier to process in this arrangement
        _x = x.view(-1, self.ndim)

        if _x.shape[0] > bsize:
            J = []
            for i in range(0, _x.shape[0], bsize):
                J.append(self._backend(_x[i:i+bsize], params))
            J = torch.cat(J, 0)
        else:
            J = self._backend(_x, params)

        return J.view(x.shape+(self.npars,))  # shape: (..., ndim, npars)

    def _backend(self, x: Tensor, params: Tensor) -> Tensor:
        """Calculate the gradient at space points.
        """

        # aliases/references for our convenience
        npars = self.npars
        ndim = self.ndim
        nverts = self.nverts
        nx = x.shape[:-1]  # a tuple; the shape of the points
        eps = self.eps
        eps2 = self.eps2
        epsx = self.epsx
        epsx2 = self.epsx2

        # empty containers
        H = torch.zeros(nx+(ndim, ndim), device=self.gridlines[0].device)
        G = torch.zeros(nx+(ndim, npars), device=self.gridlines[0].device)
        J = torch.zeros(nx+(ndim, npars), device=self.gridlines[0].device)

        # construct H and G
        for i, (vi, dxi) in enumerate(zip(self.gridlines, self.dx)):  # i-th conditional

            # expand and copy (NOTE: memory inefficient!!)
            xk = x.view(nx+(1, ndim)).expand(nx+(nverts[i], ndim)).clone()

            # conditioning
            xk[..., i] = vi  # xk shape: (nx, nverts[i], ndim)

            # construct H
            for j in range(ndim):  # looping over spatial dimensions

                # keep a copy of the original data points' values
                if i == j:
                    # no need for normalize CDF; only need the normalization factor
                    norm = _getcdf(self.pdf(xk, params).view(xk.shape[:-1]), dxi)[1]

                    # calculate normalized PDF at x directly
                    torch.divide(self.pdf(x, params).view(nx), norm, out=H[..., i, j])
                    norm = None
                else:
                    xj = x[..., j].clone().view(nx+(1,)).expand(nx+(nverts[i],))

                    # alias
                    vj = self.gridlines[j]

                    # positive perturb the j-th spatial dimension
                    torch.add(xj, epsx[j], out=xk[..., j])
                    torch.clip(xk[..., j], vj[0], vj[-1], out=xk[..., j])
                    cdfp = _getcdf(self.pdf(xk, params).view(xk.shape[:-1]), dxi)[0]

                    # negative perturb the j-th spatial dimension
                    torch.subtract(xj, epsx[j], out=xk[..., j])
                    torch.clip(xk[..., j], vj[0], vj[-1], out=xk[..., j])
                    cdfm = _getcdf(self.pdf(xk, params).view(xk.shape[:-1]), dxi)[0]

                    # reusing cdfp mem space; cdfp = d F_i / d x_j
                    torch.subtract(cdfp, cdfm, out=cdfp)
                    torch.divide(cdfp, epsx2[j], out=cdfp)  # shape (nx, nverts[i])

                    # multiple 1D interp; (nx,), (nverts[i],), (nx, nverts[i]) -> (nx,)
                    H[..., i, j] = _minterp(x[..., i], vi, dxi, cdfp)

                    # restore xk
                    xk[..., j] = xj

                    # release memory
                    cdfm = None
                    cdfp = None
                    xj = None

            # construct G
            for j in range(npars):  # looping over parameters

                pars = params.clone()

                # params[j] += eps
                pars[j] = params[j] + eps[j]
                cdfp = _getcdf(self.pdf(xk, pars).view(xk.shape[:-1]), dxi)[0]

                # params[j] -= eps
                pars[j] = params[j] - eps[j]
                cdfm = _getcdf(self.pdf(xk, pars).view(xk.shape[:-1]), dxi)[0]

                # reusing cdfp mem space; cdfp = d F_i / d param_j
                torch.subtract(cdfp, cdfm, out=cdfp)
                torch.divide(cdfp, eps2[j], out=cdfp)  # shape (nx, nverts[i])

                # multiple 1D interp; (nx,), (nverts[i],), (nx, nverts[i]) -> (nx,)
                G[..., i, j] = _minterp(x[..., i], vi, dxi, cdfp)

                # release memory
                cdfm = None
                cdfp = None
                pars = None

            # release memory
            xk = None

        # solve the linear systems (avoid singular matrices)
        valid = torch.linalg.matrix_rank(H) >= ndim
        J[valid, :, :] = torch.linalg.solve(H[valid], G[valid])

        # apply the negative sign on the solution directly
        torch.negative(J, out=J)

        # return shape: (nx, ndim, npars)
        return J
