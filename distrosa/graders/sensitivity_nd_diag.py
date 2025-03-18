#!/usr/bin/env python3
# vim:fenc=utf-8

"""Implementations of a calculator for N-D sensitivity using the diagonal approximation.
"""
from typing import Callable
from typing import Sequence
from torch import Tensor
import torch
from .._misc import minterp as _minterp
from .._misc import getcdf as _getcdf
from . import SensitivityBase


class SensitivityNDDiag(SensitivityBase):
    """Sensitivity calculator for a N-D distribution via diagonal approximation.

    Arguments
    ---------
    npars : int
        Number of parameters in the PDF.

    gridlines : Sequence[Tensor]
        Vertices along a 1D gridline in all dimensions. `len(gridlines)` must be equal
        to the number of dimensions. And `len(gridlines[i])` is the number of vertices
        in the i-th dimension. Vertices in each dimension must be in ascending order.

    eps : float | Tensor
        Finite difference step size(s). If a scalar, it is used for all parameters.
        Otherwise, it must have the same length as `params`.

    Notes
    -----
    * All init inputs are hard copied.
    * To make code more readable, not much sanity checks are done.
    """

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

        # empty containers
        J = torch.zeros(nx+(ndim, npars), device=self.gridlines[0].device)

        # construct \partial F_i / \partial param_j
        for i, (vi, dxi) in enumerate(zip(self.gridlines, self.dx)):  # i-th conditional

            # expand and copy (NOTE: memory inefficient!!)
            xk = x.view(nx+(1, ndim)).expand(nx+(nverts[i], ndim)).clone()

            # conditioning
            xk[..., i] = vi  # xk shape: (nx, nverts[i], ndim)

            for j in range(npars):  # loop over parameters

                pars = params.clone()

                # params[j] += eps
                pars[j] = params[j] + eps[j]
                cdfp = _getcdf(self.pdf(xk, pars).view(xk.shape[:-1]), dxi)[0]

                # params[j] -= eps
                pars[j] = params[j] - eps[j]
                cdfm = _getcdf(self.pdf(xk, pars).view(xk.shape[:-1]), dxi)[0]

                # reusing cdfp mem space; cdfp = d F_i / d param_j
                torch.subtract(cdfp, cdfm, out=cdfp)
                torch.divide(cdfp, eps2[j], out=cdfp)  # shape (Nx, Ki)

                # 1D interpolation; shape change: (Nx,), (Ki,), (Nx, Ki) -> (Nx,)
                J[..., i, j] = _minterp(x[..., i], vi, dxi, cdfp)

                # clear memory
                pars = None
                cdfp = None
                cdfm = None

            # we don't need to normalize the CDF, just need the normalization factor
            norm = _getcdf(self.pdf(xk, params).view(xk.shape[:-1]), dxi)[1]

            # calculate normalized PDF at x directly (rather than via interpolation)
            pdfvals = self.pdf(x, params).view(nx)  # some 1D PDF returns (Nx, 1)
            torch.divide(pdfvals, norm, out=pdfvals)

            # scale J[..., i, :] by -1 / f(x)
            torch.divide(J[..., i, :], pdfvals.view(nx+(1,)), out=J[..., i, :])
            torch.negative(J[..., i, :], out=J[..., i, :])

            # release memory
            xk = None
            pdfvals = None
            norm = None

        # return shape: (nx, ndim, npars)
        return J
