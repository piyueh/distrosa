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

        if x.shape[0] > bsize:
            J = []
            for i in range(0, _x.shape[0], bsize):
                J.append(self._backend(_x[i:i+bsize], params))
            J = torch.cat(J, 0)
        else:
            J = self._backend(_x, params)

        return J.view(*x.shape, self.npars)  # shape: (..., ndim, npars)

    def _backend(self, x: Tensor, params: Tensor) -> Tensor:
        """Calculate the gradient at space points.
        """

        # aliases/references for our convenience
        v = self.gridlines
        dx = self.dx
        npars = self.npars
        ndim = self.ndim
        nverts = self.nverts
        nx = x.shape[:-1]  # a tuple; the shape of the points
        eps = self.eps
        eps2 = self.eps2

        # empty containers
        J = torch.zeros(nx+(ndim, npars))

        # construct \partial F_i / \partial param_j
        for i in range(ndim):  # loop over 1D conditionals

            # expand and copy (NOTE: memory inefficient!!)
            xk = x.view(nx+(1, ndim)).expand(nx+(nverts[i], ndim))

            # conditioning
            xk[..., i] = v[i]  # xk shape: (nx, nverts[i], ndim)

            for j in range(npars):  # loop over parameters

                pars = params.clone()

                # params[j] += eps
                pars[j] = params[j] + eps[j]
                cdfp, _ = _getcdf(self.pdf, xk, pars, dx[i])

                # params[j] -= eps
                pars[j] = params[j] - eps[j]
                cdfm, _ = _getcdf(self.pdf, xk, pars, dx[i])

                # reusing cdfp mem space; cdfp = d F_i / d param_j
                torch.subtract(cdfp, cdfm, out=cdfp)
                torch.divide(cdfp, eps2[j], out=cdfp)  # shape (Nx, Ki)

                # 1D interpolation; shape change: (Nx,), (Ki,), (Nx, Ki) -> (Nx,)
                J[..., i, j] = _minterp(x[..., i], v[i], dx[i], cdfp)

            # we don't need to normalize the CDF, just need the normalization factor
            _, norm = _getcdf(self.pdf, xk, params, dx[i])

            # calculate normalized PDF at x directly (rather than via interpolation)
            _pdf = self.pdf(x, params).view(nx)  # some 1D PDF returns (Nx, 1)
            torch.divide(_pdf, norm, out=_pdf)

            # scale J[..., i, :] by -1 / f(x)
            torch.divide(J[..., i, :], _pdf.view(nx+(1,)), out=J[..., i, :])
            torch.negative(J[..., i, :], out=J[..., i, :])

        # return shape: (nx, ndim, npars)
        return J
