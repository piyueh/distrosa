#!/usr/bin/env python3
# vim:fenc=utf-8

"""Implementations of a calculator for N-D sensitivity via interpolation.
"""
from typing import Callable
from typing import Sequence
from torch import Tensor
import torch
from .._misc import getconditionals as _getconditionals
from .._misc import centraldiff as _centraldiff
from .._misc import forwarddiff as _forwarddiff
from .._misc import backwarddiff as _backwarddiff
from .._misc import interpnd as _interpnd
from . import SensitivityBase


class SensitivityNDInterp(SensitivityBase):
    """Sensitivity interpolater for a N-D distribution via multilinear interpolation.

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

    def __init__(self, npars: int, gridlines: Sequence[Tensor], eps: float|Tensor):

        super().__init__(npars, gridlines, eps)

        # to make static type checkers happy
        self._params: Tensor
        self._J: Tensor

        # cache
        self.register_buffer("_params", torch.zeros(npars))
        self.register_buffer("_J", torch.zeros(self.nverts+(self.ndim, npars)))

        self.requires_grad_(False)

    def _construct(self, params: Tensor) -> None:
        """Construct values at vertices for later being used in interpolations.
        """

        # aliases for readability
        v = self.gridlines
        dx = self.dx
        npars = self.npars
        ndim = self.ndim
        nverts = self.nverts
        eps = self.eps
        eps2 = self.eps2

        # initialize arrays
        H = torch.zeros(nverts+(ndim, ndim), device=self.gridlines[0].device)
        G = torch.zeros(nverts+(ndim, npars), device=self.gridlines[0].device)

        # get all 1D conditional CDFs at all N-D vertices
        cdfs = _getconditionals(
            self.pdf(
                torch.stack(torch.meshgrid(*v, indexing="ij"), -1),
                params
            ).view(nverts),
            dx
        )[1]  # only needs normalized conditional CDFs

        # [preparing H]
        for j in range(ndim):  # loop over each spatial direction
            for i in range(ndim):  # loop over each conditional
                _centraldiff(cdfs[i], dx[j], j, out=H[..., i, j])  # internal points
                _forwarddiff(cdfs[i], dx[j], j, out=H[..., i, j])  # lower boundary
                _backwarddiff(cdfs[i], dx[j], j, out=H[..., i, j])  # upper boundary

                # release memory
                cdfs[i] = None

        # [preparing G]
        for j in range(npars):  # loop over each parameter
            perturb = params.clone()

            perturb[j] = params[j]+ eps[j]
            cdfsp = _getconditionals(
                self.pdf(
                    torch.stack(torch.meshgrid(*v, indexing="ij"), -1),
                    perturb
                ).view(nverts),
                dx
            )[1]  # type: ignore

            perturb[j] = params[j]- eps[j]
            cdfsm = _getconditionals(
                self.pdf(
                    torch.stack(torch.meshgrid(*v, indexing="ij"), -1),
                    perturb
                ).view(nverts),
                dx
            )[1]  # type: ignore

            for i in range(ndim):  # loop over each conditional
                G[..., i, j] = (cdfsp[i] - cdfsm[i]) / eps2[j]

                # immediately release the memory
                cdfsp[i] = None
                cdfsm[i] = None

            # release memory
            perturb = None

        # solve the linear systems (avoid singular matrices)
        valid = torch.linalg.matrix_rank(H) >= ndim
        self._J[...] = 0.0
        self._J[valid, :, :] = torch.linalg.solve(H[valid], G[valid])

        # apply the negative sign on the solution directly
        torch.negative(self._J, out=self._J)

        # update the cached parameters
        self._params[...] = params

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        # docstring is inherited from SensitivityBase.forward

        # check if self._J needs to be reconstructed
        if self.needupdate(params):
            self._construct(params)

        # easier to process in this arrangement even for 1D
        _x = x.view(-1, self.ndim)

        # to hold the outputs
        Jx = torch.zeros(_x.shape+(self.npars,), device=x.device)

        for i in range(self.ndim):
            for j in range(self.npars):
                Jx[..., i, j] = _interpnd(_x, self.gridlines, self._J[..., i, j])

        return Jx.view(x.shape+(self.npars,))  # restore the original shape

    def needupdate(self, params: Tensor) -> bool:
        """Check if the internal data needs to be updated.

        Arguments
        ---------
        params : Tensor
            Parameters of the PDF as a 1D array.

        Returns
        -------
        bool
            Whether the internal data needs to be updated.
        """

        # we may have other criteria in the future
        return (not torch.allclose(params, self._params, 0.0, 1e-9, True))
