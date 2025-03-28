#!/usr/bin/env python3
# vim:fenc=utf-8

"""N-D sensitivity calculator via interpolation and the diagonal approximation.
"""
import itertools
from typing import Callable
from typing import Sequence
from torch import Tensor
import torch
from .._misc import getconditionals as _getconditionals
from .._misc import interpnd as _interpnd
from . import SensitivityBase


class SensitivityNDDiagInterp(SensitivityBase):
    """N-D sensitivity calculator w/ diagonal approximation and interpolation.

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

        # to make static type checkers happy
        self._params: Tensor
        self._deltas: torch.nn.ParameterList
        self._pdfvals: torch.nn.ParameterList

        # data for interpolations
        self._deltas = torch.nn.ParameterList([
            torch.nn.ParameterList([
                torch.zeros(self.nverts)
                for _2 in range(self.ndim)
            ])
            for _1 in range(self.npars)
        ])

        self._pdfvals = torch.nn.ParameterList([
            torch.zeros(self.nverts) for _ in range(self.ndim)
        ])

        # cached params used as a hash to check if self._J needs to be reconstructed
        self.register_buffer("_params", torch.zeros(npars))

        self.requires_grad_(False)

    def _construct(self, params: Tensor) -> None:
        """Construct values at vertices for later being used in interpolations.

        `self._pdfvals` and `self._deltas` are constructed and updated here.
        """

        # aliases for readability
        v = self.gridlines
        dx = self.dx
        npars = self.npars
        ndim = self.ndim
        nverts = self.nverts
        eps = self.eps
        eps2 = self.eps2

        # get all 1D conditional PDFs at all N-D vertices
        _getconditionals(
            self.pdf(
                torch.stack(torch.meshgrid(*v, indexing="ij"), -1),
                params
            ).view(nverts),
            dx,
            cpdfs=self._pdfvals
        )

        # get all 1D conditional CDFs at all N-D vertices under perturbed parameters
        for j in range(npars):
            perturb = params.clone()

            perturb[j] = params[j]+ eps[j]
            _getconditionals(
                self.pdf(
                    torch.stack(torch.meshgrid(*v, indexing="ij"), -1),
                    perturb
                ).view(nverts),
                dx,
                ccdfs=self._deltas[j]
            )

            perturb[j] = params[j]- eps[j]
            cdfsm = _getconditionals(
                self.pdf(
                    torch.stack(torch.meshgrid(*v, indexing="ij"), -1),
                    perturb
                ).view(nverts),
                dx
            )[1]

            for i in range(ndim):
                torch.subtract(self._deltas[j][i], cdfsm[i], out=self._deltas[j][i])
                torch.divide(self._deltas[j][i], eps2[j], out=self._deltas[j][i])

                # immediately release the memory
                cdfsm[i] = None  # type: ignore

        # link the attributes to the constructed values; no-copy, just referencing
        self._params[...] = params.detach()

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        # docstring is inherited from SensitivityBase.forward

        # check if self._delta needs to be reconstructed
        if self.needupdate(params):
            self._construct(params)

        # easier to process in this arrangement even for 1D
        _x = x.view(-1, self.ndim)

        # to hold the outputs
        J = torch.zeros(_x.shape+(self.npars,), device=x.device)

        for i in range(self.ndim):
            fx = _interpnd(_x, self.gridlines, self._pdfvals[i])
            for j in range(self.npars):
                derv = _interpnd(_x, self.gridlines, self._deltas[j][i])
                J[..., i, j] = - derv / fx

        return J.view(x.shape+(self.npars,))  # restore the original shape

    def needupdate(self, params: Tensor) -> bool:
        """Check if the internal data needs to be updated.

        Arguments
        ---------
        params : NDArray
            Parameters of the PDF as a 1D array.

        Returns
        -------
        bool
            Whether the internal data needs to be updated.
        """

        eps = torch.finfo(self.eps.dtype).eps * 10

        # we may have other criteria in the future
        return (not torch.allclose(params, self._params, 0.0, eps, True))
