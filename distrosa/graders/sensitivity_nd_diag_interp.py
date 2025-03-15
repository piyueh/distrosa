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

    Notes
    -----
    * All init inputs are hard copied.
    * To make code more readable, not much sanity checks are done.
    """

    def __init__(self, npars: int, gridlines: Sequence[Tensor], eps: float|Tensor):

        super().__init__(npars, gridlines, eps)

        # to make static type checkers happy
        self._params: Tensor
        self._deltas: torch.nn.ParameterDict
        self._pdfvals: torch.nn.ParameterList

        # data for interpolations
        self._deltas = torch.nn.ParameterDict({
            f"{(i, j)}": torch.zeros(self.nverts)
            for (i, j) in itertools.product(range(self.ndim), range(self.npars))
        })

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
        data = _getconditionals(self.pdf, v, dx, params)[0]  # type: ignore
        for i in range(ndim):
            self._pdfvals[i][...] = data[i]
            data[i] = None  # immediately release the memory

        # get all 1D conditional CDFs at all N-D vertices under perturbed parameters
        for j in range(npars):
            perturb = params.clone()

            perturb[j] = params[j]+ eps[j]
            cdfsp = _getconditionals(self.pdf, v, dx, perturb)[1]  # type: ignore

            perturb[j] = params[j]- eps[j]
            cdfsm = _getconditionals(self.pdf, v, dx, perturb)[1]  # type: ignore

            for i in range(ndim):
                self._deltas[f"{(i, j)}"][...] = (cdfsp[i] - cdfsm[i]) / eps2[j]

                # immediately release the memory
                cdfsp[i] = None
                cdfsm[i] = None

        # link the attributes to the constructed values; no-copy, just referencing
        self._params[...] = params

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        # docstring is inherited from SensitivityBase.forward

        # check if self._delta needs to be reconstructed
        if self.needupdate(params):
            self._construct(params)

        # easier to process in this arrangement even for 1D
        _x = x.view(-1, self.ndim)

        # to hold the outputs
        J = torch.zeros(_x.shape+(self.npars,))

        for i in range(self.ndim):
            for j in range(self.npars):
                fx = _interpnd(_x, self.gridlines, self._pdfvals[i])
                derv = _interpnd(_x, self.gridlines, self._deltas[f"{(i, j)}"])
                J[..., i, j] = - derv / fx

        return J.view(*x.shape, self.npars)

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

        if not torch.allclose(params, self._params, 0, 1e-9, True):
            return True

        return False
