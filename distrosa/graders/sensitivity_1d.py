#!/usr/bin/env python3
# vim:fenc=utf-8

"""Implementation of the 1D sensitivity analysis.
"""
from typing import Callable
from torch import Tensor
import torch
from .._misc import getcdf as _getcdf
from .._misc import interp as _interp
from . import SensitivityBase


class Sensitivity1D(SensitivityBase):
    """Sensitivity/gradient calculator for a 1D distribution.

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

    Notes
    -----
    * All init inputs are hard copied.
    * To make code more readable, not much sanity checks are done.
    """

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        # docstring is inherited from SensitivityBase.forward

        # aliases for our convenience
        v = self.gridlines[0]  # this is 1D
        dx = self.dx[0]  # this is 1D
        npars = self.npars

        # get normalized PDF at x
        f_at_x = self.pdf(x, params) / _getcdf(self.pdf(v, params), dx)[1]

        # will be holding d F / d params[j] for all j
        gj = []

        for j in range(self.npars):

            _pars = params.clone()

            _pars[j] = params[j] + self.eps[j]
            _cdfp = _getcdf(self.pdf(v, _pars), dx)[0]

            _pars[j] = params[j] - self.eps[j]
            _cdfm = _getcdf(self.pdf(v, _pars), dx)[0]

            # reusing the memory space of `_cdfp` to hold the numerical derivatives
            torch.subtract(_cdfp, _cdfm, out=_cdfp)
            torch.divide(_cdfp, self.eps2[j], out=_cdfp)

            # let's trust that CuPy's interpolation is efficient enough for now
            gj.append(_interp(x, v, dx, _cdfp))

            # clear memory
            _pars = None
            _cdfp = None
            _cdfm = None

        # stack the results to have shape x.shape+(P,)
        gj = torch.stack(gj, -1)

        # calculate J = - (d F / d p) / f(x)
        torch.negative(gj, out=gj)
        torch.divide(gj, f_at_x.view(-1, 1), out=gj)

        # stack the results and return
        return gj
