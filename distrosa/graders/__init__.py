#!/usr/bin/env python3
# vim:fenc=utf-8

"""Collection of sensitivity calculators.
"""
from typing import Callable
from typing import Tuple
from typing import Sequence
from torch import Tensor
import torch


class SensitivityBase(torch.nn.Module):
    """Base class for sensitivity calculators.

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

        # type hints to make static type checkers (and TorchScript) happy
        self.gridlines: torch.nn.ParameterList
        self.dx: torch.nn.ParameterList
        self.ndim: int
        self.npars: int
        self.nverts: Tuple[int, ...]
        self.eps: Tensor
        self.eps2: Tensor
        self.pdf: Callable[[Tensor, Tensor], Tensor]

        # mandatory for all torch.nn.Module subclasses
        super().__init__()

        self.gridlines = torch.nn.ParameterList([_.clone() for _ in gridlines])
        self.dx = torch.nn.ParameterList([_[1:]-_[:-1] for _ in gridlines])
        self.register_buffer("eps", torch.zeros(npars))
        self.register_buffer("eps2", torch.zeros(npars))
        self.ndim = len(self.gridlines)
        self.npars = npars
        self.nverts = tuple(len(_) for _ in gridlines)

        self.eps[:] = eps
        self.eps2[:] = eps * 2.0

        # dummy; users should register the actual PDF with `.register(...)`
        self.pdf = lambda x, params: torch.zeros_like(x)

        # this is not a real "differentiable" layer
        self.requires_grad_(False)

    def register(self, func: Callable[[Tensor, Tensor], Tensor]) -> None:
        """Register the PDF function.

        Arguments
        ---------
        func : Callable, (x: Tensor, params: Tensor) -> pdfvals: Tensor
            Parametric probability density function (PDF). Potentially unnormalized.
            Broadcast should be supported for arbitrary shapes of `x`. Except fo 1D,
            the function should expect x.shape[-1] to be the dimensionality and should
            return a Tensor with a shape of x.shape[:-1]. And for 1D, the function
            should always return a Tensor with a shape of x.shape.
        """
        self.pdf = func

    def forward(self, x: Tensor, params: Tensor) -> Tensor:
        """Calculate the gradient w.r.t. params at space points.

        Arguments
        ---------
        x : Tensor
            Where to evaluate the sensitivity. The shape of `x` can be arbitrary.

        params : Tensor
            Parameters of the PDF as a 1D tensor.

        Returns
        -------
        Tensor
            Gradient values at `x`. Its shape is `x.shape+(self.npars,)`. Note that
            `x.shape[-1]` is the dimensionality of the PDF.

        Notes
        -----
        All things must be `torch.Tensor`s.
        """
        raise NotImplementedError
