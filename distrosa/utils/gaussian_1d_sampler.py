#!/usr/bin/env python3
# vim:fenc=utf-8

"""Differentiable Gaussian sampler in 1D as a PyTorch module/layer.
"""
import torch
from torch import Tensor
from typing import Callable
from ..graders import SensitivityBase


class Gaussian1DSampler(torch.nn.Module):
    """Sampler of a 1D Gaussian distribution as a torch module/layer.
    """

    def __init__(self, ndraw: int, grader: SensitivityBase):
        super().__init__()
        self.ndraw: int = ndraw
        self.grader: SensitivityBase = grader
        self.pdf: Callable[[Tensor, Tensor], Tensor] = gaussian_1d_pdf
        self.grader.register(self.pdf)

    def forward(self, params: Tensor):  # type: ignore
        return gaussian_1d_sampler.apply(params, self.ndraw, self.grader)


class gaussian_1d_sampler(torch.autograd.Function):
    """Functional backend of the 1D Gaussian sampler.
    """
    @staticmethod
    def forward(ctx, params: Tensor, ndraw: int, grader: SensitivityBase) -> Tensor:
        """Generate samples from the 1D Gaussian distribution.
        """

        with torch.no_grad():
            # convert uniform random numbers to 1D Gaussian samples
            u = torch.rand(ndraw, dtype=params.dtype, device=params.device)
            torch.multiply(u, 2.0, out=u)  # type: ignore
            torch.subtract(u, 1.0, out=u)  # type: ignore
            torch.special.erfinv(u, out=u)
            torch.multiply(u, params[1]*2.0**0.5, out=u)  # type: ignore
            torch.add(u, params[0], out=u)

        # save information for backward pass
        ctx.save_for_backward(params, u)  # tensors
        ctx.grader = grader

        return u

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> Tensor:  # type: ignore
        """Gradient of samples w.r.t. params (multiplying downstream gradients).
        """

        # retrieve saved tensors
        params, x = ctx.saved_tensors

        # retieve sensitivity calculator
        grader = ctx.grader

        # grad w.r.t. params
        with torch.no_grad():
            # shape: (..., 1) * (..., 5) -> (..., 5)
            grad1 = grad_output.unsqueeze(-1) * grader(x, params)

        return grad1, None, None  # type: ignore


@torch.jit.script
def gaussian_1d_pdf(x: Tensor, params: Tensor) -> Tensor:
    """Implementation of the 1D Gaussian PDF.
    """

    # constants
    negtwo = torch.tensor(-2.0)
    twopisqrt = torch.tensor((2.*torch.pi)**0.5)

    outs = torch.zeros_like(x)
    torch.subtract(x, params[0], out=outs)
    torch.divide(outs, params[1], out=outs)
    torch.square(outs, out=outs)
    torch.divide(outs, negtwo, out=outs)  # type: ignore
    torch.exp(outs, out=outs)
    torch.divide(outs, params[1]*twopisqrt, out=outs)  # type: ignore

    return outs
