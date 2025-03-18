#!/usr/bin/env python3
# vim:fenc=utf-8

"""Differentiable Gaussian sampler in 1D as a PyTorch module/layer.
"""
import torch
from torch import Tensor
from ..graders import SensitivityBase


@torch.jit.script
def gaussian_pdf_1d(x: Tensor, params: Tensor) -> Tensor:
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


class gaussian_sampler_1d(torch.autograd.Function):
    """Functional backend of the 1D Gaussian sampler.
    """
    @staticmethod
    def forward(ctx, params: Tensor, ndraw: int, grader: SensitivityBase) -> Tensor:
        """Generate samples from the 1D Gaussian distribution.
        """

        with torch.inference_mode():
            # convert uniform random numbers to 1D Gaussian samples
            u = torch.rand(ndraw, dtype=params.dtype, device=params.device)
            torch.multiply(u, 2.0, out=u)  # type: ignore
            torch.subtract(u, 1.0, out=u)  # type: ignore
            torch.special.erfinv(u, out=u)
            torch.multiply(u, params[1]*2.0**0.5, out=u)  # type: ignore
            torch.add(u, params[0], out=u)

        # save information for backward pass
        ctx.save_for_backward(params, u)  # tensors
        ctx.ndraw = ndraw
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
        with torch.inference_mode():
            grad1 = grad_output.view(-1, 1) * grader(x, params)

        # grad w.r.t. ndraw
        grad2 = None

        # grad w.r.t. grader
        grad3 = None

        return grad1, grad2, grad3  # type: ignore


class GaussianSampler1D(torch.nn.Module):
    """Sampler of a 1D Gaussian distribution as a torch module/layer.
    """

    def __init__(self, ndraw, grader):
        super().__init__()
        self.ndraw = ndraw
        self.grader = grader
        self.grader.register(self.pdf)

    def pdf(self, x: Tensor, params: Tensor):
        """Probability density function.
        """
        return gaussian_pdf_1d(x, params)

    def forward(self, params: Tensor):  # type: ignore
        return gaussian_sampler_1d.apply(params, self.ndraw, self.grader)
