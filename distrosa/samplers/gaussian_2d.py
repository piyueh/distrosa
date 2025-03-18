#!/usr/bin/env python3
# vim:fenc=utf-8

"""Differentiable Gaussian sampler in 2D as a PyTorch module/layer.
"""
import torch
from torch import Tensor
from torch.distributions.multivariate_normal import MultivariateNormal
from ..graders import SensitivityBase


@torch.jit.script
def gaussian_pdf_2d(x: Tensor, params: Tensor) -> Tensor:
    """Implementation of the 2D Gaussian PDF.

    Arguments
    ---------
    x : Tensor
        The input tensor. Regardless the dimensionality of `x`, `x.shape[-1]` must be 2.

    params : Tensor
        The parameters tensor. `params.shape` must be `(5,)`, corresponding to mu1, mu2,
        sigma1, sigma2, and rho.

    Returns
    -------
    pdfvals : Tensor
        The PDF values at `x` with the given parameters. `pdfvals.shape` is the same as
        `x.shape[:-1]`.
    """

    tmp = 1.0 - params[4] * params[4]

    z1 = (x[..., 0] - params[0]) / params[2]
    z2 = (x[..., 1] - params[1]) / params[3]

    pdfvals = - (z1 * z1 - 2 * params[4] * z1 * z2 + z2 * z2)
    pdfvals = pdfvals / (2.0 * tmp)
    pdfvals = torch.exp(pdfvals)
    pdfvals = pdfvals / (2.0 * torch.pi * params[2] * params[3] * tmp**0.5)

    return pdfvals


class gaussian_sampler_2d(torch.autograd.Function):
    """Functional backend of the 2D Gaussian sampler.
    """
    @staticmethod
    def forward(ctx, params: Tensor, ndraw: int, grader: SensitivityBase) -> Tensor:
        """Generate samples from the 1D Gaussian distribution.
        """

        with torch.inference_mode():
            cov = torch.zeros(2, 2, dtype=params.dtype, device=params.device)
            cov[0, 0] = params[2]**2
            cov[1, 1] = params[3]**2
            cov[0, 1] = cov[1, 0] = params[2] * params[3] * params[4]
            dist = MultivariateNormal(params[:2], cov)
            x = dist.sample((ndraw,))  # will return shape (ndraw, 2)

        # save information for backward pass
        ctx.save_for_backward(params, x)  # tensors
        ctx.ndraw = ndraw
        ctx.grader = grader

        return x

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
            # shape: (..., 2) * (..., 2, 5) -> (..., 5)
            grad1 = grad_output * grader(x, params)

        # grad w.r.t. ndraw
        grad2 = None

        # grad w.r.t. grader
        grad3 = None

        return grad1, grad2, grad3  # type: ignore


class GaussianSampler2D(torch.nn.Module):
    """Sampler of a 2D Gaussian distribution as a torch module/layer.
    """

    def __init__(self, ndraw, grader):
        super().__init__()
        self.ndraw = ndraw
        self.grader = grader
        self.grader.register(self.pdf)

    def pdf(self, x: Tensor, params: Tensor):
        """Probability density function.
        """
        return gaussian_pdf_2d(x, params)

    def forward(self, params: Tensor):  # type: ignore
        return gaussian_sampler_2d.apply(params, self.ndraw, self.grader)
