#!/usr/bin/env python3
# vim:fenc=utf-8

"""Differentiable sampler in for 1D beta distribution.
"""
import torch
from torch import Tensor
from torch.distributions.beta import Beta
from typing import Callable
from ..graders import SensitivityBase


class Beta1DSampler(torch.nn.Module):
    """A differentiable sampler for 1D beta distribution as a PyTorch layer/module.
    """

    def __init__(self, grader: SensitivityBase):
        super().__init__()
        self.pdf: Callable[[Tensor, Tensor], Tensor] = beta_1d_pdf  # type: ignore
        self.grader: SensitivityBase = grader
        self.grader.register(self.pdf)

    def forward(self, ndraw: int, params: Tensor):  # type: ignore
        return beta_1d_sampler.apply(params, ndraw, self.grader)


class beta_1d_sampler(torch.autograd.Function):
    """Functional version of the differentiable 1D beta sampler.
    """

    @staticmethod
    def forward(ctx, params: Tensor, ndraw: int, grader: SensitivityBase) -> Tensor:
        """Generate samples from the 1D beta distribution.
        """

        with torch.no_grad():
            dist = Beta(params[0], params[1])
            x = dist.sample((ndraw,))

        # save information for backward pass
        ctx.save_for_backward(params.detach(), x.detach())
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
        with torch.no_grad():
            # shape: (..., 1), (..., len(params)) -> (len(params),)
            grad1 = grad_output.unsqueeze(-1)
            grad1 = grad1 * grader(x, params)
            grad1 = grad1.sum(dim=list(range(x.ndim)))

        assert grad1.shape == params.shape

        return grad1, None, None  # type: ignore


class _beta_1d_pdf(torch.autograd.Function):
    """Functional version of 1D Beta PDF with custom backward pass.

    Instead of relying automatic differentiation, we the backward pass is done with
    analytical gradients directly. Hopefully this speeds up the performance.

    Arguments
    ---------
    x : Tensor
        Space points in (0, 1). Can be of any shape.

    params : Tensor
        Parameters of the 1D beta distribution. Must be of shape (2,).

    Returns
    -------
    Tensor
        PDF values at `x`.

    Notes
    -----
    * x should > 0 and < 1, but this routine does not do the sanity checks.
    * Hopfully this would be faster than using `torch.distributions.beta.Beta`.
    """
    @staticmethod
    def forward(ctx, x: Tensor, params: Tensor) -> Tensor:
        """Implementation of the 1D Beta PDF.
        """
        with torch.no_grad():
            val = _pdf_backend(x, params)

        ctx.save_for_backward(x, params, val)
        return val

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, Tensor]:
        """Implementation of the backward pass of 1D Beta PDF.
        """
        x, params, pdfvals = ctx.saved_tensors

        with torch.no_grad():
            # grad1 final shape: x.shape
            grad1 = _d_pdf_d_x_backend(x, params, pdfvals)  # x.shape
            grad1 = grad1 * grad_output  # x.shape

            # grad2 final shape: (len(params),)
            grad2 = _d_pdf_d_params_backend(x, params, pdfvals)
            grad2 = (grad2 * grad_output.unsqueeze(-1)).sum(dim=list(range(x.ndim)))

        assert grad1.shape == x.shape
        assert grad2.shape == params.shape

        return grad1, grad2


@torch.jit.script
def _pdf_backend(x: Tensor, params: Tensor) -> Tensor:
    """Implementation of the 1D Beta PDF.

    Notes
    -----
    * x should > 0 and < 1, but this routine does not do the sanity checks.
    * Hopfully this would be faster than using `torch.distributions.beta.Beta`.
    """

    # tmp is a scalar
    tmp = torch.exp(torch.special.gammaln(params[0]+params[1]))
    tmp = tmp / torch.exp(torch.special.gammaln(params[0]))
    tmp = tmp / torch.exp(torch.special.gammaln(params[1]))

    # try our best not to allocate new memory
    val = 1.0 - x
    val = val**(params[1] - 1.0)
    val = torch.pow(x, params[0]-1.0) * val
    val = val * tmp
    return val


@torch.jit.script
def _d_pdf_d_params_backend(x: Tensor, params: Tensor, pdfvals: Tensor) -> Tensor:
    """Implementatin of the gradient of PDF w.r.t. params using analytical gradients.
    """
    digamma_ab = torch.special.digamma(params.sum())
    digamma_a = torch.special.digamma(params[0])
    digamma_b = torch.special.digamma(params[1])
    dfda = (torch.log(x) + digamma_ab - digamma_a) * pdfvals
    dfdb = (torch.log(1.0-x) + digamma_ab - digamma_b) * pdfvals
    return torch.stack([dfda, dfdb], dim=-1)


@torch.jit.script
def _d_pdf_d_x_backend(x: Tensor, params: Tensor, pdfvals: Tensor) -> Tensor:
    """Implementatin of the gradient of PDF w.r.t. x using analytical gradients.
    """
    val = (params[0] - 1.0) / x - (params[1] - 1.0) / (1.0 - x)
    val = val * pdfvals
    return val


# a differentiable 1D bete PDF function for end users
beta_1d_pdf = _beta_1d_pdf.apply
