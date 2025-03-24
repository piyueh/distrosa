#!/usr/bin/env python3
# vim:fenc=utf-8

"""A differentiable rejection sampler.
"""
import torch
from torch import Tensor
from torch.distributions.uniform import Uniform
from typing import Callable
from ..graders import SensitivityBase


class RejectionSampler(torch.nn.Module):
    """A differentiable rejection sampler as a PyTorch layer/module.
    """

    def __init__(
        self,
        ndraw: int,
        bounds: Tensor,
        grader: SensitivityBase,
        pdf: Callable[[Tensor, Tensor], Tensor] | None = None,
        maxiter: int = 1000,
    ):
        super().__init__()

        # type declarations to make type checkers happy
        self.pdf: None | Callable[[Tensor, Tensor], Tensor] = None
        self.ndraw: int
        self.bounds: Tensor
        self.maxiter: int
        self.grader: SensitivityBase

        # assign attributes
        self.ndraw = ndraw
        self.bounds = bounds
        self.grader = grader
        self.maxiter = maxiter
        self.pdf = pdf

        if self.pdf is not None:
            self.grader.register(self.pdf)

    def register(self, pdf: Callable[[Tensor, Tensor], Tensor]) -> None:
        """Register the PDF function.
        """
        self.pdf = pdf
        self.grader.register(pdf)

    def forward(self, params: Tensor) -> Tensor:
        """Generate samples from the PDF using the rejection sampler.
        """
        return rejection_sampler.apply(  # type: ignore
            self.pdf, self.ndraw, params, self.bounds, self.maxiter, self.grader
        )


class rejection_sampler(torch.autograd.Function):
    """The functional version of the differentiable rejection sampler.
    """

    @staticmethod
    def forward(
        ctx,
        pdf: Callable[[Tensor, Tensor], Tensor],
        ndraw: int,
        params: Tensor,
        bounds: Tensor,
        maxiter: int,
        grader: SensitivityBase,
    ):
        """Generate samples from the PDF using the rejection sampler.
        """

        with torch.no_grad():
            x = rejection_sampler_impl(pdf, ndraw, params, bounds, maxiter)

        # save information for backward pass
        ctx.save_for_backward(params.detach(), x.detach())
        ctx.grader = grader

        return x  # shape (ndraw, bounds.shape[0]) or (ndraw,)

    def backward(ctx, grad_output: Tensor):  # type: ignore
        """The backward pass of the rejection sampler.
        """

        # retrieve saved tensors
        params, x = ctx.saved_tensors
        grader = ctx.grader

        with torch.no_grad():

            if x.ndim == 1:
                grad = grad_output.unsqueeze(-1) * grader(x, params)
            else:
                grad = (grad_output.unsqueeze(-1) * grader(x, params)).sum(dim=-2)

        return None, None, grad, None, None, None


def rejection_sampler_impl(
    pdf: Callable[[Tensor, Tensor], Tensor],
    ndraw: int,
    params: Tensor,
    bounds: Tensor,
    maxiter: int,
) -> Tensor:
    """An extremely naive and simple rejection sampler for a demonstration purpose.

    This uses a uniform proposal distribution as the proposal distribution.

    Parameters
    ----------
    pdf : Callable[[Tensor, Tensor], Tensor]
        A parametric probability density function.

    ndraw : int
        The number of samples to draw.

    params : Tensor
        The parameters of the PDF.

    bounds : Tensor
        The bounds of the PDF. `bounds.ndim==2` and `bounds.shape[-1]==2`.

    maxiter : int
        The maximum number of iterations to collect enough samples.

    Returns
    -------
    Tensor
        The realizations of the PDF with a shape of `(ndraw, bounds.shape[0])`.
    """

    # 1D, special case
    if bounds.ndim == 1 and bounds.shape[0] == 2:
        bounds = bounds.unsqueeze(0)  # make it 2D with shape (1, 2)

    # sanity checks; can be turned off via command-line flag `-O`
    assert bounds.ndim == 2
    assert bounds.shape[-1] == 2

    # aliases
    ndim = bounds.size(0)
    dtype = bounds.dtype
    device = bounds.device

    # this is non-differentiable, so we need to detach inputs' graph
    params = params.detach()
    bounds = bounds.detach()
    ndraw = int(ndraw)

    # proposal functions
    proposers = [Uniform(bounds[i, 0], bounds[i, 1]) for i in range(ndim)]

    # minimum probability
    pmin = 0.0

    # maximum probability
    pmax = _get_pmax(pdf, params, bounds)[1]

    # a uniform random number between pmin and pmax
    rangen = Uniform(pmin, pmax)

    # initialize the containers for final samples
    out = torch.full((ndraw, ndim), torch.nan, dtype=dtype, device=device)

    bg, ed = 0, ndraw
    for trials in range(maxiter):

        # how many samples are needed
        missing = ed - bg

        # make proposals (already within the bounds because of the uniform proposals)
        suggests = torch.stack([_.sample((missing,)) for _ in proposers], dim=-1)

        # evaluate target density at the proposed samples
        pdfvals = pdf(suggests, params)

        # a uniform random number between pmin and pmax
        criteria = rangen.sample((missing,))

        # MH-rule accept/reject
        yes = (pdfvals > criteria)

        # update the collection
        ed = bg + int(yes.to(torch.int).sum())
        out[bg:ed, :] = suggests[yes, :]

        # check if we got enough samples
        if ed == ndraw:
            break

        # update counters
        bg, ed = ed, ndraw
    else:
        raise RuntimeError(f"{ndraw} wanted; got {bg} in {maxiter} trials.")

    # 1D, special case
    if ndim == 1:
        out = out.view(ndraw)

    return out


def _get_pmax(
    pdf: Callable[[Tensor, Tensor], Tensor], params: Tensor, bounds: Tensor,
    maxiter: int = 1000
) -> tuple[Tensor, Tensor]:
    """Find the maximum location and PDF value of a distribution.
    """

    q = torch.rand((bounds.size(0),), dtype=bounds.dtype, device=bounds.device)
    q.requires_grad_(True)
    pars = params.detach().clone()
    pars.requires_grad_(False)
    optimizer = torch.optim.LBFGS([q,], line_search_fn="strong_wolfe")

    def lossfn(_in):
        x = (bounds[:, 1] - bounds[:, 0]) * torch.nn.functional.sigmoid(_in)
        x = x + bounds[:, 0]
        loss = - torch.log(pdf(x, pars))
        return loss

    def closure():
        optimizer.zero_grad()
        loss = lossfn(q)
        loss.backward()
        return loss

    for c in range(maxiter):
        old = optimizer.step(closure).detach().cpu().item()
        cur = lossfn(q).detach().cpu().item()
        if abs(old - cur) < 1e-6:
            break
    else:
        raise RuntimeError(f"Failed to converge in {maxiter} iterations.")

    with torch.no_grad():
        x = (bounds[:, 1] - bounds[:, 0]) * torch.nn.functional.sigmoid(q)
        x = x + bounds[:, 0]
        pmax = pdf(x, pars)

    x = x.detach()
    x.requires_grad_(False)

    pmax = pmax.detach()
    pmax.requires_grad_(False)

    return x, pmax
