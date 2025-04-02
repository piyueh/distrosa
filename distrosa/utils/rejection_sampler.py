#!/usr/bin/env python3
# vim:fenc=utf-8

"""A general purpose differentiable piecewise rejection sampler.
"""
import itertools
import numpy
import torch
from torch import Tensor
from torch.distributions.uniform import Uniform
from typing import Callable
from typing import Sequence
from distrosa import SensitivityBase


class RejectionSampler(torch.nn.Module):
    """A differentiable rejection sampler as a PyTorch layer/module.
    """

    def __init__(
        self,
        pdf: Callable[[Tensor, Tensor], Tensor],
        gridlines: Sequence[Tensor],
        grader: SensitivityBase | None,
        maxiter: int = 1000
    ):
        super().__init__()

        self.pdf = pdf
        self.gridlines = [_.detach().clone() for _ in gridlines]
        self.grader = grader
        self.proposal = PiecewiseConstantProposal(pdf, self.gridlines)
        self.maxiter = maxiter

    def to(self, *args, **kwargs):  # override the paraent's method
        self.gridlines = [_.to(*args, **kwargs) for _ in self.gridlines]
        self.proposal = self.proposal.to(*args, **kwargs)
        if self.grader is not None:
            self.geader = self.grader.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def forward(self, ndraw: int, params: Tensor) -> Tensor:
        return rejection_sampler.apply(  # type: ignore
            self.pdf, ndraw, params, self.proposal, self.maxiter, self.grader
        )


class PiecewiseConstantProposal(torch.nn.Module):
    """Piecewise constant proposal distribution.

    This is a piecewise constant approximation of a given PDF. The constant value of
    each piece (i.e., cell) is the maximum value of the PDF within that cell. However,
    the weight of each cell is determined by the integral value of PDF in that cell.
    """

    def __init__(
        self,
        pdf: Callable[[Tensor, Tensor], Tensor],
        gridlines: Sequence[Tensor]
    ):
        super().__init__()

        # sanity check: currently only support uniform grid
        for i in range(len(gridlines)):
            dx = gridlines[i][1:] - gridlines[i][:-1]
            assert torch.allclose(dx, dx[0]), f"{dx}"

        # useless type declarations; to make static type checkers happy
        self.geidlines: Sequence[Tensor]
        self.dtype: torch.dtype
        self.device: torch.device
        self.ndim: int
        self.shape: tuple[int, ...]
        self.bounds: Tensor
        self.weights: Tensor
        self.xmax: Tensor
        self.valmax: Tensor
        self._params: Tensor

        self.pdf = pdf
        self.gridlines = gridlines

        # floating number precision and device
        self.dtype = gridlines[0].dtype
        self.device = gridlines[0].device

        # spatial dimension
        self.ndim = len(gridlines)
        ndim = self.ndim  # alias for local use

        # number of cells/pieces
        self.shape = tuple(len(_)-1 for _ in gridlines)

        # bounds of each cell/piece
        self.bounds = torch.zeros(self.shape+(self.ndim, 2))
        self.bounds = self.bounds.to(self.dtype).to(self.device)

        # generate the bounds of each cell/piece
        for i, gridline in enumerate(self.gridlines):
            # (K1, ..., Ki-1, ..., Kn) <- (1, ..., Ki-1, ..., 1)
            self.bounds[..., i, 0] = gridline[:-1].view((1,)*i+(-1,)+(1,)*(ndim-i-1))
            self.bounds[..., i, 1] = gridline[1:].view((1,)*i+(-1,)+(1,)*(ndim-i-1))

        # values to be built once we get the parameters
        self.weights = None  # weights of each cell/piece  # type: ignore
        self.xmax = None  # type: ignore
        self.valmax = None  # maximum value of PDF in each cell/piece  # type: ignore

        # cached parameters that can serve as a hash
        self._params = None  # type: ignore

    def to(self, *args, **kwargs):  # override the paraent's method
        self.gridlines = [_.to(*args, **kwargs) for _ in self.gridlines]
        self.dtype = self.gridlines[0].dtype
        self.device = self.gridlines[0].device
        self.bounds = self.bounds.to(*args, **kwargs)

        if self._params is not None:
            self.weights = self.weights.to(*args, **kwargs)
            self.xmax = self.xmax.to(*args, **kwargs)
            self.valmax = self.valmax.to(*args, **kwargs)
            self._params = self._params.to(*args, **kwargs)

        return super().to(*args, **kwargs)

    def build(self, params: Tensor) -> None:
        """Build parameter-dependent values.

        The following attributes are built here:
        * self.weights
        * self.xmax
        * self.valmax
        * self._params
        """

        # find the maximum location and maximal PDF value in each cell
        _pdf = lambda x: self.pdf(x, params)
        self.xmax, self.valmax = _get_pmax_all(_pdf, self.bounds)

        # normalized weights
        self.weights = self.valmax / torch.sum(self.valmax)

        # cache the parameters to be used as a hash
        self._params = params.detach().clone()

    def propose(self, ndraw: int, params: Tensor) -> tuple[Tensor, Tensor]:
        """Propose samples from the piecewise constant distribution.
        """

        # build required values
        if self.needupdate(params):
            self.build(params)

        # determine the bin of each sample
        sources = torch.multinomial(self.weights.view(-1), ndraw, replacement=True)

        # determine the number of samples in each bin
        counts = torch.bincount(sources, minlength=self.weights.numel())

        # reshape
        counts = counts.view(self.shape)

        # output holder
        samples = []
        pdfvals = []

        # generate samples per bin
        for idx in itertools.product(*[range(_) for _ in self.shape]):
            # the Uniform class theoretically should respect bounds' dtype
            dist = Uniform(self.bounds[idx][:, 0], self.bounds[idx][:, 1])
            samples.append(dist.sample([int(counts[idx]),]))
            pdfvals.append(self.valmax[idx].view(1).expand(int(counts[idx])))

        # combine to one single tensor
        samples = torch.concat(samples, dim=0)
        pdfvals = torch.concat(pdfvals, dim=0)

        return samples, pdfvals

    def needupdate(self, params: Tensor) -> bool:
        """Check if the internal data needs to be updated.

        The tolerance is set to be 10 times the machine precision of the dtype.

        Arguments
        ---------
        params : Tensor
            Parameters of the PDF as a 1D array.

        Returns
        -------
        bool
            Whether the internal data needs to be updated.
        """

        eps = torch.finfo(self.dtype).eps * 10

        # we may have other criteria in the future
        if self._params is None:
            return True

        return (not torch.allclose(params, self._params, 0.0, eps*10, True))


class rejection_sampler(torch.autograd.Function):
    """The functional version of the differentiable rejection sampler.
    """

    @staticmethod
    def forward(
        ctx,
        pdf: Callable[[Tensor, Tensor], Tensor],
        ndraw: int,
        params: Tensor,
        proposal: PiecewiseConstantProposal,
        maxiter: int,
        grader: SensitivityBase | None,
    ):

        with torch.no_grad():
            x = _rejection_sampler_impl(pdf, ndraw, params, proposal, maxiter)

        # save information for backward pass
        ctx.save_for_backward(params.detach(), x.detach())
        ctx.grader = grader

        return x  # shape (ndraw, bounds.shape[0]) or (ndraw,)

    def backward(ctx, grad: Tensor):  # type: ignore

        # retrieve saved tensors
        params, x = ctx.saved_tensors
        grader = ctx.grader

        with torch.no_grad():
            out = grad.unsqueeze(-1) * grader(x, params)
            out = torch.sum(out, dim=tuple(range(out.ndim-1)))

        return None, None, out, None, None, None


@torch.jit.script
def _lossfn(_pdfvals: Tensor) -> Tensor:
    loss = torch.nan_to_num(torch.log(_pdfvals), neginf=-1e-12)
    loss = - torch.sum(loss)
    return loss


@torch.jit.script
def _scale(inputs, lowers, lengths):
    return lengths * torch.nn.functional.sigmoid(inputs) + lowers


def _get_pmax_all(
    pdf: Callable[[Tensor], Tensor],
    bounds: Tensor,
    maxiter: int = 1000
) -> tuple[Tensor, Tensor]:
    """Find the maximum location and PDF value of a distribution.

    The optimization is done in [-inf, inf]^ndim space, and the infinite-domain spatial
    coordinates are mapped into `bounds`.
    """

    # aliases
    ncells = bounds.shape[:-2]
    ndim = bounds.shape[-2]
    lengths = bounds[..., 1] - bounds[..., 0]  # length of each dimension
    lowers = bounds[..., 0]  # lower bound of each dimension

    # initial guess of unbounded location
    q = torch.rand(ncells+(ndim,), dtype=bounds.dtype, device=bounds.device)
    q.requires_grad_(True)

    optimizer = torch.optim.LBFGS([q,], line_search_fn="strong_wolfe")

    def _closure():
        optimizer.zero_grad()
        x = _scale(q, lowers, lengths)
        loss = _lossfn(pdf(x))
        loss.backward()
        return loss

    for c in range(maxiter):

        old = optimizer.step(_closure).detach().cpu().item()

        with torch.no_grad():
            x = _scale(q, lowers, lengths)
            cur = _lossfn(pdf(x))

        if abs((old-cur)/old) < 1e-10:
            break
    else:
        raise RuntimeError(f"Failed to converge in {maxiter} iterations.")

    with torch.no_grad():
        # scale to the true spatial bounds and search in the log sapce
        x = lengths * torch.nn.functional.sigmoid(q) + lowers
        pmax = pdf(x)  # shape: ncells

    x = x.detach()
    x.requires_grad_(False)

    pmax = pmax.detach()
    pmax.requires_grad_(False)

    return x, pmax


def _rejection_sampler_impl(
    pdf: Callable[[Tensor, Tensor], Tensor],
    ndraw: int,
    params: Tensor,
    proposal: PiecewiseConstantProposal,
    maxiter: int,
) -> Tensor:
    """A very simple rejection sampler for a demonstration purpose.

    This uses a uniform proposal distribution as the proposal distribution.

    Parameters
    ----------
    pdf : Callable[[Tensor, Tensor], Tensor]
        A parametric probability density function.

    ndraw : int
        The number of samples to draw.

    params : Tensor
        The parameters of the PDF.

    proposal : PiecewiseConstantProposal
        A piecewise-constant proposal.

    maxiter : int
        The maximum number of iterations to collect enough samples.

    Returns
    -------
    Tensor
        The realizations of the PDF with a shape of `(ndraw, bounds.shape[0])`.
    """

    # this is non-differentiable, so we need to detach inputs' graph
    params = params.detach()
    ndraw = int(ndraw)
    ndim = proposal.ndim
    dtype = proposal.dtype
    device = proposal.device

    # initialize the containers for final samples
    out = torch.full((ndraw, ndim), torch.nan, dtype=dtype, device=device)

    bg, ed = 0, ndraw
    for trials in range(maxiter):

        # how many samples are needed
        missing = ed - bg

        # make proposals (already within the bounds because of the uniform proposals)
        suggests, pmaxs = proposal.propose(missing, params)

        # evaluate target density at the proposed samples
        pdfvals = pdf(suggests, params)

        # a uniform random number between pmin and pmax
        criteria = Uniform(0.0, pmaxs).sample()

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
