#!/usr/bin/env python3
# vim:fenc=utf-8

"""Differentiable empirical and analytical energy score implementations.
"""
import psutil
import numpy
import torch
from typing import Callable
from torch import Tensor


class AnalyticalEnergyScore(torch.nn.Module):
    """Analytical energy loss by numerically integrating the PDF.

    The input to the forward method is the parameters of the PDF. So the
    differentiability here means the gradient w.r.t. the PDF parameters.
    """

    def __init__(
        self,
        bounds: Tensor,
        nq: int,
        pdf: None | Callable[[Tensor, Tensor], Tensor] = None,
    ) -> None:
        super().__init__()

        # to make type checkers happy
        self.bounds: Tensor
        self.qs: Tensor
        self.ws: Tensor
        self.nq: int
        self.no: int
        self.ndim: int
        self.pdf: None | Callable[[Tensor, Tensor], Tensor]

        assert bounds.ndim == 2
        assert bounds.shape[1] == 2

        self.register_buffer("bounds", bounds)
        self.nq = nq
        self.ndim = bounds.shape[0]

        # get quadrature points and weights
        q, w = numpy.polynomial.legendre.leggauss(nq)  # q in [-1, 1]

        # scaled q and w in each dimension
        qs = []
        ws = []
        _lims = bounds.cpu().detach().numpy()
        for i in range(self.ndim):
            qs.append((q+1.0)*(_lims[i][1]-_lims[i][0])/2.0+_lims[i][0])
            ws.append(w*(_lims[i][1]-_lims[i][0])/2.0)

        # generate the whole quadrature gridline; qs: (nq**ndim, ndim), ws: (nq**ndim,)
        qs = numpy.meshgrid(*qs, indexing="ij")
        qs = numpy.stack(qs, axis=-1).reshape(-1, self.ndim)
        ws = numpy.meshgrid(*ws, indexing="ij")
        ws = numpy.prod(numpy.stack(ws, axis=-1), axis=-1)

        # move to torch.Tensor and save as in buffers
        self.register_buffer("qs", torch.tensor(qs))
        self.register_buffer("ws", torch.tensor(ws))

        # if pdf is None, users later must register one with `register` method
        self.pdf = pdf

    def register(self, pdf: Callable[[Tensor, Tensor], Tensor]) -> None:
        """Register the PDF function.
        """
        self.pdf = pdf

    def forward(self, params: Tensor, y: Tensor) -> Tensor:

        # even in 1D, we want to work with shape (..., 1)
        if y.ndim == 1:
            y = y.view(-1, 1)

        assert y.ndim == 2
        assert self.pdf is not None
        pdfvals = self.pdf(self.qs, params).view(-1)  # pdfvals shape (nq,)
        score1 = torch.cdist(self.qs, y, 2.0).sum(dim=1)  # shape: (nq,)
        score1 = (score1 * pdfvals * self.ws).sum() / y.shape[0]  # shape: scalar
        score2 = torch.cdist(self.qs, self.qs, 2.0)  # shape: (nq, nq)
        score2 = score2 * pdfvals.view(-1, 1) * pdfvals.view(1, -1)  # shape: (nq, nq)
        score2 = score2 * self.ws.view(-1, 1) * self.ws.view(1, -1)  # shape: (nq, nq)
        score2 = score2.sum() / 2.0  # scalar
        score = score1 - score2  # scalar
        return score


class EmpiricalEnergyScore(torch.nn.Module):
    """Differentiable empirical energy loss.

    The input to the forward method is the samples. So the differentiability here
    means the gradient w.r.t. the samples.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return empirical_energy_score.apply(x, y)  # type: ignore


class empirical_energy_score(torch.autograd.Function):
    """Differentiable empirical energy loss.

    Hopefully this provides a faster differentiation than using PyTorch's AD.

    Arguments
    ---------
    x : torch.Tensor
        The input tensor. Must be either 1D or 2D. If 2D, the shape must be (nx, ndim),
        where `nx` is the number of points and `ndim` is the spatial dimensionality.
    y : torch.Tensor
        The target tensor. Must be either 1D or 2D. If 2D, the shape must be (ny, ndim),
        where `ny` is the number of points and `ndim` is the spatial dimensionality.

    Returns
    -------
    torch.Tensor
        The energy score. It should be a 0-D tensor.

    Notes
    -----
    * `x.ndim == 2` and `y.ndim == 2`.
    * `x.shape[1] == y.shape[1]`.
    * Blocked algorithm is used to save runtim peak RAM usage. The size of a block is
      determined by the available free memory.
    * `y` is always assumed to be non-differentiable. So the backward pass will always
      return zeros for the gradient w.r.t. `y`.
    """

    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:
        """Energy score.
        """

        # remember the original shape of `x`
        xshape = x.shape

        # if 1D and users do not use shape (nx, 1)
        if x.ndim == 1:
            x = x.view(-1, 1)

        # if 1D and users do not use shape (ny, 1)
        if y.ndim == 1:
            y = y.view(-1, 1)

        # sanity checks; can be turned off using `-O` flag
        assert x.ndim == 2
        assert y.ndim == 2
        assert x.shape[1] == y.shape[1]

        # determine the block size
        if x.device.type == "cuda":
            avail = torch.cuda.mem_get_info()[0]  # in bytes
            avail /= 2.5
            bsize = int((avail/(x.shape[1]*8))**0.5)
        else:  # assume CPU
            avail = psutil.virtual_memory().available  # in bytes
            avail /= 2.5
            bsize = int((avail/(x.shape[1]*8))**0.5)

        # calling the block-based energy score calculation
        score, jac = blocked_energy_score(x, y, bsize)

        # respect the original x's shape (especially for 1D)
        jac = jac.view(xshape)

        ctx.save_for_backward(jac)

        return score

    @staticmethod
    def backward(ctx, grad_output: Tensor) -> tuple[Tensor, None]:  # type: ignore
        """Backward pass.
        """

        assert grad_output.ndim == 0
        jac, = ctx.saved_tensors  # (nx, ndim)
        grad1 = grad_output * jac

        # only `x` is considerred differentiable
        return grad1, None


@torch.jit.script
def blocked_energy_score(x: Tensor, y: Tensor, bsize: int) -> tuple[Tensor, Tensor]:
    """Blocked implementation for empirical energy loss with jacobian w.r.t. `x`.
    """

    # sanity checks can be turned off using `-O` command-line flag
    assert x.ndim == 2
    assert y.ndim == 2
    assert x.shape[1] == y.shape[1]

    nx = x.shape[0]
    ny = y.shape[0]
    ndim = x.shape[1]
    coeff1 = nx * ny
    coeff2 = nx * (nx - 1)
    coeff3 = coeff2 * 2.0

    score = torch.tensor(0.0, dtype=x.dtype, device=x.device)
    jac = torch.zeros((nx, ndim), dtype=x.dtype, device=x.device)

    for i in range(0, nx, bsize):
        xbatch = x[i:i+bsize]

        for j in range(0, ny, bsize):
            ybatch = y[j:j+bsize]

            # dealing with dist1[i:i+bsize, j:j+bsize, :]
            rvec = xbatch.view(-1, 1, ndim) - ybatch.view(1, -1, ndim)  # (nx, ny, ndim)
            r = torch.norm(rvec, dim=2)  # (nx, ny)
            torch.divide(rvec, r.unsqueeze(-1), out=rvec)  # (nx, ny, ndim)
            rvec = torch.sum(rvec, dim=1)  # (nx, ndim)

            jac[i:i+bsize, :] += (rvec / coeff1)
            score += (torch.sum(r) / coeff1)
            rvec = None
            r = None

    for i in range(0, nx, bsize):
        xbatch = x[i:i+bsize]

        for j in range(0, nx, bsize):
            ybatch = x[j:j+bsize]

            # dealing with dist2[i:i+bsize, j:j+bsize, :]
            rvec = xbatch.view(-1, 1, ndim) - ybatch.view(1, -1, ndim)
            r = torch.norm(rvec, dim=2)  # (nx, ny)
            torch.divide(rvec, r.unsqueeze(-1), out=rvec)  # (nx, ny, ndim)
            rvec[r == 0.0] = 0.0
            rvec = torch.sum(rvec, dim=1)  # (nx, ndim)

            jac[i:i+bsize, :] -= (rvec / coeff2)
            score -= (torch.sum(r) / coeff3)
            rvec = None
            r = None

    return score, jac
