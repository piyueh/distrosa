#!/usr/bin/env python3
# vim:fenc=utf-8

"""Differentiable empirical and analytical energy score implementations.
"""
import pathlib
import psutil
import numpy
import cupy
import torch
from typing import Callable
from typing import Sequence
from torch import Tensor


if torch.cuda.is_available():
    cusrc = pathlib.Path(__file__).resolve().parent.joinpath("energy_score.cu")
    with open(cusrc, "r") as fp:
        code = fp.read()
    _m = cupy.RawModule(code=code, jitify=True, options=("--dopt=on",))
    _cu_kernel_xy = _m.get_function("_kernel_xy")
else:
    def _cu_kernel_xy(*args, **kwargs):
        raise NotImplementedError("CUDA not available")


class AnalyticalEnergyScore(torch.nn.Module):
    """Analytical energy loss by numerically integrating the PDF.

    The input to the forward method is the parameters of the PDF. So the
    differentiability here means the gradient w.r.t. the PDF parameters.

    This function will take care of the normalization of the density function.
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
        ws = numpy.prod(numpy.stack(ws, axis=-1), axis=-1).reshape(-1)

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

        # get PDF values at quadrature points
        pdfvals = self.pdf(self.qs, params).view(-1)  # pdfvals shape (nq,)

        # calculate the normalization factor
        norm = (pdfvals * self.ws).sum()

        # normalize the PDF values
        pdfvals = pdfvals / norm

        # scores
        score1 = torch.cdist(self.qs, y, 2.0).sum(dim=1)  # shape: (nq,)
        score1 = (score1 * pdfvals * self.ws).sum() / y.shape[0]  # shape: scalar
        score2 = torch.cdist(self.qs, self.qs, 2.0)  # shape: (nq, nq)
        score2 = score2 * pdfvals.view(-1, 1) * pdfvals.view(1, -1)  # shape: (nq, nq)
        score2 = score2 * self.ws.view(-1, 1) * self.ws.view(1, -1)  # shape: (nq, nq)
        score2 = score2.sum() / 2.0  # scalar
        score = score1 - score2  # scalar

        return score


class BlockAnalyticalEnergyScore(torch.nn.Module):
    """Analytical energy loss by numerically integrating the PDF.

    This one is different from `AnalyticalEnergyScore` in that it uses piecewise
    integral, which make the loss more accurate but more computational expensive
    in terms of both speed and RAM consumption. For the purpose of training, the two
    do not give a significant difference because the optimizer can still find the
    minimal location even if the loss is not accurate (as long as the inaccurate loss
    has the same minimal location as the super-accurate one).

    The input to the forward method is the parameters of the PDF. So the
    differentiability here means the gradient w.r.t. the PDF parameters.

    This function will take care of the normalization of the density function.
    """

    def __init__(
        self,
        gridlines: Sequence[Tensor],
        nq: int,
        pdf: None | Callable[[Tensor, Tensor], Tensor] = None,
    ) -> None:
        super().__init__()

        # to make type checkers happy
        self.pdf: None | Callable[[Tensor, Tensor], Tensor]
        self.gridlines: Sequence[Tensor]
        self.ndim: int
        self.nq: int
        self.qs: Tensor
        self.ws: Tensor

        # if pdf is None, users later must register one with `register` method
        self.pdf = pdf
        self.gridlines = [_.detach().clone() for _ in gridlines]
        self.ndim = len(self.gridlines)
        self.nq = nq

        dtype = self.gridlines[0].dtype
        device = self.gridlines[0].device

        # get quadrature points and weights
        q, w = numpy.polynomial.legendre.leggauss(nq)  # q in [-1, 1]
        q = torch.tensor(q, dtype=dtype, device=device)
        w = torch.tensor(w, dtype=dtype, device=device)

        # scaled q and w in each dimension
        qs = []
        ws = []
        for i in range(self.ndim):
            v = self.gridlines[i]  # alias
            qs.append((q+1.0)*(v[1:]-v[:-1]).view(-1, 1)/2.0+v[:-1].view(-1, 1))
            qs[-1] = qs[-1].view(-1)
            ws.append(w*(v[1:]-v[:-1]).view(-1, 1)/2.0)
            ws[-1] = ws[-1].view(-1)

        # generate the whole quadrature gridline; qs: (nq**ndim, ndim), ws: (nq**ndim,)
        qs = torch.meshgrid(*qs, indexing="ij")
        ws = torch.meshgrid(*ws, indexing="ij")

        self.qs = torch.stack(qs, dim=-1).view(-1, self.ndim)
        self.ws = torch.prod(torch.stack(ws, dim=-1), dim=-1).view(-1)

    def to(self, *args, **kwargs):
        self.gridlines = [_.to(*args, **kwargs) for _ in self.gridlines]
        self.qs = self.qs.to(*args, **kwargs)
        self.ws = self.ws.to(*args, **kwargs)
        return super().to(*args, **kwargs)

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

        # get PDF values at quadrature points
        pdfvals = self.pdf(self.qs, params).view(-1)  # pdfvals shape (nq,)

        # calculate the normalization factor
        norm = (pdfvals * self.ws).sum()

        # normalize the PDF values
        pdfvals = pdfvals / norm

        # determine the block size
        if pdfvals.device.type == "cuda":
            avail = torch.cuda.mem_get_info()[0]  # in bytes
            avail /= 4
            bsize = int((avail/(self.ndim*8))**0.5)
        else:  # assume CPU
            avail = psutil.virtual_memory().available  # in bytes
            avail /= 4
            bsize = int((avail/(self.ndim*8))**0.5)

        # score 1
        score1 = torch.tensor(0.0, dtype=params.dtype, device=params.device)
        for bi in range(0, self.qs.shape[0], bsize):
            x = self.qs[bi:bi+bsize]
            w = self.ws[bi:bi+bsize]
            vals = pdfvals[bi:bi+bsize]
            for bj in range(0, y.shape[0], bsize):
                ybatch = y[bj:bj+bsize]
                nograd = torch.cdist(x, ybatch, 2.0).sum(dim=1) * w
                tmp = (nograd * vals).sum() / y.shape[0]
                score1 = score1 + tmp

        # score 2
        score2 = torch.tensor(0.0, dtype=params.dtype, device=params.device)
        for bi in range(0, self.qs.shape[0], bsize):
            x = self.qs[bi:bi+bsize]
            xw = self.ws[bi:bi+bsize]
            xvals = pdfvals[bi:bi+bsize]
            for bj in range(0, self.qs.shape[0], bsize):
                y = self.qs[bj:bj+bsize]
                yw = self.ws[bj:bj+bsize]
                yvals = pdfvals[bj:bj+bsize]
                nograd = torch.cdist(x, y, 2.0)
                torch.multiply(nograd, xw.view(-1, 1), out=nograd)
                torch.multiply(nograd, yw.view(1, -1), out=nograd)
                tmp = xvals.view(-1, 1) * yvals.view(1, -1)
                tmp = (tmp * nograd).sum() / 2.0
                score2 = score2 + tmp

        return score1 - score2  # scalar


class EmpiricalEnergyScore(torch.nn.Module):
    """Differentiable empirical energy loss.

    The input to the forward method is the samples. So the differentiability here
    means the gradient w.r.t. the samples.
    """

    def __init__(self) -> None:
        super().__init__()

        self._kernels = {
            "cuda": cuda_empirical_energy_score.apply,
            "cpu": torch_empirical_energy_score.apply,
        }

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self._kernels[x.device.type](x, y)  # type: ignore


class torch_empirical_energy_score(torch.autograd.Function):
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
        with torch.no_grad():
            score, jac = blocked_energy_score(x.detach(), y.detach(), bsize)

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


class cuda_empirical_energy_score(torch.autograd.Function):
    """Differentiable empirical energy loss.

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

        # calling the block-based energy score calculation
        with torch.no_grad():
            score, jac = cuda_energy_score(x.detach(), y.detach())

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


def cuda_energy_score(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Wrapper of the CUDA C implementation of the energy score.

    Notes
    -----
    * The CUDA C implementation is only for double precision.
    * Currently can't handle large arrays requiring more than 65535 blocks.
    * Currently the max dimensionality can be handled is 12.
    """

    nx = int(x.shape[0])
    ny = int(y.shape[0])
    ndim = int(x.shape[1])
    coeff1 = float(nx*ny)
    coeff2 = float(-(nx*(nx-1))*2)
    coeff3 = float(-(nx*(nx-1)))
    assert ndim <= 12, "Currently only supports up to 12-D space points"

    # number of threads per block; number of blocks; number of grids
    nths = 16
    nblkx = (nx + nths - 1) // nths
    nblky = (ny + nths - 1) // nths
    nblkmax = 65535  # maximum number of blocks per grid
    assert nblkx <= nblkmax, "len(x) too large"
    assert nblky <= nblkmax, "len(y) too large"

    # final shared memory size; should not exceed 48KB; but no sanity check here
    memsize = int(((ndim+1)*nths**2+2*ndim*nths)*8)

    # non-copy conversion to cupy data type
    cux = cupy.asarray(x)
    cuy = cupy.asarray(y)

    # allocate memory for the jacobian and the score
    cujac = cupy.zeros((nx, ndim), dtype=cupy.float64)
    cuscore = cupy.zeros((1,), dtype=cupy.float64)  # scalar as size-1 1D array

    # score 1
    _cu_kernel_xy(
        (nblkx, nblky),  # number of blocks
        (nths, nths),  # number of threads per block
        (cux, cuy, cujac, cuscore, nx, ny, ndim, coeff1, coeff1),
        shared_mem=memsize
    )

    # score 2
    _cu_kernel_xy(
        (nblkx, nblkx),  # number of blocks
        (nths, nths),  # number of threads per block
        (cux, cux, cujac, cuscore, nx, nx, ndim, coeff2, coeff3),
        shared_mem=memsize
    )

    # supposedly non-copy conversion back to torch data type
    score = torch.as_tensor(cuscore[0], dtype=x.dtype, device=x.device)
    jac = torch.as_tensor(cujac, dtype=x.dtype, device=x.device)

    return score, jac


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
            rvec[r == 0.0] = 0.0
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


# quick test
if __name__ == "__main__":
    import time

    _nx = 54321
    _ny = 12345
    _ndim = 2

    for i in range(20):

        # for cuda kernels
        _x1 = torch.rand((_nx, _ndim), dtype=torch.float64, device="cuda")
        _x1 = _x1.requires_grad_(True)
        _y1 = torch.rand((_ny, _ndim), dtype=torch.float64, device="cuda")
        _y1 = _y1.requires_grad_(False)

        # for PyTorch kernels
        _x2 = _x1.detach().clone().requires_grad_(True)
        _y2 = _y1.detach().clone().requires_grad_(False)

        # for CUDA kernels v2
        _x3 = _x1.detach().clone().requires_grad_(True)
        _y3 = _y1.detach().clone().requires_grad_(False)

        # test cuda kernels
        torch.cuda.synchronize()
        _tbg1 = time.perf_counter_ns()
        _loss1 = cuda_empirical_energy_score.apply(_x1, _y1)
        _loss1.backward()  # type: ignore
        torch.cuda.synchronize()
        _ted1 = time.perf_counter_ns()

        # test torch kernels
        torch.cuda.synchronize()
        _tbg2 = time.perf_counter_ns()
        _loss2 = torch_empirical_energy_score.apply(_x2, _y2)
        _loss2.backward()  # type: ignore
        torch.cuda.synchronize()
        _ted2 = time.perf_counter_ns()

        # test cuda kernels
        _lossfn = EmpiricalEnergyScore()
        torch.cuda.synchronize()
        _tbg3 = time.perf_counter_ns()
        _loss3 = _lossfn(_x3, _y3)
        _loss3.backward()  # type: ignore
        torch.cuda.synchronize()
        _ted3 = time.perf_counter_ns()

        _loss1 = _loss1.detach().cpu().numpy()  # type: ignore
        _loss2 = _loss2.detach().cpu().numpy()  # type: ignore
        _loss3 = _loss3.detach().cpu().numpy()  # type: ignore
        _jac1 = _x1.grad.clone().detach().cpu().numpy()  # type: ignore
        _jac2 = _x2.grad.clone().detach().cpu().numpy()  # type: ignore
        _jac3 = _x3.grad.clone().detach().cpu().numpy()  # type: ignore

        print(_loss1, _jac1.mean(), (_ted1-_tbg1)/1e9)  # type: ignore
        print(_loss2, _jac2.mean(), (_ted2-_tbg2)/1e9)  # type: ignore
        print(_loss3, _jac3.mean(), (_ted3-_tbg3)/1e9)  # type: ignore
        print()

        assert numpy.allclose(_loss1, _loss2, 0.0, 1e-10)
        assert numpy.allclose(_jac1, _jac2, 0.0, 1e-10)
        assert numpy.allclose(_loss2, _loss3, 0.0, 1e-10)
        assert numpy.allclose(_jac2, _jac3, 0.0, 1e-10)
        assert numpy.allclose(_loss1, _loss3, 0.0, 1e-12)
        assert numpy.allclose(_jac1, _jac3, 0.0, 1e-12)
