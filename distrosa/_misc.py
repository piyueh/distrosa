#!/usr/bin/env python3
# vim:fenc=utf-8

"""Miscellaneous functions/helpers.

This module is supposed to be used internally.
"""
import itertools
from typing import Callable
from typing import Sequence
from torch import Tensor  # for type hints
import torch


@torch.jit.script
def cartesianprod(N: int, device: torch.device) -> Tensor:
    """Our own implementation of cartesian product for integers.

    PyTorch's `cartesian_prod` does not work with `torch.jit.script` when it cannot
    infer the dimensions at the compile time. And `torch.jit.script` does not support
    `itertools.product` will, either. This function is a workaround.

    Arguments
    ---------
    N : int
        Number of dimensions.

    Returns
    -------
    Tensor of shape (2**N, N)
        All possible values of the Cartesian product [0, 1]**N.
    """

    result = torch.tensor([[0], [1]], dtype=torch.long, device=device)

    for _ in range(1, N):
        zeros = torch.zeros((result.size(0), 1), dtype=torch.long, device=device)
        ones = torch.ones((result.size(0), 1), dtype=torch.long, device=device)

        result = torch.cat((
            torch.cat((result, zeros), dim=1),
            torch.cat((result, ones), dim=1)
        ), dim=0)

    return result


@torch.jit.script
def interp(x: Tensor, verts: Tensor, h: Tensor, values: Tensor) -> Tensor:
    """Piecewise linear interpolation to multiple points on a 1D gridline.

    Notes
    -----
    * Points outside the gridline will be interpolated from either the 1st or the last
      piece.
    * This function assumes the inputs are already PyTorch tensors. No sanity check.

    Arguments
    ---------
    x : Tensor
        Points to be interpolated. Can be an arbitrary shape.

    verts : Tensor
        Vertices along a 1D gridline. `x.ndim == 1`. Must be sorted in ascending order.

    h : Tensor
        Cell size along the gridline. `len(h) == len(verts)-1`. `h=verts[1:]-verts[:-1]`

    values : Tensor
        Values at the vertices. `values.ndim == 1`

    Returns
    -------
    Tensor
        Interpolated values at `x`. The same shape as `x`.
    """

    # identify the upper bound vertex's index for the piece that contains each point
    idx = torch.searchsorted(verts, x, side="right")

    # idx=0, meaning x < verts[0]; then interpolated it from the 1st piece
    # idx=len(verts), meaning x >= verts[-1]; then interpolated from the last piece
    torch.clamp(idx, 1, len(verts)-1, out=idx)

    # aliases for readability
    ds = x - verts[idx-1]
    torch.divide(ds, h[idx-1], out=ds)

    low = values[idx-1]
    dval = values[idx]
    torch.subtract(dval, low, out=dval)
    torch.multiply(dval, ds, out=dval)
    torch.add(dval, low, out=dval)

    return dval


@torch.jit.script
def minterp(x: Tensor, verts: Tensor, h: Tensor, values: Tensor):
    """Multiple 1D piecewise linear interpolation happening at the same time.

    Regardless the shape/dim of `x`, it is treated like a sequence of 1D coordinates
    in different shape. So `x` can have an arbitrary shape, but the dimension of `verts`
    and `values` must be as follows:
        * verts.ndim == 1
        * values.ndim = x.ndim + 1
        * values.shape == x.shape + verts.shape

    This ie memory inefficient!!
    """

    assert verts.ndim == 1
    assert values.ndim == x.ndim + 1
    assert values.shape == x.shape + verts.shape

    # easier to work with flattened arrays
    xshape = x.shape
    x = x.view(-1)  # shape (nx,)
    values = values.view(-1, len(verts))  # shape (nx, nv)
    ix = torch.arange(x.numel())  # shape (nx,)

    # idx.shape: (nx,)
    idx = torch.searchsorted(verts, x, side="right")
    torch.clip(idx, 1, len(verts)-1, out=idx)

    # ds.shape: (nx,)
    ds = x - verts[idx-1]
    torch.divide(ds, h[idx-1], out=ds)

    # dval.shape: (nx,)
    low = values[(ix, idx.view(-1)-1)]
    dval = values[(ix, idx)] -  low

    # interpolate; shape: (nx,)
    torch.multiply(dval, ds, out=dval)
    torch.add(dval, low, out=dval)

    return dval.view(xshape)  # return to the original shape of x


@torch.jit.unused
def interpnd(x: Tensor, verts: Sequence[Tensor]|torch.nn.ParameterList, values: Tensor):
    """Piecewise multilinear interpolation to multiple points in N-D space.

    This uses the concept the 1D shape function in finie element methods for hypercubes.
    It's mathematically equivalent to piecewise multilinear interpolation but with a
    simpler mathematical description for an easier implementation.

    Notes
    -----
    * `len(verts) == values.ndim == N`
    * `verts[i].ndim == 1` for all i
    * `values.shape == (len(_) of _ in verts)`
    """

    # extract info (so values must be a torch.Tensor)
    device = x.device
    N = len(verts)  # number of dimensions
    K = [len(v) for v in verts]  # number of vertices in each dimension
    coeff = 2.0**N  # dividing coefficient for the shape functions

    # we still want to work with shape (..., N) even for N=1, i.e., 1D
    if N == 1 and x.shape[-1] != 1:
        x = x.view(x.shape+(1,))

    # hypercube indices (note the N is the leading dimension for convenience later)
    ids = torch.zeros((N,)+x.shape[:-1], dtype=torch.long, device=device)

    # local coordinates (in [-1, 1]^N) for points in their hypercubes
    local = torch.zeros(x.shape[:-1]+(N,), device=device)

    for i in range(N):  # loop over each dimension

        # aliases for readability
        vi = verts[i]

        # identify the hypercube's index in each dimension the points belong to
        tmp = torch.searchsorted(vi, x[..., i], side="right") - 1
        ids[i, ...] = torch.clip(tmp, 0, K[i]-2)
        tmp = None

        # calculate the local coordinates for points in their hypercubes
        tmp = (x[..., i] - vi[ids[i, ...]]) / (vi[ids[i, ...]+1] - vi[ids[i, ...]])
        local[..., i] = tmp * 2.0 - 1.0  # shift to [-1, 1]
        tmp = None

    # build shape functions
    out = torch.zeros(x.shape[:-1], device=x.device)

    # loop each shape function
    for key in itertools.product([0, 1], repeat=N):

        # node corresponds to this shape function for all points in x
        node = torch.asarray(key, dtype=torch.long, device=device)
        node = ids + node.view((N,)+(1,)*(x.ndim-1))

        # shape function's values at all points in x
        shapevals = torch.ones(x.shape[:-1], device=x.device)
        for k in range(N):
            shapevals = shapevals * (1.0 - local[..., k] * (-1)**key[k])

        torch.multiply(shapevals, values[tuple(node)], out=shapevals)
        torch.divide(shapevals, coeff, out=shapevals)  # type: ignore
        torch.add(out, shapevals, out=out)

    return out  # should have shape x.shape[:-1]


@torch.jit.script
def getcdf(pdfvals: Tensor, h: Tensor) -> tuple[Tensor, Tensor]:
    """Get the normalized CDF and the normalization factor.

    Mostly for internal use.
    """

    two = torch.tensor(2.0, dtype=pdfvals.dtype, device=pdfvals.device)

    vals = torch.zeros_like(pdfvals)
    torch.add(pdfvals[..., 1:], pdfvals[..., :-1], out=vals[..., 1:])
    torch.multiply(vals[..., 1:], h, out=vals[..., 1:])
    torch.divide(vals[..., 1:], two, out=vals[..., 1:])
    torch.cumsum(vals, -1, out=vals)
    norm = vals[..., -1].clone()
    torch.divide(vals, norm.view(vals.shape[:-1]+(1,)), out=vals)

    return vals, norm


@torch.jit.unused
def getconditionals(
    pdfvals: Tensor, h: Sequence[Tensor]|torch.nn.ParameterList,
    cpdfs: Sequence[Tensor]|torch.nn.ParameterList|None = None,
    ccdfs: Sequence[Tensor]|torch.nn.ParameterList|None = None,
) -> tuple[
    Sequence[Tensor]|torch.nn.ParameterList,
    Sequence[Tensor]|torch.nn.ParameterList
]:
    """Get the 1D conditional PDF and CDF values at vertices for N-D distributions.

    This function assume `pdfvals` has the shape to the background rectilinear grid.

    Arguments
    ---------
    pdfvals : Tensor
        Discrete joint PDF values at vertices.
    dx : Sequence[Tensor]
        Cell sizes along each conditional direction.
    cpdfs : Sequence[Tensor], optional
        All conditional (normalized) PDFs. If None, allocate new memory space.
    ccdfs : Sequence[Tensor], optional
        All conditional (normalized) CDFs. If None, allocate new memory space.

    Returns
    -------
    cpdfs : Sequence[Tensor]
        All conditional (normalized) PDFs.
    ccdfs : Sequence[Tensor]
        All conditional (normalized) CDFs.

    Notes
    -----
    * `pdfvals.ndim = len(dx)`
    * `pdfvals.shape = [len(dxi) for dxi in dx]`
    * `len(pdfs) = len(cdfs) = len(dx) = pdfvals.ndim`
    * `pdfs[i].shape = cdfs[i].shape = pdfvals.shape` for i=0, 1, ..., pdfvals.ndim-1
    """

    # aliases for our convenience
    N = pdfvals.ndim  # number of dimensions
    K = pdfvals.shape  # number of vertices in each dimension

    # do lists work with torch.jit.script?
    if cpdfs is None:
        cpdfs = [torch.zeros_like(pdfvals) for _ in range(N)]

    if ccdfs is None:
        ccdfs = [torch.zeros_like(pdfvals) for _ in range(N)]

    # loop over each 1D conditional direction
    for i in range(N):

        low = [slice(None) for _ in range(N)]
        low[i] = slice(None, -1)
        low = tuple(low)

        high = [slice(None) for _ in range(N)]
        high[i] = slice(1, None)
        high = tuple(high)

        bcast = [None for _ in range(N)]
        bcast[i] = slice(None)  # type: ignore
        bcast = tuple(bcast)

        normloc = [slice(None) for _ in range(N)]
        normloc[i] = slice(-1, None)
        normloc = tuple(normloc)

        ccdfs[i][low] = 0.0
        torch.add(pdfvals[low], pdfvals[high], out=ccdfs[i][high])
        torch.multiply(ccdfs[i][high], h[i][bcast], out=ccdfs[i][high])
        torch.divide(ccdfs[i][high], 2.0, out=ccdfs[i][high])  # type: ignore
        torch.cumsum(ccdfs[i], i, out=ccdfs[i])

        # workaround for pytorch, which needs a hard copy....
        norms = ccdfs[i][normloc].clone()

        # append conditional PDF
        cpdfs[i][...] = pdfvals / norms

        # reuse memory space for conditional CDF and append it
        torch.divide(ccdfs[i], norms, out=ccdfs[i])

    return cpdfs, ccdfs


@torch.jit.unused
def centraldiff(vals: Tensor, dh: Tensor, axis: int, out=None):
    """Central difference for 1st-order derivatives.
    """

    if out is None:
        out = torch.zeros_like(vals)

    N = vals.ndim

    # for broadcasting 1D arrays to N-D compatible
    bcast = tuple(-1 if _ == axis else 1 for _ in range(N))

    # vertices where central difference is applicable
    k = tuple(slice(1, -1) if _ == axis else slice(None) for _ in range(N))
    kp = tuple(slice(2, None) if _ == axis else slice(None) for _ in range(N))
    km = tuple(slice(None, -2) if _ == axis else slice(None) for _ in range(N))

    # coefficients for central difference
    div = dh[1:] * dh[:-1] * (dh[1:] + dh[:-1])
    A = - dh[1:]**2 / div  # for k-1-th terms
    B = (dh[1:] - dh[:-1]) * (dh[1:] + dh[:-1]) / div  # for k-th terms
    C = dh[:-1]**2 / div  # for k+1-th terms

    # broadcasting
    A = A.view(bcast)
    B = B.view(bcast)
    C = C.view(bcast)

    # loop over each 1D conditional direction and apply central difference
    out[k] = A * vals[km] + B * vals[k] + C * vals[kp]

    return out


@torch.jit.unused
def forwarddiff(vals: Tensor, dh: Tensor, axis: int, out=None):
    """Second-order accurate forward difference for 1st-order derivatives.
    """

    if out is None:
        out = torch.zeros_like(vals)

    N = vals.ndim

    # for broadcasting 1D arrays to N-D compatible
    bcast = tuple(-1 if _ == axis else 1 for _ in range(N))

    # vertices where forward difference is applicable
    k = tuple(slice(0, 1) if _ == axis else slice(None) for _ in range(N))
    kp1 = tuple(slice(1, 2) if _ == axis else slice(None) for _ in range(N))
    kp2 = tuple(slice(2, 3) if _ == axis else slice(None) for _ in range(N))

    # coefficients for forward difference
    div = dh[0] * dh[1] * (dh[0] + dh[1])
    A = - dh[1] * (2.0 * dh[0] + dh[1]) / div  # for k-th terms
    B = (dh[0] + dh[1])**2 / div  # for k+1-th terms
    C = - dh[0]**2 / div  # for k+2-th terms

    # broadcasting
    A = A.view(bcast)
    B = B.view(bcast)
    C = C.view(bcast)

    # loop over each 1D conditional direction and apply forward difference
    out[k] = A * vals[k] + B * vals[kp1] + C * vals[kp2]

    return out


@torch.jit.unused
def backwarddiff(vals: Tensor, dh: Tensor, axis: int, out=None):
    """Second-order accurate backward difference for 1st-order derivatives.
    """

    if out is None:
        out = torch.zeros_like(vals)

    N = vals.ndim

    # for broadcasting 1D arrays to N-D compatible
    bcast = tuple(-1 if _ == axis else 1 for _ in range(N))

    # vertices where backward difference is applicable
    k = tuple(slice(-1, None) if _ == axis else slice(None) for _ in range(N))
    km1 = tuple(slice(-2, -1) if _ == axis else slice(None) for _ in range(N))
    km2 = tuple(slice(-3, -2) if _ == axis else slice(None) for _ in range(N))

    # coefficients for backward difference
    div = dh[-1] * dh[-2] * (dh[-1] + dh[-2])
    A = dh[-2] * (2.0 * dh[-1] + dh[-2]) / div  # for k-th terms
    B = - (dh[-1] + dh[-2])**2 / div  # for k-1-th terms
    C = dh[-1]**2 / div  # for k-2-th terms

    # broadcasting
    A = A.view(bcast)
    B = B.view(bcast)
    C = C.view(bcast)

    # loop over each 1D conditional direction and apply backward difference
    out[k] = A * vals[k] + B * vals[km1] + C * vals[km2]

    return out
