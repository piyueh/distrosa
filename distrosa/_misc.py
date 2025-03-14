#!/usr/bin/env python3
# vim:fenc=utf-8

"""Miscellaneous functions/helpers.

This module is supposed to be used internally.
"""
import sys
import itertools
from typing import Callable
from typing import Sequence
from numpy.typing import NDArray


def getconditionals(
    func: Callable[[NDArray, NDArray], NDArray],
    verts: Sequence[NDArray],
    dx: Sequence[NDArray],
    params: NDArray,
):
    """Get the 1D conditional PDF and CDF values at vertices for N-D distributions.

    Arguments
    ---------
    func : Callable, (x: NDArray, params: NDArray) -> pdfvals: NDArray
        Parametric probability density function (PDF) in N-D. Potentially unnormalized.
        `pdfvals.shape` must be the same as `x.shape[:-1]`, and `x.shape[-1]` is the
        dimensionality (a.k.a., `N`).

    verts : Sequence[NDArray]
        Length-N sequence of 1D arrays for vertices along each dimension.

    dx : Sequence[NDArray]
        Length-N sequence of 1D arrays for cell size along each dimension. `len(dx[i])`
        should be `len(verts[i])-1`.

    params : NDArray
        Flattened parameters for `func`.

    Returns
    -------
    pdfs : Sequence[NDArray]
        Length-N list of N-D arrays. `pdfs[i]` holds the conditional PDF values in `i`
        direction at all vertices. Each ND array has shape `(K[0], K[1], ..., K[N])`,
        where `K[i] = len(verts[i])`.

    cdfs : Sequence[NDArray]
        Length-N list of N-D arrays. `cdfs[i]` holds the conditional CDF values in `i`
        direction at all vertices. Each ND array has shape `(K[0], K[1], ..., K[N])`,
        where `K[i] = len(verts[i])`.
    """

    # extract info (so params must already be a NDArray)
    _np = sys.modules[params.__class__.__module__]
    ftype = params.dtype
    device = params.device
    N = len(verts)  # number of dimensions
    K = [len(v) for v in verts]  # number of vertices in each dimension

    # compute joint PDF values at vertices
    if N == 1:
        eta = func(verts[0], params)  # shape: (K_1,)
    else:
        # shape: (K_1, K_2, ..., K_N)
        eta = func(_np.stack(_np.meshgrid(*verts, indexing="ij"), -1), params)

    pdfs = []
    cdfs = []

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

        tmp = _np.zeros(K, dtype=ftype)
        _np.add(eta[low], eta[high], out=tmp[high])
        _np.multiply(tmp[high], dx[i][bcast], out=tmp[high])
        _np.divide(tmp[high], 2.0, out=tmp[high])
        _np.cumsum(tmp, i, out=tmp)

        # append conditional PDF
        pdfs.append(eta/tmp[normloc])

        # reuse memory space for conditional CDF and append it
        _np.divide(tmp, tmp[normloc], out=tmp)
        cdfs.append(tmp)

    return pdfs, cdfs


def interpnd(x, verts, values):
    """Piecewise multilinear interpolation to multiple points in N-D space.

    This uses the concept the 1D shape function in finie element methods for hypercubes.
    It's mathematically equivalent to piecewise multilinear interpolation but with a
    simpler mathematical description for an easier implementation.
    """

    # extract info (so values must be a _np.ndarray)
    _np = sys.modules[values.__class__.__module__]
    ftype = values.dtype
    N = len(verts)  # number of dimensions
    K = [len(v) for v in verts]  # number of vertices in each dimension
    coeff = _np.asarray(2**N, dtype=ftype)

    # make it an array in case this is a scalar; no-op if already a ndarray
    x = _np.asarray(x, dtype=ftype)

    # we still want to work with shape (Nx, N) even for N=1, i.e., 1D
    if N == 1 and x.shape[-1] != 1:
        x = x.unsqueeze(-1)

    # for general N-D, it's users' responsibility to ensure this
    assert x.shape[-1] == N

    # hypercube indices (note the N is the leading dimension for convenience later)
    ids = _np.zeros((N,)+x.shape[:-1], dtype=_np.int64)

    # local coordinates (in [-1, 1]^N) for points in their hypercubes
    local = _np.zeros(x.shape[:-1]+(N,), dtype=ftype)

    for i in range(N):
        # aliases for readability
        vi = verts[i]

        # identify the hypercube's index in each dimension the points belong to
        tmp = _np.searchsorted(vi, x[..., i], side="right") - 1
        ids[i, ...] = _np.clip(tmp, 0, K[i]-2)

        # calculate the local coordinates for points in their hypercubes
        tmp = (x[..., i] - vi[ids[i, ...]]) / (vi[ids[i, ...]+1] - vi[ids[i, ...]])
        local[..., i] = tmp * 2.0 - 1.0  # shift to [-1, 1]

    # build shape functions
    out = 0
    for key in itertools.product([0, 1], repeat=N):

        # node corresponds to this shape function for all points in x
        node = _np.asarray(key, dtype=_np.int64)
        node = ids + node.reshape((N,)+tuple(1 for _ in range(x.ndim-1)))

        # shape function's values at all points in x
        shapevals = 1.0  # will have shape x.shape[:-1] later
        for k in range(N):
            shapevals = shapevals * (1.0 - local[..., k] * (-1)**key[k])

        out = out + shapevals * values[tuple(node)] / coeff

    return out  # should have shape x.shape[:-1]


def getcdf(
    density: Callable, x: NDArray, params: NDArray, h: NDArray
) -> tuple[NDArray, NDArray]:
    """Get the normalized CDF and the normalization factor.

    Mostly for internal use.
    """

    # determine if it's numpy or cupy using params
    _np = sys.modules[params.__class__.__module__]

    vals = density(x, params).reshape(x.shape[:-1])  # some 1D PDF return shape (Nx, 1)
    _np.add(vals[..., 1:], vals[..., :-1], out=vals[..., 1:])
    _np.multiply(vals[..., 1:], h, out=vals[..., 1:])
    _np.divide(vals[..., 1:], 2.0, out=vals[..., 1:])
    vals[..., 0] = 0.0
    _np.cumsum(vals, -1, out=vals)
    norm = vals[..., -1].copy()
    _np.divide(vals, norm.reshape(vals.shape[:-1]+(1,)), out=vals)

    return vals, norm


def minterp(x: NDArray, verts: NDArray, h: NDArray, values: NDArray):
    """Multiple 1D piecewise linear interpolation happening at the same time.

    Regardless the shape/dim of `x`, it is treated like a sequence of 1D coordinates
    in different shape. So `x` can have an arbitrary shape, but the dimension of `verts`
    and `values` must be as follows:
        * verts.ndim == 1
        * values.ndim = x.ndim + 1
        * values.shape == x.shape + verts.shape

    This ie memory inefficient!!
    """

    # determine if it's numpy or cupy using verts
    _np = sys.modules[verts.__class__.__module__]

    assert verts.ndim == 1
    assert values.ndim == x.ndim + 1
    assert values.shape == x.shape + verts.shape

    # easier to work with flattened arrays
    xshape = x.shape
    x = x.reshape(-1)  # shape (nx,)
    values = values.reshape(-1, len(verts))  # shape (nx, nv)
    ix = _np.arange(x.size)  # shape (nx,)

    # idx.shape: (nx,)
    idx = _np.searchsorted(verts, x, side="right")
    _np.clip(idx, 1, len(verts)-1, out=idx)

    # ds.shape: (nx,)
    ds = x - verts[idx-1]
    _np.divide(ds, h[idx-1], out=ds)

    # dval.shape: (nx,)
    low = values[(ix, idx.reshape(-1)-1)]
    dval = values[(ix, idx)] -  low

    # interpolate; shape: (nx,)
    _np.multiply(dval, ds, out=dval)
    _np.add(dval, low, out=dval)

    # reshape without copying; shape original x.shape
    return dval.reshape(xshape)


def centraldiff(vals, dh, axis, out=None):
    """Central difference for 1st-order derivatives.
    """

    if out is None:
        out = _np.zeros_like(vals)

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
    A = A.reshape(bcast)
    B = B.reshape(bcast)
    C = C.reshape(bcast)

    # loop over each 1D conditional direction and apply central difference
    out[k] = A * vals[km] + B * vals[k] + C * vals[kp]

    return out


def forwarddiff(vals, dh, axis, out=None):
    """Second-order accurate forward difference for 1st-order derivatives.
    """

    if out is None:
        out = _np.zeros_like(vals)

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
    A = A.reshape(bcast)
    B = B.reshape(bcast)
    C = C.reshape(bcast)

    # loop over each 1D conditional direction and apply forward difference
    out[k] = A * vals[k] + B * vals[kp1] + C * vals[kp2]

    return out


def backwarddiff(vals, dh, axis, out=None):
    """Second-order accurate backward difference for 1st-order derivatives.
    """

    if out is None:
        out = _np.zeros_like(vals)

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
    A = A.reshape(bcast)
    B = B.reshape(bcast)
    C = C.reshape(bcast)

    # loop over each 1D conditional direction and apply backward difference
    out[k] = A * vals[k] + B * vals[km1] + C * vals[km2]

    return out
