#!/usr/bin/env python3
# vim:fenc=utf-8

"""Miscellaneous functions/helpers.
"""
import numpy
import torch
import itertools


def get_cdf_1d(func, verts, params):
    """Get the CDF values at vertices of a 1D function.

    Arguments
    ---------
    func : callable, (x, params) -> value
        Parametric probability density function (PDF) in 1D. Potentially unnormalized.
        `func` must be capable of broadcasting so that `x` can be a N-D array of an
        arbitrary shape consisting of multiple space points. `params` is a flattened
        parameter vector, i.e., 1D array. The `value` corresponds to the PDF values at
        space points in `x`, so `value` have the same shape of `x`.
    verts : 1D array
        Vertices along a 1D gridline.
    params : 1D array
        Parameters of the PDF.

    Returns
    -------
    cdf : 1D array
        CDF values at the vertices.
    norm : scalar
        Normalization constant of the PDF if it's not normalized.
    """

    pdfs = func(verts, params)  # unnormalized PDF values at vertices

    # reuse `pdfs` to hold cdf values
    pdfs[1:] = (pdfs[:-1] + pdfs[1:]) * (verts[1:] - verts[:-1]) / 2.0
    pdfs[0] = 0.0  # the most left vertex always has CDF = 0
    pdfs = pdfs.cumsum(0)  # trapzoidal integration

    norm = pdfs[-1]  # normalization constant
    pdfs = pdfs / norm  # normalize the CDF (avoid in-place operation for PyTorch)

    return pdfs, norm  # note `pdfs` holds CDF values now


def get_conditionals(func, verts, params):
    """Get the 1D conditional PDF and CDF values at vertices for N-D distributions.

    Arguments
    ---------
    func : callable, (x, params) -> value
    verts : length-N list of 1D arrays
        Gridlines along each dimension.
    params : 1D array

    Returns
    -------
    pdfs : length-N list of ND arrays w/ shape (K_1, K_2, ..., K_N)
        `i`-th array in the list holds the conditional PDF values at all vertices.
    cdfs : length-N list of ND arrays w/ shape (K_1, K_2, ..., K_N)
        `i`-th array in the list holds the conditional CDF values at all vertices.

    Notes
    -----
    * `N` denotes the number of dimensions, which is inferred from `verts`'s length.
    * `K_i` for i=1,2,...,N denote the number of vertices along the `i`-th dimension.
      which are inferred from `verts[i]` for i=1,2,...,N.
    """

    verts = [torch.asarray(v) for v in verts]  # convert to PyTorch tensors
    params = torch.asarray(params)  # convert to PyTorch tensor
    N = len(verts)  # number of dimensions
    K = [len(v) for v in verts]  # number of vertices in each dimension

    # compute joint PDF values at vertices
    if N == 1:
        eta = func(verts[0], params)  # shape: (K_1,)
    else:
        coords = torch.meshgrid(*verts, indexing="ij")
        coords = torch.stack(coords, -1)  # shape: (K_1, K_2, ..., K_N, N)
        eta = func(coords, params)  # shape: (K_1, K_2, ..., K_N)

    pdfs = []
    cdfs = []

    # loop over each 1D conditional direction
    for i in range(N):

        # alias
        vi = verts[i]

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

        tmp = torch.zeros(K, dtype=params.dtype, device=params.device)
        tmp[high] = (eta[low] + eta[high]) * (vi[1:] - vi[:-1])[bcast] / 2.0
        tmp = torch.cumsum(tmp, i)

        pdfs.append(eta/tmp[normloc])
        cdfs.append(tmp/tmp[normloc])

    return pdfs, cdfs


def interp_1d(x, verts, values):
    """Piecewise linear interpolation to multiple points on a 1D gridline.

    Notes
    -----
    * Points outside the gridline will be interpolated from either the 1st or the last
      piece.
    * This function assumes the inputs are already PyTorch tensors. No sanity check.

    Arguments
    ---------
    x : 1D array
        Points to be interpolated.
    verts : 1D array
        Vertices along a 1D gridline.
    values : 1D array
        Values at the vertices.

    Returns
    -------
    interp : 1D array
        Interpolated values at `x`.
    """

    # identify the upper bound vertex's index for the piece that contains each point
    idx = torch.searchsorted(verts, x, side="right")

    # idx=0, meaning x < verts[0]; then interpolated it from the 1st piece
    # idx=len(verts), meaning x >= verts[-1]; then interpolated from the last piece
    idx = torch.clamp(idx, 1, len(verts)-1)

    # aliases for readability
    dx = verts[idx] - verts[idx-1]
    dval = values[idx] - values[idx-1]
    ds = x - verts[idx-1]

    # interpolate
    interp = values[idx-1] + dval * ds / dx

    return interp


def interp_nd(x, verts, values):
    """Piecewise linear interpolation to multiple points in N-D space.
    """

    # aliases
    N = len(verts)  # number of dimensions
    K = [len(v) for v in verts]  # number of vertices in each dimension
    coeff = torch.asarray(2**N, dtype=values.dtype, device=values.device)

    # make it a torch tensor of shape (..., N)
    x = torch.asarray(x)

    if N == 1 and x.shape[-1] != 1:  # only automatically cast the shape for 1D
        x = x.unsqueeze(-1)

    assert x.shape[-1] == N  # other dimension is users' responsibility to ensure this

    # hypercube indices (note the N is the leading dimension for convenience later)
    ids = torch.zeros((N,)+x.shape[:-1], dtype=torch.int64, device=values.device)

    # local coordinates (in [-1, 1]^N) for points in their hypercubes
    local = torch.zeros(x.shape[:-1]+(N,), dtype=x.dtype, device=values.device)

    for i in range(N):
        # aliases for readability
        vi = verts[i]

        # identify the hypercube's index in each dimension the points belong to
        tmp = torch.searchsorted(vi, x[..., i], side="right") - 1
        ids[i, ...] = torch.clamp(tmp, 0, K[i]-2)

        # calculate the local coordinates for points in their hypercubes
        tmp = (x[..., i] - vi[ids[i, ...]]) / (vi[ids[i, ...]+1] - vi[ids[i, ...]])
        local[..., i] = tmp * 2.0 - 1.0  # shift to [-1, 1]

    # build shape functions
    out = 0
    for key in itertools.product([0, 1], repeat=N):

        # node corresponds to this shape function for all points in x
        node = torch.asarray(key, dtype=torch.int64, device=values.device)
        node = ids + node.view((N,)+tuple(1 for _ in range(x.ndim-1)))

        # shape function's values at all points in x
        shapevals = 1.0  # will have shape x.shape[:-1] later
        for k in range(N):
            shapevals = shapevals * (1.0 - local[..., k] * (-1)**key[k])

        out = out + shapevals * values[tuple(node)] / coeff

    return out  # should have shape x.shape[:-1]
