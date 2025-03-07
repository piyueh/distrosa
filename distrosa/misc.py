#!/usr/bin/env python3
# vim:fenc=utf-8

"""Miscellaneous functions/helpers.
"""
import numpy
import torch


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
