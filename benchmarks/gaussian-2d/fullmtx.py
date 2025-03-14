#!/usr/bin/env python3
# vim:fenc=utf-8

"""Verification case using 2D Gaussian with the full inverse matrix.

`params` is defined as (mu_1, mu_2, sigma_1, sigma_2, rho).
"""
import sys
import time
import itertools
import numpy
import distrosa
from dist import Gaussian2D


# mapping between algorithm keys and implementations
algcls = {
    "alg-3": distrosa.SensitivityND,
    "alg-4": distrosa.SensitivityNDDiag,
    "alg-6": distrosa.SensitivityNDInterp,
    "alg-7": distrosa.SensitivityNDDiagInterp,
}


def solution(x, params):
    """Analytical sensitivity for the 2D Gaussian via full inverse matrix.
    """
    np = sys.modules[x.__class__.__module__]
    out = np.zeros(x.shape+(5,), dtype=x.dtype)

    mu_1, mu_2, sigma_1, sigma_2, rho = params

    z1 = (x[..., 0] - mu_1) / sigma_1
    z2 = (x[..., 1] - mu_2) / sigma_2

    out[..., 0, 0] = 1.0
    out[..., 0, 1] = 0.0
    out[..., 0, 2] = z1
    out[..., 0, 3] = 0.0
    out[..., 0, 4] = sigma_1 * z2 / (1 - rho * rho)

    out[..., 1, 0] = 0.0
    out[..., 1, 1] = 1.0
    out[..., 1, 2] = 0.0
    out[..., 1, 3] = z2
    out[..., 1, 4] = sigma_2 * z1 / (1 - rho * rho)

    return out


def run(params, bounds, eps, nvs, algs, res=2048):
    """Generates figures for visualizing the sensitivities.
    """

    np = sys.modules[params.__class__.__module__]

    # all solutions and algorithm evaluate sensitivity on this grid
    x = np.stack(
        np.meshgrid(
            np.linspace(*bounds[0], res, dtype=params.dtype),
            np.linspace(*bounds[1], res, dtype=params.dtype),
            indexing="ij"
        ),
        axis=-1
    )

    # initialize a Gaussian 2D instance based on the backend
    dist = Gaussian2D(params.__class__.__module__)

    outs = {}
    times = {}
    for (nv, alg) in itertools.product(nvs, algs):

        verts = [np.linspace(_[0], _[1], nv) for _ in bounds]
        calculator = algcls[alg](dist.pdf, verts, eps)

        tbg = time.perf_counter_ns()
        out = calculator(x, params)

        try:
            np.cuda.Device().synchronize()
        except AttributeError:
            pass
        ted = time.perf_counter_ns()
        print(f"({alg}, ({nv}x{nv})), time: {(ted-tbg)/1e9} s")

        try:
            out = out.get()
        except AttributeError:
            pass

        outs.setdefault(alg, []).append(out)
        times.setdefault(alg, []).append((ted-tbg)/1e9)

    # get answer
    ans = solution(x, params)
    pdfvals = dist.pdf(x, params)

    if np.__name__ == "cupy":
        x = x.get()
        ans = ans.get()
        pdfvals = pdfvals.get()

    # merge list to arrays; now these data are for sure CPU arrays
    outs = {k: numpy.stack(v, 0) for k, v in outs.items()}
    times = {k: numpy.array(v) for k, v in times.items()}
    nvs = numpy.array(nvs)

    return x, ans, pdfvals, nvs, outs, times


def get_errors(x, ans, pdfvals, computed):
    """Calculate the errors.
    """

    print("calculating errors")

    # infer whether this is numpy of cupy
    np = sys.modules[x.__class__.__module__]
    assert np.__name__ == "numpy"  # theoretically, at this stage, data are in numpy

    # hack for the API name change
    if np.__name__ == "cupy":
        np.trapezoid = np.trapz

    errs = {}
    conv = {}
    for alg, dset in computed.items():
        for data in dset:

            err = np.abs(data - ans)  # -> (res, res, 2, 5)
            errs.setdefault(alg, []).append(err)

            err = err * pdfvals.reshape(*pdfvals.shape, 1, 1)  # -> (res, res, 2, 5)
            err = np.trapezoid(err, x[..., 0].reshape(*x.shape[:2], 1, 1), axis=0)
            err = np.trapezoid(err, x[0, :, 1].reshape(-1, 1, 1), axis=0)

            conv.setdefault(alg, []).append(err)

    # conver lists to arrays
    errs = {k: np.stack(v, 0) for k, v in errs.items()}
    conv = {k: np.stack(v, 0) for k, v in conv.items()}

    return errs, conv


if __name__ == "__main__":
    import pathlib

    try:
        import cupy
    except ImportError:
        import numpy as cupy

    # ground truth
    mu1 = 0.7
    mu2 = -1.1
    sigma1 = 2.6
    sigma2 = 1.3
    rho = 0.678
    eps = 1e-6

    # bounds
    xmin1, xmin2 = mu1 - 5. * sigma1, mu2 - 5. * sigma2
    xmax1, xmax2 = mu1 + 5. * sigma1, mu2 + 5. * sigma2
    bounds = cupy.array([[xmin1, xmax1], [xmin2, xmax2] ])

    # combine parameters to a vector and convert to torch tensors
    params = cupy.array([mu1, mu2, sigma1, sigma2, rho])

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # all background resolutions we want to check (use the same resolution in x and y)
    nvs = cupy.power(2, cupy.arange(5, 12)).tolist()

    # all algorithms we want to check
    algs = ["alg-3", "alg-6"]

    # get results
    x, ans, pdfvals, nvs, outs, times = run(params, bounds, eps, nvs, algs, 2048)

    # get errors
    errs, convs = get_errors(x, ans, pdfvals, outs)

    if params.__class__.__module__ == "cupy":
        params = params.get()

    print("saving results")
    meta = dict(x=x, ans=ans, pdfvals=pdfvals, nvs=nvs, params=params)
    numpy.savez(figdir.joinpath("fullmtx.meta.npz"), **meta)
    numpy.savez(figdir.joinpath("fullmtx.outs.npz"), **outs)
    numpy.savez(figdir.joinpath("fullmtx.times.npz"), **times)
    numpy.savez(figdir.joinpath("fullmtx.errs.npz"), **errs)
    numpy.savez(figdir.joinpath("fullmtx.convs.npz"), **convs)
