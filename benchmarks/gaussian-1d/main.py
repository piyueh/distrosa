#!/usr/bin/env python3
# vim:fenc=utf-8

"""1D Gaussian verification."""

import time
import itertools
import torch
import distrosa
from distrosa.utils.gaussian_1d_sampler import gaussian_1d_pdf


# let the default floating precision to be 64bit
torch.set_default_dtype(torch.float64)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"

# mapping between algorithm keys and implementations
algcls = {
    "alg-2": distrosa.Sensitivity1D,
    "alg-3": distrosa.SensitivityND,
    "alg-4": distrosa.SensitivityNDDiag,
    "alg-6": distrosa.SensitivityNDInterp,
    "alg-7": distrosa.SensitivityNDDiagInterp,
}


def run(params, bounds, eps, nvs, algs, res=8192):
    """Run the sensitivity calculation using different algorithms and resolutions."""

    # all solutions and algorithm evaluate sensitivity on this grid
    x = torch.linspace(bounds[0], bounds[1], res).to(device)

    # make sure other tensors are on the same device
    params = params.to(device)
    bounds = bounds.to(device)

    outs = {}
    for nv, alg in itertools.product(nvs, algs):
        # background grid to discretize the distribution
        verts = torch.linspace(bounds[0], bounds[1], nv).to(device)

        # note that `verts` needs to be a list
        grader = algcls[alg](
            len(params),
            [
                verts,
            ],
            eps,
            gaussian_1d_pdf,
        ).to(device)

        # timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        tbg = time.perf_counter_ns()

        # evaluate the sensitivity at x
        with torch.no_grad():
            out = grader(x, params)

        # timer
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ted = time.perf_counter_ns()

        print(f"{(alg, nv)}, time: {(ted - tbg) / 1e9} s")

        outs.setdefault(alg, {})[nv] = out.cpu()

    # get answer
    ans = solution(x, params)
    pdfvals = gaussian_1d_pdf(x, params)

    return x.cpu(), ans.cpu(), pdfvals.cpu(), outs


def solution(x, params):
    """Analytical sensitivity for the 1D Gaussian distribution."""
    out = torch.zeros(x.shape + (2,), dtype=x.dtype, device=x.device)
    out[..., 0] = 1.0
    out[..., 1] = (x - params[0]) / params[1]
    return out


if __name__ == "__main__":
    import pathlib

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # parameters (of the Gaussian and other stuff)
    mu = 2.175
    sigma = 1.371
    eps = 1e-5
    nsigma = 5.0  # domain = mu +- nsigma * sigma
    res = 16384  # number of locations to evaluate the sensitivity

    # make tensors
    params = torch.tensor([mu, sigma]).to(device)
    bounds = torch.tensor((mu - nsigma * sigma, mu + nsigma * sigma)).to(device)

    # all background resolutions we want to investigate
    nvs = torch.pow(2, torch.arange(4, 15)).tolist()

    # all algorithms we want to check
    algs = ["alg-2", "alg-3", "alg-4", "alg-6", "alg-7"]

    # get results (all outputs should already be on CPU)
    x, ans, pdfvals, computed = run(params, bounds, eps, nvs, algs, res)

    # save to a file
    torch.save(
        {
            "x": x,
            "ans": ans,
            "pdfvals": pdfvals,
            "computed": computed,
            "nvs": nvs,
            "algs": algs,
            "params": params.cpu(),
            "bounds": bounds.cpu(),
            "eps": eps,
        },
        figdir.joinpath("results.dat"),
    )
