#!/usr/bin/env python3
# vim:fenc=utf-8

"""Verification case using 2D Gaussian with diagonal approximation.

`params` is defined as (mu_1, mu_2, sigma_1, sigma_2, rho).
"""
import time
import itertools
import torch
import distrosa
from distrosa.utils.gaussian_2d_sampler import gaussian_2d_pdf


# let the default floating precision to be 64bit
torch.set_default_dtype(torch.float64)

# device
device = "cuda" if torch.cuda.is_available() else "cpu"


# mapping between algorithm keys and implementations
algcls = {
    "alg-3": distrosa.SensitivityND,
    "alg-4": distrosa.SensitivityNDDiag,
    "alg-6": distrosa.SensitivityNDInterp,
    "alg-7": distrosa.SensitivityNDDiagInterp,
}


def run(params, bounds, eps, nvs, algs, res):
    """Generates figures for visualizing the sensitivities.
    """

    # all solutions and algorithm evaluate sensitivity on this grid
    x = torch.stack(
        torch.meshgrid(
            torch.linspace(bounds[0][0], bounds[0][1], res),
            torch.linspace(bounds[1][0], bounds[1][1], res),
            indexing="ij"
        ),
        -1
    ).to(device)

    # make sure other tensors are on the same device
    params = params.to(device)
    bounds = bounds.to(device)

    outs = {}
    times = {}
    for (nv, alg) in itertools.product(nvs, algs):

        # background grid to discretize the distribution
        verts = [torch.linspace(_[0], _[1], nv) for _ in bounds]

        # gradient/sensitivity calculator
        grader = algcls[alg](len(params), verts, eps, gaussian_2d_pdf).to(device)

        # timer
        torch.cuda.synchronize()
        tbg = time.perf_counter_ns()

        # evaluate the sensitivity at x
        with torch.inference_mode():
            out = grader(x, params)

        # timer
        torch.cuda.synchronize()
        ted = time.perf_counter_ns()

        print(f"({alg}, ({nv}x{nv})), time: {(ted-tbg)/1e9} s")

        outs.setdefault(alg, []).append(out.cpu())
        times.setdefault(alg, []).append((ted-tbg)/1e9)

        out = None

    # get answer
    ans = solution(x, params)
    pdfvals = gaussian_2d_pdf(x, params)

    return x.cpu(), ans.cpu(), pdfvals.cpu(), outs, times


def get_errors(x, ans, pdfvals, computed):
    """Calculate the errors.
    """

    print("calculating errors")

    errs = {}
    conv = {}
    for alg, dset in computed.items():
        for data in dset:

            err = torch.abs(data - ans)  # -> (res, res, 2, 5)
            errs.setdefault(alg, []).append(err)

            err = err * pdfvals.view(*pdfvals.shape, 1, 1)  # -> (res, res, 2, 5)
            err = torch.trapezoid(err, x[..., 0].view(*x.shape[:2], 1, 1), dim=0)
            err = torch.trapezoid(err, x[0, :, 1].view(-1, 1, 1), dim=0)

            conv.setdefault(alg, []).append(err)

    return errs, conv


def solution(x, params):
    """Analytical sensitivity for the 2D Gaussian via diagonal approximation.
    """
    out = torch.zeros(x.shape+(5,), dtype=x.dtype, device=x.device)

    mu_1, mu_2, sigma_1, sigma_2, rho = params

    z1 = (x[..., 0] - mu_1) / sigma_1
    z2 = (x[..., 1] - mu_2) / sigma_2

    out[..., 0, 0] = 1.0
    out[..., 0, 1] = - rho * sigma_1 / sigma_2
    out[..., 0, 2] = z1
    out[..., 0, 3] = - rho * z2 * sigma_1 / sigma_2
    out[..., 0, 4] = - sigma_1 * (rho * z1 - z2) / (1.0 - rho * rho)

    out[..., 1, 0] = - rho * sigma_2 / sigma_1
    out[..., 1, 1] = 1.0
    out[..., 1, 2] = - rho * z1 * sigma_2 / sigma_1
    out[..., 1, 3] = z2
    out[..., 1, 4] = sigma_2 * (z1 - rho * z2) / (1.0 - rho * rho)

    return out


if __name__ == "__main__":
    import pathlib

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # ground truth
    mu1 = 0.7
    mu2 = -1.1
    sigma1 = 2.6
    sigma2 = 1.3
    rho = 0.678
    eps = 1e-6
    nsigma = 5.0  # domain = mu +- nsigma * sigma
    res = 2048  # number of locations to evaluate the sensitivity

    # make tensors
    params = torch.tensor([mu1, mu2, sigma1, sigma2, rho]).to(device)

    bounds = torch.tensor((
        (mu1-nsigma*sigma1, mu1+nsigma*sigma1),
        (mu2-nsigma*sigma2, mu2+nsigma*sigma2)
    )).to(device)

    # all background resolutions we want to check (use the same resolution in x and y)
    nvs = torch.pow(2, torch.arange(5, 12)).tolist()

    # all algorithms we want to check
    algs = ["alg-4", "alg-7"]

    # get results
    x, ans, pdfvals, outs, times = run(params, bounds, eps, nvs, algs, res)

    # get errors
    errs, convs = get_errors(x, ans, pdfvals, outs)

    # save to a file
    torch.save({
        "x": x, "ans": ans, "pdfvals": pdfvals, "nvs": nvs, "params": params.cpu(),
        "algs": algs, "bounds": bounds.cpu(), "outs": outs, "times": times,
        "errs": errs, "convs": convs
    }, figdir.joinpath("diagapprox.dat"))
