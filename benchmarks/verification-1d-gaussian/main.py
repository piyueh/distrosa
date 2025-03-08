#!/usr/bin/env python3
# vim:fenc=utf-8

"""Verification case using 1D Gaussian distribution.
"""
import pathlib
import itertools
import math
import torch
import numpy
import scipy.integrate
from matplotlib import pyplot
from cycler import cycler
from distrosa.gradcalc import Sensitivity1D
from distrosa.gradcalc import SensitivityND
from distrosa.gradcalc import SensitivityNDDiag
from distrosa.gradcalc import SensitivityNDInterp
from distrosa.gradcalc import SensitivityNDInterpDiag


pyplot.rcParams.update({
    "font.size": 8,  # default font size for normal text
    "axes.labelsize": "small",
    "axes.titlesize": "medium",
    "xtick.labelsize": "small",
    "ytick.labelsize": "small",
    "legend.fontsize": "small",
    "legend.title_fontsize": "small",
    "figure.dpi": 192,
    "figure.titlesize": "medium",
    "lines.linewidth": 2.0,
})

# line styles
linestyles = ["dashdot", "dashed", (0, (1, 1))]


def ans(x, params):
    """Analytical sensitivity for the 1D Gaussian distribution.
    """
    x = torch.asarray(x)
    out = torch.zeros(x.shape+(2,), dtype=x.dtype, device=x.device)
    out[..., 0] = 1.0
    out[..., 1] = (x - params[0]) / params[1]
    return out


def pdf(x, params):
    """Probability density function for 1D Gaussian distribution.
    """
    mu, sigma = params
    coeff_1 = - 2.0 * sigma * sigma
    coeff_2 = sigma * torch.sqrt(torch.asarray(2.0*math.pi))
    return torch.exp((x-mu)**2/coeff_1) / coeff_2


def cdf(x, params):
    """Cumulative distribution function for 1D Gaussian distribution.
    """
    mu, sigma = params
    coeff = torch.sqrt(torch.asarray(2.0)) * sigma
    return (1.0 + torch.special.erf((x-mu)/coeff)) / 2.0


def sensitivity_demo(nv, xmin, xmax, params, eps, calculators, labels, figdir):
    """Generates figures for visualizing the sensitivities.
    """

    # for plotting the sensitivity
    v = torch.linspace(xmin, xmax, nv+1, dtype=torch.float64, device="cpu")
    theo = ans(v, params)  # theoretical values

    # loop over algorithms
    outs = {}
    for key, cls in calculators.items():
        if key == "alg-2":
            calculator = cls(pdf, v, params, eps)
        else:
            calculator = cls(pdf, [v,], params, eps)

        outs[key] = calculator(v)  # sensitivity at vertices

    # plot \partial x / \partial \mu
    lscycler = itertools.cycle(linestyles)
    fig, ax = pyplot.subplots(1, 1, figsize=(2.5, 2.5), layout="constrained")
    ax.plot(v, theo[..., 0], label="Analytical", lw=2, color="k")
    for key, out in outs.items():
        ax.plot(v, out[..., 0], label=labels[key], alpha=0.85, ls=next(lscycler))
    ax.legend(loc=0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\partial x \slash \partial \mu$")
    fig.savefig(figdir.joinpath("sensitivity_mu.pdf"), dpi=192, bbox_inches="tight")

    # plot \partial x / \partial \sigma
    lscycler = itertools.cycle(linestyles)
    fig, ax = pyplot.subplots(1, 1, figsize=(2.5, 2.5), layout="constrained")
    ax.plot(v, theo[..., 1], label="Analytical", lw=2, color="k")
    for key, out in outs.items():
        ax.plot(v, out[..., 1], label=labels[key], alpha=0.85, ls=next(lscycler))
    ax.legend(loc=0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$\partial x \slash \partial \sigma$")
    fig.savefig(figdir.joinpath("sensitivity_sigma.pdf"), dpi=192, bbox_inches="tight")


def error_estimation(nvs, xmin, xmax, params, eps, calculators, labels, figdir):
    """Generates figures for error convergence.
    """

    errs_0 = {}
    errs_1 = {}

    for key, cls in calculators.items():

        for nv in nvs:

            print(nv)

            v = torch.linspace(xmin, xmax, nv+1, dtype=torch.float64, device="cpu")

            if key == "alg-2":
                calculator = cls(pdf, v, params, eps)
            else:
                calculator = cls(pdf, [v,], params, eps)

            mu, sigma = params
            xint = torch.linspace(
                mu-3.0*sigma, mu+3.0*sigma, nvs[-1]+1, dtype=torch.float64, device="cpu"
            )
            dx = 6.0 * sigma / nvs[-1]
            vals1 = calculator(xint)
            vals2 = ans(xint, params)
            err = torch.abs(vals1-vals2)
            w = pdf(xint, params)
            errs_0.setdefault(key, []).append(scipy.integrate.romb(err[:, 0]*w, dx))
            errs_1.setdefault(key, []).append(scipy.integrate.romb(err[:, 1]*w, dx))

    lscycler = itertools.cycle(linestyles)
    fig, ax = pyplot.subplots(1, 1, figsize=(2.5, 2.5), layout="constrained")
    for key, err in errs_0.items():
        ax.plot(nvs, err, label=labels[key], ls=next(lscycler))
    ax.grid(True, which="both", zorder=-1, lw=0.5)
    ax.legend(loc=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"Number of Vertices")
    ax.set_yscale("log")
    ax.set_ylabel(r"$L_1$ Error of $\partial x \slash \partial \mu$")
    fig.savefig(figdir.joinpath("error_mu.pdf"), dpi=192, bbox_inches="tight")

    lscycler = itertools.cycle(linestyles)
    fig, ax = pyplot.subplots(1, 1, figsize=(2.5, 2.5), layout="constrained")
    for key, err in errs_1.items():
        ax.plot(nvs, err, label=labels[key], ls=next(lscycler))
    ax.grid(True, which="both", zorder=-1, lw=0.5)
    ax.legend(loc=0)
    ax.set_xscale("log")
    ax.set_xlabel(r"Number of Vertices")
    ax.set_yscale("log")
    ax.set_ylabel(r"$L_1$ Error of $\partial x \slash \partial \sigma$")
    fig.savefig(figdir.joinpath("error_sigma.pdf"), dpi=192, bbox_inches="tight")


if __name__ == "__main__":

    # parameters of the Gaussian distribution
    mu = 2.175
    sigma = 1.371
    xmin = mu - 5.0 * sigma
    xmax = mu + 5.0 * sigma
    eps = 1e-5

    # combine parameters to a vector and convert to torch tensors
    params = torch.asarray([mu, sigma], dtype=torch.float64, device="cpu")

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # labels for the each algorithm in plots
    labels = {
        "alg-2": "Subroutine 2",
        "alg-3": "Subroutine 3",
        "alg-4": "Subroutine 4",
        "alg-6": "Subroutine 6",
        "alg-7": "Subroutine 7",
    }

    # gradient calculator for each algorithm
    calculators = {
        "alg-2": Sensitivity1D,
        "alg-3": SensitivityND,
        "alg-4": SensitivityNDDiag,
        "alg-6": SensitivityNDInterp,
        "alg-7": SensitivityNDInterpDiag,
    }

    # visualize the sensitivities from all algorithms
    sensitivity_demo(1024, xmin, xmax, params, eps, calculators, labels, figdir)

    # different background resolutions
    nvs = numpy.power(2, numpy.arange(5, 14))
    error_estimation(nvs, xmin, xmax, params, eps, calculators, labels, figdir)
    pyplot.show()
