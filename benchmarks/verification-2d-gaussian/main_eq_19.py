#!/usr/bin/env python3
# vim:fenc=utf-8

"""Verification case using 2D Gaussian distribution.

`params` is defined as (mu_1, mu_2, sigma_1, sigma_2, rho).
"""
import pathlib
import itertools
import math
import matplotlib
import torch
import numpy
import scipy.integrate
from matplotlib import pyplot
from matplotlib import colors as mcolors
from matplotlib import cm as mcm
from matplotlib import ticker as mticker
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
    "figure.constrained_layout.use": True,
    "lines.linewidth": 2.0,
    "image.cmap": "turbo",
    "savefig.dpi": 192,
    "savefig.format": "pdf",
    "savefig.bbox": "tight"
})

# line styles
linestyles = ["dashdot", "dashed", (0, (1, 1))]


jactags = {
    (0, 0): r"$\partial x_1 / \partial \mu_1$",
    (0, 1): r"$\partial x_1 / \partial \mu_2$",
    (0, 2): r"$\partial x_1 / \partial \sigma_1$",
    (0, 3): r"$\partial x_1 / \partial \sigma_2$",
    (0, 4): r"$\partial x_1 / \partial \rho$",
    (1, 0): r"$\partial x_2 / \partial \mu_1$",
    (1, 1): r"$\partial x_2 / \partial \mu_2$",
    (1, 2): r"$\partial x_2 / \partial \sigma_1$",
    (1, 3): r"$\partial x_2 / \partial \sigma_2$",
    (1, 4): r"$\partial x_2 / \partial \rho$",
}


def ans(x, params):
    """Analytical sensitivity for the 1D Gaussian distribution.
    """
    x = torch.asarray(x)
    assert x.shape[-1] == 2
    out = torch.zeros(x.shape+(5,), dtype=x.dtype, device=x.device)

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


def pdf(x, params):
    """Probability density function for 2D Gaussian distribution.
    """
    mu_1, mu_2, sigma_1, sigma_2, rho = params

    mean = torch.tensor([mu_1, mu_2], dtype=params.dtype, device=params.device)

    covmtx = torch.tensor([
        [sigma_1**2, rho*sigma_1*sigma_2],
        [rho*sigma_1*sigma_2, sigma_2**2]
    ], dtype=params.dtype, device=params.device)

    dist = torch.distributions.multivariate_normal.MultivariateNormal(mean, covmtx)

    x = torch.asarray(x)
    assert x.shape[-1] == 2

    return torch.exp(dist.log_prob(x))


def get_ctf_args(params):
    """Return the configurations for contourf plots.
    """

    _, _, sigma1, sigma2, rho = params.detach().cpu().numpy()
    coeff1 = sigma1 / (1 - rho * rho)
    coeff2 = sigma2 / (1 - rho * rho)

    zeroscale = numpy.logspace(-3, -1, 3)
    zeroscale = numpy.concatenate([-zeroscale[::-1], zeroscale])

    ctf_kwargs = {
        (0, 0): dict(
            levels=1.0+zeroscale,
            norm=mcolors.BoundaryNorm(1.0+zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        ),
        (0, 1): dict(
            levels=zeroscale,
            norm=mcolors.BoundaryNorm(zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        ),
        (0, 2): dict(
            levels=numpy.linspace(-5, 5, 64),
            norm=mcolors.Normalize(vmin=-5.0, vmax=5.0),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        ),
        (0, 3): dict(
            levels=zeroscale,
            norm=mcolors.BoundaryNorm(zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        ),
        (0, 4): dict(
            levels=numpy.linspace(-coeff1*5.0, coeff1*5.0, 64),
            norm=mcolors.Normalize(-coeff1*5.0, coeff1*5.0, clip=False),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        ),
        (1, 0): dict(
            levels=zeroscale,
            norm=mcolors.BoundaryNorm(zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        ),
        (1, 1): dict(
            levels=1.0+zeroscale,
            norm=mcolors.BoundaryNorm(1.0+zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        ),
        (1, 2): dict(
            levels=zeroscale,
            norm=mcolors.BoundaryNorm(zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        ),
        (1, 3): dict(
            levels=numpy.linspace(-5, 5, 64),
            norm=mcolors.Normalize(vmin=-5.0, vmax=5.0),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        ),
        (1, 4): dict(
            levels=numpy.linspace(-coeff2*5.0, coeff2*5.0, 64),
            norm=mcolors.Normalize(-coeff2*5.0, coeff2*5.0, clip=False),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w", under="w", over="w"),
            extend="both"
        )
    }

    oneformatter = \
        lambda x, pos: rf"${'-' if x-1 <= 0 else '+'}10^{{{int(numpy.log10(abs(x-1))-0.4)}}}$"
    zeroformatter = \
        lambda x, pos: rf"${'-' if x <= 0 else '+'}10^{{{int(numpy.log10(abs(x))-0.4)}}}$"
    otherformatter = \
        lambda x, pos: rf"${x: 4.1f}$"

    cbar_kwargs = {}
    for key in ctf_kwargs.keys():
        cbar_kwargs.setdefault(key, {})["mappable"] = mcm.ScalarMappable(
            norm=ctf_kwargs[key]["norm"], cmap=ctf_kwargs[key]["cmap"]
        )
        cbar_kwargs[key]["orientation"] = "horizontal"
        cbar_kwargs[key]["extend"] = ctf_kwargs[key]["extend"]

    cbar_kwargs[(0, 0)]["format"] = mticker.FuncFormatter(oneformatter)
    cbar_kwargs[(0, 0)]["format"].set_offset_string("+1")

    cbar_kwargs[(0, 1)]["format"] = mticker.FuncFormatter(zeroformatter)
    cbar_kwargs[(0, 1)]["format"].set_offset_string("  ")

    cbar_kwargs[(0, 3)]["format"] = mticker.FuncFormatter(zeroformatter)
    cbar_kwargs[(0, 3)]["format"].set_offset_string("  ")

    cbar_kwargs[(1, 0)]["format"] = mticker.FuncFormatter(zeroformatter)
    cbar_kwargs[(1, 0)]["format"].set_offset_string("  ")

    cbar_kwargs[(1, 1)]["format"] = mticker.FuncFormatter(oneformatter)
    cbar_kwargs[(1, 1)]["format"].set_offset_string("+1")

    cbar_kwargs[(1, 2)]["format"] = mticker.FuncFormatter(zeroformatter)
    cbar_kwargs[(1, 2)]["format"].set_offset_string("  ")

    return ctf_kwargs, cbar_kwargs


def get_sensitivity(nvs, bounds, params, eps, calculators):
    """Generates figures for visualizing the sensitivities.
    """

    # unpack data for our convenience
    xmin1, xmax1 = bounds[0]
    xmin2, xmax2 = bounds[1]
    device = params.device
    ftype = params.dtype

    # grid lines
    grid = [
        torch.linspace(xmin1, xmax1, nvs[0], dtype=ftype, device=device),
        torch.linspace(xmin2, xmax2, nvs[1], dtype=ftype, device=device),
    ]

    # meshgrid
    v = torch.stack(torch.meshgrid(grid, indexing="ij"), -1)
    vx = v[..., 0]
    vy = v[..., 1]

    # loop over algorithms
    outs = {}
    for key, cls in calculators.items():
        calculator = cls(pdf, grid, params, eps)
        outs[key] = calculator(v)  # sensitivity at vertices

    # theoretical values
    outs["ans"] = ans(v, params)

    # move everything to CPU as a numpy array for plotting
    vx = vx.detach().cpu().numpy()
    vy = vy.detach().cpu().numpy()
    outs = {key: vals.detach().cpu().numpy() for key, vals in outs.items()}

    return vx, vy, outs


def plot_sensitivity(params, vx, vy, outs, labels, figdir):
    """Plot the sensitivities.
    """

    # colormap normalizer
    ctfargs, cbarargs = get_ctf_args(params)

    # plotting
    for key, vals in outs.items():
        for ij in itertools.product(range(2), range(5)):
            i, j = ij
            fig = pyplot.figure(figsize=(2.5, 3.0))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            ax.contourf(vx, vy, vals[..., i, j], **ctfargs[ij])
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(jactags[ij])
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            fig.colorbar(**cbarargs[ij], ax=ax)
            fig.savefig(figdir.joinpath(f"{key}_{i}_{j}"))

    return


def plot_dist(params, bounds, figdir):
    """Plot the 2D Gaussian distribution.
    """

    # unpack data for our convenience
    xmin1, xmax1 = bounds[0]
    xmin2, xmax2 = bounds[1]
    device = params.device
    ftype = params.dtype

    # grid lines
    grid = [
        torch.linspace(xmin1, xmax1, 1024+1, dtype=ftype, device=device),
        torch.linspace(xmin2, xmax2, 1024+1, dtype=ftype, device=device),
    ]

    # meshgrid
    v = torch.stack(torch.meshgrid(grid, indexing="ij"), -1)
    vx = v[..., 0]
    vy = v[..., 1]

    # probability density values
    vals = pdf(v, params)

    # contourf configurations
    ctfargs = dict(
        levels=numpy.linspace(0.0, 0.1, 64),
        norm=mcolors.Normalize(vmin=vals.min().item(), vmax=vals.max().item()),
        cmap=pyplot.get_cmap("turbo"),
        extend="neither"
    )

    # colorbar configurations
    cbarargs = dict(
        format=mticker.ScalarFormatter(),
        mappable=mcm.ScalarMappable(ctfargs["norm"], ctfargs["cmap"]),  # type: ignore
        orientation="horizontal",
        extend="neither"
    )

    # move everything to CPU as a numpy array for plotting
    vx = vx.detach().cpu().numpy()
    vy = vy.detach().cpu().numpy()
    vals = vals.detach().cpu().numpy()

    # plot
    fig = pyplot.figure(figsize=(2.5, 3.0))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.contourf(vx, vy, vals, **ctfargs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    fig.colorbar(**cbarargs, ax=ax)  # type: ignore
    fig.savefig(figdir.joinpath(f"probability_density"))

    return


if __name__ == "__main__":

    # torch device
    device = "cpu"  # torch.device("cuda")
    ftype = torch.float64

    # ground truth
    mu1 = 0.7
    mu2 = -1.1
    sigma1 = 2.6
    sigma2 = 1.3
    rho = 0.678

    # finite difference step size
    eps = 1e-6

    # bounds
    xmin1, xmin2 = mu1 - 5.0 * sigma1, mu2 - 5.0 * sigma2
    xmax1, xmax2 = mu1 + 5.0 * sigma1, mu2 + 5.0 * sigma2
    bounds = torch.tensor([[xmin1, xmax1], [xmin2, xmax2] ], dtype=ftype, device=device)

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # ground truth (mu1, mu2, sigma1, sigma2, rho)
    truth = torch.asarray([mu1, mu2, sigma1, sigma2, rho], dtype=ftype, device=device)

    # labels for the each algorithm in plots
    labels = {
        "alg-3": "Subroutine 3",
        "alg-4": "Subroutine 4",
        "alg-6": "Subroutine 6",
        "alg-7": "Subroutine 7",
    }

    # gradient calculator for each algorithm
    calculators = {
        "alg-3": SensitivityND,
        # "alg-6": SensitivityNDInterp,
    }

    # # first plot the distribution as a reference
    # plot_dist(truth, bounds, figdir)

    vx, vy, outs = get_sensitivity([128+1, 128+1], bounds, truth, eps, calculators)
    plot_sensitivity(truth, vx, vy, outs, labels, figdir)
    # pyplot.show()
