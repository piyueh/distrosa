#!/usr/bin/env python3
# vim:fenc=utf-8

"""Verification case using 2D Gaussian distribution.

`params` is defined as (mu_1, mu_2, sigma_1, sigma_2, rho).
"""
import argparse
import pathlib
import time
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
    "figure.dpi": 768,
    "figure.titlesize": "medium",
    "figure.constrained_layout.use": True,
    "lines.linewidth": 2.0,
    "image.cmap": "turbo",
    "savefig.dpi": 768,
    "savefig.format": "png",
    "savefig.bbox": "tight"
})

# line styles
linestyles = ["dashdot", "dashed", (0, (1, 1))]

# nsigma: number of sigma we want in our domain
nsigma = 5.0

# torch device
device = "cuda" if torch.cuda.is_available() else "cpu"
ftype = torch.float64


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


@torch.jit.script
def pdf(x: torch.Tensor, params: torch.Tensor):
    """Probability density function for 2D Gaussian distribution.
    """
    sigma_1 = params[2]
    sigma_2 = params[3]
    rho = params[4]

    z = (x - params[:2]) / params[2:4]

    out = (
        torch.exp(
            -
            (z[..., 0]**2 - 2 * rho * z[..., 0] * z[..., 1] + z[..., 1]**2)
            /
            (2*(1-rho**2))
        )
        /
        (2 * math.pi * sigma_1 * sigma_2 * torch.sqrt(1-rho**2))
    )

    return out


def get_ctf_args(params):
    """Return the configurations for contourf plots.
    """

    _, _, sigma1, sigma2, rho = params.detach().cpu().numpy()
    coeff1 = sigma1 / (1 - rho * rho)
    coeff2 = sigma2 / (1 - rho * rho)

    zeroscale = numpy.logspace(-3, -1, 3)
    zeroscale = numpy.concatenate([-zeroscale[::-1], zeroscale])

    ans01 = - rho * sigma1 / sigma2
    ans03 = nsigma * rho * sigma1 / sigma2
    ans04_l = - sigma1 * (rho * nsigma + nsigma) / (1 - rho * rho)
    ans04_h = sigma1 * (rho * nsigma + nsigma) / (1 - rho * rho)
    ans10 = - rho * sigma2 / sigma1
    ans12 = nsigma * rho * sigma2 / sigma1
    ans14_l = - sigma2 * (nsigma + rho * nsigma) / (1 - rho * rho)
    ans14_h = sigma2 * (nsigma + rho * nsigma) / (1 - rho * rho)

    ctf_kwargs = {
        (0, 0): dict(  # ans: 1.0
            levels=1.0+zeroscale,
            norm=mcolors.BoundaryNorm(1.0+zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (0, 1): dict(  # ans: -rho * sigma1 / sigma2
            levels=ans01+zeroscale,
            norm=mcolors.BoundaryNorm(ans01+zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (0, 2): dict(  # ans: z1
            levels=numpy.linspace(-nsigma, nsigma, 64),
            norm=mcolors.Normalize(vmin=-nsigma, vmax=nsigma),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (0, 3): dict(  # ans: -rho * z2 * sigma1 / sigma2
            levels=numpy.linspace(-ans03, ans03, 64),
            norm=mcolors.Normalize(-ans03, ans03),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (0, 4): dict(  # ans: -sigma1 * (rho * z1 - z2) / (1 - rho * rho)
            levels=numpy.linspace(ans04_l, ans04_h, 64),
            norm=mcolors.Normalize(ans04_l, ans04_h),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 0): dict(  # ans: -rho * sigma2 / sigma1
            levels=ans10+zeroscale,
            norm=mcolors.BoundaryNorm(ans10+zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 1): dict(  # ans: 1.0
            levels=1.0+zeroscale,
            norm=mcolors.BoundaryNorm(1.0+zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 2): dict(  # ans: -rho * z1 * sigma2 / sigma1
            levels=numpy.linspace(-ans12, ans12, 64),
            norm=mcolors.Normalize(-ans12, ans12),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 3): dict(  # ans: z2
            levels=numpy.linspace(-nsigma, nsigma, 64),
            norm=mcolors.Normalize(vmin=-nsigma, vmax=nsigma),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 4): dict(
            levels=numpy.linspace(ans14_l, ans14_h, 64),
            norm=mcolors.Normalize(ans14_l, ans14_h),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        )
    }

    oneformatter = \
        lambda x, pos: rf"${'-' if x-1 <= 0 else '+'}10^{{{int(numpy.log10(abs(x-1))-0.4)}}}$"
    zeroformatter = \
        lambda x, pos: rf"${'-' if x <= 0 else '+'}10^{{{int(numpy.log10(abs(x))-0.4)}}}$"
    otherformatter = \
        lambda x, pos: rf"${x: 4.1f}$"

    ans01formatter = \
        lambda x, pos: rf"${'-' if x-1 <= 0 else '+'}10^{{{int(numpy.log10(abs(x-ans01))-0.4)}}}$"

    ans10formatter = \
        lambda x, pos: rf"${'-' if x-1 <= 0 else '+'}10^{{{int(numpy.log10(abs(x-ans10))-0.4)}}}$"

    cbar_kwargs = {}
    for key in ctf_kwargs.keys():
        cbar_kwargs.setdefault(key, {})["mappable"] = mcm.ScalarMappable(
            norm=ctf_kwargs[key]["norm"], cmap=ctf_kwargs[key]["cmap"]
        )
        cbar_kwargs[key]["orientation"] = "horizontal"
        cbar_kwargs[key]["extend"] = ctf_kwargs[key]["extend"]

    cbar_kwargs[(0, 0)]["format"] = mticker.FuncFormatter(oneformatter)
    cbar_kwargs[(0, 0)]["format"].set_offset_string("+1")

    cbar_kwargs[(0, 1)]["format"] = mticker.FuncFormatter(ans01formatter)
    cbar_kwargs[(0, 1)]["format"].set_offset_string(f"{ans01:.2f}")

    cbar_kwargs[(1, 0)]["format"] = mticker.FuncFormatter(ans10formatter)
    cbar_kwargs[(1, 0)]["format"].set_offset_string(f"{ans10:.2f}")

    cbar_kwargs[(1, 1)]["format"] = mticker.FuncFormatter(oneformatter)
    cbar_kwargs[(1, 1)]["format"].set_offset_string("+1")

    return ctf_kwargs, cbar_kwargs


def get_sensitivity(nvs, bounds, params, eps, calculators):
    """Generates figures for visualizing the sensitivities.
    """

    # unpack data for our convenience
    xmin1, xmax1 = bounds[0]
    xmin2, xmax2 = bounds[1]
    device = params.device
    ftype = params.dtype

    # where to evaluate the senesitivity, always 2048 x 2048
    x = torch.stack(
        torch.meshgrid(
            torch.linspace(xmin1, xmax1, 2048, dtype=ftype, device=device),
            torch.linspace(xmin2, xmax2, 2048, dtype=ftype, device=device),
            indexing="ij"
        ),
        -1
    )

    # output dictionary
    outs = {"x": x.detach().cpu()}

    # theoretical values
    outs["ans"] = ans(x, params).detach().cpu()

    # other information
    outs["bounds"] = bounds.detach().cpu()
    outs["params"] = params.detach().cpu()
    outs["eps"] = eps

    # loop over different resolutions
    for nv in nvs:

        # background gridlines
        grid = [
            torch.linspace(xmin1, xmax1, nv[0]+1, dtype=ftype, device=device),
            torch.linspace(xmin2, xmax2, nv[1]+1, dtype=ftype, device=device),
        ]

        # loop over algorithms
        for key, cls in calculators.items():
            print(f"Running {key}-{nv}")
            torch.cuda.synchronize()
            tbg = time.perf_counter_ns()
            calculator = cls(pdf, grid, params, eps)
            out = calculator(x)  # sensitivity at the 2048 x 2048 grid
            torch.cuda.synchronize()
            ted = time.perf_counter_ns()
            print(f"Time taken: {(ted-tbg)/1e9:.2f} s")

            # store the resolutions
            outs.setdefault(nv, {})[key] = out.detach().cpu()  # type: ignore

    return outs


def get_errors(outs):
    """Calculate the errors.
    """

    params = outs["params"].to(device)
    bounds = outs["bounds"].to(device)
    ans = outs["ans"].to(device)
    x = outs["x"].to(device)

    pdfs = pdf(x, params)  # (Nx, Ny)

    converge = {}
    errors = {"x": x}

    # loop over different background grid resolutions
    for nv, data in outs.items():

        if nv in ["bounds", "params", "eps", "x", "ans"]:
            continue

        # loop over algorithms
        for alg, val in data.items():

            # absolute error
            err = torch.abs(val.to(device) - ans)  # (Nx, Ny, 2, 5)

            # save nv and errors for contour plots
            errors.setdefault(nv, {})[alg] = err.detach().cpu()

            # expectation
            err = err * pdfs.view(pdfs.shape+(1, 1))  # (Nx, Ny, 2, 5)

            # integrate over x; resulting shape (Ny, 2, 5)
            err = torch.trapezoid(
                err, x[..., 0].view(x.shape[:2]+(1, 1)).expand(-1, -1, 2, 5), dim=0)

            # integrate over y; resulting shape (2, 5)
            err = torch.trapezoid(
                err, x[-1, :, 1].view(x.shape[1], 1, 1).expand(-1, 2, 5), dim=0)

            converge.setdefault(alg, {}).setdefault("vals", []).append(err.detach().cpu())
            converge.setdefault(alg, {}).setdefault("nv", []).append(nv[0]*nv[1])

    # convert to numpy arrays
    for alg, data in converge.items():
        # overwrite the original list
        converge[alg]["vals"] = numpy.stack(data["vals"], -1)
        converge[alg]["nv"] = numpy.array(data["nv"])

    return errors, converge


def plot_sensitivity(outs, labels, figdir, res):
    """Plot the sensitivities.
    """

    # retrieve the grid
    x = outs["x"].cpu().numpy()

    # colormap normalizer
    ctfargs, cbarargs = get_ctf_args(outs["params"])

    # plotting
    for alg, vals in outs[res].items():

        for ij in itertools.product(range(2), range(5)):

            i, j = ij

            fig = pyplot.figure(figsize=(2.5, 3.0))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            ax.contourf(x[..., 0], x[..., 1], vals[..., i, j].numpy(), **ctfargs[ij])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            fig.colorbar(**cbarargs[ij], ax=ax)
            fig.savefig(figdir.joinpath(f"{alg}_{i}_{j}_{res[0]}x{res[1]}"))

            pyplot.close(fig)

    # plot theoretical solutions
    vals = outs["ans"].cpu().numpy()
    for ij in itertools.product(range(2), range(5)):

        i, j = ij

        fig = pyplot.figure(figsize=(2.5, 3.0))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.contourf(x[..., 0], x[..., 1], vals[..., i, j], **ctfargs[ij])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.colorbar(**cbarargs[ij], ax=ax)
        fig.savefig(figdir.joinpath(f"ans_{i}_{j}_{res[0]}x{res[1]}"))

        pyplot.close(fig)

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
        torch.linspace(xmin1, xmax1, 2048+1, dtype=ftype, device=device),
        torch.linspace(xmin2, xmax2, 2048+1, dtype=ftype, device=device),
    ]

    # meshgrid
    x = torch.stack(torch.meshgrid(grid, indexing="ij"), -1)

    # probability density values
    vals = pdf(x, params)

    # contourf configurations
    ctfargs = dict(
        levels=64,
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
    x = x.detach().cpu().numpy()
    vals = vals.detach().cpu().numpy()

    # plot
    fig = pyplot.figure(figsize=(2.5, 3.0))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])
    ax.contourf(x[..., 0], x[..., 1], vals, **ctfargs)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    fig.colorbar(**cbarargs, ax=ax)  # type: ignore
    fig.savefig(figdir.joinpath(f"probability_density"))

    pyplot.close(fig)
    return


def plot_error_converges(converge, labels, figdir):
    """Plot the errors.
    """

    for ij in itertools.product(range(2), range(5)):

        i, j = ij
        lscycler = itertools.cycle(linestyles)

        fig = pyplot.figure(figsize=(2.5, 2.5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        for alg, data in converge.items():
            ls = next(lscycler)
            ax.plot(numpy.sqrt(data["nv"]), data["vals"][i, j, :], label=labels[alg], ls=ls)

        ax.grid(True, which="both", zorder=-1, lw=0.5)

        ax.set_xscale("log")
        ax.set_xlabel(r"$N$")
        ax.set_yscale("log")
        ax.set_ylabel(rf"$L_1$ Error of {jactags[(i, j)]}")

        # plot 1st order reference
        ox1, oy1 = ax.transAxes.transform((0.1, 1.05))  # axes to display coordinates
        ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display to data coordinates
        ox2, oy2 = ax.transAxes.transform((0.5, 1.05))  # axes to display coordinates
        ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display to data coordinates
        oy2 = ox1 * oy1 / ox2
        ax.plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
        ax.text(
            ox2*10**0.1, oy2/10**0.2, r"$\mathcal{O}(N^{-1})$", fontsize="x-small", ha="right",
            va="top", bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75)
        )

        # plot 2nd order reference
        ox1, oy1 = ax.transAxes.transform((0.1, 0.75))  # axes to display coordinates
        ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display to data coordinates
        ox2, oy2 = ax.transAxes.transform((0.5, 0.75))  # axes to display coordinates
        ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display to data coordinates
        oy2 = (ox1 / ox2)**2 * oy1
        ax.plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
        ax.text(
            ox1*2.1, oy1*(oy2/oy1)**0.5, r"$\mathcal{O}(N^{-2})$", fontsize="x-small",
            ha="right", va="top",
            bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75)
        )

        ax.legend(loc=0)

        fig.savefig(figdir.joinpath(f"converge_{i}_{j}"))

        pyplot.close(fig)


def plot_errors(errors, params, labels, figdir, res):
    """Plot the sensitivities.
    """

    # get the grid and resolution for 1024x1024
    x = errors["x"].cpu().numpy()

    # plotting
    for alg, vals in errors[res].items():

        for ij in itertools.product(range(2), range(5)):
            i, j = ij

            err = vals[..., i, j].numpy()

            # to avoid log(0)
            err = numpy.where(err < 1e-15, 1e-15, err)

            vmin = int(numpy.log10(err.min()))
            vmax = int(numpy.log10(err.max()))

            # contourf configurations
            ctfargs = dict(
                levels=numpy.power(10., numpy.arange(vmin, vmax+1)),
                norm=mcolors.LogNorm(vmin=10**vmin, vmax=10**vmax),
                cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
                extend="both"
            )

            # colorbar configurations
            cbarargs = dict(
                mappable=mcm.ScalarMappable(ctfargs["norm"], ctfargs["cmap"]),  # type: ignore
                orientation="horizontal",
                extend="both"
            )

            fig = pyplot.figure(figsize=(2.5, 3.0))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            cf = ax.contourf(x[..., 0], x[..., 1], err, **ctfargs)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            fig.colorbar(ax=ax, **cbarargs)  # type: ignore
            fig.savefig(figdir.joinpath(f"error_{alg}_{i}_{j}_{res[0]}x{res[1]}"))

            pyplot.close(fig)

    return


if __name__ == "__main__":

    torch.set_num_threads(1)

    parser = argparse.ArgumentParser(description="2D Gaussian Verification")
    parser.add_argument("--new-run", action="store_true", help="Re-run calculations")
    args = parser.parse_args()

    # ground truth
    mu1 = 0.7
    mu2 = -1.1
    sigma1 = 2.6
    sigma2 = 1.3
    rho = 0.678

    # finite difference step size
    eps = 1e-6

    # grid resolutions to study
    nvs = [(int(_), int(_)) for _ in numpy.power(2, range(5, 12))]

    # bounds
    xmin1, xmin2 = mu1 - nsigma * sigma1, mu2 - nsigma * sigma2
    xmax1, xmax2 = mu1 + nsigma * sigma1, mu2 + nsigma * sigma2
    bounds = torch.tensor([[xmin1, xmax1], [xmin2, xmax2] ], dtype=ftype, device=device)

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # ground truth (mu1, mu2, sigma1, sigma2, rho)
    truth = torch.asarray([mu1, mu2, sigma1, sigma2, rho], dtype=ftype, device=device)

    # labels for the each algorithm in plots
    labels = {
        "alg-4": "Subroutine 4",
        "alg-7": "Subroutine 7",
    }

    # gradient calculator for each algorithm
    calculators = {
        "alg-4": SensitivityNDDiag,
        "alg-7": SensitivityNDInterpDiag
    }

    # get sensitivity
    if args.new_run:
        outs = get_sensitivity(nvs, bounds, truth, eps, calculators)
        torch.save(outs, figdir.joinpath("eq_20_res.data"))
    else:
        outs = torch.load(figdir.joinpath("eq_20_res.data"))

    # get errors
    errors, converges = get_errors(outs)

    # plot sensitivity for demo
    # plot_sensitivity(outs, labels, figdir, (2048, 2048))

    # plot error contours
    plot_errors(errors, truth, labels, figdir, (2048, 2048))

    # plot error convergence
    plot_error_converges(converges, labels, figdir)

    # first plot the distribution as a reference
    plot_dist(truth, bounds, figdir)
