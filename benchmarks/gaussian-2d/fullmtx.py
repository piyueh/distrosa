#!/usr/bin/env python3
# vim:fenc=utf-8

"""Verification case using 2D Gaussian distribution.

`params` is defined as (mu_1, mu_2, sigma_1, sigma_2, rho).
"""
import sys
import time
import itertools
import numpy
import matplotlib.pyplot as pyplot
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import matplotlib.ticker as mticker
import distrosa
import distrosa.dists


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

# mapping between algorithm keys and implementations
algcls = {
    "alg-3": distrosa.SensitivityND,
    "alg-4": distrosa.SensitivityNDDiag,
    "alg-6": distrosa.SensitivityNDInterp,
    "alg-7": distrosa.SensitivityNDDiagInterp,
}

# mapping between algorithm keys and names in plots
alglbls = {
    "alg-3": "Subroutine 3",
    "alg-4": "Subroutine 4",
    "alg-6": "Subroutine 6",
    "alg-7": "Subroutine 7",
}

# dependent variable names
vnames = ["x_1", "x_2"]

# parameter names
pnames = ["mu_1", "mu_2", "sigma_1", "sigma_2", "rho"]


def solution(x, params):
    """Analytical sensitivity for the 1D Gaussian distribution.
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


def get_ctf_args(params):
    """Return the configurations for contourf plots.
    """

    # infer whether this is numpy of cupy
    np = sys.modules[params.__class__.__module__]

    _, _, sigma1, sigma2, rho = params
    coeff1 = sigma1 / (1 - rho * rho)
    coeff2 = sigma2 / (1 - rho * rho)

    zeroscale = np.logspace(-3, -1, 3)
    zeroscale = np.concatenate([-zeroscale[::-1], zeroscale])

    ctf_kwargs = {
        (0, 0): dict(
            levels=1.0+zeroscale,
            norm=mcolors.BoundaryNorm(1.0+zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (0, 1): dict(
            levels=zeroscale,
            norm=mcolors.BoundaryNorm(zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (0, 2): dict(
            levels=numpy.linspace(-nsigma, nsigma, 64),
            norm=mcolors.Normalize(vmin=-nsigma, vmax=nsigma),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (0, 3): dict(
            levels=zeroscale,
            norm=mcolors.BoundaryNorm(zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (0, 4): dict(
            levels=numpy.linspace(-coeff1*nsigma, coeff1*nsigma, 64),
            norm=mcolors.Normalize(-coeff1*nsigma, coeff1*nsigma, clip=False),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 0): dict(
            levels=zeroscale,
            norm=mcolors.BoundaryNorm(zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 1): dict(
            levels=1.0+zeroscale,
            norm=mcolors.BoundaryNorm(1.0+zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 2): dict(
            levels=zeroscale,
            norm=mcolors.BoundaryNorm(zeroscale, 256),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 3): dict(
            levels=numpy.linspace(-nsigma, nsigma, 64),
            norm=mcolors.Normalize(vmin=-nsigma, vmax=nsigma),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 4): dict(
            levels=numpy.linspace(-coeff2*nsigma, coeff2*nsigma, 64),
            norm=mcolors.Normalize(-coeff2*nsigma, coeff2*nsigma, clip=False),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        )
    }

    def oneformatter(x, pos):
        return rf"${'-' if x-1 <= 0 else '+'}10^{{{int(np.log10(abs(x-1))-0.4)}}}$"

    def zeroformatter(x, pos):
        return rf"${'-' if x <= 0 else '+'}10^{{{int(np.log10(abs(x))-0.4)}}}$"

    def otherformatter(x, pos):
        return rf"${x: 4.1f}$"

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
    dist = distrosa.dists.Gaussian2D(params.__class__.__module__)

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


def plot_dist(x, vals, figdir):
    """Plot the 2D Gaussian distribution.
    """

    # contourf configurations
    ctfargs = dict(
        levels=64,
        norm=mcolors.Normalize(vmin=vals.min(), vmax=vals.max()),
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

    print("plotting pdf")

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


def plot_sensitivity(x, ans, params, computed, figdir):
    """Plot the sensitivities.
    """

    # colormap normalizer
    ctfargs, cbarargs = get_ctf_args(params)

    # plotting
    for ij in itertools.product(range(2), range(5)):
        i, j = ij
        for alg, dset in computed.items():
            val = dset[-1]
            res = val.shape[0]
            print(f"plotting {alg}-{res}x{res}-({i}, {j})")

            fig = pyplot.figure(figsize=(2.5, 3.0))
            gs = fig.add_gridspec(1, 1)
            ax = fig.add_subplot(gs[0, 0])
            ax.contourf(x[..., 0], x[..., 1], val[..., i, j], **ctfargs[ij])
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel(r"$x_1$")
            ax.set_ylabel(r"$x_2$")
            fig.colorbar(**cbarargs[ij], ax=ax)
            fig.savefig(figdir.joinpath(f"{alg}_{i}_{j}_{res}x{res}"))
            pyplot.close(fig)

    # plot theoretical solutions
    for ij in itertools.product(range(2), range(5)):
        i, j = ij
        print(f"plotting ans-({i}, {j})")

        fig = pyplot.figure(figsize=(2.5, 3.0))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        ax.contourf(x[..., 0], x[..., 1], ans[..., i, j], **ctfargs[ij])
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.colorbar(**cbarargs[ij], ax=ax)
        fig.savefig(figdir.joinpath(f"ans_{i}_{j}_{x.shape[0]}x{x.shape[1]}"))
        pyplot.close(fig)

    return


def plot_errors(x, errors, figdir):
    """Plot the sensitivities.
    """

    # plotting
    for key in itertools.product(errors.items(), range(2), range(5)):

        (alg, dset), i, j = key
        res = dset[-1].shape[0]
        err = dset[-1][..., i, j]
        print(f"plotting error {alg}-{res}x{res}-({i}, {j})")

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
        fig.savefig(figdir.joinpath(f"error_{alg}_{i}_{j}_{res}x{res}"))

        pyplot.close(fig)

    return


def plot_convergences(nvs, convs, figdir):
    """Plot the error convergence.
    """

    for ij in itertools.product(range(2), range(5)):

        i, j = ij
        lscycler = itertools.cycle(linestyles)

        print(f"plotting convergence ({i}, {j})")

        fig = pyplot.figure(figsize=(2.5, 2.5))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])

        for alg, dset in convs.items():
            ls = next(lscycler)
            ax.plot(nvs, dset[:, i, j], label=alglbls[alg], ls=ls)

        ax.grid(True, which="both", zorder=-1, lw=0.5)

        ax.set_xscale("log")
        ax.set_xlabel(r"$N$")
        ax.set_yscale("log")
        ax.set_ylabel(rf"$L_1$ Error of $\partial {vnames[i]}/\partial \{pnames[j]}$")

        # plot 1st order reference
        ox1, oy1 = ax.transAxes.transform((0.1, 1.05))  # axes -> display
        ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display -> data
        ox2, oy2 = ax.transAxes.transform((0.5, 1.05))  # axes -> display
        ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display -> data
        oy2 = ox1 * oy1 / ox2
        ax.plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
        ax.text(
            ox2*10**0.1, oy2/10**0.2, r"$\mathcal{O}(N^{-1})$", fontsize="x-small",
            ha="right", va="top",
            bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75)
        )

        # plot 2nd order reference
        ox1, oy1 = ax.transAxes.transform((0.1, 0.75))  # axes -> display
        ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display -> data
        ox2, oy2 = ax.transAxes.transform((0.5, 0.75))  # axes -> display
        ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display -> data
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


if __name__ == "__main__":
    import pathlib
    import argparse

    try:
        import cupy
    except ImportError:
        import numpy as cupy

    parser = argparse.ArgumentParser()
    parser.add_argument("--new-run", action="store_true")
    args = parser.parse_args()

    # ground truth
    mu1 = 0.7
    mu2 = -1.1
    sigma1 = 2.6
    sigma2 = 1.3
    rho = 0.678
    eps = 1e-6

    # bounds
    xmin1, xmin2 = mu1 - nsigma * sigma1, mu2 - nsigma * sigma2
    xmax1, xmax2 = mu1 + nsigma * sigma1, mu2 + nsigma * sigma2
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
    if args.new_run:
        x, ans, pdfvals, nvs, outs, times = run(params, bounds, eps, nvs, algs, 2048)

        # get errors
        errs, convs = get_errors(x, ans, pdfvals, outs)

        if params.__class__.__module__ == "cupy":
            params = params.get()

        print("saving results")
        meta = dict(x=x, ans=ans, pdfvals=pdfvals, nvs=nvs, params=params)
        numpy.savez_compressed(figdir.joinpath("fullmtx.meta.npz"), **meta)
        numpy.savez_compressed(figdir.joinpath("fullmtx.outs.npz"), **outs)
        numpy.savez_compressed(figdir.joinpath("fullmtx.times.npz"), **times)
        numpy.savez_compressed(figdir.joinpath("fullmtx.errs.npz"), **errs)
        numpy.savez_compressed(figdir.joinpath("fullmtx.convs.npz"), **convs)
    else:
        print("loading results")
        with numpy.load(figdir.joinpath("fullmtx.meta.npz")) as dset:
            x, ans, pdfvals = dset["x"], dset["ans"], dset["pdfvals"]
            nvs, params = dset["nvs"], dset["params"]

        with numpy.load(figdir.joinpath("fullmtx.outs.npz")) as dset:
            outs = {alg: dset[alg] for alg in algs}

        with numpy.load(figdir.joinpath("fullmtx.errs.npz")) as dset:
            errs = {alg: dset[alg] for alg in algs}

        with numpy.load(figdir.joinpath("fullmtx.convs.npz")) as dset:
            convs = {alg: dset[alg] for alg in algs}

    # first plot the distribution as a reference
    plot_dist(x, pdfvals, figdir)

    # plot sensitivity for demo
    plot_sensitivity(x, ans, params, outs, figdir)

    # plot error contours
    plot_errors(x, errs, figdir)

    # plot error convergence
    plot_convergences(nvs, convs, figdir)
