#!/usr/bin/env python3
# vim:fenc=utf-8

"""1D Gaussian verification.
"""
import sys
import itertools
import time
import matplotlib.pyplot as pyplot
import distrosa

# default plotting settings
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

# mapping between algorithm keys and implementations
algcls = {
    "alg-2": distrosa.Sensitivity1D,
    "alg-3": distrosa.SensitivityND,
    "alg-4": distrosa.SensitivityNDDiag,
    "alg-6": distrosa.SensitivityNDInterp,
    "alg-7": distrosa.SensitivityNDDiagInterp,
}

# mapping between algorithm keys and names in plots
alglbls = {
    "alg-2": "Subroutine 2",
    "alg-3": "Subroutine 3",
    "alg-4": "Subroutine 4",
    "alg-6": "Subroutine 6",
    "alg-7": "Subroutine 7",
}


def solution(x, params):
    """Analytical sensitivity for the 1D Gaussian distribution.
    """
    np = sys.modules[x.__class__.__module__]
    out = np.zeros(x.shape+(2,), dtype=x.dtype)
    out[..., 0] = 1.0
    out[..., 1] = (x - params[0]) / params[1]
    return out


def pdf(x, params):
    """Probability density function for 1D Gaussian distribution.
    """
    # infer whether this is numpy of cupy
    np = sys.modules[x.__class__.__module__]

    mu, sigma = params

    pdfs = np.zeros_like(x)
    np.subtract(x, mu, out=pdfs)
    np.divide(pdfs, sigma, out=pdfs)
    np.square(pdfs, out=pdfs)
    np.divide(pdfs, -2, out=pdfs)
    np.exp(pdfs, out=pdfs)
    np.divide(pdfs, sigma*np.sqrt(2*np.pi), out=pdfs)  # pyright: ignore

    return pdfs


def run(params, bounds, eps, nvs, algs, res=8192):
    """Run the sensitivity calculation using different algorithms and resolutions.
    """

    # infer whether this is numpy of cupy
    np = sys.modules[params.__class__.__module__]

    # all solutions and algorithm evaluate sensitivity on this grid
    x = np.linspace(bounds[0], bounds[1], res)

    outs = {}
    for (nv, alg) in itertools.product(nvs, algs):

        verts = np.linspace(bounds[0], bounds[1], nv)

        if alg == "alg-2":
            calculator = algcls[alg](pdf, verts, eps)
        else:
            calculator = algcls[alg](pdf, [verts,], eps)

        tbg = time.perf_counter_ns()
        out = calculator(x, params)

        try:
            np.cuda.Device().synchronize()
        except AttributeError:
            pass
        ted = time.perf_counter_ns()
        print(f"{(alg, nv)}, time: {(ted-tbg)/1e9} s")

        outs.setdefault(alg, {})[nv] = out

    # get answer
    ans = solution(x, params)
    pdfvals = pdf(x, params)

    # convert to numpy if cupy
    if x.__class__.__module__ == "cupy":
        x = x.get()
        ans = ans.get()
        pdfvals = pdfvals.get()
        for alg, data in outs.items():
            for res, out in data.items():
                outs[alg][res] = out.get()

    return x, ans, pdfvals, outs


def plot_sensitivity(x, ans, computed, figdir):
    """Plot the sensitivity results.
    """

    # name for derivatives
    names = [r"mu", r"sigma"]

    for i in range(2):

        # restart the line style iterator
        lscycler = itertools.cycle(linestyles)

        fig, ax = pyplot.subplots(1, 1, figsize=(2.5, 2.5))
        ax.plot(x, ans[..., i], label="Analytical", lw=2, color="k")

        for alg, data in computed.items():

            # line label and styles
            pltprops = dict(label=alglbls[alg], alpha=0.85, ls=next(lscycler))

            # only plot the highest background resolution
            ax.plot(x, data[max(data.keys())][..., i], **pltprops)  # type: ignore

        ax.set_xlabel(r"$x$")
        ax.set_ylabel(rf"$\partial x \slash \partial \{names[i]}$")
        ax.legend(loc=0)
        fig.savefig(figdir.joinpath(f"sensitivity_{names[i]}"))
        pyplot.close(fig)


def plot_errors(x, ans, pdfvals, computed, figdir):
    """Calculate and plot L1 errors.
    """

    # infer whether this is numpy of cupy
    np = sys.modules[x.__class__.__module__]

    errs = [{}, {}]
    nvs = {}
    for alg, dset in computed.items():
        for res, data in dset.items():
            out = np.abs(data - ans)
            out = out * pdfvals.reshape(-1, 1)
            out = np.trapezoid(out, x, axis=0)
            errs[0].setdefault(alg, []).append(out[0])
            errs[1].setdefault(alg, []).append(out[1])
            nvs.setdefault(alg, []).append(res)

    # tunes location for legends
    lgdcfg = [
        dict(loc="lower left", bbox_to_anchor=(0.001, 0.001)),
        dict(loc="center right", bbox_to_anchor=(0.999, 0.5))
    ]

    # name for derivatives
    names = [r"mu", r"sigma"]

    for i in range(2):

        # restart the line style iterator
        lscycler = itertools.cycle(linestyles)

        fig, ax = pyplot.subplots(1, 1, figsize=(2.5, 2.5))
        for alg, err in errs[i].items():
            pltprops = dict(label=alglbls[alg], ls=next(lscycler))
            ax.plot(nvs[alg], err, **pltprops)  # type: ignore

        ax.grid(True, which="both", zorder=-1, lw=0.5)
        ax.set_xscale("log")
        ax.set_xlabel(r"$N$")
        ax.set_yscale("log")
        ax.set_ylabel(rf"$L_1$ Error of $\partial x \slash \partial \{names[i]}$")

        # plot 1st order reference
        ox1, oy1 = ax.transAxes.transform((0.1, 1.05))  # axes -> display
        ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display -> data
        ox2, oy2 = ax.transAxes.transform((0.5, 1.05))  # axes -> display
        ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display -> data
        oy2 = ox1 * oy1 / ox2
        ax.plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
        ax.text(
            ox2*10**0.1, oy2, r"$\mathcal{O}(N^{-1})$", fontsize="x-small", ha="left",
            va="bottom", bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75)
        )

        # plot 2nd order reference
        ox1, oy1 = ax.transAxes.transform((0.1, 0.75))  # axes -> display
        ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display -> data
        ox2, oy2 = ax.transAxes.transform((0.5, 0.75))  # axes -> display
        ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display -> data
        oy2 = (ox1 / ox2)**2 * oy1
        ax.plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
        ax.text(
            ox1*3.16, oy1*(oy2/oy1)**0.5, r"$\mathcal{O}(N^{-2})$", fontsize="x-small",
            ha="right", va="top",
            bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75)
        )

        ax.legend(**lgdcfg[i])

        fig.savefig(figdir.joinpath(f"error_{names[i]}"))
        pyplot.close(fig)


if __name__ == "__main__":
    import pathlib

    try:
        import cupy
    except ImportError:
        import numpy as cupy

    # parameters of the Gaussian distribution
    mu = 2.175
    sigma = 1.371
    nsigma = 5.0
    bounds = cupy.array((mu-nsigma*sigma, mu+nsigma*sigma))
    eps = 1e-5

    # combine parameters to a vector and convert to torch tensors
    params = cupy.array([mu, sigma])

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # all background resolutions we want to check
    nvs = cupy.power(2, cupy.arange(4, 15)).tolist()

    # all algorithms we want to check
    algs = ["alg-2", "alg-3", "alg-4", "alg-6", "alg-7"]

    # get results
    x, ans, pdfvals, computed = run(params, bounds, eps, nvs, algs, nvs[-1])

    # plot
    plot_sensitivity(x, ans, computed, figdir)
    plot_errors(x, ans, pdfvals, computed, figdir)
