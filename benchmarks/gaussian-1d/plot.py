#!/usr/bin/env python3
# vim:fenc=utf-8

"""1D Gaussian verification.
"""
import itertools
import matplotlib.pyplot as pyplot
import numpy
import torch

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
    "lines.linewidth": 1.5,
    "image.cmap": "turbo",
    "savefig.dpi": 768,
    "savefig.format": "png",
    "savefig.bbox": "tight"
})

# line styles
linestyles = ["dashdot", "dashed", (0, (1, 1))]

# mapping between algorithm keys and names in plots
alglbls = {
    "alg-2": "1D Alg",
    "alg-3": "Full Inv",
    "alg-4": "Diag Approx",
    "alg-6": "Interp Full",
    "alg-7": "Interp Diag",
}


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
    # these were saved as torch.tensors; now convert to numpy
    x = x.numpy()
    ans = ans.numpy()
    pdfvals = pdfvals.numpy()

    errs = [{}, {}]
    nvs = {}
    for alg, dset in computed.items():
        for res, data in dset.items():
            data = data.numpy()
            out = numpy.abs(data - ans)
            out = out * pdfvals.reshape(-1, 1)
            out = numpy.trapezoid(out, x, axis=0)
            errs[0].setdefault(alg, []).append(out[0])
            errs[1].setdefault(alg, []).append(out[1])
            nvs.setdefault(alg, []).append(res)

    # tunes location for legends
    lgdcfg = [
        dict(loc="lower left", bbox_to_anchor=(0.001, 0.001)),
        dict(loc="center right", bbox_to_anchor=(0.999, 0.45))
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

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # read in data
    dset = torch.load(figdir.joinpath("results.dat"))

    # plot
    plot_sensitivity(dset["x"], dset["ans"], dset["computed"], figdir)
    plot_errors(dset["x"], dset["ans"], dset["pdfvals"], dset["computed"], figdir)
