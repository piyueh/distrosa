#!/usr/bin/env python3
# vim:fenc=utf-8

"""Plot error convergences.
"""
import itertools
import numpy
import matplotlib.pyplot as pyplot


# line styles
linestyles = ["dashdot", "dashed", (0, (1, 1))]

# mapping between algorithm keys and names in plots
alglbls = {
    "alg-3": "Full Inv",
    "alg-4": "Diag Approx",
    "alg-6": "Interp Full",
    "alg-7": "Interp Diag",
}

# dependent variable names
vnames = ["x_1", "x_2"]

# parameter names
pnames = ["mu_1", "mu_2", "sigma_1", "sigma_2", "rho"]


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
            ox2/10**0.2, oy2*10**0.9, r"$\mathcal{O}(N^{-1})$", fontsize="x-small",
            ha="right", va="top",
            bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75)
        )

        # plot 2nd order reference
        ox1, oy1 = ax.transAxes.transform((0.1, 0.5))  # axes -> display
        ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display -> data
        ox2, oy2 = ax.transAxes.transform((0.5, 0.5))  # axes -> display
        ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display -> data
        oy2 = (ox1 / ox2)**2 * oy1
        ax.plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
        ax.text(
            ox1*2.1, oy1*(oy2/oy1)**0.5, r"$\mathcal{O}(N^{-2})$", fontsize="x-small",
            ha="right", va="top",
            bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75)
        )

        ax.legend(loc="upper right", bbox_to_anchor=(1.02, 1.02))

        fig.savefig(figdir.joinpath(f"converge_{i}_{j}"))

        pyplot.close(fig)


if __name__ == "__main__":
    import pathlib

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    print("loading style sheet")
    pyplot.style.use(figdir.parent.joinpath("plot.mplstyle"))

    print("loading fullmtx results")

    with numpy.load(figdir.joinpath("fullmtx.meta.npz")) as dset:
        nvs = dset["nvs"]

    with numpy.load(figdir.joinpath("fullmtx.convs.npz")) as dset:
        convs = dict(dset)

    print("loading diagapprox results")

    with numpy.load(figdir.joinpath("diagapprox.meta.npz")) as dset:
        nvs = dset["nvs"]

    with numpy.load(figdir.joinpath("diagapprox.convs.npz")) as dset:
        convs.update(dict(dset))

    print("plotting error convergences")
    plot_convergences(nvs, convs, figdir)
