#!/usr/bin/env python3
# vim:fenc=utf-8

"""Plot computational costs.
"""
import itertools
import numpy
import torch
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


def plot_costs(nvs, data, figdir):
    """Plot the computational costs.
    """

    keys = sorted(data.keys(), key=lambda x: int(x[-1]))
    data = {k: data[k] for k in keys}

    lscycler = itertools.cycle(linestyles)

    fig = pyplot.figure(figsize=(2.5, 2.5))
    gs = fig.add_gridspec(1, 1)
    ax = fig.add_subplot(gs[0, 0])

    for key, cost in data.items():
        ax.plot(nvs, cost, label=alglbls[key], ls=next(lscycler))

    ax.grid(True, which="both", zorder=-1, lw=0.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"$N$")

    ax.set_yscale("log")
    ax.set_ylabel(rf"Computational Time (Seconds)")

    ax.legend(loc="upper left", bbox_to_anchor=(-0.02, 1.02))

    fig.savefig(figdir.joinpath(f"comp_time"))
    pyplot.close(fig)


if __name__ == "__main__":
    import pathlib

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    print("loading style sheet")
    pyplot.style.use(figdir.parent.joinpath("plot.mplstyle"))

    print("loading fullmtx results")
    dset = torch.load(figdir.joinpath("fullmtx.dat"))
    nvs = dset["nvs"]
    costs = dset["times"]
    del dset

    print("loading diagapprox results")
    dset = torch.load(figdir.joinpath("diagapprox.dat"))
    nvs = dset["nvs"]
    costs.update(dset["times"])
    del dset

    print("plotting")
    plot_costs(nvs, costs, figdir)
