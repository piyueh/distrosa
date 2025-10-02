#!/usr/bin/env python3
# vim:fenc=utf-8

"""Plotting script."""

import itertools
from os import waitid_result
import torch
import numpy
import matplotlib.pyplot as pyplot
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerPolyCollection


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

# names of the parameters
parname = [r"\theta_1", r"\theta_2"]


class HandlerMedianInterval(HandlerPolyCollection):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        # docstring inherited
        p = Rectangle(xy=(-xdescent, -ydescent), width=width, height=height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)

        x, y = p.xy
        w = p.get_width()
        h = p.get_height()
        c = p.get_facecolor()
        l = Line2D([x, x + w], [y + h / 2.0, y + h / 2.0], color=c, alpha=1.0)
        return [p, l]


def ploterrors(data, figdir):
    """Plot errors."""

    ndraws = numpy.array(data["ndraws"])
    ans = data["ans"]["grad"].numpy()

    grads = {}
    errs = {}
    for alg, dset in data["computed"].items():
        grads[alg] = numpy.array([dset[1024][_].numpy() for _ in ndraws])
        errs[alg] = abs((grads[alg] - ans) / ans)

    fdgrads1 = numpy.array([data["fd1"][_].numpy() for _ in ndraws])
    fderrs1 = abs((fdgrads1 - ans) / ans)

    fdgrads2 = numpy.array([data["fd2"][_].numpy() for _ in ndraws])
    fderrs2 = abs((fdgrads2 - ans) / ans)

    fig = pyplot.figure(figsize=(6.5, 2.0))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.3], wspace=0.1)
    axs = [fig.add_subplot(gs[0]), fig.add_subplot(gs[1])]
    lgax = fig.add_subplot(gs[2])

    for i in range(2):
        bands = []
        for alg, err in errs.items():
            band = axs[i].fill_between(
                ndraws,
                numpy.quantile(errs[alg], 0.05, 1)[:, i],
                numpy.quantile(errs[alg], 0.95, 1)[:, i],
                alpha=0.5,
                zorder=1,
            )
            axs[i].plot(ndraws, numpy.quantile(errs[alg], 0.5, 1)[:, i], zorder=2)
            bands.append(band)

        # finite difference w/ step size 1e-3
        band = axs[i].fill_between(
            ndraws,
            numpy.quantile(fderrs1, 0.05, 1)[:, i],
            numpy.quantile(fderrs1, 0.95, 1)[:, i],
            alpha=0.5,
            zorder=1,
        )
        axs[i].plot(ndraws, numpy.quantile(fderrs1, 0.5, 1)[:, i], zorder=2)
        bands.append(band)

        # finite difference w/ step size 1e-5
        band = axs[i].fill_between(
            ndraws,
            numpy.quantile(fderrs2, 0.05, 1)[:, i],
            numpy.quantile(fderrs2, 0.95, 1)[:, i],
            alpha=0.5,
            zorder=1,
        )
        axs[i].plot(ndraws, numpy.quantile(fderrs2, 0.5, 1)[:, i], zorder=2)
        bands.append(band)

        axs[i].set_xscale("log")
        axs[i].set_xlabel(r"$M_{x}$")
        axs[i].set_yscale("log")
        axs[i].set_ylabel(rf"$\partial L \slash \partial {parname[i]}$")

        # plot 1st order reference
        ox1, oy1 = axs[i].transAxes.transform((0.1, 0.1))  # axes -> display
        ox1, oy1 = axs[i].transData.inverted().transform((ox1, oy1))  # display -> data
        ox2, oy2 = axs[i].transAxes.transform((0.75, 0.1))  # axes -> display
        ox2, oy2 = axs[i].transData.inverted().transform((ox2, oy2))  # display -> data
        oy2 = (ox1 / ox2) ** 0.5 * oy1
        axs[i].plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
        axs[i].text(
            ox2 / 10**1.5,
            oy1 / 10**0.5,
            r"$\mathcal{O}(M_x^{-0.5})$",
            fontsize="x-small",
            ha="right",
            va="top",
            bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75),
        )

        axs[i].grid(True, which="major", lw=0.25, zorder=-1, color="gainsboro")
        axs[i].set_axisbelow(True)

    lgax.legend(
        handles=bands,  # type: ignore
        labels=[alglbls[_] for _ in errs.keys()]
        + [r"FD ($\Delta=10^{-3}$)", r"FD ($\Delta=10^{-5}$)"],
        handler_map={_: HandlerMedianInterval() for _ in bands},  # type: ignore
        loc="center",
        ncol=1,
        columnspacing=0.6,
        borderaxespad=0.0,
    )
    lgax.axis("off")

    fig.savefig(figdir / f"grads")


def plot_gradients_1d(anspars, params, grads: dict, figdir):
    """Plot"""

    distrosa25 = numpy.quantile(grads["distrosa"].numpy(), 0.25, axis=1)[:, 0]
    distrosa50 = numpy.median(grads["distrosa"].numpy(), axis=1)[:, 0]
    distrosa75 = numpy.quantile(grads["distrosa"].numpy(), 0.75, axis=1)[:, 0]

    fd25 = numpy.quantile(grads["fd"].numpy(), 0.25, axis=1)[:, 0]
    fd50 = numpy.median(grads["fd"].numpy(), axis=1)[:, 0]
    fd75 = numpy.quantile(grads["fd"].numpy(), 0.75, axis=1)[:, 0]

    ans = anspars[0]
    pars = params[:, 0]

    fig, ax = pyplot.subplots(1, 1, figsize=(3.25, 2.0))

    line_1 = ax.axvline(ans, c="k", ls="--", lw=1, zorder=5)

    band_2 = ax.fill_between(pars, fd25, fd75, color="tab:orange", alpha=0.3, zorder=1)
    ax.plot(pars, fd50, "tab:orange", lw=1, zorder=2)

    band_3 = ax.fill_between(
        pars, distrosa25, distrosa75, color="tab:blue", alpha=0.3, zorder=3
    )
    ax.plot(pars, distrosa50, "tab:blue", lw=1.0, zorder=4)

    ax.axhline(0.0, c="gray", ls="-", lw=0.5, zorder=0)

    ax.set_xlim(pars[0], pars[-1])
    ax.set_xlabel(r"$\theta_1$")

    ax.set_ylim(-0.05, 0.03)

    ax.set_ylabel(r"$\partial L \slash \partial \theta_1$")

    ax.legend(
        handles=[line_1, band_2, band_3],
        labels=[
            r"$\theta_1$ when $L$ minimized",
            "FD (Median & IQR)",
            "1D Alg (Median & IQR)",
        ],
        handler_map={
            band_2: HandlerMedianInterval(),
            band_3: HandlerMedianInterval(),
        },
        loc="lower right",
    )

    # # inset axis
    # insax = ax.inset_axes((0.39, 0.625, 0.6, 0.4))
    # insax.fill_between(pars, fd25, fd75, color="tab:orange", alpha=0.3, zorder=1)
    # insax.plot(pars, fd50, "tab:orange", zorder=2)
    # insax.fill_between(pars, distrosa25, distrosa75, color="tab:blue", alpha=0.3, zorder=3)
    # insax.plot(pars, distrosa50, "tab:blue", zorder=4)
    # insax.axvline(anspars[0], 0.0, 1.0, c="k", ls="--", zorder=5)
    # insax.set_xlim(2.250, 2.375)
    # insax.set_ylim(-0.004, 0.004)
    # insax.grid(True, which="both")
    # insax.tick_params(axis="both", which="both", labelsize="xx-small")

    fig.savefig(figdir / "distrosa_vs_fd_derivatives")


def plot_train_hists(params, losses, anspars, distrosahist, fdhist, figdir):
    """Plot the loss surface and train history."""

    fig, ax = pyplot.subplots(1, 1, figsize=(3.25, 2.0))
    ax.contour(
        params[..., 0],
        params[..., 1],
        losses,
        numpy.linspace(losses.min(), losses.max(), 128),
        cmap="turbo",
        norm="log",
        alpha=0.3,
        linewidths=0.5,
    )
    ax.scatter(anspars[0], anspars[1], 20, "k", marker="^", label="Truth")
    ax.plot(distrosahist[:, 0], distrosahist[:, 1], "tab:blue", label="1D Alg")
    ax.plot(fdhist[:, 0], fdhist[:, 1], "tab:red", label="FD")
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.legend(loc="lower right", bbox_to_anchor=(1.01, -0.02))
    fig.savefig(figdir / "train_hist")


if __name__ == "__main__":
    import pathlib

    _figdir = pathlib.Path(__file__).resolve().parent.joinpath("figs")

    pyplot.style.use(_figdir.parent.joinpath("plot.mplstyle"))

    _out = torch.load(_figdir / "out.dat")
    ploterrors(_out, _figdir)

    # results for plotting gradients of only one parameter changing
    _grad1d1par = torch.load(_figdir / "grad1d1par.dat")
    anspar = _grad1d1par["anspar"]
    params = _grad1d1par["params2"]  # only load FD w/ delta = 1e-3
    grads = _grad1d1par["grads2"]
    plot_gradients_1d(anspar, params, grads, _figdir)

    # results for training history
    _trainhist = torch.load(_figdir / "train.dat")
    params = _trainhist["params"]
    losses = _trainhist["losses"]
    anspars = _trainhist["anspars"]
    distrosahist = _trainhist["distrosa_hist"]
    fdhist = _trainhist["fd_hist"]
    plot_train_hists(params, losses, anspars, distrosahist, fdhist, _figdir)
