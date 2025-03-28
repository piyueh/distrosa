#!/usr/bin/env python3
# vim:fenc=utf-8

"""Plot absolute error contours.
"""
import itertools
import numpy
import torch
import matplotlib.pyplot as pyplot
import matplotlib.colors as mcolors
import matplotlib.cm as mcm


def plot_errors(x, errors, mask, figdir):
    """Plot the sensitivities.
    """

    # plotting
    for key in itertools.product(errors.items(), range(2), range(5)):

        (alg, dset), i, j = key  # type: ignore
        res = dset[-1].shape[0]
        err = dset[-1][..., i, j]
        print(f"plotting error {alg}-{res}x{res}-({i}, {j})")

        # to avoid log(0)
        err = numpy.where(err < 1e-15, 1e-15, err)
        merr = numpy.ma.array(err, mask=~mask)

        # 5% and 95% quantile
        q05 = numpy.quantile(err, 0.05)
        q95 = numpy.quantile(err, 0.95)

        vmin = max(int(numpy.log10(q05)), -6)
        vmax = min(int(numpy.log10(q95)), 1)

        # contourf configurations
        ctfargs = dict(
            levels=numpy.power(10., numpy.arange(vmin, vmax+1)),
            norm=mcolors.LogNorm(vmin=10**vmin, vmax=10**vmax),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        )

        # colorbar configurations
        cbarargs = dict(
            mappable=mcm.ScalarMappable(ctfargs["norm"], ctfargs["cmap"]),# type: ignore
            orientation="horizontal",
            extend="both"
        )

        fig = pyplot.figure(figsize=(2.5, 3.0))
        gs = fig.add_gridspec(1, 1)
        ax = fig.add_subplot(gs[0, 0])
        cf = ax.contourf(x[..., 0], x[..., 1], merr, **ctfargs)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(r"$x_1$")
        ax.set_ylabel(r"$x_2$")
        fig.colorbar(ax=ax, **cbarargs)  # type: ignore
        fig.savefig(figdir.joinpath(f"error_{alg}_{i}_{j}_{res}x{res}"))

        pyplot.close(fig)

    return


if __name__ == "__main__":
    import pathlib

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    print("loading style sheet")
    pyplot.style.use(figdir.parent.joinpath("plot.mplstyle"))

    print("loading fullmtx results")
    dset = torch.load(figdir.joinpath("fullmtx.dat"))
    x = dset["x"].numpy()
    errs = {alg: torch.stack(val).numpy() for alg, val in dset["errs"].items()}
    mask = dset["mask"].numpy()
    del dset

    print("plotting fullmtx errors")
    plot_errors(x, errs, mask, figdir)

    print("loading diagapprox results")
    dset = torch.load(figdir.joinpath("diagapprox.dat"))
    x = dset["x"].numpy()
    errs = {alg: torch.stack(val).numpy() for alg, val in dset["errs"].items()}
    mask = dset["mask"].numpy()
    del dset

    print("plotting diagapprox errors")
    plot_errors(x, errs, mask, figdir)
