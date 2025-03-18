#!/usr/bin/env python3
# vim:fenc=utf-8

"""Plot sensitivities.
"""
import itertools
import numpy
import torch
import matplotlib.pyplot as pyplot
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as mcm


def fullmtxcfg(params):
    """Return the configurations for contourf plots.
    """

    _, _, sigma1, sigma2, rho = params
    coeff1 = sigma1 / (1 - rho * rho)
    coeff2 = sigma2 / (1 - rho * rho)

    zeroscale = numpy.logspace(-3, -1, 3)
    zeroscale = numpy.concatenate([-zeroscale[::-1], zeroscale])

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
            levels=numpy.linspace(-5., 5., 64),
            norm=mcolors.Normalize(vmin=-5., vmax=5.),
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
            levels=numpy.linspace(-coeff1*5., coeff1*5., 64),
            norm=mcolors.Normalize(-coeff1*5., coeff1*5., clip=False),
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
            levels=numpy.linspace(-5., 5., 64),
            norm=mcolors.Normalize(vmin=-5., vmax=5.),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        ),
        (1, 4): dict(
            levels=numpy.linspace(-coeff2*5., coeff2*5., 64),
            norm=mcolors.Normalize(-coeff2*5., coeff2*5., clip=False),
            cmap=pyplot.get_cmap("turbo").with_extremes(bad="w"),
            extend="both"
        )
    }

    def oneformatter(x, pos):
        return rf"${'-' if x-1 <= 0 else '+'}10^{{{int(numpy.log10(abs(x-1))-0.4)}}}$"

    def zeroformatter(x, pos):
        return rf"${'-' if x <= 0 else '+'}10^{{{int(numpy.log10(abs(x))-0.4)}}}$"

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


def diagapprox(params):
    """Return the configurations for contourf plots.
    """

    _, _, sigma1, sigma2, rho = params
    coeff1 = sigma1 / (1 - rho * rho)
    coeff2 = sigma2 / (1 - rho * rho)

    zeroscale = numpy.logspace(-3, -1, 3)
    zeroscale = numpy.concatenate([-zeroscale[::-1], zeroscale])

    ans01 = - rho * sigma1 / sigma2
    ans03 = 5. * rho * sigma1 / sigma2
    ans04_l = - sigma1 * (rho * 5. + 5.) / (1 - rho * rho)
    ans04_h = sigma1 * (rho * 5. + 5.) / (1 - rho * rho)
    ans10 = - rho * sigma2 / sigma1
    ans12 = 5. * rho * sigma2 / sigma1
    ans14_l = - sigma2 * (5. + rho * 5.) / (1 - rho * rho)
    ans14_h = sigma2 * (5. + rho * 5.) / (1 - rho * rho)

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
            levels=numpy.linspace(-5., 5., 64),
            norm=mcolors.Normalize(vmin=-5., vmax=5.),
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
            levels=numpy.linspace(-5., 5., 64),
            norm=mcolors.Normalize(vmin=-5., vmax=5.),
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

    def oneformatter(x, pos):
        val = int(numpy.log10(abs(x-1))-0.4)
        return rf"${'-' if x-1 <= 0 else '+'}10^{{{val}}}$"

    def ans01formatter(x, pos):
        val = int(numpy.log10(abs(x-ans01))-0.4)
        return rf"${'-' if x-1 <= 0 else '+'}10^{{{val}}}$"

    def ans10formatter(x, pos):
        val = int(numpy.log10(abs(x-ans10))-0.4)
        return rf"${'-' if x-1 <= 0 else '+'}10^{{{val}}}$"

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


def plot_sensitivity(x, ans, params, computed, figdir, variant):
    """Plot the sensitivities.
    """

    getcfg = dict(fullmtxcfg=fullmtxcfg, diagapprox=diagapprox)

    # colormap normalizer
    ctfargs, cbarargs = getcfg[variant](params)

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
        fig.savefig(figdir.joinpath(f"{variant}_{i}_{j}_{x.shape[0]}x{x.shape[1]}"))
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
    ans = dset["ans"].numpy()
    pdfvals = dset["pdfvals"].numpy()
    params = dset["params"].numpy()
    outs = {alg: torch.stack(val).numpy() for alg, val in dset["outs"].items()}
    del dset

    print("plotting pdf")
    plot_dist(x, pdfvals, figdir)

    print("plotting fullmtx results")
    plot_sensitivity(x, ans, params, outs, figdir, "fullmtxcfg")

    print("loading diagapprox results")
    dset = torch.load(figdir.joinpath("diagapprox.dat"))
    x = dset["x"].numpy()
    ans = dset["ans"].numpy()
    pdfvals = dset["pdfvals"].numpy()
    params = dset["params"].numpy()
    outs = {alg: torch.stack(val).numpy() for alg, val in dset["outs"].items()}
    del dset

    print("plotting diagapprox results")
    plot_sensitivity(x, ans, params, outs, figdir, "diagapprox")
