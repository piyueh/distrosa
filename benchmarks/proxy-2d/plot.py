#!/usr/bin/env python3
# vim:fenc=utf-8

"""Plotting."""

import re
import numpy
import torch
from matplotlib import pyplot
from matplotlib import colors as mcolors
from matplotlib import cm as mcm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.legend_handler import HandlerPolyCollection
from matplotlib import ticker as mticker
import distrosa
import distrosa.utils
from density import proxy_2d


torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=5, linewidth=120)
numpy.set_printoptions(precision=5, linewidth=120)


# line styles
linestyles = ["dashdot", "dashed", (0, (1, 1))]

# mapping between algorithm keys and names in plots
alglbls = {
    "alg-3": "Full Inv",
    "alg-4": "Diag Approx",
    "alg-6": "Interp Full",
    "alg-7": "Interp Diag",
    "analytical": "Continuous",
}

# names of the parameters
parnames = [r"\alpha_1", r"\alpha_2", r"\alpha_3", r"\alpha_4", r"\alpha_5"]


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


def integrate(verts, func):
    """Get the normalization factor."""

    vals = torch.zeros(
        (verts[0].shape[0] - 1, verts[1].shape[0] - 1), dtype=verts[0].dtype
    )
    vals = vals.to(verts[0].device)
    for i in range(verts[0].shape[0] - 1):
        for j in range(verts[1].shape[0] - 1):
            q, w = numpy.polynomial.legendre.leggauss(5)
            q = torch.tensor(q, dtype=verts[0].dtype, device=verts[0].device)
            w = torch.tensor(w, dtype=verts[0].dtype, device=verts[0].device)
            qs = [
                (q + 1.0) / 2.0 * (verts[0][i + 1] - verts[0][i]) + verts[0][i],
                (q + 1.0) / 2.0 * (verts[1][j + 1] - verts[1][j]) + verts[1][j],
            ]
            ws = [
                w / 2.0 * (verts[0][i + 1] - verts[0][i]),
                w / 2.0 * (verts[1][j + 1] - verts[1][j]),
            ]
            qs = torch.stack(torch.meshgrid(*qs, indexing="ij"), -1).view(-1, 2)
            ws = torch.stack(torch.meshgrid(*ws, indexing="ij"), -1).view(-1, 2)
            ws = torch.prod(ws, -1)
            vals[i, j] = torch.sum(func(qs) * ws)
    norm = vals.sum()
    return norm


def read_grad_data(outdir):
    """Read gradient data."""

    losses = {"alg-3": {}, "alg-4": {}, "alg-6": {}, "alg-7": {}}
    grads = {"alg-3": {}, "alg-4": {}, "alg-6": {}, "alg-7": {}}
    for alg in ["alg-3", "alg-4", "alg-6", "alg-7"]:
        # remove what are already done from all identifiers
        files = list(_.name for _ in outdir.glob(f"grad-{alg}-*.dat"))
        for f in files:
            # extract info from filename
            res = re.search(rf"grad-{alg}-(\d+?)-(\d+?).dat", f)
            res = res.groups()  # type: ignore
            nx, irep = int(res[0]), int(res[1])

            # read data
            with open(outdir / f, "rb") as fp:
                tmp = pickle.load(fp)

            # verify
            assert nx == tmp["ndraw"]
            assert irep == tmp["irep"]
            assert alg == tmp["alg"]

            # make sure we have a list for this nx
            if nx not in losses[alg]:
                losses[alg][nx] = []
                grads[alg][nx] = []

            losses[alg][nx].append(tmp["loss"])
            grads[alg][nx].append(tmp["grad"].numpy())

        # sort
        losses[alg] = dict(sorted(losses[alg].items(), key=lambda _1: _1[0]))
        grads[alg] = dict(sorted(grads[alg].items(), key=lambda _1: _1[0]))

        # to big numpy array
        losses[alg] = numpy.array(list(losses[alg].values()))
        grads[alg] = numpy.array(list(grads[alg].values()))

    return losses, grads


def read_trained_data(outdir):
    """Read trained data."""
    lossout = {}
    paramout = {}
    timeout = {}
    itersout = {}

    for alg in ["alg-3", "alg-4", "alg-6", "alg-7", "analytical"]:
        # aliases
        lossout[alg] = _losses = {}
        paramout[alg] = _pars = {}
        timeout[alg] = _times = {}
        itersout[alg] = _iters = {}

        files = list(_.name for _ in outdir.glob(f"train-{alg}-*.dat"))
        for f in files:
            # extract info from filename
            res = re.search(rf"train-{alg}-(\d+?)-(\d+?)-(\d+?).dat", f)
            res = res.groups()  # type: ignore
            ntrain, ibt, irep = int(res[0]), int(res[1]), int(res[2])

            # read data
            with open(outdir / f, "rb") as fp:
                tmp = pickle.load(fp)

            # verify
            assert ibt == tmp["ibt"]
            assert irep == tmp["irep"]
            assert alg == tmp["alg"]

            # make sure keys exist
            _losses.setdefault(ntrain, {}).setdefault(ibt, numpy.inf)
            _pars.setdefault(ntrain, {}).setdefault(ibt, numpy.full(5, numpy.nan))
            _times.setdefault(ntrain, []).append(tmp["time"])
            _iters.setdefault(ntrain, []).append(tmp["iters"])

            # only keep the current best
            if tmp["loss"].item() < _losses[ntrain][ibt]:
                _losses[ntrain][ibt] = tmp["loss"].item()
                _pars[ntrain][ibt][...] = tmp["pars"].numpy()

        # convert to numpy arrays
        for ntrain in _losses.keys():
            lossout[alg][ntrain] = numpy.array(list(_losses[ntrain].values()))
            paramout[alg][ntrain] = numpy.array(list(_pars[ntrain].values()))
            timeout[alg][ntrain] = numpy.array(_times[ntrain])
            itersout[alg][ntrain] = numpy.array(_iters[ntrain], dtype=float)

    return lossout, paramout, timeout, itersout


def plot_convergence(meta, losses, grads, figdir):
    """Plot errors of gradients."""

    ndraws = numpy.array(meta["ndraws"])
    ansloss = meta["truegrads"]["loss"].numpy()
    ansgrad = meta["truegrads"]["grad"].numpy()

    errs = {}
    for alg in ["alg-3", "alg-4", "alg-6", "alg-7"]:
        errloss = abs((losses[alg] - ansloss) / ansloss)  # (6, 200)
        errgrad = abs((grads[alg] - ansgrad) / ansgrad)  # (6, 200, 5)

        errs[alg] = {}
        for q in [25, 50, 75]:
            errs[alg][f"q{q:02d}"] = numpy.concat(
                [
                    numpy.quantile(errloss, q / 100, axis=1)[:, None],  # shape (6, 1)
                    numpy.quantile(errgrad, q / 100, axis=1),  # shape (6, 5)
                ],
                axis=1,
            )  # shape (6, 6)

    fig = pyplot.figure(figsize=(6.5, 3.8))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 0.1])
    axs = [[None, None, None], [None, None, None]]
    lax = fig.add_subplot(gs[2, :])  # last row, all columns

    for i in range(6):
        row, col = i // 3, i % 3

        if i in [0, 1, 2]:
            ax = axs[row][col] = fig.add_subplot(gs[row, col])  # pyright: ignore
        else:
            ax = axs[row][col] = fig.add_subplot(gs[row, col], sharex=axs[0][col])

        bands = []
        for alg, err in errs.items():
            band = ax.fill_between(
                ndraws,
                errs[alg]["q25"][:, i],
                errs[alg]["q75"][:, i],
                alpha=0.3,
                zorder=1,
            )
            ax.plot(ndraws, errs[alg]["q50"][:, i], zorder=2)
            bands.append(band)

        ax.set_xscale("log")
        ax.set_yscale("log")

        if i in [0, 1, 2]:
            ax.tick_params("x", which="both", labelbottom=False)

        if i in [3, 4, 5]:
            ax.set_xlabel(r"$M_{x}$")

        if i in [0, 3]:
            ax.set_ylabel("Relative Error")

        if i == 0:
            ax.set_title(r"$L$")
        else:
            ax.set_title(rf"$\partial L \slash \partial {parnames[i - 1]}$")

        # plot reference convergence order
        ox1, oy1 = ax.transAxes.transform((0.3, 0.9))  # axes -> display
        ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display -> data
        ox2, oy2 = ax.transAxes.transform((0.75, 0.1))  # axes -> display
        ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display -> data
        oy2 = (ox1 / ox2) ** 0.5 * oy1
        ax.plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
        ax.text(
            ox1 * 10**0.6,
            oy2 * 10**0.5,
            r"$\mathcal{O}(M_x^{-0.5})$",
            fontsize="small",
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75),
        )

        ax.grid(True, which="both", lw=0.25, zorder=-1, color="gainsboro")
        ax.set_axisbelow(True)

        lax.legend(
            handles=bands,  # type: ignore
            labels=[alglbls[_] for _ in errs.keys()],
            handler_map={_: HandlerMedianInterval() for _ in bands},  # type: ignore
            loc="center",
            ncol=4,
            columnspacing=0.6,
            borderaxespad=0.0,
        )
        lax.set_axis_off()

        fig.savefig(figdir / "convergence")


def plot_trained_params(params, ans, figdir):
    """Plot training results."""

    algs = list(params.keys())
    ntrains = list(params[algs[0]].keys())
    params = {k1: {k2: v2 / ans for k2, v2 in v1.items()} for k1, v1 in params.items()}

    cmap = pyplot.get_cmap("tab10")

    # plot loss
    fig, axs = pyplot.subplots(
        1,
        5,
        figsize=(6.5, 3.0),
        squeeze=False,
    )
    for col in range(5):  # loop over parameters
        for i, ntrain in enumerate(sorted(ntrains)):
            axs[0, col].boxplot(
                [params[_][ntrain][:, col] for _ in algs],
                vert=True,
                widths=0.6,
                patch_artist=True,
                showfliers=False,
                showmeans=False,
                capprops={"color": cmap(3 * i), "lw": 1.0},
                boxprops={"fc": cmap(3 * i), "alpha": 0.7, "lw": 0},
                whiskerprops={"color": cmap(3 * i), "lw": 1.0},
                medianprops={"lw": 2.0, "color": cmap(3 * i)},
                tick_labels=list(alglbls[_] for _ in algs),
            )

        axs[0, col].axhline(1.0, color="k", ls="-.", lw=1.5)
        axs[0, col].tick_params(axis="x", labelsize=8, rotation=90)
        axs[0, col].tick_params(axis="y", labelsize=8, rotation=-60)
        axs[0, col].yaxis.set_major_formatter(mticker.ScalarFormatter(False))
        axs[0, col].set_title(f"${parnames[col]}$", fontsize=12)

    fig.supylabel("Relative to Truth")
    fig.savefig(figdir / "trained_params")


def get_pdf_errors(params, anspars, fname):
    """Get PDF errors"""

    if fname.is_file():
        errs = torch.load(fname)
        return errs

    algs = list(params.keys())
    ntrains = list(params[algs[0]].keys())

    verts = [torch.linspace(1e-6, 1.0 - 1e-6, 51) for _ in range(2)]
    anspars = torch.as_tensor(anspars)
    ansnorm = integrate(verts, lambda x: proxy_2d(x, anspars))

    # convert to tensor
    params = {
        k1: {k2: torch.as_tensor(v2) for k2, v2 in v1.items()}
        for k1, v1 in params.items()
    }

    errs = {}
    for alg, ntrain in itertools.product(algs, ntrains):
        # initialize data holder if it has not been created
        errs.setdefault(alg, {})
        errs[alg].setdefault(ntrain, torch.zeros(params[alg][ntrain].shape[0]))

        for i, pars in enumerate(params[alg][ntrain]):
            pars = pars.clone().detach()
            norm = integrate(verts, lambda _x: proxy_2d(_x, pars))

            def kernel(_x):
                _anspdf = proxy_2d(_x, anspars) / ansnorm
                _trainedpdf = proxy_2d(_x, pars) / norm
                _err = torch.abs(_trainedpdf - _anspdf)  # absolute
                return _err

            with torch.no_grad():
                errs[alg][ntrain][i] = integrate(verts, kernel).detach().cpu()

            print(alg, ntrain, i, errs[alg][ntrain][i])

    torch.save(errs, fname)
    return errs


def plot_pdf_errors(errs, figdir):
    """Plot PDF errors."""

    algs = list(errs.keys())
    ntrains = list(errs[algs[0]].keys())
    cmap = pyplot.get_cmap("tab10")

    fig, axs = pyplot.subplots(1, 1, figsize=(3.25, 1.75), squeeze=False)

    for i, ntrain in enumerate(sorted(ntrains)):
        axs[0, 0].boxplot(
            [errs[_][ntrain] for _ in algs],
            vert=False,
            widths=0.6,
            patch_artist=True,
            showfliers=False,
            showmeans=False,
            capprops={"color": cmap(3 * i), "lw": 1.0},
            boxprops={"fc": cmap(3 * i), "alpha": 0.7, "lw": 0},
            whiskerprops={"color": cmap(3 * i), "lw": 1.0},
            medianprops={"lw": 2.0, "color": cmap(3 * i)},
            tick_labels=list(alglbls[_] for _ in algs),
        )
    axs[0, 0].set_xlabel(r"$L_1$ Error of PDF")
    axs[0, 0].tick_params(axis="x", labelsize=8, rotation=0)
    axs[0, 0].tick_params(axis="y", labelsize=8, rotation=0)
    axs[0, 0].xaxis.set_major_formatter(mticker.ScalarFormatter(False))

    fig.savefig(figdir / "trained_pdf_errs")


def examine_1(curpars, ndraw, figdir, fname):
    """Examine"""

    # fixed random seed
    torch.manual_seed(0)

    # operate in PyTorch
    curpars = torch.as_tensor(curpars).to("cpu")

    # sampler and samples
    verts = [torch.linspace(1e-6, 1.0 - 1e-6, 51) for _ in range(2)]
    sampler = distrosa.utils.RejectionSampler(proxy_2d, verts, None)
    samples = sampler(ndraw, curpars)

    # calculate normalization factor
    norm = integrate(verts, lambda x: proxy_2d(x, curpars))

    # get normalized PDF at vertices
    vals = proxy_2d(torch.stack(torch.meshgrid(verts, indexing="ij"), -1), curpars)
    vals = vals / norm

    # normalized PDF at samples
    svals = proxy_2d(samples, curpars) / norm  # type: ignore

    # a coarser grid for histogram
    hverts = [torch.linspace(1e-6, 1.0 - 1e-6, 33) for _ in range(2)]

    # move to numpy
    vals = vals.detach().cpu().numpy()
    svals = svals.detach().cpu().numpy()
    samples = samples.detach().cpu().numpy()
    hist = numpy.histogram2d(samples[:, 0], samples[:, 1], hverts, density=True)[0]
    hist = hist / hist.sum()
    hist = hist / (((1.0 - 1e-6 - 1e-6) / (33 - 1)) ** 2)

    # to unify the colorbar
    vmin = numpy.quantile(vals, 0.01).item()
    vmax = numpy.quantile(vals, 0.99).item()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mcm.ScalarMappable(norm, "turbo")

    # plot
    fig, axs = pyplot.subplots(1, 3, figsize=(7.5, 3.3))

    axs[0].pcolormesh(
        verts[0], verts[1], vals.T, shading="gouraud", norm=norm, cmap="turbo"
    )
    axs[0].set_title("PDF")
    axs[0].set_xlabel(r"$x_1$")
    axs[0].set_xlim(0, 1)
    axs[0].set_ylim(0, 1)

    axs[1].scatter(samples[:, 0], samples[:, 1], 3, marker="o", ec="k", alpha=0.1)
    axs[1].set_title("Samples")
    axs[1].set_xlabel(r"$x_1$")
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].tick_params(axis="y", which="both", left=False, labelleft=False)

    axs[2].pcolormesh(hverts[0], hverts[1], hist.T, norm=norm, cmap="turbo")
    axs[2].set_title("Histogram")
    axs[2].set_xlabel(r"$x_1$")
    axs[2].set_xlim(0, 1)
    axs[2].set_ylim(0, 1)
    axs[2].tick_params(axis="y", which="both", left=False, labelleft=False)

    cbar = fig.colorbar(mappable, ax=axs, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Normalized Probability Density")
    fig.supylabel(r"$x_2$")
    fig.savefig(figdir / fname)


def examine_2(curpars, figdir, fname):
    """Examine"""

    # operate in PyTorch
    curpars = torch.as_tensor(curpars).to("cpu")

    # sampler and samples
    verts = [torch.logspace(-6, numpy.log10(1.0 - 1e-6), 201) for _ in range(2)]

    # calculate normalization factor
    norm = integrate(verts, lambda x: proxy_2d(x, curpars))

    # get normalized PDF at vertices
    vals = proxy_2d(torch.stack(torch.meshgrid(verts, indexing="ij"), -1), curpars)
    vals = vals / norm

    # move to numpy
    vals = vals.detach().cpu().numpy()

    # to unify the colorbar
    vmin = numpy.quantile(vals, 0.01).item()
    vmax = numpy.quantile(vals, 0.99).item()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mcm.ScalarMappable(norm, "turbo")

    # plot
    fig, ax = pyplot.subplots(1, 1, figsize=(2.5, 3.0))
    ax.pcolormesh(
        verts[0], verts[1], vals.T, shading="gouraud", norm=norm, cmap="turbo"
    )
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cbar = fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Normalized Probability Density")
    fig.savefig(fig / fname)


if __name__ == "__main__":
    import pathlib
    import pickle
    import pprint

    _figdir = pathlib.Path(__file__).resolve().parent.joinpath("figs")
    pyplot.style.use(_figdir.parent.joinpath("plot.mplstyle"))

    # load meta data
    _meta = torch.load(_figdir / "meta.dat")

    # load losses and gradients for plotting convergence
    _losses, _grads = read_grad_data(_figdir)

    # plot the convergence of losses and gradients
    plot_convergence(_meta, _losses, _grads, _figdir)

    # load trained results
    _losses, _params, _times, _iters = read_trained_data(_figdir)

    # print averaged time
    pprint.pprint(
        {
            k1: {k2: numpy.mean(v2).item() for k2, v2 in v1.items()}
            for k1, v1 in _times.items()
        }
    )

    # print time std
    pprint.pprint(
        {
            k1: {k2: numpy.std(v2).item() for k2, v2 in v1.items()}
            for k1, v1 in _times.items()
        }
    )

    # print averaged iteration per optimization
    pprint.pprint(
        {
            k1: {k2: numpy.mean(v2).item() for k2, v2 in v1.items()}
            for k1, v1 in _iters.items()
        }
    )

    # print iteration per optimization STD
    pprint.pprint(
        {
            k1: {k2: numpy.std(v2).item() for k2, v2 in v1.items()}
            for k1, v1 in _iters.items()
        }
    )

    # plot trained results
    plot_trained_params(_params, _meta["anspars"].numpy(), _figdir)

    # get PDF absolute errors
    _errs = get_pdf_errors(_params, _meta["anspars"], _figdir / "pdferrs.dat")

    # plot PDF errors
    plot_pdf_errors(_errs, _figdir)

    # plot contours of PDF, samples, and the histogram of the samples
    examine_1(_meta["anspars"], 10000, _figdir, "true_pdf")
    examine_1(_meta["curpars"], 10000, _figdir, "current_pdf")
    examine_1(
        numpy.median(_params["alg-6"][10000], axis=0), 10000, _figdir, "trained_pdf"
    )
