#!/usr/bin/env python3
# vim:fenc=utf-8

"""Plotting.
"""
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

numpy.set_printoptions(precision=5, linewidth=120)


# line styles
linestyles = ["dashdot", "dashed", (0, (1, 1))]

# mapping between algorithm keys and names in plots
alglbls = {
    "alg-3": "Full Inv",
    "alg-4": "Diag Approx",
    "alg-6": "Interp Full",
    "alg-7": "Interp Diag",
}

# names of the parameters
parname = [r"\alpha_1", r"\alpha_2", r"\alpha_3", r"\alpha_4", r"\alpha_5"]


class HandlerMedianInterval(HandlerPolyCollection):
    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize,
        trans
    ):
        # docstring inherited
        p = Rectangle(xy=(-xdescent, -ydescent), width=width, height=height)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)

        x, y = p.xy
        w = p.get_width()
        h = p.get_height()
        c = p.get_facecolor()
        l = Line2D([x, x+w], [y+h/2.0, y+h/2.0], color=c, alpha=1.0)
        return [p, l]


def plot_loss_errors(data, figdir):
    """Plot errors of losses.
    """
    ndraws = numpy.array(data["ndraws"])
    res = data["res1"]
    ans = data["truegrads"]["loss"].numpy()

    grads = {}
    errs = {}
    for alg in ["alg-3", "alg-4", "alg-6", "alg-7"]:
        dset = data[alg]["loss"]
        grads[alg] = numpy.array([dset[_].numpy() for _ in ndraws])
        err = abs((grads[alg]-ans)/ans)

        errs[alg] = {}
        for q in [25, 50, 75]:
            errs[alg][f"q{q:02d}"] = numpy.quantile(err, q/100, axis=1)

    fig = pyplot.figure(figsize=(2.5, 2.5))
    ax = pyplot.gca()

    bands = []
    for alg, err in errs.items():
        band = ax.fill_between(
            ndraws, errs[alg]["q25"], errs[alg]["q75"], alpha=0.5, zorder=1,)
        ax.plot(ndraws, errs[alg]["q50"], zorder=2)
        bands.append(band)

    ax.set_xscale("log")
    ax.set_xlabel(r"$M_{x}$")

    ax.set_yscale("log")
    ax.set_ylabel(rf"Relative Error of Energy Loss")

    # plot 1st order reference
    ox1, oy1 = ax.transAxes.transform((0.3, 0.9))  # axes -> display
    ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display -> data
    ox2, oy2 = ax.transAxes.transform((0.75, 0.2))  # axes -> display
    ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display -> data
    oy2 = (ox1 / ox2)**0.5 * oy1
    ax.plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
    ax.text(
        ox1*10**0.6, oy2*10**0.6, r"$\mathcal{O}(M_x^{-0.5})$", fontsize="x-small",
        ha="left", va="bottom",
        bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75)
    )

    ax.grid(True, which="both", lw=0.1, zorder=-1)
    ax.set_axisbelow(True)

    ax.legend(
        handles=bands,  # type: ignore
        labels=[alglbls[_] for _ in errs.keys()],
        handler_map={_: HandlerMedianInterval() for _ in bands},  # type: ignore
        loc="lower left", bbox_to_anchor=(0.01, 0.01),
        ncol=1, columnspacing=0.6, borderaxespad=0.0,
    )

    fig.savefig(figdir/f"loss_err")


def plot_grad_errors(data, figdir):
    """Plot errors of gradients.
    """

    ndraws = numpy.array(data["ndraws"])
    res = data["res1"]
    ans = data["truegrads"]["grad"].numpy()

    grads = {}
    errs = {}
    for alg in ["alg-3", "alg-4", "alg-6", "alg-7"]:
        dset = data[alg]["grad"]
        grads[alg] = numpy.array([dset[_].numpy() for _ in ndraws])
        err = abs((grads[alg]-ans)/ans)

        errs[alg] = {}
        for q in [25, 50, 75]:
            errs[alg][f"q{q:02d}"] = numpy.quantile(err, q/100, axis=1)

    for i in range(5):

        fig = pyplot.figure(figsize=(2.5, 2.5))
        ax = pyplot.gca()

        bands = []
        for alg, err in errs.items():
            band = ax.fill_between(
                ndraws, errs[alg]["q25"][:, i], errs[alg]["q75"][:, i],
                alpha=0.3, zorder=1,
            )
            ax.plot(ndraws, errs[alg]["q50"][:, i], zorder=2)
            bands.append(band)

        ax.set_xscale("log")
        ax.set_xlabel(r"$M_{x}$")

        ax.set_yscale("log")
        ax.set_ylabel(rf"Relative Error of $\partial L / \partial {parname[i]}$")

        # plot 1st order reference
        ox1, oy1 = ax.transAxes.transform((0.3, 0.9))  # axes -> display
        ox1, oy1 = ax.transData.inverted().transform((ox1, oy1))  # display -> data
        ox2, oy2 = ax.transAxes.transform((0.75, 0.1))  # axes -> display
        ox2, oy2 = ax.transData.inverted().transform((ox2, oy2))  # display -> data
        oy2 = (ox1 / ox2)**0.5 * oy1
        ax.plot([ox1, ox2], [oy1, oy2], "k--", lw=1.0)
        ax.text(
            ox1*10**0.6, oy2*10**0.6, r"$\mathcal{O}(M_x^{-0.5})$", fontsize="x-small",
            ha="left", va="bottom",
            bbox=dict(boxstyle="square", color="w", lw=None, alpha=0.75)
        )

        ax.grid(True, which="both", lw=0.1, zorder=-1)
        ax.set_axisbelow(True)

        ax.legend(
            handles=bands,  # type: ignore
            labels=[alglbls[_] for _ in errs.keys()],
            handler_map={_: HandlerMedianInterval() for _ in bands},  # type: ignore
            loc="lower left", bbox_to_anchor=(0.01, 0.01),
            ncol=1, columnspacing=0.6, borderaxespad=0.0,
        )

        fig.savefig(figdir/f"grad_err_{i}")


def integrate(verts, func):
    """Get the normalization factor.
    """

    vals = torch.zeros((verts[0].shape[0]-1, verts[1].shape[0]-1), dtype=verts[0].dtype)
    vals = vals.to(verts[0].device)
    for i in range(verts[0].shape[0]-1):
        for j in range(verts[1].shape[0]-1):
            q, w = numpy.polynomial.legendre.leggauss(5)
            q = torch.tensor(q, dtype=verts[0].dtype, device=verts[0].device)
            w = torch.tensor(w, dtype=verts[0].dtype, device=verts[0].device)
            qs = [
                (q + 1.0) / 2.0 * (verts[0][i+1] - verts[0][i]) + verts[0][i],
                (q + 1.0) / 2.0 * (verts[1][j+1] - verts[1][j]) + verts[1][j],
            ]
            ws = [
                w / 2.0 * (verts[0][i+1] - verts[0][i]),
                w / 2.0 * (verts[1][j+1] - verts[1][j]),
            ]
            qs = torch.stack(torch.meshgrid(*qs, indexing="ij"), -1).view(-1, 2)
            ws = torch.stack(torch.meshgrid(*ws, indexing="ij"), -1).view(-1, 2)
            ws = torch.prod(ws, -1)
            vals[i, j] = torch.sum(func(qs)*ws)
    norm = vals.sum()
    return norm


def examine(curpars, ndraw, fname):
    """Examine
    """

    # fixed random seed
    torch.manual_seed(0)

    # operate in PyTorch
    curpars = torch.as_tensor(curpars).to("cpu")

    # sampler and samples
    verts = [torch.linspace(1e-6, 1.-1e-6, 51) for _ in range(2)]
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
    hverts = [torch.linspace(1e-6, 1.-1e-6, 33) for _ in range(2)]

    # move to numpy
    vals = vals.detach().cpu().numpy()
    svals = svals.detach().cpu().numpy()
    samples = samples.detach().cpu().numpy()
    hist = numpy.histogram2d(samples[:, 0], samples[:, 1], hverts, density=True)[0]
    hist = hist / hist.sum()
    hist = hist / (((1. - 1e-6 - 1e-6) / (33 - 1))**2)

    # to unify the colorbar
    vmin = numpy.quantile(vals, 0.01).item()
    vmax = numpy.quantile(vals, 0.99).item()
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    mappable = mcm.ScalarMappable(norm, "turbo")

    # plot
    fig, axs = pyplot.subplots(1, 3, figsize=(7.5, 3.3))

    axs[0].pcolormesh(verts[0], verts[1], vals.T, shading="gouraud", norm=norm, cmap="turbo")
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
    fig.savefig(fname)


def plot_train_results(params, ans, figdir):
    """Plot training results.
    """

    parnames = [rf"$\alpha_{i}$" for i in range(1, 6)]
    params = {k: v / ans for k, v in params.items()}
    algs = ["alg-3", "alg-4", "alg-6", "alg-7", "analytical"]
    labels = ["Full Inv", "Diag Approx", "Interp Full", "Interp Diag", "Continuous"]

    # plot loss
    fig, axs = pyplot.subplots(1, 5, figsize=(7.5, 3.0), squeeze=False)
    for col in range(5):

        # determine the plotting y range
        valmax = max(*list(_[:, col].max() for _ in params.values()))
        valmin = min(*list(_[:, col].min() for _ in params.values()))
        delta = (valmax - valmin) * 0.01
        valmax += delta
        valmin -= delta

        axs[0, col].boxplot(
            [params[_][:, col] for _ in algs],
            vert=True, widths=0.6,
            medianprops={"lw": 1.5,},
            showmeans=True, meanprops={"ms": 4,},
            flierprops=dict(marker="o", mec="grey", alpha=0.3, ms=3),
            tick_labels=labels,
        )
        axs[0, col].axhline(1.0, color="tab:red", ls="-.", lw=1.5)
        axs[0, col].tick_params(axis="x", labelsize=8, rotation=90)
        axs[0, col].tick_params(axis="y", labelsize=8, rotation=-60)
        axs[0, col].yaxis.set_major_formatter(mticker.ScalarFormatter(False))
        axs[0, col].set_ylim(valmin, valmax)
        axs[0, col].set_title(parnames[col], fontsize=12)

    fig.supxlabel("DistroSA Algorithms")
    fig.supylabel("Relative to Truth")
    fig.set_facecolor("whitesmoke")
    fig.savefig(figdir/"trained_params")


def read_trained_data(figdir):
    """Read trained data.
    """

    lossout = {}
    paramout = {}
    timeout = {}
    itersout = {}
    for alg in ["alg-3", "alg-4", "alg-6", "alg-7", "analytical"]:
        tmp = []
        with open(_figdir/f"train-{alg}.dat", "rb") as fp:
            while True:
                try:
                    tmp.append(pickle.load(fp))
                except EOFError:
                    break

        # we only need the best loss and params among repetitions of each bootstrap
        nbts = max(tmp, key=lambda inp: inp["ibt"])["ibt"] + 1
        losses = numpy.full(nbts, numpy.inf, dtype=float)
        params = numpy.full((nbts, 5), numpy.inf, dtype=float)

        # times and numbers of iterations are from ALL optimizations
        times = numpy.full(len(tmp), numpy.inf, dtype=float)
        iters = numpy.full(len(tmp), -9999, dtype=int)

        for i, record in enumerate(tmp):
            if record["loss"] < losses[record["ibt"]]:
                losses[record["ibt"]] = record["loss"].numpy()
                params[record["ibt"], :] = record["pars"].numpy()
            times[i] = record["time"]  # in seconds
            iters[i] = record["iters"]

        lossout[alg] = losses
        paramout[alg] = params
        timeout[alg] = times
        itersout[alg] = iters

    return lossout, paramout, timeout, itersout


def get_pdf_errors(trainedpars, anspars, fname):
    """Get PDF errors
    """

    if fname.is_file():
        errs = torch.load(fname)
        return errs

    algs =  ["alg-3", "alg-4", "alg-6", "alg-7", "analytical"]
    verts = [torch.linspace(1e-6, 1.-1e-6, 51) for _ in range(2)]
    verts = [_.to("cpu") for _ in verts]
    anspars = torch.as_tensor(anspars).to("cpu")
    ansnorm = integrate(verts, lambda x: proxy_2d(x, anspars))

    errs = {}
    for alg in algs:

        if alg not in errs:
            errs[alg] = torch.zeros(trainedpars[alg].shape[0], dtype=torch.float64)
            errs[alg] = errs[alg].to("cpu")

        for i, pars in enumerate(trainedpars[alg]):
            pars = torch.tensor(pars).to("cpu")
            norm = integrate(verts, lambda _x: proxy_2d(_x, pars))

            def kernel(_x):
                _anspdf = proxy_2d(_x, anspars) / ansnorm
                _trainedpdf = proxy_2d(_x, pars) / norm
                _err = torch.abs(_trainedpdf-_anspdf)  # absolute
                _err = torch.where(_anspdf == 0, _err, _err/_anspdf)  # relative
                return _err

            with torch.no_grad():
                errs[alg][i] = integrate(verts, kernel).detach().cpu()

            print(alg, i, errs[alg][i])

    torch.save(errs, fname)
    return errs


def plot_pdf_errors(errs, figdir):
    """Plot PDF errors.
    """

    algs =  ["alg-3", "alg-4", "alg-6", "alg-7", "analytical"]
    labels = ["Full Inv", "Diag Approx", "Interp Full", "Interp Diag", "Continuous"]

    fig, axs = pyplot.subplots(1, 1, figsize=(2.5, 2.5), squeeze=False)

    axs[0, 0].boxplot(
        errs.values(),
        vert=True, widths=0.6,
        showmeans=False,
        capprops={"color": "tab:blue"},
        boxprops={"color": "tab:blue"},
        whiskerprops={"color": "tab:blue"},
        flierprops=dict(marker="o", mec="tab:blue", alpha=0.3, ms=3),
        medianprops={"lw": 1.5, "color": "tab:blue"},
        meanprops={"ms": 4, "mfc": "tab:blue", "mec": "tab:blue"},
        tick_labels=labels,
    )
    axs[0, 0].tick_params(axis="x", labelsize=8, rotation=90)
    axs[0, 0].tick_params(axis="y", labelsize=8, rotation=-60)
    axs[0, 0].yaxis.set_major_formatter(mticker.ScalarFormatter(False))
    axs[0, 0].set_ylim(0.0, 0.6)

    fig.supxlabel("DistroSA Algorithms")
    fig.supylabel(r"$L_1$ Error of PDF")
    fig.set_facecolor("whitesmoke")
    fig.savefig(figdir/"pdf_errs")


if __name__ == "__main__":
    import pathlib
    import pickle
    import pprint
    _figdir = pathlib.Path(__file__).resolve().parent.joinpath("figs")
    pyplot.style.use(_figdir.parent.joinpath("plot.mplstyle"))

    _out = torch.load(_figdir/"meta.dat")
    _out["alg-3"] = torch.load(_figdir/"alg-3.dat")
    _out["alg-4"] = torch.load(_figdir/"alg-4.dat")
    _out["alg-6"] = torch.load(_figdir/"alg-6.dat")
    _out["alg-7"] = torch.load(_figdir/"alg-7.dat")

    examine(_out["anspars"], 10000, _figdir/"obsrvs_hist")
    examine(_out["curpars"], 10000, _figdir/"current_hist")
    plot_loss_errors(_out, _figdir)
    plot_grad_errors(_out, _figdir)

    _losses, _params, _times, _iters = read_trained_data(_figdir)

    pprint.pprint({k: numpy.mean(v).item() for k, v in _times.items()})
    pprint.pprint({k: numpy.mean(v).item() for k, v in _iters.items()})
    plot_train_results(_params, _out["anspars"].numpy(), _figdir)
    examine(numpy.median(_params["alg-6"], axis=0), 10000, _figdir/"trained_hist")

    _errs_10k = get_pdf_errors(_params, _out["anspars"], _figdir/"pdferrs.dat")
    plot_pdf_errors(_errs_10k, _figdir)
