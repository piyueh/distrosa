#!/usr/bin/env python3
# vim:fenc=utf-8

"""Plotting.
"""
import re
import itertools
import numpy
import torch
from matplotlib import pyplot
from matplotlib import colors as mcolors
from matplotlib import cm as mcm
from matplotlib import ticker as mticker
from density import proxy_2d


torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=5, linewidth=120)
numpy.set_printoptions(precision=5, linewidth=120)


# mapping between algorithm keys and names in plots
alglbls = {
    "alg-3": "Full Inv", "alg-4": "Diag Approx", "alg-6": "Interp Full",
    "alg-7": "Interp Diag", "analytical": "Continuous",
}

# names of the parameters
parnames = [r"\alpha_1", r"\alpha_2", r"\alpha_3", r"\alpha_4", r"\alpha_5"]


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


def examine(curpars, fname):
    """Examine
    """

    # operate in PyTorch
    curpars = torch.as_tensor(curpars).to("cpu")

    # sampler and samples
    verts = [torch.logspace(-6, numpy.log10(1.-1e-6), 201) for _ in range(2)]

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
    ax.pcolormesh(verts[0], verts[1], vals.T, shading="gouraud", norm=norm, cmap="turbo")
    ax.set_xlabel(r"$x_1$")
    ax.set_ylabel(r"$x_2$")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    cbar = fig.colorbar(mappable, ax=ax, orientation="horizontal", pad=0.05, aspect=50)
    cbar.set_label("Normalized Probability Density")
    fig.savefig(fname)


def read_trained_data(outdir):
    """Read trained data.
    """
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

        # remove what are already done from all identifiers
        files = list(_.name for _ in outdir.glob(f"train-{alg}-*.dat"))
        for f in files:

            # extract info from filename
            res = re.search(rf"train-{alg}-(\d+?)-(\d+?)-(\d+?).dat", f)
            res = res.groups()  # type: ignore
            ntrain, ibt, irep = int(res[0]), int(res[1]), int(res[2])

            # read data
            with open(outdir/f, "rb") as fp:
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


def plot_trained_params(params, ans, fname):
    """Plot training results.
    """

    algs = list(params.keys())
    ntrains = list(params[algs[0]].keys())
    params = {k1: {k2: v2 / ans for k2, v2 in v1.items()} for k1, v1 in params.items()}

    cmap = pyplot.get_cmap("tab10")

    # plot loss
    fig, axs = pyplot.subplots(1, 5, figsize=(7.5, 3.0), squeeze=False)
    for col in range(5):  # loop over parameters

        for i, ntrain in enumerate(ntrains):

            axs[0, col].boxplot(
                [params[_][ntrain][:, col] for _ in algs],
                vert=True, widths=0.6,
                patch_artist=True,
                showfliers=False,
                showmeans=False,
                capprops={"color": cmap(3*i), "lw": 1.0},
                boxprops={"fc": cmap(3*i), "alpha": 0.7, "lw": 0},
                whiskerprops={"color": cmap(3*i), "lw": 1.0},
                medianprops={"lw": 2.0, "color": cmap(3*i)},
                tick_labels=list(alglbls[_] for _ in algs),
            )

        axs[0, col].axhline(1.0, color="k", ls="-.", lw=1.5)
        axs[0, col].tick_params(axis="x", labelsize=8, rotation=90)
        axs[0, col].tick_params(axis="y", labelsize=8, rotation=-60)
        axs[0, col].yaxis.set_major_formatter(mticker.ScalarFormatter(False))
        axs[0, col].set_title(f"${parnames[col]}$", fontsize=12)

    fig.supxlabel("DistroSA Algorithms")
    fig.supylabel("Relative to Truth")
    fig.set_facecolor("whitesmoke")
    fig.savefig(fname)


def get_pdf_errors(params, anspars, fname):
    """Get PDF errors
    """

    if fname.is_file():
        errs = torch.load(fname)
        return errs

    algs = list(params.keys())
    ntrains = list(params[algs[0]].keys())

    verts = [torch.linspace(1e-6, 1.-1e-6, 51) for _ in range(2)]
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
                _err = torch.abs(_trainedpdf-_anspdf)  # absolute
                return _err

            with torch.no_grad():
                errs[alg][ntrain][i] = integrate(verts, kernel).detach().cpu()

            print(alg, ntrain, i, errs[alg][ntrain][i])

    torch.save(errs, fname)
    return errs


def plot_pdf_errors(errs, fname):
    """Plot PDF errors.
    """

    algs = list(errs.keys())
    ntrains = list(errs[algs[0]].keys())
    cmap = pyplot.get_cmap("tab10")

    fig, axs = pyplot.subplots(1, 1, figsize=(2.5, 3.0), squeeze=False)

    for i, ntrain in enumerate(ntrains):
        axs[0, 0].boxplot(
            [errs[_][ntrain] for _ in algs],
            vert=True, widths=0.6,
            patch_artist=True,
            showfliers=False,
            showmeans=False,
            capprops={"color": cmap(3*i), "lw": 1.0},
            boxprops={"fc": cmap(3*i), "alpha": 0.7, "lw": 0},
            whiskerprops={"color": cmap(3*i), "lw": 1.0},
            medianprops={"lw": 2.0, "color": cmap(3*i)},
            tick_labels=list(alglbls[_] for _ in algs),
        )
    axs[0, 0].tick_params(axis="x", labelsize=8, rotation=90)
    axs[0, 0].tick_params(axis="y", labelsize=8, rotation=-60)
    axs[0, 0].yaxis.set_major_formatter(mticker.ScalarFormatter(False))

    fig.supxlabel("DistroSA Algorithms")
    fig.supylabel(r"$L_1$ Error of PDF")
    fig.set_facecolor("whitesmoke")
    fig.savefig(fname)


if __name__ == "__main__":
    import pathlib
    import pickle
    import pprint
    _figdir = pathlib.Path(__file__).resolve().parent.joinpath("figs")
    pyplot.style.use(_figdir.parent.joinpath("plot.mplstyle"))

    # load meta data
    _out = torch.load(_figdir/"meta.dat")

    # load trained data
    _, _params, _times, _iters = read_trained_data(_figdir)

    # print averaged time
    pprint.pprint({
        k1: {k2: numpy.mean(v2).item() for k2, v2 in v1.items()}
        for k1, v1 in _times.items()
    })

    # print time std
    pprint.pprint({
        k1: {k2: numpy.std(v2).item() for k2, v2 in v1.items()}
        for k1, v1 in _times.items()
    })

    # print averaged iteration per optimization
    pprint.pprint({
        k1: {k2: numpy.mean(v2).item() for k2, v2 in v1.items()}
        for k1, v1 in _iters.items()
    })

    # print iteration per optimization STD
    pprint.pprint({
        k1: {k2: numpy.std(v2).item() for k2, v2 in v1.items()}
        for k1, v1 in _iters.items()
    })

    # plot trained parameters
    plot_trained_params(_params, _out["anspars"].numpy(), _figdir/"trained_params")

    # get PDF absolute errors
    _errs = get_pdf_errors(_params, _out["anspars"], _figdir/"pdferrs.dat")

    # plot PDF errors
    plot_pdf_errors(_errs, _figdir/"trained_pdf_errs")

    # showcase a PDF contourf
    examine(numpy.median(_params["alg-6"][10000], axis=0), _figdir/"trained_pdf")
