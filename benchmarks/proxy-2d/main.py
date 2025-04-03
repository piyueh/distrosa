#!/usr/bin/env python3
# vim:fenc=utf-8

"""Check gradient of loss and train the parameters of the 2D proxy distribution.

The density function is:

    f(x; p) = x_0^{p_0} (1 - x_0)^{p_1} x_1^{p_2} (1 - x_1)^{p_3} (1 + p_4 x_0 x_1)

where 0 < x_0, x_1 < 1; x := [x_0, x_1]; and p := [p_0, p_1, p_2, p_3, p_4].

The parameters have upper and lower bounds:

* parmin: [-0.5, 2.75, 0.0, 3.0, 0.0]
* parmax: [1.0, 4.0, 1.3, 4.5, 1.5]

The ground truth parameters are: ans = [0.5, 3.0, 0.3, 4.0, 0.75].
"""
import pathlib
import math
import time
import itertools
import pickle
from typing import Sequence
import numpy
import torch
from torch import Tensor
from torch import multiprocessing as mp
import distrosa
import distrosa.utils
from density import proxy_2d

# figure folder
figdir = pathlib.Path(__file__).parent.joinpath("figs")
figdir.mkdir(exist_ok=True)

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=15)
numpy.set_printoptions(precision=5, formatter={"float": "{:.5e}".format})
torch.cpu.manual_seed = torch.manual_seed  # type: ignore

# mapping between algorithm keys and implementations
algcls = {
    "alg-3": distrosa.SensitivityND,
    "alg-4": distrosa.SensitivityNDDiag,
    "alg-6": distrosa.SensitivityNDInterp,
    "alg-7": distrosa.SensitivityNDDiagInterp,
}


def gentrain(ndraw: int, tol: float, params: Tensor) -> Tensor:
    """Generate training dataset.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    params = params.to(device)

    # backup the current random state and manually set the seed
    if device.type == "cuda":
        state = torch.cuda.get_rng_state(device)
        torch.cuda.manual_seed(100)
    else:
        state = torch.get_rng_state()
        torch.manual_seed(100)

    with torch.no_grad():
        verts = [torch.linspace(tol, 1.-tol, 11).to(device) for _ in range(2)]
        sampler = distrosa.utils.RejectionSampler(proxy_2d, verts, None).to(device)
        samples = sampler(ndraw, params)

    # restore the random state
    if device.type == "cuda":
        torch.cuda.set_rng_state(state, device)
    else:
        torch.set_rng_state(state)

    return samples.detach()


def get_analytical_grads(params: Tensor, bounds: Tensor, traindset: Tensor):
    """Analytical solution of the energy loss w.r.t. the parameters of a 1D beta.

    Arguments
    ---------
    params : Tensor
    traindset : Tensor

    Returns
    -------
    loss : Tensor
        The energy loss. A 0-D tensor.
    grad : Tensor
        The gradient of loss w.r.t. the parameters. Shape `(2,)`.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    traindset = traindset.to(device)

    # the "current" parameters; `device` must be inside to make it a leaf tensor
    curpars = params.detach().clone().to(device).requires_grad_(True)

    # gridlines for piecewise integration
    gridlines = [torch.logspace(math.log10(_[0]), math.log10(_[1]), 51) for _ in bounds]
    gridlines = [_.to(device) for _ in gridlines]

    # analytical energy loss function
    lossfn = distrosa.utils.BlockAnalyticalEnergyScore(gridlines, 3, proxy_2d)
    lossfn = lossfn.to(device)
    loss = lossfn(curpars, traindset)
    loss.backward()
    grad = curpars.grad.detach().cpu()  # type: ignore

    print("Analytical:")
    print(loss.item(), grad.numpy())

    return {"loss": loss.detach().cpu(), "grad": grad}


def get_numerical_grads(_tdset, _pars, _xbds, _nv, _eps, _nreps, _ndraws, _alg, _dvce):

    # for reproducibility
    if _dvce.type == "cuda":
        state = torch.cuda.get_rng_state(_dvce)
        torch.cuda.manual_seed(8888)
    else:
        state = torch.get_rng_state()
        torch.manual_seed(8888)

    _grads = {}
    _losses = {}

    # move training data to the device
    _tdset = _tdset.to(_dvce)

    # the "current" parameters; `device` must be inside to make it a leaf tensor
    _pars = _pars.detach().clone().to(_dvce).requires_grad_(True)

    # sensitivity calculator
    _gd1 = [torch.logspace(math.log10(_[0]), math.log10(_[1]), _nv) for _ in _xbds]
    _grader = algcls[_alg](len(_pars), _gd1, _eps, proxy_2d).to(_dvce)

    # grid for piecewise constant rejection sampler
    _gd2 = [torch.linspace(_[0], _[1], 11) for _ in _xbds]
    _sampler = distrosa.utils.RejectionSampler(proxy_2d, _gd2, _grader).to(_dvce)

    # loss function and optimizer
    _lossfn = distrosa.utils.EmpiricalEnergyScore().to(_dvce)

    for _ndr, _irep in itertools.product(_ndraws, range(_nreps)):

        if _ndr not in _grads:
            _grads[_ndr] = torch.zeros((_nreps, _pars.numel()))
            _losses[_ndr] = torch.full((_nreps,), numpy.nan)

        # clear the gradient
        _pars.grad = None

        # generate predicitons via differentiable Beta sampler
        _preds = _sampler(_ndr, _pars)

        # calculate the loss
        _loss = _lossfn(_preds, _tdset)

        # backpropagate
        _loss.backward()

        # we only store detached CPU tensors
        _grad = _pars.grad.detach().cpu()  # type: ignore
        _grads[_ndr][_irep, :] = _grad
        _losses[_ndr][_irep] = _loss.item()

        print(f"[{_alg}, {_nv}^2, {_ndr}, {_irep}] {_loss.item()}; {_grad.numpy()}")

        # release mem
        _loss = None; _preds = None;

    # save data
    torch.save({"loss": _losses, "grad": _grads}, figdir/f"{_alg}.dat")

    # restore the random state
    if _dvce.type == "cuda":
        torch.cuda.set_rng_state(state, _dvce)
    else:
        torch.set_rng_state(state)

    return None  # end


def trainer_analytical(traindset, pbounds, xbounds, maxiters, *args, **kwargs):
    """Train the parameters of the 2D proxy distribution.
    """

    # aliases
    device = traindset.device
    parmins = pbounds[:, 0]
    parlens = pbounds[:, 1] - pbounds[:, 0]

    # initial guesses for parameters
    theta = torch.randn((pbounds.shape[0],), dtype=pbounds.dtype, device=device)
    theta = theta.requires_grad_(True)

    # gridlines for piecewise integration
    grid = [torch.logspace(math.log10(_[0]), math.log10(_[1]), 51) for _ in xbounds]
    grid = [_.to(device) for _ in grid]

    # analytical energy loss function
    lossfn = distrosa.utils.BlockAnalyticalEnergyScore(grid, 3, proxy_2d)
    lossfn = lossfn.to(device)

    optimizer = torch.optim.LBFGS([theta,], line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()  # clear the gradient
        pars = parmins + parlens * torch.sigmoid(theta)
        loss = lossfn(pars, traindset)  # calculate the loss
        loss.backward()  # backpropagate
        return loss

    # print initial state
    pars = parmins + parlens * torch.sigmoid(theta.detach().clone())

    # synchronous for timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tbg = time.perf_counter_ns()

    for counter in range(maxiters):

        optimizer.step(closure)

        newpars = parmins + parlens * torch.sigmoid(theta.detach().clone())
        deltas = torch.abs((newpars-pars)/pars)
        pars = newpars

        if torch.all(deltas < 1e-6):
            break

    # synchronous for timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ted = time.perf_counter_ns()
    tcost = (ted - tbg) / 1e9

    loss = lossfn(pars, traindset)

    out = {
        "iters": counter+1,  # type: ignore
        "time": tcost,  # in seconds
        "loss": loss.detach().clone().cpu(),
        "pars": pars.detach().clone().cpu(),
    }

    return out


def trainer_empirical(traindset, pbounds, xbounds, maxiters, nverts, eps, alg):
    """Train the parameters of the 2D proxy distribution.
    """

    device = traindset.device

    # number of predicitons
    npreds = traindset.shape[0]

    # aliases
    parmins = pbounds[:, 0]
    parlens = pbounds[:, 1] - pbounds[:, 0]

    # initial parameters in the infinite domain
    theta = torch.randn((pbounds.shape[0],), dtype=pbounds.dtype, device=device)
    theta = theta.requires_grad_(True)

    # sensitivity calculator
    gd1 = [torch.logspace(math.log10(_[0]), math.log10(_[1]), nverts) for _ in xbounds]
    grader = algcls[alg](len(theta), gd1, eps, proxy_2d).to(device)

    # grid for piecewise constant rejection sampler
    gd2 = [torch.linspace(_[0], _[1], 11) for _ in xbounds]
    sampler = distrosa.utils.RejectionSampler(proxy_2d, gd2, grader, npreds).to(device)

    # loss function and optimizer
    lossfn = distrosa.utils.EmpiricalEnergyScore().to(device)
    optimizer = torch.optim.LBFGS([theta,], line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()  # clear the gradient
        pars = parmins + parlens * torch.sigmoid(theta)
        preds = sampler(npreds, pars)
        loss = lossfn(preds, traindset)  # calculate the loss
        loss.backward()  # backpropagate
        return loss

    # print initial state
    pars = parmins + parlens * torch.sigmoid(theta.detach().clone())

    # synchronous for timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    tbg = time.perf_counter_ns()

    for counter in range(maxiters):
        optimizer.step(closure)
        newpars = parmins + parlens * torch.sigmoid(theta.detach().clone())
        deltas = torch.abs((newpars-pars)/pars)
        pars = newpars

        if torch.all(deltas < 1e-6):
            break

    # synchronous for timing
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    ted = time.perf_counter_ns()
    tcost = (ted - tbg) / 1e9

    preds = sampler(npreds, pars)
    loss = lossfn(preds, traindset)

    out = {
        "iters": counter+1,  # type: ignore
        "time": tcost,  # in seconds
        "loss": loss.detach().clone().cpu(),
        "pars": pars.detach().clone().cpu(),
    }

    return out


def bootstrapping(nbts, nreps, tdset, pbds, xbds, nv, eps, alg, device):
    """Bootstrapping analysis.
    """

    print(f"starting {alg}")

    trainer = trainer_analytical if alg == "analytical" else trainer_empirical

    # move inputs to the device
    tdset = tdset.to(device)
    pbds = pbds.to(device)
    xbds = xbds.to(device)

    # weights of bootstrapping data using multinomial distribution
    wts = torch.ones(tdset.shape[0], dtype=tdset.dtype).to(device)

    # all bootstraps and repetitions
    cases = set(itertools.product(range(nbts), range(nreps)))

    # gether what are already done
    if figdir.joinpath(f"train-{alg}.dat").exists():
        with open(figdir/f"train-{alg}.dat", "rb") as f:
            while True:
                try:
                    _result = pickle.load(f)
                    cases.remove((_result["ibt"], _result["irep"]))
                except EOFError:
                    break
    else:
        figdir.joinpath(f"train-{alg}.dat").touch()

    print(f"{alg} remaining cases: {len(cases)}")

    # turn into soreted list
    cases = sorted(cases, key=lambda x: x[0])

    # loop over bootstraps
    for ibt, irep in cases:

        if ibt == 0:
            _ids = torch.arange(tdset.shape[0])
        else:
            getattr(torch, device.type).manual_seed(ibt)
            _ids = torch.multinomial(wts, tdset.shape[0], replacement=True)

        # cases with the same ibt share the same training dataset
        _dset = tdset[_ids]

        # unique seed for each ibt-irep conbination
        getattr(torch, device.type).manual_seed(ibt*1000+irep)
        _result = trainer(_dset, pbds, xbds, 25, nv, eps, alg)

        # add extra information
        _result.update({"ibt": ibt, "irep": irep, "alg": alg})

        # save
        with open(figdir/f"train-{alg}.dat", "ab") as f:
            pickle.dump(_result, f)

        msg = f"{alg};"
        msg += f"{ibt+1}/{nbts};{irep+1}/{nreps};"
        msg += f"loss:{_result['loss'].item():.5e};"
        msg += f"time:{_result['time']:.3e}s;"
        msg += f"iters:{_result['iters']};"
        msg += f"pars:{_result['pars'].numpy()}"
        print(msg)

    return None  # end


if __name__ == "__main__":

    # configurations
    _tol = 1e-6  # tol <= x <= 1-tol
    _eps = 1e-6  # finite difference step
    _nbts1 = 200  # number of bootstraps to get stats for error convergence
    _nbts2 = 30  # number of bootstraps to get stats for training performance
    _nreps = 10  # 10  # number of repetitions to get stats
    _ndraws = [128, 512, 2048, 8192, 32768, 131072]
    _ntrain1 = 10000  # number of data points in the training dataset
    _ntrain2 = 10000  # number of data points in the training dataset
    _res1 = 1024  # resolution of the background grid
    _res2 = 64  # resolution of the background grid
    _algs = ["alg-7", "alg-4", "alg-6", "alg-3"]
    _curpars = torch.tensor([0.25, 3.375, 0.65, 3.75, 0.1])

    # ground truth parameters and bounds
    _anspars = torch.tensor([0.5, 3.0, 0.3, 4.0, 0.75])
    _parmin = torch.tensor([-0.5, 2.75, 0.0, 3.0, 0.0])
    _parmax = torch.tensor([1.0, 4.0, 1.3, 4.5, 1.5])
    _pbds = torch.stack([_parmin, _parmax], dim=1)

    # bounds
    _xbds = torch.tensor([[_tol, 1-_tol], [_tol, 1-_tol]])

    # get training dataset
    _train1 = gentrain(_ntrain1, _tol, _anspars)
    _train2 = gentrain(_ntrain2, _tol, _anspars)

    # get analytical gradients
    _truegrads = get_analytical_grads(_curpars, _xbds, _train1)

    # determine the way to parallelize the computation
    if torch.cuda.is_available():
        mp.set_start_method("spawn")  # CUDA needs to use spawn
        worldsize = torch.cuda.device_count()
        devices = [torch.device(f"cuda:{_}") for _ in range(worldsize)]
    else:  # cpu
        worldsize = mp.cpu_count() // 2  # in case of hyperthreading
        devices = [torch.device("cpu") for _ in range(worldsize)]

    # infinite device/worker iterators
    worker = itertools.cycle(devices)

    # get numerical gradients
    args = (_train1, _curpars, _xbds, _res1, _eps, _nbts1, _ndraws)
    with mp.Pool(worldsize) as pool:
        pool.starmap(get_numerical_grads, [args+(_, next(worker)) for _ in _algs])

    # get training performance
    args = (_nbts2, _nreps, _train2, _pbds, _xbds, _res2, _eps)
    cases = [args + (_, next(worker)) for _ in _algs]
    cases.extend([args + ("analytical", next(worker))])
    with mp.Pool(worldsize) as pool:
        pool.starmap(bootstrapping, cases)

    # save data
    torch.save({
        "truegrads": _truegrads, "ndraws": _ndraws,
        "ntrain1": _ntrain1, "ntrain2": _ntrain2,
        "train1": _train1, "train2": _train2,
        "res1": _res1, "res2": _res2,
        "algs": _algs, "curpars": _curpars, "anspars": _anspars, "pbounds": _pbds,
        "xbounds": _xbds, "eps": _eps, "tol": _tol,
    }, figdir.joinpath("meta.dat"))
