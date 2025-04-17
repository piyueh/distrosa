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
import re
import pathlib
import math
import time
import itertools
import pickle
import multiprocessing as mp
import numpy
import torch
from torch import Tensor
import distrosa
import distrosa.utils
from density import proxy_2d

# figure folder
figdir = pathlib.Path(__file__).parent.joinpath("figs")
figdir.mkdir(exist_ok=True)

torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=5)
numpy.set_printoptions(precision=5, formatter={"float": "{:.5e}".format})
torch.cpu.manual_seed = torch.manual_seed  # type: ignore

# no need for any built-in multithreading, which interfere with multiprocessing
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

# mapping between algorithm keys and implementations
algcls = {
    "alg-3": distrosa.SensitivityND,
    "alg-4": distrosa.SensitivityNDDiag,
    "alg-6": distrosa.SensitivityNDInterp,
    "alg-7": distrosa.SensitivityNDDiagInterp,
}


def gentrain(ndraw: int, xbounds: Tensor, params: Tensor) -> Tensor:
    """Generate training dataset.

    Always on CPU.
    """

    xbounds = xbounds.to("cpu")
    params = params.to("cpu")

    # backup the current random state and manually set the seed
    state = torch.get_rng_state()
    torch.manual_seed(100)

    with torch.no_grad():
        verts = [torch.linspace(_[0], _[1], 11, device="cpu") for _ in xbounds]
        sampler = distrosa.utils.RejectionSampler(proxy_2d, verts, None)
        samples = sampler(ndraw, params)

    # restore the random state
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


def gradworker(inqueue, device, curpars, trainset, xbds, nv, eps):
    """Worker of evaluating gradients.
    """

    proc = mp.current_process()
    print(f"[{proc.name}] Started. Device: {device}")

    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)

    # they are supposed to be on CPU before
    xbds = xbds.clone().to(device)
    trainset = trainset.clone().to(device)
    curpars = curpars.detach().clone().to(device).requires_grad_(True)

    # sensitivity calculator for all algorithms
    vert1 = [torch.logspace(math.log10(_[0]), math.log10(_[1]), nv) for _ in xbds]
    graders = {
        alg: algcls[alg](len(curpars), vert1, eps, proxy_2d).to(device)
        for alg in ["alg-3", "alg-4", "alg-6", "alg-7"]
    }

    # samplers for all algorithms
    vert2 = [torch.linspace(_[0], _[1], 11) for _ in xbds]
    samplers = {
        alg: distrosa.utils.RejectionSampler(proxy_2d, vert2, graders[alg]).to(device)
        for alg in ["alg-3", "alg-4", "alg-6", "alg-7"]
    }

    # loss function and optimizer
    lossfn = distrosa.utils.EmpiricalEnergyScore().to(device)

    while True:
        try:
            case = inqueue.get(True, 5)
            if case is None:
                inqueue.task_done()
                break
        except mp.queues.Empty:  # type: ignore
            break

        alg, ndraw, irep = case  # extract info

        # reset the random number generator using ndraw and irep
        getattr(torch, curpars.device.type).manual_seed(ndraw+irep)

        # clear the gradient
        curpars.grad = None

        # generate predicitons via differentiable Beta sampler
        preds = samplers[alg](ndraw, curpars)

        # calculate the loss
        loss = lossfn(preds, trainset)

        # backpropagate
        loss.backward()

        # we only store detached CPU tensors
        grad = curpars.grad.detach().cpu()  # type: ignore
        loss = loss.item()

        print(f"[{proc.name}] {(alg, ndraw, irep)}, {loss}, {grad.numpy()}")

        # save
        with open(figdir/f"grad-{alg}-{ndraw}-{irep:03d}.dat", "wb") as f:
            pickle.dump(dict(alg=alg, ndraw=ndraw, irep=irep, loss=loss, grad=grad), f)

        del alg, ndraw, irep, case, loss, grad

        inqueue.task_done()

    print(f"[{proc.name}] Ended.")
    return None


def get_numerical_grads(tdset, pars, xbds, nv, eps, nreps, ndraws, algs: list[str]):
    """Get numerical gradients using DistroSA.
    """

    # working on CPUs first
    tdset = tdset.cpu()
    pars = pars.cpu()
    xbds = xbds.cpu()

    # identifiers for all bootstraps and repetitions
    cases = set(itertools.product(algs, ndraws, range(nreps)))

    # remove what are already done from all identifiers
    files = list(_.name for _ in figdir.glob(f"grad-*.dat"))
    for f in files:
        res = re.search(r"grad-(.*)-(\d+?)-(\d+?).dat", f).groups()  # type: ignore
        cases.remove((res[0], int(res[1]), int(res[2])))

    # determine the way to parallelize the computation
    if torch.cuda.is_available():
        worldsize = torch.cuda.device_count()
        devices = [f"cuda:{_}" for _ in range(worldsize)]
    else:  # cpu
        worldsize = mp.cpu_count() // 2  # in case of hyperthreading
        devices = ["cpu" for _ in range(worldsize)]

    # prepare a shared queue to hold all cases
    inpq = mp.JoinableQueue()
    for case in cases:
        inpq.put(case)

    for rank in range(worldsize):
        inpq.put(None)

    # put them together for a shorter code later
    workerargs = (pars, tdset, xbds, nv, eps)

    # start workers
    procs = []
    for rank in range(worldsize):
        p = mp.Process(target=gradworker, args=(inpq, devices[rank])+workerargs)
        p.start()
        procs.append(p)

    # wait until the queue is empty
    inpq.join()

    return None


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


def btworker(inqueue, device, pbds, xbds, nv, eps, trainset, btids):
    """A worker for bootstrapping analysis.
    """

    proc = mp.current_process()
    print(f"[{proc.name}] Started. Device: {device}")

    if torch.device(device).type == "cuda":
        torch.cuda.set_device(device)

    # they are supposed to be on CPU before
    pbds = pbds.clone().to(device)
    xbds = xbds.clone().to(device)
    trainset = {k: v.clone().to(device) for k, v in trainset.items()}
    btids = {k: v.clone().to(device) for k, v in btids.items()}

    while True:
        try:
            case = inqueue.get(True, 5)
            if case is None:
                inqueue.task_done()
                break
        except mp.queues.Empty:  # type: ignore
            break

        alg, ntrain, ibt, irep = case  # extract info
        trainer = trainer_analytical if alg == "analytical" else trainer_empirical
        dset = trainset[ntrain][btids[ntrain][ibt]].clone()
        getattr(torch, dset.device.type).manual_seed(ibt*1000+irep)
        result = trainer(dset, pbds, xbds, 25, nv, eps, alg)
        result.update({"ibt": ibt, "irep": irep, "alg": alg})

        print(
            f"[{proc.name}] " +
            f"{(alg, ntrain, ibt, irep)}, {result["loss"]}, {result["pars"].numpy()}"
        )

        # save
        with open(figdir/f"train-{alg}-{ntrain}-{ibt:03d}-{irep:02d}.dat", "wb") as f:
            pickle.dump(result, f)

        del alg, ntrain, ibt, irep, case, dset, trainer, result

        inqueue.task_done()

    print(f"[{proc.name}] Ended.")
    return None


def bootstrapper(algs: list[str], ntrains, nbts, nreps, pbds, xbds, nv, anspars, eps):
    """Bootstrapping analysis.

    Arguments
    ---------
    algs : list[str]
    ntrains : int
    nbts : int
    nreps : int
    pbds : P by 2 tensor
    xbds : N by 2 tensor
    nv : int
    eps : float
    """

    # working on CPUs first
    xbds = xbds.cpu()
    pbds = pbds.cpu()
    anspars = anspars.cpu()

    # get training data and IDs for bootstrapped training data
    trainset = {}  # training dataset
    btids = {}  # bootstrap indices
    for ntrain in ntrains:
        trainset[ntrain] = gentrain(ntrain, xbds, _anspars)
        torch.manual_seed(ntrain)
        wts = torch.ones(ntrain, dtype=anspars.dtype)
        btids[ntrain] = torch.zeros((nbts, ntrain), dtype=torch.int64,)
        btids[ntrain][0, ...] = torch.arange(ntrain)
        for ibt in range(1, nbts):
            btids[ntrain][ibt, ...] = torch.multinomial(wts, ntrain, replacement=True)

    # identifiers for all bootstraps and repetitions
    cases = set(itertools.product(algs, ntrains, range(nbts), range(nreps)))

    # remove what are already done from all identifiers
    files = list(_.name for _ in figdir.glob(f"train-*.dat"))
    for f in files:
        res = re.search(r"train-(.*)-(\d+?)-(\d+?)-(\d+?).dat", f)
        res = res.groups()  # type: ignore
        alg, ntrain, ibt, irep = res[0], int(res[1]), int(res[2]), int(res[3])
        cases.remove((alg, ntrain, ibt, irep))

    # determine the way to parallelize the computation
    if torch.cuda.is_available():
        worldsize = torch.cuda.device_count()
        devices = [f"cuda:{_}" for _ in range(worldsize)]
    else:  # cpu
        worldsize = mp.cpu_count() // 2  # in case of hyperthreading
        devices = ["cpu" for _ in range(worldsize)]

    # prepare a shared queue to hold all cases
    inpq = mp.JoinableQueue()
    for case in cases:
        inpq.put(case)

    for rank in range(worldsize):
        inpq.put(None)

    # put them together for a shorter code later
    workerargs = (pbds, xbds, nv, eps, trainset, btids)

    # start workers
    procs = []
    for rank in range(worldsize):
        p = mp.Process(target=btworker, args=(inpq, devices[rank])+workerargs)
        p.start()
        procs.append(p)

    # wait until the queue is empty
    inpq.join()

    return None


if __name__ == "__main__":

    # CUDA backend can not use `fork` (this line must be inside the main block)
    mp.set_start_method("forkserver")

    # configurations
    _tol = 1e-6  # tol <= x <= 1-tol
    _eps = 1e-6  # finite difference step
    _nbts1 = 200  # number of bootstraps to get stats for error convergence
    _nbts2 = 100  # number of bootstraps to get stats for training performance
    _nreps = 10  # 10  # number of repetitions to get stats
    _ndraws = [128, 512, 2048, 8192, 32768, 131072]
    _ntrain1 = 10000  # number of data points in the training dataset
    _ntrain2 = [10000, 50000]  # numbers of data points in the training dataset
    _res1 = 1024  # resolution of the background grid
    _res2 = 64  # resolution of the background grid
    _algs1 = ["alg-7", "alg-4", "alg-6", "alg-3"]
    _algs2 = ["alg-7", "alg-4", "alg-6", "alg-3", "analytical"]
    _curpars = torch.tensor([0.25, 3.375, 0.65, 3.75, 0.1])

    # ground truth parameters and bounds
    _anspars = torch.tensor([0.5, 3.0, 0.3, 4.0, 0.75])
    _parmin = torch.tensor([-0.5, 2.75, 0.0, 3.0, 0.0])
    _parmax = torch.tensor([1.0, 4.0, 1.3, 4.5, 1.5])
    _pbds = torch.stack([_parmin, _parmax], dim=1)

    # bounds
    _xbds = torch.tensor([[_tol, 1-_tol], [_tol, 1-_tol]])

    # get training dataset for gradient calculation
    _train1 = gentrain(_ntrain1, _xbds, _anspars)

    # get analytical gradients
    _truegrads = get_analytical_grads(_curpars, _xbds, _train1)

    # get numerical gradients
    get_numerical_grads(_train1, _curpars, _xbds, _res1, _eps, _nbts1, _ndraws, _algs1)

    # get simulation-based inference
    bootstrapper(_algs2, _ntrain2, _nbts2, _nreps, _pbds, _xbds, _res2, _anspars, _eps)

    # save data
    torch.save({
        "truegrads": _truegrads, "ndraws": _ndraws,
        "ntrain1": _ntrain1, "ntrain2": _ntrain2, "train1": _train1,
        "res1": _res1, "res2": _res2,
        "algs1": _algs1, "_algs2": _algs2,
        "curpars": _curpars, "anspars": _anspars, "pbounds": _pbds,
        "xbounds": _xbds, "eps": _eps, "tol": _tol,
    }, figdir.joinpath("meta.dat"))
