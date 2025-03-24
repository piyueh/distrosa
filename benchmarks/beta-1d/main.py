#!/usr/bin/env python3
# vim:fenc=utf-8

"""Validating the gradients of the energy loss w.r.t. a 1D beta's parameters.
"""
import itertools
import numpy
import torch
from torch import Tensor
from matplotlib import pyplot
import distrosa
import distrosa.utils


# global settings
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=15)
numpy.set_printoptions(precision=15)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# the domain of the 1D beta distribution is always fixed and always in [0, 1]
bounds = torch.tensor([[0.0, 1.0],]).to(device)

# mapping between algorithm keys and implementations
algcls = {
    "alg-2": distrosa.Sensitivity1D,
    "alg-3": distrosa.SensitivityND,
    "alg-4": distrosa.SensitivityNDDiag,
    "alg-6": distrosa.SensitivityNDInterp,
    "alg-7": distrosa.SensitivityNDDiagInterp,
}


def run(
    a: float, b: float, nvs: list[int], algs: list[str], ndraws: list[int],
    traindset: Tensor, eps: float
) -> dict:
    """Run and get numerical gradients of loss w.r.t. the parameters of a 1D beta.
    """

    nrepeats = 20
    grads = {}
    for alg, res, ndraw in itertools.product(algs, nvs, ndraws):

        if alg not in grads:
            grads[alg] = {}

        if res not in grads[alg]:
            grads[alg][res] = {}

        if ndraw not in grads[alg][res]:
            grads[alg][res][ndraw] = torch.zeros((nrepeats, 2), device="cpu")

        for i in range(nrepeats):

            print(f"Running {alg} with {res} vertices and {ndraw} draws [{i}].")

            # the "current" parameters; `device` must be inside to make it a leaf tensor
            curpars = torch.tensor([a, b], requires_grad=True, device=device)

            # background grid for the sensitivity calculator
            verts = [torch.linspace(_[0], _[1], res).to(device) for _ in bounds]

            # sensitivity calculator
            grader = algcls[alg](2, verts, eps).to(device)

            # a differentiable Beta sampler
            sampler = distrosa.utils.Beta1DSampler(grader).to(device)

            # the loss function
            lossfn = distrosa.utils.EmpiricalEnergyScore().to(device)

            # clear the gradient (just to play it safe)
            curpars.grad = None

            # generate predicitons via differentiable Beta sampler
            preds = sampler(ndraw, curpars)

            # calculate the loss
            loss = lossfn(preds, traindset)

            # backpropagate
            loss.backward()

            # we only store detached CPU tensors
            grads[alg][res][ndraw][i, :] = curpars.grad.detach().cpu()  # type: ignore

            # release mem
            loss = None; grad = None; preds = None; lossfn = None; sampler = None;
            grader = None; verts = None; curpars = None;

    return grads


def solution(a: float, b: float, traindset: Tensor) -> dict:
    """Analytical solution of the energy loss w.r.t. the parameters of a 1D beta.

    Arguments
    ---------
    a : float
        The first parameter of the beta distribution.
    b : float
        The second parameter of the beta distribution.

    Returns
    -------
    loss : Tensor
        The energy loss. A 0-D tensor.
    grad : Tensor
        The gradient of loss w.r.t. the parameters. Shape `(2,)`.
    """

    # the "current" parameters; `device` must be inside to make it a leaf tensor
    curpars = torch.tensor([a, b], requires_grad=True, device=device)

    # analytical energy loss function
    lossfn = distrosa.utils.AnalyticalEnergyScore(bounds, 16).to("cuda")
    lossfn.register(distrosa.utils.beta_1d_pdf)  # type: ignore

    # clear the gradient
    curpars.grad = None

    # calculate the loss
    loss = lossfn(curpars, traindset)

    # backpropagate
    loss.backward()

    # gradient w.r.t. the parameters
    grad = curpars.grad.detach().cpu()  # type: ignore

    return {"loss": loss.detach().cpu(), "grad": grad}


def finitedifference(a: float, b: float, ndraws: list[int], traindset: Tensor, eps: float):
    """Get the gradient of the energy loss w.r.t. the parameters via finite difference.
    """

    # number of repeats to get stats
    nrepeats = 20

    # current parameters
    curpars = torch.tensor([a, b], device=device)

    # the loss function is the same across all runs
    lossfn = distrosa.utils.EmpiricalEnergyScore().to(device)

    outs = {}
    for ndraw, i, j in itertools.product(ndraws, range(nrepeats), range(2)):

        print(ndraw, i, j)

        if ndraw not in outs:
            outs[ndraw] = torch.zeros((nrepeats, 2), device="cpu")

        with torch.no_grad():
            # positive perturbed parameters (in a)
            pars = curpars.detach().clone()
            pars[j] = pars[j] + eps
            preds = torch.distributions.beta.Beta(*pars).sample((ndraw,)).to(device)
            lossp = lossfn(preds, traindset)

            # negative perturbed parameters (in a)
            pars = curpars.detach().clone()
            pars[j] = pars[j] - eps
            preds = torch.distributions.beta.Beta(*pars).sample((ndraw,)).to(device)
            lossm = lossfn(preds, traindset)

        # gradient w.r.t. curpars[j]
        outs[ndraw][i, j] = ((lossp - lossm) / (2.0*eps)).cpu()

    return outs


if __name__ == "__main__":
    import pathlib

    # figure folder
    figdir = pathlib.Path(__file__).parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # the true parameters that generates training data
    _anspars = torch.tensor([2.31, 1.627]).to(device)

    # number of data point in the training dataset
    _ntrain = 10000

    # number of realizations in both training data and prediction during training
    _ndraws = [100, 1000, 10000, 100000]

    # resolution of the background grid for the sensitivity calculator
    _res = [1024,]

    # finite-difference step size
    _eps1 = 1e-3
    _eps2 = 1e-5

    # current Beta's parameters where we want to evaluate the gradient against
    _a, _b = 3.0, 1.4

    # algorithms we want to check
    _algs = ["alg-2", "alg-3", "alg-4", "alg-6", "alg-7"]

    # the training data
    _traindset = torch.distributions.beta.Beta(*_anspars).sample((_ntrain,)).to(device)

    # get analytical solution
    ans = solution(_a, _b, _traindset)

    # get finite-difference solutions
    fd1 = finitedifference(_a, _b, _ndraws, _traindset, _eps1)
    fd2 = finitedifference(_a, _b, _ndraws, _traindset, _eps2)

    # get numerical solutions
    computed = run(_a, _b, _res, _algs, _ndraws, _traindset, _eps2)

    # save data
    torch.save({
        "ans": ans, "ndraws": _ndraws, "computed": computed, "ntrain": _ntrain,
        "res": _res, "algs": _algs, "fd1": fd1, "fd2": fd2,
        "eps1": _eps1, "eps2": _eps2,
    }, figdir.joinpath("out.dat"))
