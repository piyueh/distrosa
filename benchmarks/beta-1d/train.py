#!/usr/bin/env python3
# vim:fenc=utf-8

"""Validating the gradients of the energy loss w.r.t. a 1D beta's parameters.
"""
import itertools
import numpy
import torch
import torch.distributions as dists
from torch import Tensor
import distrosa
import distrosa.utils
from matplotlib import pyplot


# global settings
torch.set_default_dtype(torch.float64)
torch.set_printoptions(precision=15)
numpy.set_printoptions(precision=15)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1D beta has a fixed finite-domain of [0, 1]
bounds = torch.tensor([[0.0, 1.0],]).to(device)


def closure_fd(pars: Tensor, train: Tensor, eps):
    """Closure function for the optimizer using finite-difference.
    """

    # we use the same number of prediction as the training set
    ndraw = train.shape[0]

    # the loss function is the same across all runs
    lossfn = distrosa.utils.EmpiricalEnergyScore().to(device)

    # loss at the current parameters
    preds = torch.distributions.beta.Beta(*pars).sample((ndraw,)).to(device)
    loss = lossfn(preds, train)

    # gradient holder
    grad = torch.zeros_like(pars)

    for j in range(len(pars)):

        # positive perturbed parameters (in a)
        pars = pars.detach().clone()
        pars[j] = pars[j] + eps
        preds = torch.distributions.beta.Beta(*pars).sample((ndraw,)).to(device)
        lossp = lossfn(preds, train)

        # negative perturbed parameters (in a)
        pars = pars.detach().clone()
        pars[j] = pars[j] - eps
        preds = torch.distributions.beta.Beta(*pars).sample((ndraw,)).to(device)
        lossm = lossfn(preds, train)

        # gradient w.r.t. curpars[j]
        grad[j] = ((lossp - lossm) / (2.0*eps))

    return loss, grad


def closure_distrosa(pars: Tensor, train: Tensor, res, eps=1e-3):
    """Closure function for the optimizer using DistroSA.

    We assume `pars.requires_grad` is already `True`.
    """

    # we use the same number of prediction as the training set
    ndraw = train.shape[0]

    # clear the gradient (just to play it safe)
    pars.grad = None

    # background grid for the sensitivity calculator
    verts = [torch.linspace(_[0], _[1], res).to(device) for _ in bounds]

    # sensitivity calculator
    grader = distrosa.Sensitivity1D(2, verts, eps).to(device)

    # a differentiable Beta sampler
    sampler = distrosa.utils.Beta1DSampler(grader).to(device)

    # the loss function
    lossfn = distrosa.utils.EmpiricalEnergyScore().to(device)

    # generate predicitons via differentiable Beta sampler
    preds = sampler(ndraw, pars)

    # calculate the loss
    loss = lossfn(preds, train)

    # backpropagate
    loss.backward()

    return loss, pars.grad


def closure_analytical(pars: Tensor, train: Tensor):
    """Closure function for the optimizer using analytical calculation.

    We assume `pars.requires_grad` is already `True`.
    """

    # clear the gradient (just to play it safe)
    pars.grad = None

    # analytical energy loss function
    lossfn = distrosa.utils.AnalyticalEnergyScore(bounds, 16).to(device)
    lossfn.register(distrosa.utils.beta_1d_pdf)  # type: ignore

    # calculate the loss
    loss = lossfn(pars, train)

    # backpropagate
    loss.backward()

    return loss, pars.grad


def get_loss_surface(pbounds: Tensor, train: Tensor):
    """Get the surface of the loss function w.r.t. the parameters.
    """

    # all possible parameter values
    params = torch.stack(torch.meshgrid(
        torch.linspace(pbounds[0, 0], pbounds[0, 1], 21, device=device),
        torch.linspace(pbounds[1, 0], pbounds[1, 1], 21, device=device),
        indexing="ij"
    ), dim=-1).requires_grad_(True)

    # result holder
    lossvals = torch.zeros(params.shape[:-1]).to(device)

    # loss function
    lossfn = distrosa.utils.AnalyticalEnergyScore(bounds, 16).to(device)

    # analytical energy loss function
    lossfn.register(distrosa.utils.beta_1d_pdf)  # type: ignore

    for i, j in itertools.product(range(params.shape[0]), range(params.shape[1])):
        with torch.no_grad():
            lossvals[i, j] = lossfn(params[i, j], train)

    return params.detach().cpu(), lossvals.detach().cpu()


def get_gradients_1d(pbounds: Tensor, train: Tensor, b: float, res: int, eps: float):
    """Get the gradients at different values of one parameter.

    Other parameters are fixed.
    """

    # number of repeats to get stats
    nreps = 20

    # number of parameters in a
    npars = 51

    # all possible parameter values
    params = torch.stack(torch.meshgrid(
        torch.linspace(pbounds[0, 0], pbounds[0, 1], npars),
        torch.asarray([b,]),
        indexing="ij"
    ), dim=-1).reshape(-1, 2).to(device)

    # result holder
    losses = {
        "distrosa": torch.zeros((npars, nreps), device="cpu"),
        "fd": torch.zeros((npars, nreps), device="cpu")
    }
    grads = {
        "distrosa": torch.zeros((npars, nreps, 2), device="cpu"),
        "fd": torch.zeros((npars, nreps, 2), device="cpu")
    }

    for ipar, par in enumerate(params):
        for irep in range(nreps):

            par = par.clone().detach().requires_grad_(True)

            val, jac = closure_distrosa(par, train, res)
            losses["distrosa"][ipar, irep] = val.detach().cpu()
            grads["distrosa"][ipar, irep, :] = jac.detach().cpu()  # type: ignore

            val, jac = closure_fd(par, train, eps)
            losses["fd"][ipar, irep] = val.detach().cpu()
            grads["fd"][ipar, irep, :] = jac.detach().cpu()  # type: ignore

            print(ipar, irep, par.detach().cpu().numpy())

    return params.detach().cpu(), losses, grads


def get_distrosa_train_hists(initpar: Tensor, traindset: Tensor, res: int, eps: float):
    """Get the training history when using DistroSA.
    """

    # train the parameters in z-space, i.e., (-inf, inf)
    z = torch.log(initpar.clone().detach()-1.0)
    z = z.requires_grad_(True)

    # optimizer
    optimizer = torch.optim.Adam([z,], lr=1e-2)

    # closure function for the optimizer
    def closure():
        optimizer.zero_grad()
        pars = torch.exp(z) + 1.0
        loss, grad = closure_distrosa(pars, traindset, res, eps)
        return loss

    # fixed the random seed for reproducibility
    torch.manual_seed(2)

    # history holder
    hist = [initpar.clone().detach().cpu()]

    # training
    new = torch.exp(z.clone().detach()) + 1.0
    deltas = [False, False, False, False, False]
    counter = 0
    while (not all(deltas)) and counter < 3000:
        optimizer.step(closure)
        old = new
        new = torch.exp(z.clone().detach()) + 1.0
        delta = torch.all(abs((new-old)/old) < 1e-5).detach().cpu().numpy().item()
        deltas.pop(0)
        deltas.append(delta)  # type: ignore
        loss = closure()
        hist.append(new.clone().detach().cpu())
        counter += 1
        print(counter, loss.item(), deltas, new.detach().cpu().numpy(),)

    return hist


def get_fd_train_hists(initpar: Tensor, traindset: Tensor, eps: float):
    """Get the training history when using finite difference.
    """

    # train the parameters in z-space, i.e., (-inf, inf)
    z = torch.log(initpar.clone().detach()-1.0)
    z = z.requires_grad_(True)

    # optimizer
    optimizer = torch.optim.Adam([z,], lr=1e-2)

    # closure function for the optimizer
    def closure():
        optimizer.zero_grad()
        pars = torch.exp(z) + 1.0
        loss, grad = closure_fd(pars.clone().detach(), traindset, eps)
        grad = grad.clone().detach() * torch.exp(z).clone().detach()
        z.grad = grad
        return loss

    # fixed the random seed for reproducibility
    torch.manual_seed(2)

    # history holder
    hist = [initpar.clone().detach().cpu()]

    # training
    new = torch.exp(z.clone().detach()) + 1.0
    deltas = [False, False, False, False, False]
    counter = 0
    while (not all(deltas)) and counter < 3000:
        optimizer.step(closure)
        old = new
        new = torch.exp(z.clone().detach()) + 1.0
        delta = torch.all(abs((new-old)/old) < 1e-5).detach().cpu().numpy().item()
        deltas.pop(0)
        deltas.append(delta)  # type: ignore
        loss = closure()
        hist.append(new.clone().detach().cpu())
        counter += 1
        print(counter, loss.item(), deltas, new.detach().cpu().numpy(),)

    return hist


if __name__ == "__main__":
    import pathlib
    figdir = pathlib.Path(__file__).resolve().parent.joinpath("figs")
    figdir.mkdir(exist_ok=True)

    # fixed the random seed for reproducibility
    torch.manual_seed(0)

    # settings
    _anspar = torch.tensor([2.31, 1.627]).to(device)
    _initpar = torch.tensor([2.0, 3.5]).to(device)  # initial values for the optimizer
    _nevents = 10000
    _pbounds1 = torch.tensor([[1.75, 4.0], [1.4, 3.65]]).to(device)  # parameter space
    _pbounds2 = torch.tensor([[2.21, 2.41], [1.1, 4.0]]).to(device)  # parameter space
    _traindset = dists.beta.Beta(*_anspar).sample((_nevents,)).to(device)
    _res = 512
    _eps1 = 1e-5
    _eps2 = 1e-3

    # get gradients with only 1 changing parameter at different FD step sizes
    outs1 = get_gradients_1d(_pbounds2, _traindset, _anspar[1].item(), 1024, _eps1)
    outs2 = get_gradients_1d(_pbounds2, _traindset, _anspar[1].item(), 1024, _eps2)

    # save 1 changing parameter gradients
    torch.save({
        "anspar": _anspar.detach().cpu(), "initpar": _initpar.detach().cpu(),
        "pbounds": _pbounds2.detach().cpu(), "res": _res, "nevents": _nevents,
        "eps1": _eps1, "params1": outs1[0], "losses1": outs1[1], "grads1": outs1[2],
        "eps2": _eps2, "params2": outs2[0], "losses2": outs2[1], "grads2": outs2[2]
    }, figdir/"grad1d1par.dat")

    # get the loss surface
    params, losses = get_loss_surface(_pbounds1, _traindset)

    # get the training history using DistroSA
    distrosa_hist = get_distrosa_train_hists(_initpar, _traindset, _res, _eps1)
    distrosa_hist = torch.stack(distrosa_hist, dim=0).detach().cpu()

    # get the training history using Finite-Difference
    fd_hist = get_fd_train_hists(_initpar, _traindset, _eps2)
    fd_hist = torch.stack(fd_hist, dim=0).detach().cpu()

    # save 1 changing parameter gradients
    torch.save({
        "params": params, "losses": losses, "distrosa_hist": distrosa_hist,
        "fd_hist": fd_hist, "anspars": _anspar.detach().cpu()
    }, figdir/"train.dat")
