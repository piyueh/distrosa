#!/usr/bin/env python3
# vim:fenc=utf-8

"""Trivial example of how to use a gradient calculator from DistroSA.
"""
import torch
import torch.distributions.multivariate_normal as dist
import distrosa


def pdf(x, params):
    """Probability density function (PDF) of the distribution.

    Arguments
    ---------
    x : Length-2 1D torch.Tensor or Nx-by-2 2D torch.Tensor.
    params : Length-5 torch.Tensor for the 2D Gaussian's parameters.

    Returns
    -------
    torch.Tensor: PDF values.

    Notes
    -----
    Automatic differentiation is disabled to show that DistroSA does not it to work.
    """
    means = params[:2]
    covmtx = torch.tensor([
        [params[2]**2, params[2]*params[3]*params[4]],
        [params[2]*params[3]*params[4], params[3]**2]
    ])
    return torch.exp(dist.MultivariateNormal(means, covmtx).log_prob(x))


def sampler(nx, params):
    """A trivial sampler for a parametric 2D Gaussian distribution.

    Arguments
    ---------
    nx : tuple of torch.Size denoting the shape of output sample.
    params : Length-5 torch.Tensor for the 2D Gaussian's parameters.

    Returns
    -------
    torch.Tensor : Realizations.
    """
    means = params[:2]
    covmtx = torch.tensor([
        [params[2]**2, params[2]*params[3]*params[4]],
        [params[2]*params[3]*params[4], params[3]**2]
    ])
    return dist.MultivariateNormal(means, covmtx).sample(nx)


if __name__ == "__main__":

    # default floating precision: 64bit
    torch.set_default_dtype(torch.float64)

    # configure the gradient calculator
    vertices = [torch.linspace(-8., 8., 128), torch.linspace(-8., 8., 128)]
    grader = distrosa.SensitivityND(5, vertices, 1e-5, pdf)

    # target parameters of the 2D Gaussian
    pars = torch.tensor([0.5, -0.3, 1.5, 0.7, 0.75])

    # generate realizations
    x = sampler((5,), pars)

    # gradients of x with respect to parameters
    dxdp = grader(x, pars)

    assert dxdp.shape == (5, 2, 5)  # (nx, ndim, npars)
    print(dxdp)
