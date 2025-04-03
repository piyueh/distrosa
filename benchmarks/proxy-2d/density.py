#!/usr/bin/env python3
# vim:fenc=utf-8

"""Implementation of the proxy 2D distribution.
"""
import torch
from torch import Tensor
from typing import Callable


class _proxy_2d(torch.autograd.Function):
    """Unnormalized 2D proxy PDF with custom backward pass.

    f(x; p) = x_0^{p_0} (1 - x_0)^{p_1} x_1^{p_2} (1 - x_1)^{p_3} (1 + p_4 x_0 x_1)

    for 0 < x_0, x_1 < 1; x := [x_0, x_1]; and p := [p_0, p_1, p_2, p_3, p_4].

    Arguments
    ---------
    x : Tensor
        The locations at where to evaluate the PDF. Shape can be arbitrary, but the
        last dimension must be 2, i.e., `x.shape[-1] == 2`.

    params : Tensor
        1D tensor of length 5.

    Notes
    -----
    The forward mode implements the calculation of PDF. The backward mode implements the
    vjp operation (vector-Jacobian product).
    """
    @staticmethod
    def forward(ctx, x, params):

        # sanity check; can be turned off during runtime w/ the `-O` flag
        assert torch.all(x > 0.0)
        assert torch.all(x < 1.0)

        with torch.no_grad():
            vals = _forward(x, params)
        ctx.save_for_backward(x, params, vals)
        return vals

    @staticmethod
    def backward(ctx, grad: Tensor):  # type: ignore

        # retrive saved tensors
        x, p, vals = ctx.saved_tensors

        with torch.no_grad():
            jacx, jacp = _backward(x, p, vals, grad)

        return jacx, jacp

@torch.jit.script
def _forward(x: Tensor, params: Tensor) -> Tensor:
    """Unnormalized 2D proxy PDF.

    Assumes x in (0, 1) and params of shape (5,).
    """
    x0 = x[..., 0]
    x1 = x[..., 1]

    one = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    # compute components using fused multiplications
    term1 = torch.pow(x0, params[0])
    term2 = torch.pow(one - x0, params[1])
    term3 = torch.pow(x1, params[2])
    term4 = torch.pow(one - x1, params[3])
    interaction = one + params[4] * x0 * x1

    return term1 * term2 * term3 * term4 * interaction


@torch.jit.script
def _backward(
    x: Tensor, params: Tensor, vals: Tensor, grad: Tensor
) -> tuple[Tensor, Tensor]:
    """The implementation of the vjp operation for 2D proxy.

    Note that x should be > 0 and < 1. Though no sanity check here.
    """

    x0 = x[..., 0]
    x1 = x[..., 1]
    one = torch.tensor(1.0, device=x.device, dtype=x.dtype)

    # compute shared intermediate: tmp = 1 + p4 * x0 * x1
    tmp = one + params[4] * x0 * x1
    tmp_inv = 1.0 / tmp  # reuse this for both x0 and x1 derivative terms

    # jacobian wrt x
    jx0 = params[0] / x0 - params[1] / (one - x0) + params[4] * x1 * tmp_inv
    jx1 = params[2] / x1 - params[3] / (one - x1) + params[4] * x0 * tmp_inv
    jacx = torch.stack((jx0, jx1), dim=-1)
    jacx = grad.unsqueeze(-1) * jacx * vals.unsqueeze(-1)

    # jacobian wrt params
    log_x0_safe = torch.nan_to_num(torch.log(x0), neginf=0.0)
    log_1mx0_safe = torch.nan_to_num(torch.log(one-x0), neginf=0.0)
    log_x1_safe = torch.nan_to_num(torch.log(x1), neginf=0.0)
    log_1mx1_safe = torch.nan_to_num(torch.log(one-x1), neginf=0.0)

    # use fused reductions instead of summing intermediates one at a time
    product = vals * grad

    jacp = torch.empty_like(params)
    jacp[0] = torch.sum(log_x0_safe*product)
    jacp[1] = torch.sum(log_1mx0_safe*product)
    jacp[2] = torch.sum(log_x1_safe*product)
    jacp[3] = torch.sum(log_1mx1_safe*product)
    jacp[4] = torch.sum(x0*x1*product*tmp_inv)

    return jacx, jacp


# export the 2D proxy PDF
proxy_2d: Callable[[Tensor, Tensor], Tensor] = _proxy_2d.apply  # type: ignore
