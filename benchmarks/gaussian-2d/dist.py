#!/usr/bin/env python3
# vim:fenc=utf-8

"""Infinite-domain Gaussian in 2D.
"""
import sys
from typing import Callable
from numpy.typing import NDArray


_cupy_pdf_kernel = """
    out = (x - params[0]) / params[2];
    T z2 = (y - params[1]) / params[3];
    T tmp = 1.0 - params[4] * params[4];
    out = - (out * out - 2 * params[4] * out * z2 + z2 * z2);
    z2 = (2.0 * tmp);
    out /= z2;
    out = exp(out);
    z2 = (2.0 * M_PI * params[2] * params[3] * sqrt(tmp));
    out /= z2;
"""


def _numpy_pdf_kernel(x: NDArray, y: NDArray, params: NDArray):
    """Naive implementation for 2D Gaussian PDF.
    """

    # x.__class__.__module__ should give "numpy"
    np = sys.modules[x.__class__.__module__]

    sigma_1 = params[2]
    sigma_2 = params[3]
    rho = params[4]

    z1 = (x - params[0]) / params[2]
    z2 = (y - params[1]) / params[3]
    tmp = 1 - params[4] * params[4]

    out = np.exp(-(z1**2-2*params[4]*z1*z2+z1**2)/(2*tmp))
    out /= (2 * np.pi * sigma_1 * sigma_2 * np.sqrt(tmp))

    return out


class Gaussian2D:
    """Infinite-domain Gaussian in 2D.

    Arguments
    ---------
    backend: str
        Backend to use. Can be either 'numpy' or 'cupy'.
    """

    def __init__(self, backend: str):
        self.backend = backend
        assert backend in ['numpy', 'cupy']
        assert backend in sys.modules.keys()

        self._pdf: Callable = lambda: None

        if backend == 'cupy':
            cupy = sys.modules[backend]
            self._pdf = cupy.ElementwiseKernel(
                "T x, T y, raw T params", "T out", _cupy_pdf_kernel
            )
        else:  # assume numpy
            self._pdf = _numpy_pdf_kernel

    def pdf(self, x: NDArray, params:NDArray) -> NDArray:
        """Probability density function for 2D Gaussian distribution.
        """
        return self._pdf(x[..., 0], x[..., 1], params)
