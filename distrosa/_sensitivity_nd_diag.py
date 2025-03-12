#!/usr/bin/env python3
# vim:fenc=utf-8

"""Implementations of a calculator for N-D sensitivity using the diagonal approximation.
"""
from typing import Callable
from typing import Sequence
from numpy.typing import NDArray
from numpy.typing import DTypeLike
from numpy.typing import ArrayLike
from . import _np
from ._misc import minterp as _minterp
from ._misc import getcdf as _getcdf


class SensitivityNDDiag:
    """Sensitivity calculator for a N-D distribution via diagonal approximation.

    Arguments
    ---------
    func : Callable, (x: NDArray, params: NDArray) -> pdfvals: NDArray
        Parametric probability density function (PDF) in N-D. Potentially unnormalized.
        `pdfvals.shape` must be the same as `x.shape[:-1]`, and `x.shape[-1]` is the
        dimensionality.

    gridlines : Sequence[NDArray]
        Vertices along the gridline in each spatial dimension. `len(gridlines)` must
        be the dimensionality.

    eps : float | NDArray
        Finite difference step size(s) for paramerters.

    Notes
    -----
    * This class works also for 1D, but it is better to define the 1D PDF as if it's
      for N-D distribution. That is, the PDF should expect the input `x` to have its
      last dimension being 1. And the return valus of the PDF should have shape
      `x.shape[:-1]`.
    * Except for `func`, all other class init inputs are hard copied.
    * The floating point precision is determined by the dtype of `gridlines[0]`.
    """

    def __init__(
        self,
        func: Callable[[NDArray, NDArray], NDArray],
        gridlines: Sequence[NDArray],
        eps: float | NDArray,
    ):

        self._density = func

        # dealting w/ the gridlines and vertices
        self._gridlines: Sequence[NDArray] = [_np.array(_) for _ in gridlines]
        self._ftype: DTypeLike = self._gridlines[0].dtype
        self._N: int = len(self._gridlines)  # num. of spatial dimensions
        self._K: tuple[int, ...] = tuple(len(_) for _ in gridlines)  # num. of vertices
        self._dx: Sequence[NDArray] = [_[1:] - _[:-1] for _ in self._gridlines]

        # dealing w/ the finite difference step size
        self._eps = _np.array(eps, dtype=self._ftype)

    def __call__(self, x: NDArray, params: NDArray) -> NDArray:
        """Calculate the gradient at space points.

        Arguments
        ---------
        x : NDArray
            Where to evaluate the sensitivity. The shape of `x` can be arbitrary, but
            the last dimension, `x.shape[-1]` must be the same as dimensionality (except
            when 1D, i.e., when `self._N = 1`).

        params : NDArray
            Parameters of the PDF as a 1D array.

        Returns
        -------
        g : NDArray
            Gradient values at `x`. Its shape is `x.shape + (len(params),)`

        Notes
        -----
        * No sanity checks at all to keep code simple.
        * To avoid OOM, we split `x` to chuncks based on the size of background grid.
        * Unlike in Sensitivity1D, the `x` here must be already a NDArray.
        * Parameters can still be anything that implements the array interface.
        """

        # make sure we're always using shape (..., N) even if N = 1
        if self._N == 1 and x.shape[-1] != 1:
            x = x.reshape(x.shape+(1,))  # non-copy view

        nelms_per_1g = 134217728  # number of doubles per 1 GB
        bsize = nelms_per_1g // max(self._K) // self._N

        if x.size // self._N > bsize:
            J = []
            _x = x.reshape(-1, self._N)  # flatten to 2D with shape (Nx, N)
            for i in range(0, _x.shape[0], bsize):
                J.append(self._backend(_x[i:i+bsize], params))
            J = _np.concatenate(J, 0).reshape(*x.shape, -1)  # (Nx, N, P) -> (..., N, P)
        else:
            J = self._backend(x, params)

        # 1D is special... we don't want the shape to be (Nx, 1, P)
        return J.reshape(*x.shape[:-1], -1,) if self._N == 1 else J

    def _backend(self, x: NDArray, params: NDArray) -> NDArray:
        """Calculate the gradient at space points.
        """

        # in case this is a list; it's a no-op if already a ndarray w/ correct dtype
        params = _np.asarray(params, dtype=self._ftype)

        # number of parameters
        P = len(params)

        # treatment of eps
        eps: NDArray = _np.repeat(self._eps, P) if self._eps.ndim == 0 else self._eps
        eps2: NDArray = 2.0 * eps

        # aliases for readability
        N = self._N
        K = self._K
        v = self._gridlines
        nx = x.shape[:-1]  # the number/shape of the points

        # empty containers
        J = _np.zeros(nx+(N, P), dtype=self._ftype)

        # construct \partial F_i / \partial param_j
        for i in range(N):

            # expand and copy (NOTE: memory inefficient!!)
            xk = _np.repeat(x.reshape(nx+(1, N)), K[i], -2)  # xk shape: (Nx, Ki, N)

            # conditioning
            xk[..., i] = v[i]  # xk shape: (Nx, Ki, N)

            for j in range(P):

                pars = params.copy()

                # params[j] += eps
                pars[j] += eps[j]
                cdfp, _ = _getcdf(self._density, xk, pars, self._dx[i])

                # params[j] -= eps
                pars[j] -= eps2[j]
                cdfm, _ = _getcdf(self._density, xk, pars, self._dx[i])

                # reusing cdfp mem space; cdfp = d F_i / d param_j
                _np.subtract(cdfp, cdfm, out=cdfp)
                _np.divide(cdfp, eps2[j], out=cdfp)  # shape (Nx, Ki)

                # 1D interpolation; shape change: (Nx,), (Ki,), (Nx, Ki) -> (Nx,)
                J[..., i, j] = _minterp(x[..., i], v[i], self._dx[i], cdfp)

            # we don't need to normalize the CDF, just need the normalization factor
            _, norm = _getcdf(self._density, xk, params, self._dx[i])

            # calculate normalized PDF at x directly (rather than via interpolation)
            _pdf = self._density(x, params).reshape(nx)  # some 1D PDF returns (Nx, 1)
            _np.divide(_pdf, norm, out=_pdf)

            # scale J[..., i, :] by -1 / f(x)
            _np.divide(J[..., i, :], _pdf.reshape(nx+(1,)), out=J[..., i, :])
            _np.negative(J[..., i, :], out=J[..., i, :])

        return J
