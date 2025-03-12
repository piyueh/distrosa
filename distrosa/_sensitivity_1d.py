#!/usr/bin/env python3
# vim:fenc=utf-8

"""Implementation of the 1D sensitivity analysis.
"""
import sys
from typing import Callable
from numpy.typing import NDArray
from numpy.typing import DTypeLike
from numpy.typing import ArrayLike


class Sensitivity1D:
    """Sensitivity/gradient calculator for a 1D distribution.

    Arguments
    ---------
    func : callable, (x: NDArray, params: NDArray) -> pdfvals: NDArray
        Parametric probability density function (PDF) in 1D. Potentially unnormalized.
        Broadcast should be supported for arbitrary shapes of `x`.

    gridline : NDArray
        Vertices along a 1D gridline. Must be a 1D array in ascending order.
        We use `gridline` to determine if the backend is cupy or numpy. So `gridline`
        must be a numpy/cupy array.

    eps : float | NDArray
        Finite difference step size(s). If a scalar, it is used for all parameters.
        Otherwise, it must have the same length as `params`.

    Notes
    -----
    * Except for `func`, all other class init inputs are hard copied.
    * The floating point precision is determined by gridline's dtype.
    """

    def __init__(
        self,
        func: Callable[[NDArray, NDArray], NDArray],
        gridline: NDArray,
        eps: float | NDArray
    ):

        self._np = sys.modules[gridline.__class__.__module__]

        self._density = func

        # dealting w/ the gridlines and vertices
        self._gridline: NDArray = self._np.array(gridline)
        self._ftype: DTypeLike = self._gridline.dtype
        self._N: int = 1  # this class is for 1D distribution only
        self._dx: NDArray = self._gridline[1:] - self._gridline[:-1]

        # dealing w/ the finite difference step size
        self._eps: NDArray = self._np.array(eps, dtype=self._ftype)

    def __call__(self, x: NDArray, params: NDArray) -> NDArray:
        """Calculate the gradient at space points.

        Arguments
        ---------
        x : int | float | ArrayLike
            Where to evaluate the sensitivity. The shape of `x` can be arbitrary.

        params : ArrayLike
            Parameters of the PDF as a 1D array.

        Returns
        -------
        g : NDArray
            Gradient values at `x`. Its shape is `x.shape+(self._P,)`

        Notes
        -----
        * If `x` has a different floating point precision than the gridline, it will be
          hard copied and converted. The same applies to the `params`.
        * The inputs accept anything that implements the array interface, but the output
          is always a numpy/cupy ndarray.
        """

        _np = self._np

        # in case this is a scalar; it's a no-op if already a ndarray w/ correct dtype
        x = _np.asarray(x, dtype=self._ftype)

        # in case this is a list; it's a no-op if already a ndarray w/ correct dtype
        params = _np.asarray(params, dtype=self._ftype)

        # number of parameters
        P = len(params)

        # treatment of eps
        eps: NDArray = _np.repeat(self._eps, P) if self._eps.ndim == 0 else self._eps
        eps2: NDArray = 2.0 * eps

        # normalized PDF at x
        f_at_x: NDArray = self._density(x, params)  / self._cdf(params)[1]

        # will be holding dF_j/dp for all j
        gj = []

        for j in range(P):

            _params = params.copy()
            _params[j] = _params[j] + eps[j]
            _cdfp, _ = self._cdf(_params)  # we don't need the normalization constant

            _params = params.copy()
            _params[j] = _params[j] - eps[j]
            _cdfm, _ = self._cdf(_params)  # we don't need the normalization constant

            # reusing the memory space of `_cdfp` to hold the numerical derivatives
            _np.subtract(_cdfp, _cdfm, out=_cdfp)
            _np.divide(_cdfp, eps2[j], out=_cdfp)

            # let's trust that CuPy's interpolation is efficient enough for now
            gj.append(_np.interp(x, self._gridline, _cdfp))

        # stack the results to have shape x.shape+(P,)
        gj = _np.stack(gj, axis=-1)

        # calculate J = - (d F / d p) / f(x)
        _np.negative(gj, out=gj)
        _np.divide(gj, f_at_x.reshape(-1, 1), out=gj)

        # stack the results and return
        return gj

    def _cdf(self, params: NDArray) -> tuple[NDArray, NDArray]:
        """Get 1D CDF numerically.
        """

        _np = self._np

        # unnormalized PDF values at vertices
        pdfs = self._density(self._gridline, params)

        # integration to get CDF but reusing `pdfs` memory space
        _np.add(pdfs[:-1], pdfs[1:], out=pdfs[1:])
        _np.multiply(pdfs[1:], self._dx, out=pdfs[1:])
        _np.divide(pdfs[1:], 2.0, out=pdfs[1:])
        pdfs[0] = 0.0  # the most left vertex always has CDF = 0
        _np.cumsum(pdfs, out=pdfs)

        # normalization constant
        norm = pdfs[-1]

        # normalize the PDFs
        _np.divide(pdfs, norm, out=pdfs)

        return pdfs, norm  # `pdfs` holds CDF values
