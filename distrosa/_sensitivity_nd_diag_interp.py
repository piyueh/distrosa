#!/usr/bin/env python3
# vim:fenc=utf-8

"""N-D sensitivity calculator via interpolation and the diagonal approximation.
"""
import sys
from typing import Callable
from typing import Sequence
from numpy.typing import NDArray
from numpy.typing import DTypeLike
from numpy.typing import ArrayLike
from ._misc import getconditionals as _getconditionals
from ._misc import interpnd as _interpnd


class SensitivityNDDiagInterp:
    """N-D sensitivity calculator w/ diagonal approximation and interpolation.

    Arguments
    ---------
    func : Callable, (x: NDArray, params: NDArray) -> pdfvals: NDArray
        Parametric probability density function (PDF) in N-D. Potentially unnormalized.
        `pdfvals.shape` must be the same as `x.shape[:-1]`, and `x.shape[-1]` is the
        dimensionality.

    gridlines : Sequence[NDArray]
        Vertices along the gridline in each spatial dimension. `len(gridlines)` must
        be the dimensionality. We use `gridlines[0]` to determine if the backend is
        cupy or numpy. So it must be a numpy/cupy array.

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

        self._np = sys.modules[gridlines[0].__class__.__module__]

        self._density = func

        # dealting w/ the gridlines and vertices
        self._gridlines: Sequence[NDArray] = [self._np.array(_) for _ in gridlines]
        self._ftype: DTypeLike = self._gridlines[0].dtype
        self._N: int = len(self._gridlines)  # num. of spatial dimensions
        self._K: tuple[int, ...] = tuple(len(_) for _ in gridlines)  # num. of vertices
        self._dx: Sequence[NDArray] = [_[1:] - _[:-1] for _ in self._gridlines]

        # dealing w/ the finite difference step size
        self._eps = self._np.array(eps, dtype=self._ftype)

        # data for interpolations
        self._deltas: None | Sequence[Sequence[NDArray]] = None  # (d CDF_i / d param_j)
        self._pdfs: None | Sequence[NDArray] = None  # 1D conditional PDFs

        # cached params used as a hash to check if self._J needs to be reconstructed
        self._params: None | NDArray = None

    def _construct(self, params: NDArray) -> None:
        """Construct values at vertices for later being used in interpolations.

        `self._pdfs` and `self._deltas` are constructed and updated here.
        """

        _np = self._np

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
        dhs = self._dx
        density = self._density

        # get all 1D conditional PDFs at all N-D vertices
        pdfs, _ = _getconditionals(density, v, dhs, params)

        # get all 1D conditional CDFs at all N-D vertices under perturbed parameters
        cdfsp = []
        cdfsm = []
        for j in range(P):
            perturbed = params.copy()

            perturbed[j] = params[j]+ eps[j]
            cdfsp.append(_getconditionals(density, v, dhs, perturbed)[1])

            perturbed[j] = params[j]- eps[j]
            cdfsm.append(_getconditionals(density, v, dhs, perturbed)[1])

        # empty container for (d CDF_i / d param_j) at all vertices
        deltas = []

        # loop over each 1D conditional direction
        for i in range(N):
            deltas.append([])
            # loop over each parameter
            for j in range(P):
                # delta <- d CDF_i / d param_j at all vertices
                deltas[i].append((cdfsp[j][i]-cdfsm[j][i])/eps2[j])

        # link the attributes to the constructed values; no-copy, just referencing
        self._params = params
        self._pdfs = pdfs
        self._deltas = deltas

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

        _np = self._np

        # check if self._delta needs to be reconstructed
        if self.needupdate(params):
            self._construct(params)

        # make sure we're always using shape (..., N) even if N = 1
        if self._N == 1 and x.shape[-1] != 1:
            x = x.reshape(x.shape+(1,))  # non-copy view

        # to hold the outputs
        J = _np.zeros(x.shape+(len(params),), dtype=self._ftype)

        for i in range(self._N):
            for j in range(len(params)):
                fx = _interpnd(x, self._gridlines, self._pdfs[i])
                derv = _interpnd(x, self._gridlines, self._deltas[i][j])
                J[:, i, j] = - derv / fx

        # 1D is special... we don't want the shape to be (Nx, 1, P)
        return J.reshape(*x.shape[:-1], -1,) if self._N == 1 else J

    def needupdate(self, params: NDArray) -> bool:
        """Check if the internal data needs to be updated.

        Arguments
        ---------
        params : NDArray
            Parameters of the PDF as a 1D array.

        Returns
        -------
        bool
            Whether the internal data needs to be updated.
        """

        _np = self._np

        if self._deltas is None:
            return True

        if self._pdfs is None:
            return True

        if not _np.allclose(params, self._params, 0, 1e-9, True):
            return True

        return False
