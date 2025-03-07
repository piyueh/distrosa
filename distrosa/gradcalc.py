#!/usr/bin/env python3
# vim:fenc=utf-8

"""Gradient calculators.
"""
import torch
from . import misc as _misc


class Sensitivity1D:
    """Sensitivity/gradient calculator for a 1D distribution.

    Arguments
    ---------
    func : callable, (x, params) -> value
        Parametric probability density function (PDF) in 1D. Potentially unnormalized.
        `func` must be capable of broadcasting so that `x` can have an arbitrary shape
        consisting of multiple space points. `params` is a flattened
        parameter vector, i.e., 1D array. The `value` corresponds to the PDF values at
        space points in `x`, so `value` have the same shape of `x`.
    gridline : 1D array
        Vertices along a 1D gridline.
    params : 1D array
        Parameters of the PDF.
    eps : float or 1D array of the same shape as `params`
        Finite difference step size(s) for gradient calculation.
    """

    def __init__(self, func, gridline, params, eps):
        self._density = func
        self._gridline = torch.asarray(gridline)
        self._params = torch.asarray(params)
        self._eps = torch.asarray(eps)

        if self._eps.ndim == 0:
            self._eps = self._eps.expand(self._params.shape)
        elif self._eps.shape != self._params.shape:
            raise ValueError("`eps` must be a scalar or array of `params`'s shape")

        # derived attributes
        self._P = len(params)  # number of parameters\
        self._N = 1  # this class is for 1D distribution only
        self._two_eps = 2.0 * self._eps

    def __call__(self, x):
        """Calculate the gradient at space points.

        Arguments
        ---------
        x : multidimensional array
            Space points where the gradient is calculated. Regardless of the shape of
            `x`, each element in this array is a space point as this class deals with
            1D distribution only.

        Returns
        -------
        g : N-D array of shape x.shape + (P,)
            Gradient values at space points.
        """

        x = torch.asarray(x)

        c = _misc.get_cdf_1d(self._density, self._gridline, self._params)[1]

        f_at_x = self._density(x, self._params) / c

        g = torch.zeros(x.shape + (self._P,), dtype=x.dtype)

        for j in range(self._P):

            _params = self._params.clone()
            _params[j] = _params[j] + self._eps[j]
            cdf_plus = _misc.get_cdf_1d(self._density, self._gridline, _params)[0]

            _params = self._params.clone()
            _params[j] = _params[j] - self._eps[j]
            cdf_minus = _misc.get_cdf_1d(self._density, self._gridline, _params)[0]

            delta = (cdf_plus - cdf_minus) / self._two_eps[j]

            g[..., j] = - _misc.interp_1d(x, self._gridline, delta) / f_at_x

        return g


class SensitivityND:
    """Sensitivity/gradient calculator for a N-D distribution.

    Arguments
    ---------
    func : callable, (x, params) -> value
        Parametric probability density function (PDF) in N-D. Potentially unnormalized.
        `func` must be capable of broadcasting so that `x` can have an arbitrary
        shape---as long as the last dimension in `x`'s shape is N. For example,
        `x.shape = (3, 4, N)` means 12 space point in N-D arranged in a shape of (3, 4).
        `params` is a flattened parameter vector, i.e., 1D array. The `value`
        corresponds to the PDF values at space points in `x`, so
        `value.shape = x.shape[:-1]`.
    gridlines : a sequence of 1D arrays
        Vertices along a the gridline in each spatial dimension.
    params : 1D array
        Parameters of the PDF.
    eps : float or 1D array of the same shape as `params`
        Finite difference step size(s) for paramerters.
    eps_x : None | float | 1D array of length-N
        Finite difference step size(s) for spatial derivatives. If None, infer from the
        gridlines.

    Notes
    -----
    * This class works also for 1D, but it is better to define the 1D PDF as if it's
      for N-D distribution. That is, the PDF should expect the input `x` to have its
      last dimension being 1. And the return valus of the PDF should have shape
      `x.shape[:-1]`.
    """

    def __init__(self, func, gridlines, params, eps, eps_x=None):
        self._density = func
        self._gridlines = [torch.asarray(_) for _ in gridlines]
        self._params = torch.asarray(params)
        self._eps_p = torch.asarray(eps)
        self._eps_x = torch.asarray(eps_x) if eps_x is not None else None

        # derived attributes
        self._P = len(params)  # number of parameters\
        self._N = len(gridlines)  # number of spatial dimensions

        if self._eps_p.ndim == 0:
            self._eps_p = self._eps_p.expand(self._params.shape)
        elif self._eps_p.shape != self._params.shape:
            raise ValueError("`eps_p` must be a scalar or array of `params`'s shape")

        # infer eps_x from gridlines
        if self._eps_x is None:
            self._eps_x = torch.zeros(self._N, dtype=self._eps_p.dtype)
            for i in range(self._N):
                self._eps_x[i] = (self._gridlines[i][-1] - self._gridlines[i][0]) * 1e-5

        if self._eps_x.ndim == 0:
            self._eps_x = self._eps_x.expand(self._N)
        elif self._eps_x.shape != (self._N,):
            raise ValueError("`eps_x` must be a scalar or array of length N")

        self._two_eps_p = 2.0 * self._eps_p
        self._two_eps_x = 2.0 * self._eps_x

    def __call__(self, x):
        """Calculate the gradient at space points.

        Arguments
        ---------
        x : N-D array
            Space points where the gradient is calculated. The last dimension of `x`
            must be N, which is the number of spatial dimensions.

        Returns
        -------
        g : N-D array of shape x.shape + (P,)
            Gradient values at space points.

        Notes
        -----
        * To keep the code simple, we do not check whether the last dimension of `x`
          is N or not.
        """

        x = torch.asarray(x)
        shape = x.shape  # save the original shape
        x = x.view(-1, self._N)  # non-copy view

        J = torch.zeros(x.shape+(self._P,), dtype=x.dtype)

        for ix in range(x.shape[0]):
            H = torch.zeros((self._N, self._N), dtype=x.dtype)
            G = torch.zeros((self._N, self._P), dtype=x.dtype)

            for i in range(self._N):

                def fi(s, params):
                    s = torch.asarray(s)
                    _x = torch.tile(x[ix], (len(s), 1))
                    _x[:, i] = s
                    return self._density(_x, params).view(len(s))

                for j in range(self._N):

                    def fp(s, params):
                        s = torch.asarray(s)
                        _x = torch.tile(x[ix], (len(s), 1))
                        _x[:, i] = s
                        _x[:, j] = _x[:, j] + self._eps_x[j]  # type: ignore
                        return self._density(_x, params).view(len(s))

                    def fm(s, params):
                        s = torch.asarray(s)
                        _x = torch.tile(x[ix], (len(s), 1))
                        _x[:, i] = s
                        _x[:, j] = _x[:, j] - self._eps_x[j]  # type: ignore
                        return self._density(_x, params).view(len(s))

                    cdf_plus = _misc.get_cdf_1d(fp, self._gridlines[i], self._params)[0]
                    cdf_minus = _misc.get_cdf_1d(fm, self._gridlines[i], self._params)[0]
                    delta = (cdf_plus - cdf_minus) / self._two_eps_x[j]
                    H[i, j] = - _misc.interp_1d(x[ix, i], self._gridlines[i], delta)

                for j in range(self._P):

                    _params = self._params.clone()
                    _params[j] = _params[j] + self._eps_p[j]
                    cdf_plus = _misc.get_cdf_1d(fi, self._gridlines[i], _params)[0]

                    _params = self._params.clone()
                    _params[j] = _params[j] - self._eps_p[j]
                    cdf_minus = _misc.get_cdf_1d(fi, self._gridlines[i], _params)[0]

                    delta = (cdf_plus - cdf_minus) / self._two_eps_p[j]
                    G[i, j] = _misc.interp_1d(x[ix, i], self._gridlines[i], delta)

            # solve the linear system
            try:
                J[ix, :, :] = torch.linalg.solve(H, G)
            except torch._C._LinAlgError as err:  # pyright: ignore
                # if the matrix is singular, set the gradients to zero
                if "singular" in str(err):
                    J[ix, :, :] = 0.0
                else:
                    raise

        # restore the original shape but do not copy
        return J.view(shape+(self._P,))


class SensitivityNDDiag:
    """Sensitivity/gradient calculator for a N-D distribution w/ diagonal approximation.
    """

    def __init__(self, func, gridlines, params, eps):
        self._density = func
        self._gridlines = [torch.asarray(_) for _ in gridlines]
        self._params = torch.asarray(params)
        self._eps = torch.asarray(eps)

        if self._eps.ndim == 0:
            self._eps = self._eps.expand(self._params.shape)
        elif self._eps.shape != self._params.shape:
            raise ValueError("`eps` must be a scalar or array of `params`'s shape")

        # derived attributes
        self._P = len(params)  # number of parameters\
        self._N = len(gridlines)  # number of spatial dimensions
        self._two_eps = 2.0 * self._eps

    def __call__(self, x):
        """Calculate the gradient at space points.

        Arguments
        ---------
        x : N-D array
            Space points where the gradient is calculated. The last dimension of `x`
            must be N, which is the number of spatial dimensions.

        Returns
        -------
        g : N-D array of shape x.shape + (P,)
            Gradient values at space points.

        Notes
        -----
        * To keep the code simple, we do not check whether the last dimension of `x`
          is N or not.
        """



        x = torch.asarray(x)
        shape = x.shape  # save the original shape
        x = x.view(-1, self._N)  # non-copy view

        J = torch.zeros(x.shape+(self._P,), dtype=x.dtype)

        for ix in range(x.shape[0]):
            for i in range(self._N):

                # local function as a 1D conditional PDF
                def fi(s, params):
                    s = torch.asarray(s)
                    _x = torch.tile(x[ix], s.view(-1, 1).shape)
                    _x[:, i] = s
                    return self._density(_x, params).view(s.shape)

                cond = Sensitivity1D(fi, self._gridlines[i], self._params, self._eps)
                J[ix, i, :] = cond(x[ix, i])

        # restore the original shape but do not copy
        return J.view(shape+(self._P,))
