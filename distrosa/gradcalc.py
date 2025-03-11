#!/usr/bin/env python3
# vim:fenc=utf-8

"""Gradient calculators.
"""
import itertools
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
        self._params = params.clone()
        self._device = self._params.device
        self._ftype = self._params.dtype
        self._density = func
        self._gridlines = [
            torch.asarray(_, dtype=self._ftype, device=self._device)
            for _ in gridlines
        ]
        self._eps_p = torch.asarray(eps, dtype=self._ftype, device=self._device)

        if eps_x is not None:
            self._eps_x = torch.asarray(eps_x, dtype=self._ftype, device=self._device)
        else:
            self._eps_x = None

        # derived attributes
        self._P = len(params)  # number of parameters\
        self._N = len(gridlines)  # number of spatial dimensions
        self._K = tuple(len(_) for _ in gridlines)  # number of vertices
        self._h = [_[1:] - _[:-1] for _ in self._gridlines]  # cell sized

        if self._eps_p.ndim == 0:
            self._eps_p = self._eps_p.expand(self._params.shape)
        elif self._eps_p.shape != self._params.shape:
            raise ValueError("`eps_p` must be a scalar or array of `params`'s shape")

        # infer eps_x from gridlines
        if self._eps_x is None:
            self._eps_x = torch.zeros(self._N, dtype=self._ftype, device=self._device)
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
        * To avoid OOM, we split `x` to smaller chuncks according to the size of
          background grid.
        """

        # in case x is a scalar or a built-in list
        x = torch.asarray(x, dtype=self._ftype, device=self._device)

        # make sure we're always using shape (..., N) even if N = 1
        if self._N == 1 and x.shape[-1] != 1:
            x = x.view(x.shape+(1,))  # non-copy view
        else:
            assert x.shape[-1] == self._N

        nelms_per_1g = 134217728  # number of doubles per 1 GB
        bsize = nelms_per_1g // max(self._K) // self._N

        if x.numel() // self._N > bsize:
            J = []
            _x = x.view(-1, self._N)
            for i in range(0, _x.shape[0], bsize):
                J.append(self._backend(_x[i:i+bsize]))
            J = torch.cat(J, dim=0).view(x.shape[:-1]+(self._N, self._P))
        else:
            J = self._backend(x)

        return J

    def _backend(self, x):
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

        # aliases for readability
        N = self._N
        P = self._P
        K = self._K
        v = self._gridlines
        xshape = x.shape[:-1]  # the number/shape of the points

        # empty containers
        H = torch.zeros(xshape+(N, N), dtype=self._ftype, device=self._device)
        G = torch.zeros(xshape+(N, P), dtype=self._ftype, device=self._device)
        J = torch.zeros(xshape+(N, P), dtype=self._ftype, device=self._device)

        # construct H and G
        for i in range(N):

            # _x.shape = (Nx, Ki, N); _x[..., k, i] = v[i][k]; `expand` is also a view
            _x = x.view(xshape+(1, N)).expand(xshape+(K[i], N)).clone()
            _x[..., i] = v[i]

            # construct H
            for j in range(N):

                # _x[..., k, j] += eps; perturb the j-th spatial dimension
                _x[..., j] = _x[..., j] + self._eps_x[j]  # type: ignore

                # _pdf.shape = (Nx, Ki); _pdf[..., k] = f(v[i][k] | x_{-i})
                _pdf = self._density(_x, self._params).view(xshape+(K[i],))

                # _cdf.shape = (Nx, Ki); _cdf[..., k] = F(v[i][k] | x_{-i})
                _cdfp = torch.zeros_like(_pdf)
                _cdfp[..., 1:] = (_pdf[..., :-1] + _pdf[..., 1:]) * self._h[i] / 2.0
                _cdfp[..., 1:] = torch.cumsum(_cdfp[..., 1:], dim=-1)
                _cdfp = _cdfp / _cdfp[..., -1].view(xshape+(1,))

                # _x[..., k, j] -= 2 * eps; perturb the j-th spatial dimension
                _x[..., j] = _x[..., j] - self._two_eps_x[j]  # type: ignore

                # _pdf.shape = (Nx, Ki); _pdf[..., k] = f(v[i][k] | x_{-i})
                _pdf = self._density(_x, self._params).view(xshape+(K[i],))

                # _cdf.shape = (Nx, Ki); _cdf[..., k] = F(v[i][k] | x_{-i})
                _cdfm = torch.zeros_like(_pdf)
                _cdfm[..., 1:] = (_pdf[..., :-1] + _pdf[..., 1:]) * self._h[i] / 2.0
                _cdfm[..., 1:] = torch.cumsum(_cdfm[..., 1:], dim=-1)
                _cdfm = _cdfm / _cdfm[..., -1].view(xshape+(1,))

                # delta.shape = (Nx, Ki); delta approx \partial F_i / \partial x_j
                diff = (_cdfp - _cdfm) / self._two_eps_x[j]

                # 1D interpolation; shape change: (Nx,), (Ki,), (Nx, Ki) -> (Nx,)
                H[..., i, j] = - _misc.multi_interp_1d(x[..., i], v[i], diff)

                # restore _x
                if i == j:
                    _x[..., j] = v[i]
                else:
                    _x[..., j] = x[..., j].view(xshape+(1,))

            # construct G
            for j in range(P):

                _pars = self._params.clone()

                # params[j] += eps
                _pars[j] = _pars[j] + self._eps_p[j]

                # _pdf.shape = (Nx, Ki); _pdf[..., k] = f(v[i][k] | x_{-i})
                _pdf = self._density(_x, _pars).view(xshape+(K[i],))

                # _cdf.shape = (Nx, Ki); _cdf[..., k] = F(v[i][k] | x_{-i})
                _cdfp = torch.zeros_like(_pdf)
                _cdfp[..., 1:] = (_pdf[..., :-1] + _pdf[..., 1:]) * self._h[i] / 2.0
                _cdfp[..., 1:] = torch.cumsum(_cdfp[..., 1:], dim=-1)
                _cdfp = _cdfp / _cdfp[..., -1].view(xshape+(1,))

                # params[j] -= eps
                _pars[j] = _pars[j] - self._two_eps_p[j]

                # _pdf.shape = (Nx, Ki); _pdf[..., k] = f(v[i][k] | x_{-i})
                _pdf = self._density(_x, _pars).view(xshape+(K[i],))

                # _cdf.shape = (Nx, Ki); _cdf[..., k] = F(v[i][k] | x_{-i})
                _cdfm = torch.zeros_like(_pdf)
                _cdfm[..., 1:] = (_pdf[..., :-1] + _pdf[..., 1:]) * self._h[i] / 2.0
                _cdfm[..., 1:] = torch.cumsum(_cdfm[..., 1:], dim=-1)
                _cdfm = _cdfm / _cdfm[..., -1].view(xshape+(1,))

                # delta.shape = (Nx, Ki); delta approx \partial F_i / \partial x_j
                diff = (_cdfp - _cdfm) / self._two_eps_p[j]

                # 1D interpolation; shape change: (Nx,), (Ki,), (Nx, Ki) -> (Nx,)
                G[..., i, j] = _misc.multi_interp_1d(x[..., i], v[i], diff)

        # solve the linear systems (avoid singular matrices)
        valid = torch.linalg.matrix_rank(H) >= N
        J[valid, :, :] = torch.linalg.solve(H[valid], G[valid])

        # 1D is special... we don't want the shape to be (Nx, 1, P)
        if N == 1:
            return J.view(xshape+(P,))

        return J


class SensitivityNDDiag:
    """Sensitivity/gradient calculator for a N-D distribution w/ diagonal approximation.
    """

    def __init__(self, func, gridlines, params, eps):
        self._params = params.clone()
        self._device = self._params.device
        self._ftype = self._params.dtype
        self._density = func
        self._gridlines = [
            torch.asarray(_, dtype=self._ftype, device=self._device)
            for _ in gridlines
        ]
        self._eps = torch.asarray(eps, dtype=self._ftype, device=self._device)

        if self._eps.ndim == 0:
            self._eps = self._eps.expand(self._params.shape)
        elif self._eps.shape != self._params.shape:
            raise ValueError("`eps` must be a scalar or array of `params`'s shape")

        # derived attributes
        self._P = len(params)  # number of parameters\
        self._N = len(gridlines)  # number of spatial dimensions
        self._K = tuple(len(_) for _ in gridlines)  # number of vertices
        self._h = [_[1:] - _[:-1] for _ in self._gridlines]  # cell sized
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
        * To avoid OOM, we split `x` to smaller chuncks according to the size of
          background grid.
        """

        # in case x is a scalar or a built-in list
        x = torch.asarray(x, dtype=self._ftype, device=self._device)

        # make sure we're always using shape (..., N) even if N = 1
        if self._N == 1 and x.shape[-1] != 1:
            x = x.view(x.shape+(1,))  # non-copy view
        else:
            assert x.shape[-1] == self._N

        nelms_per_1g = 134217728  # number of doubles per 1 GB
        bsize = nelms_per_1g // max(self._K) // self._N

        if x.numel() // self._N > bsize:
            J = []
            _x = x.view(-1, self._N)
            for i in range(0, _x.shape[0], bsize):
                J.append(self._backend(_x[i:i+bsize]))
            J = torch.cat(J, dim=0).view(x.shape[:-1]+(self._N, self._P))
        else:
            J = self._backend(x)

        return J

    def _backend(self, x):
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

        # aliases for readability
        N = self._N
        P = self._P
        K = self._K
        v = self._gridlines
        xshape = x.shape[:-1]  # the number/shape of the points

        # empty containers
        J = torch.zeros(xshape+(N, P), dtype=self._ftype, device=self._device)

        # construct \partial F_i / \partial param_j
        for i in range(N):

            # _x.shape = (Nx, Ki, N); _x[..., k, i] = v[i][k]; `expand` is also a view
            _x = x.view(xshape+(1, N)).expand(xshape+(K[i], N)).clone()
            _x[..., i] = v[i]

            for j in range(P):

                _pars = self._params.clone()

                # params[j] += eps
                _pars[j] = _pars[j] + self._eps[j]

                # _pdf.shape = (Nx, Ki); _pdf[..., k] = f(v[i][k] | x_{-i})
                _pdf = self._density(_x, _pars).view(xshape+(K[i],))

                # _cdf.shape = (Nx, Ki); _cdf[..., k] = F(v[i][k] | x_{-i})
                _cdfp = torch.zeros_like(_pdf)
                _cdfp[..., 1:] = (_pdf[..., :-1] + _pdf[..., 1:]) * self._h[i] / 2.0
                _cdfp[..., 1:] = torch.cumsum(_cdfp[..., 1:], dim=-1)
                _cdfp = _cdfp / _cdfp[..., -1].view(xshape+(1,))

                # params[j] -= eps
                _pars[j] = _pars[j] - self._two_eps[j]

                # _pdf.shape = (Nx, Ki); _pdf[..., k] = f(v[i][k] | x_{-i})
                _pdf = self._density(_x, _pars).view(xshape+(K[i],))

                # _cdf.shape = (Nx, Ki); _cdf[..., k] = F(v[i][k] | x_{-i})
                _cdfm = torch.zeros_like(_pdf)
                _cdfm[..., 1:] = (_pdf[..., :-1] + _pdf[..., 1:]) * self._h[i] / 2.0
                _cdfm[..., 1:] = torch.cumsum(_cdfm[..., 1:], dim=-1)
                _cdfm = _cdfm / _cdfm[..., -1].view(xshape+(1,))

                # delta.shape = (Nx, Ki); delta approx \partial F_i / \partial x_j
                diff = (_cdfp - _cdfm) / self._two_eps[j]

                # 1D interpolation; shape change: (Nx,), (Ki,), (Nx, Ki) -> (Nx,)
                J[..., i, j] = _misc.multi_interp_1d(x[..., i], v[i], diff)

            # now we calculate the normalized PDF at x under original parameters

            # _pdf.shape = (Nx, Ki); _pdf[..., k] = f(v[i][k] | x_{-i})
            _pdf = self._density(_x, self._params).view(xshape+(K[i],))

            # _cdf.shape = (Nx, Ki); _cdf[..., k] = F(v[i][k] | x_{-i})
            _cdf = torch.zeros_like(_pdf)
            _cdf[..., 1:] = (_pdf[..., :-1] + _pdf[..., 1:]) * self._h[i] / 2.0
            _cdf[..., 1:] = torch.cumsum(_cdf[..., 1:], dim=-1)

            # we don't need to normalize the CDF, just need the normalization factor
            norm = _cdf[..., -1]  # shape: (Nx,)

            # calculate normalized PDF at x directly (rather than interpolation)
            _pdf = self._density(x, self._params).view(xshape)  # shape (Nx,)
            _pdf = _pdf / norm

            # scale J[..., i, :] by -1 / f(x)
            J[..., i, :] = - J[..., i, :] / _pdf.view(xshape+(1,))

        # 1D is special... we don't want the shape to be (Nx, 1, P)
        if N == 1:
            return J.view(xshape+(P,))

        return J


class SensitivityNDInterpDiag:
    """Sensitivity interpolater for a N-D distribution w/ diagonal approximation.
    """

    def __init__(self, func, gridlines, params, eps):
        self._params = params.clone()
        self._device = self._params.device
        self._ftype = self._params.dtype
        self._density = func
        self._gridlines = [
            torch.asarray(_, dtype=self._ftype, device=self._device)
            for _ in gridlines
        ]
        self._eps = torch.asarray(eps, dtype=self._ftype, device=self._device)

        if self._eps.ndim == 0:
            self._eps = self._eps.expand(self._params.shape)
        elif self._eps.shape != self._params.shape:
            raise ValueError("`eps` must be a scalar or array of `params`'s shape")

        # derived attributes
        self._P = len(params)  # number of parameters\
        self._N = len(gridlines)  # number of spatial dimensions
        self._two_eps = 2.0 * self._eps

        # get all 1D conditional PDFs at all N-D vertices
        self.pdfs = _misc.get_conditionals(
            self._density, self._gridlines, self._params)[0]

        # get all 1D conditional CDFs at all N-D vertices under perturbed parameters
        cdfs_p = []
        cdfs_m = []
        for j in range(self._P):
            _pars = self._params.clone()
            _pars[j] = _pars[j] + self._eps[j]
            cdfs_p.append(
                _misc.get_conditionals(self._density, self._gridlines, _pars)[1])

            _pars = self._params.clone()
            _pars[j] = _pars[j] - self._eps[j]
            cdfs_m.append(
                _misc.get_conditionals(self._density, self._gridlines, _pars)[1])

        # empty container for d CDF_i / d param_j at all vertices
        self.deltas = []

        # loop over each 1D conditional direction
        for i in range(self._N):
            self.deltas.append([])
            # loop over each parameter
            for j in range(self._P):

                # delta <- d CDF_i / d param_j at all vertices
                self.deltas[i].append((cdfs_p[j][i]-cdfs_m[j][i])/self._two_eps[j])

    def __call__(self, x):

        # in case this is a scalar or a built-in list
        x = torch.asarray(x, dtype=self._ftype, device=self._device)

        shape = x.shape  # save the original shape
        x = x.view(-1, self._N)  # non-copy view

        J = torch.zeros(x.shape+(self._P,), dtype=self._ftype, device=self._device)

        for i in range(self._N):
            for j in range(self._P):
                fx = _misc.interp_nd(x, self._gridlines, self.pdfs[i])
                derv = _misc.interp_nd(x, self._gridlines, self.deltas[i][j])
                J[:, i, j] = - derv / fx

        # restore the original shape but do not copy
        return J.view(shape+(self._P,))


class SensitivityNDInterp:
    """Sensitivity interpolater for a N-D distribution.
    """

    def __init__(self, func, gridlines, params, eps):
        self._params = params.clone()
        self._device = self._params.device
        self._ftype = self._params.dtype
        self._density = func
        self._gridlines = [
            torch.asarray(_, dtype=self._ftype, device=self._device)
            for _ in gridlines
        ]
        self._eps = torch.asarray(eps, dtype=self._ftype, device=self._device)

        if self._eps.ndim == 0:
            self._eps = self._eps.expand(self._params.shape)
        elif self._eps.shape != self._params.shape:
            raise ValueError("`eps` must be a scalar or array of `params`'s shape")

        # derived attributes
        self._P = len(params)  # number of parameters\
        self._N = len(gridlines)  # number of spatial dimensions
        self._K = tuple(len(_) for _ in gridlines)  # number of vertices
        self._two_eps = 2.0 * self._eps

        # data for interpolations
        self._J = self._construct_discrete_models()

    def _construct_discrete_models(self):
        """Construct values at vertices for later being used in interpolations.
        """

        # aliases for readability (should not do hard copying)
        N = self._N
        P = self._P
        K = self._K
        density = self._density
        gridlines = self._gridlines
        params = self._params

        # get all 1D conditional PDFs at all N-D vertices
        cdfs = _misc.get_conditionals(density, gridlines, params)[1]

        # get all 1D conditional CDFs at all N-D vertices under perturbed parameters
        cdfs_p = []
        cdfs_m = []
        for j in range(self._P):
            perturbed = params.clone()
            perturbed[j] = perturbed[j] + self._eps[j]
            cdfs_p.append(_misc.get_conditionals(density, gridlines, perturbed)[1])

            perturbed = params.clone()
            perturbed[j] = perturbed[j] - self._eps[j]
            cdfs_m.append(_misc.get_conditionals(density, gridlines, perturbed)[1])

        # initialize arrays
        H = torch.zeros(K + (N, N), dtype=self._ftype, device=self._device)
        G = torch.zeros(K + (N, P), dtype=self._ftype, device=self._device)
        J = torch.zeros(K + (N, P), dtype=self._ftype, device=self._device)

        # [preparing H]: loop over each spatial direction
        for j in range(self._N):

            # for broadcasting 1D arrays to N-D compatible
            bcast = tuple(-1 if _ == j else 1 for _ in range(N))

            # step sizes
            h = self._gridlines[j][1:] - self._gridlines[j][:-1]

            # vertices where central difference is applicable
            k = tuple(slice(1, -1) if _ == j else slice(None) for _ in range(N))
            kp = tuple(slice(2, None) if _ == j else slice(None) for _ in range(N))
            km = tuple(slice(None, -2) if _ == j else slice(None) for _ in range(N))

            # coefficients for central difference
            div = h[1:] * h[:-1] * (h[1:] + h[:-1])
            A = - h[1:]**2 / div  # for k-1-th terms
            B = (h[1:] - h[:-1]) * (h[1:] + h[:-1]) / div  # for k-th terms
            C = h[:-1]**2 / div  # for k+1-th terms

            # broadcasting
            A = A.view(bcast)
            B = B.view(bcast)
            C = C.view(bcast)

            # loop over each 1D conditional direction and apply central difference
            for i in range(self._N):
                H[k+(i, j)] = A * cdfs[i][km] + B * cdfs[i][k] + C * cdfs[i][kp]

            # vertices where forward difference is applicable
            k = tuple(slice(0, 1) if _ == j else slice(None) for _ in range(N))
            kp1 = tuple(slice(1, 2) if _ == j else slice(None) for _ in range(N))
            kp2 = tuple(slice(2, 3) if _ == j else slice(None) for _ in range(N))

            # coefficients for forward difference
            div = h[0] * h[1] * (h[0] + h[1])
            A = - h[1] * (2.0 * h[0] + h[1]) / div  # for k-th terms
            B = (h[0] + h[1])**2 / div  # for k+1-th terms
            C = - h[0]**2 / div  # for k+2-th terms

            # broadcasting
            A = A.view(bcast)
            B = B.view(bcast)
            C = C.view(bcast)

            # loop over each 1D conditional direction and apply forward difference
            for i in range(self._N):
                H[k+(i, j)] = A * cdfs[i][k] + B * cdfs[i][kp1] + C * cdfs[i][kp2]

            # vertices where backward difference is applicable
            k = tuple(slice(-1, None) if _ == j else slice(None) for _ in range(N))
            km1 = tuple(slice(-2, -1) if _ == j else slice(None) for _ in range(N))
            km2 = tuple(slice(-3, -2) if _ == j else slice(None) for _ in range(N))

            # coefficients for backward difference
            div = h[-1] * h[-2] * (h[-1] + h[-2])
            A = h[-2] * (2.0 * h[-1] + h[-2]) / div  # for k-th terms
            B = - (h[-1] + h[-2])**2 / div  # for k-1-th terms
            C = h[-1]**2 / div  # for k-2-th terms

            # broadcasting
            A = A.view(bcast)
            B = B.view(bcast)
            C = C.view(bcast)

            # loop over each 1D conditional direction and apply backward difference
            for i in range(self._N):
                H[k+(i, j)] = A * cdfs[i][k] + B * cdfs[i][km1] + C * cdfs[i][km2]

        # [preparing G]: loop over each conditional
        for i in range(N):
            for j in range(P):  # loop over each parameter
                G[..., i, j] = (cdfs_p[j][i] - cdfs_m[j][i]) / self._two_eps[j]

        # solve the linear systems (avoid singular matrices)
        valid = torch.linalg.matrix_rank(H) >= N
        J[valid, :, :] = torch.linalg.solve(H[valid], G[valid])

        # apply the negative sign
        J = torch.neg(J)

        return J.view(K+(N, P))

    def __call__(self, x):

        # in case this is a scalar or a built-in list
        x = torch.asarray(x, dtype=self._ftype, device=self._device)

        shape = x.shape  # save the original shape
        x = x.view(-1, self._N)  # non-copy view

        Jx = torch.zeros(x.shape+(self._P,), dtype=self._ftype, device=self._device)

        for i in range(self._N):
            for j in range(self._P):
                Jx[:, i, j] = _misc.interp_nd(x, self._gridlines, self._J[..., i, j])

        # restore the original shape but do not copy
        return Jx.view(shape+(self._P,))
