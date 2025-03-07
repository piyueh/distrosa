#!/usr/bin/env python3
# vim:fenc=utf-8

"""Testing functions in distrosa/misc.py.
"""
import unittest


class Test_get_cdf_1d(unittest.TestCase):

    def test_numpy_1(self):
        import numpy
        from distrosa.misc import get_cdf_1d

        def density(x, params):
            a, b, c = params
            return a * x**2 + b * x + c

        def ans(x, params):
            a, b, c = params
            vals = a / 3. * x**3 + b / 2. * x**2 + c * x
            vals -= vals[0]
            norm = vals[-1]
            vals /= norm
            return vals, norm

        params = numpy.array([1.1, -0.3, 0.5])

        err_cdfs = []
        err_norms = []
        for n in [128, 256, 512, 1024, 2048]:
            verts = numpy.linspace(-2.1, 7.6, n+1)
            cdfs, norm = get_cdf_1d(density, verts, params)
            ans_cdfs, ans_norm = ans(verts, params)
            err_cdfs.append(numpy.sqrt(numpy.trapezoid((cdfs-ans_cdfs)**2, verts)))
            err_norms.append(numpy.abs(norm-ans_norm))

        err_cdfs = numpy.asarray(err_cdfs)
        err_norms = numpy.asarray(err_norms)
        self.assertTrue(numpy.all(numpy.abs(err_cdfs[:-1]/err_cdfs[1:]-4.0) < 1e-3))
        self.assertTrue(numpy.all(numpy.abs(err_norms[:-1]/err_norms[1:]-4.0) < 1e-3))

    def test_torch_cpu_1(self):
        import numpy
        import torch
        from distrosa.misc import get_cdf_1d

        def density(x, params):
            a, b, c = params
            return a * x**2 + b * x + c

        def ans(x, params):
            a, b, c = params
            vals = a / 3. * x**3 + b / 2. * x**2 + c * x
            vals = vals - vals[0]
            norm = vals[-1]
            vals = vals / norm
            return vals, norm

        params = torch.tensor([1.1, -0.3, 0.5], dtype=torch.float64, device="cpu")

        err_cdfs = []
        err_norms = []
        for n in [128, 256, 512, 1024, 2048]:
            verts = torch.linspace(-2.1, 7.6, n+1, dtype=torch.float64, device="cpu")
            cdfs, norm = get_cdf_1d(density, verts, params)
            ans_cdfs, ans_norm = ans(verts, params)
            err = torch.sqrt(torch.trapezoid((cdfs-ans_cdfs)**2, verts)).item()
            err_cdfs.append(err)
            err_norms.append(torch.abs(norm-ans_norm).item())

        err_cdfs = numpy.asarray(err_cdfs)
        err_norms = numpy.asarray(err_norms)
        self.assertTrue(numpy.all(numpy.abs(err_cdfs[:-1]/err_cdfs[1:]-4.0) < 1e-3))
        self.assertTrue(numpy.all(numpy.abs(err_norms[:-1]/err_norms[1:]-4.0) < 1e-3))


class Test_interp_1d(unittest.TestCase):

    def test_torch_cpu_1(self):
        import torch
        from distrosa.misc import interp_1d

        verts = torch.linspace(0.0, 4.0, 5, dtype=torch.float64, device="cpu")
        values = torch.linspace(1.0, 5.0, 5, dtype=torch.float64, device="cpu")

        x = torch.linspace(-0.5, 4.5, 11, dtype=torch.float64, device="cpu")
        ans = torch.linspace(0.5, 5.5, 11, dtype=torch.float64, device="cpu")

        y = interp_1d(x, verts, values)
        self.assertTrue(torch.allclose(y, ans))


if __name__ == "__main__":
    unittest.main()
