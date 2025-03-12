#! /usr/bin/env python3
# vim:fenc=utf-8
#
# Copyright © 2025 Pi-Yueh Chuang <pychuang@pm.me>
#
# Distributed under terms of the BSD 3-Clause license.

"""DistroSA: a Python package for distributional sensitivity analysis.
"""
try:
    import cupy as _np
    device = "cuda"
except ImportError:
    import numpy as _np
    device = "cpu"

from ._sensitivity_1d import Sensitivity1D
from ._sensitivity_nd import SensitivityND
from ._sensitivity_nd_diag import SensitivityNDDiag
from ._sensitivity_nd_interp import SensitivityNDInterp
from ._sensitivity_nd_diag_interp import SensitivityNDDiagInterp
