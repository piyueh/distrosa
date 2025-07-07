# DistroSA: <ins>Distr</ins>ibuti<ins>o</ins>nal <ins>S</ins>ensitivity <ins>A</ins>nalysis

This package implements gradient estimation for random variables/vectors with respect to
distributional parameters in black-box probability density functions (PDFs) and
sampling routines. It reimplements
[this code](https://gitlab.com/ahmedattia/distributional_sensitivity_analysis) for
integration with PyTorch's automatic differentiation framework.

**Note**: This is proof-of-concept code and may contain issues.

## Contacts

* Ahmed Attia <aattia@anl.gov>
* Pi-Yueh Chuang <pchuang@anl.gov>
* Emil Constantinescu <emconsta@anl.gov>

## Dependencies

Note that package names are based on the PyPI registry and may differ in Conda (e.g.,
CuPy has a different package name in the Conda ecosystem).

### Mandatory

* `numpy>=2.1`
* `torch~=2.6`

### Optional

These optional dependencies are required for certain utilities in `distrosa.utils` and
some benchmarks in folder `benchmarks`:

* `psutil>=7.0`
* `matplotlib>=3.10`

For running benchmarks with GPUs:

* `cupy-cuda12x >= 13.4`

## Installation

To install, clone the repository, navigate into it, and run the following command:

```bash
pip install .
```

This will install only the mandatory dependencies and this package.

If you also want to install the optional dependencies, instead, use

```bash
pip install ".[opts]"
```

If you have GPU, you can use

```bash
pip install ".[gpu,opts]"
```

to install `cupy` as well. Note the core functionality of `distrosa` does not need
`cupy` to run on GPU. It fully relies on PyTorch for GPU support. `cupy` is only used
for running benchmarks on GPUs.

## Usage

This package provides four main gradient calculators for realizations (or space points)
of a random vector with respect to distributional parameters:

* `SensitivityND`
* `SensitivityNDDiag`
* `SensitivityNDInterp`
* `SensitivityNDDiagInterp`

They all share the same signature for creating a new instance. Refer to
`examples/basic.py` for a basic example.

## Caveats

### Floating Precision

Temporary tensors created during calculations default to 32-bit floats unless specified
otherwise, as this is PyTorch's default. If 64-bit precision is desired, set PyTorch's
default floating precision to `float64` using `torch.set_default_dtype(torch.float64)`
at the start of your application. Note that changing the precision of input tensors
alone will not affect the precision of temporary tensors.

### Vectorization

This is a limitation of PyTorch rather than the code. When using
`torch.autograd.functional.jacobian` to compute the Jacobian matrix of multiple space
points with respect to distribution parameters, calculations occur point by point, which
does not benefit from vectorization. However, using methods like `loss.backward()` for
scalar loss gradients with respect to distribution parameters is efficient.

### `torch.jit.script`

Not all calculators are compatible with `torch.jit.script`. Further investigation is
needed, though performance optimization is not the current focus.

### Peak Memory Consumption

As this code serves as a proof-of-concept for proposed numerical methods in our
publications, memory optimization has not been prioritized. Be mindful of memory
consumption.
