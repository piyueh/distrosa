# DistroSA: *Distr*ibuti*o*nal *S*ensitivity *A*nalysis

This package provides implementations for estimating the gradients of random variables with respect to distributional parameters for black-box probability density functions (PDFs) and sampling subroutines. It is a re-implementation of [this code](https://gitlab.com/ahmedattia/distributional_sensitivity_analysis), with the goal of integrating into PyTorch's automatic differentiation.

## License Notice

This code currently does not have an open-source license. If you have access to this code, please ensure you obtain explicit permission from one of the following authors:
* Ahmed Attia (`aattia at anl.gov`)
* Pi-Yueh Chuang (`pchuang at anl.gov`)
* Emil Constantinescu (`emconsta at anl.gov`)

## Dependencies

Note that package names are based on the PyPI registry and may differ in Conda (e.g., CuPy has a different package name in the Conda ecosystem).

### Mandatory

* `numpy>=2.1`
* `torch~=2.6`

### Optional

These optional dependencies are required for certain utilities in `distrosa.utils` and some cases in `benchmarks`:
* `psutil>=7.0`
* `cupy-cuda12x>=13`

For plotting scripts in `benchmarks`:
* `matplotlib>=3.10`

## Installation

To install, clone the repository, navigate into it, and run the following command:

```bash
$ pip install ./
```

## Usage

This package provides four main gradient calculators for realizations (or space points) of a random vector with respect to distributional parameters:

* `SensitivityND`
* `SensitivityNDDiag`
* `SensitivityNDInterp`
* `SensitivityNDDiagInterp`

They all share the same signature for creating a new instance. Refer to `examples/basic.py` for a basic example.

## Caveats

### Floating Precision

Temporary tensors created during calculations default to 32-bit floats unless specified otherwise, as this is PyTorch's default. If 64-bit precision is desired, set PyTorch's default floating precision to `float64` using `torch.set_default_dtype(torch.float64)` at the start of your application. Note that changing the precision of input tensors alone will not affect the precision of temporary tensors.

### Vectorization

This is a limitation of PyTorch rather than the code. When using `torch.autograd.functional.jacobian` to compute the Jacobian matrix of multiple space points with respect to distribution parameters, calculations occur point by point, which does not benefit from vectorization. However, using methods like `loss.backward()` for scalar loss gradients with respect to distribution parameters is efficient.

### `torch.jit.script`

Not all calculators are compatible with `torch.jit.script`. Further investigation is needed, though performance optimization is not the current focus.

### Peak Memory Consumption

As this code serves as a proof-of-concept for proposed numerical methods in our publications, memory optimization has not been prioritized. Be mindful of memory consumption.

