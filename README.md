# Distributional Sensitivity Analysis

This package develops novel algorithmic approaches for distributional sensitivity analysis. 
Specifically, we develop a new empirical algorithm for calculating derivative of any probability distribution
parameters with respect to perturbations in realization of the random variable. 
This is critical for machine learning applications that seek to fit a probability distribution 
to observational data.

## Caveats

The temporary tensors created and used during all calculations do not explicitly specify
the floating precision, and will default to 32bit floats if not specified (because this
is PyTorch's default).

So users should explicitly change PyTorch's default floating precision to `float64` if
that is desired (via `torch.set_default_dtype(torch.float64)` in the very beginning of
an application code).

This will ensure that all temporary tensors created and used during calculations will be
of type 64bit floats.

Note that merely changing the precision of user-provided input tensors will not change
the precision for these temporary tensors.

Using `.to(torch.float64)` after a PyTorch module is created will not help either, as
these temporary or hidden tensors are created with `float32` initially.

## Installation:

### Create virual environment (Only Once)
```
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Install the package under name `distrosa`  (Only when you pull a newer version)
```
pip install -e .
```

##Usage
In your code just load the package and/or its modules, e.g., 
```
from distrosa import derivatives_calculators
```
