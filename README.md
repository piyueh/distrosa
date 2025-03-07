# Distributional Sensitivity Analysis

This package develops novel algorithmic approaches for distributional sensitivity analysis. 
Specifically, we develop a new empirical algorithm for calculating derivative of any probability distribution
parameters with respect to perturbations in realization of the random variable. 
This is critical for machine learning applications that seek to fit a probability distribution 
to observational data.

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
