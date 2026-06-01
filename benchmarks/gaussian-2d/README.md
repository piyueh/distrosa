The scripts for generating data and plotting figures are separate.

To generate data:

```bash
$ python fullmtx.py
$ python diagapprox.py
```

These two commands will create the `figs` directory and save the results (two `.dat` files) in it.

The total execution time is roughly 2 hours using an NVIDIA GeForce RTX 4070 (12GB
GPU RAM).
However, if a quick run is desired, you can modify the `nruns` variable in both `fullmtx.py` and `diagapprox.py` to `1`, which reduces the runtime to roughly 12 minutes.
`nruns` simply controls how many times each calculation is repeated to obtain better wall-time statistics.
The peak GPU RAM consumption is about 9GB.

If no GPU is found, the scripts will fall back to CPU execution.
However, CPU runs are not recommended unless you are patient or reduce the
number of points in the scripts.
No execution-time estimate is available for CPU runs.

To plot figures:

```bash
$ python plotsensitivities.py
$ python ploterrors.py
$ python plotconvergence.py
$ python plottimes.py
```

The plotting scripts run on a single CPU core, and the peak CPU RAM consumption is about
9GB.
It is possible to reduce RAM consumption by loading only the data for the current plot,
but you will need to modify the scripts yourself.
