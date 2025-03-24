The scripts to obtain data and to plot figures are separated.

To get data:
```bash
$ python fullmtx.py
$ python diagapprox.py
```

The total execution time is less than 10 minutes using NVIDIA GeForce RTX 4070 (12GB
RAM).
And the peak GPU RAM consumption is about 9GB.
These two commands will create the `figs` directory and save two `.dat` files in it.

If no GPU is found, the scripts will fall back to CPU execution.
However, CPU runs are not recommended, unless either you're patient or you lower down
the number of points in the scripts.
No estimation of execution time is available for CPU runs.

To plot figures:
```bash
$ python plotsensitivities.py
$ python ploterrors.py
$ python plotconvergence.py
$ python plottimes.py
```

The plotting scripts run on a single CPU core, and the peak CPU RAM consumption is about
9GB.
It's possible to reduce the RAM consumption by not loading all data at once.
Instead, load only the data for the current plot.
But you'll need to modify the scripts yourself.
