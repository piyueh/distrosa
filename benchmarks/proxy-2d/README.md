To get data, run
```shell
$ python main.py
```

This test case takse a non-trivial time to complete. It can use multiple GPUs on a
single node. On a machine with 8 NVIDIA A100 GPUs, it takes roughly 8.5 hours to finish.
If the I/O is slow, it may take longer, as this test case outputs about 14800
small-sized files to the disk.

The parallelization is done via manager-worker dynamic scheduling. Currently each worker
has a full control of one GPU. It's possible to further improve the GPU utilization by
making multiple tasks/workers sharing one GPU. Interested users can tweak the device
tag associated to each worker in `main.py`.

To do the post-processing, run
```shell
$ python plotgrads.py
```
and
```shell
$ python plottraining.py
```
