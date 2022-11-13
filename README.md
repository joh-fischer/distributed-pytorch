# Distributed PyTorch Training


Small helper file to handle distributed, multi-GPU training in PyTorch.


In case the training gets interrupted without freeing the port, run
```
kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')
```
to kill all `multiprocessing.spawn` related processes. 