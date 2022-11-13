# Distributed PyTorch Training


In `min_DDP.py` you can find a minimum working example of single-node, multi-gpu training with PyTorch.
The whole launch, multi-process spawn, and process-communication is handled by the functions defined
in `distributed.py`.

```python
import torch
import torch.nn as nn
import distributed as dist

from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
```

First, you need to specify a main worker. This function is executed on every GPU.

```python
def main_worker(gpu, world_size, args):
    args.gpu = gpu
    
    if args.distributed:
        dist.init_process_group(gpu, world_size)

    """ Data """
    dataset = ...       # your dataset
    sampler = dist.data_sampler(dataset, args.distributed, shuffle=False)   # returns sampler, if distributed
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=(sampler is None), sampler=sampler)

    """ Model """
    model = ...         # your model

    # determine device
    if not torch.cuda.is_available():               # cpu
        device = torch.device('cpu')
    else:                                           # single or multi gpu
        device = torch.device(f'cuda:{args.gpu}')
    model.to(device)
    
    # wrap model in DistributedDataParallel
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])

    """ Optimizer and Loss """
    # optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), 0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    """ Run Epochs """
    for epoch in range(args.epochs):
        if dist.is_primary():       # only print on main process
            print(f"------- Epoch {epoch+1} - rank {gpu}")
        
        if args.distributed:
            sampler.set_epoch(epoch)

        # training
        train(model, loader, criterion, optimizer, device)

    # kill process group
    dist.cleanup()
```

Then you can specify the trainings loop.

```python
def train(model, loader, criterion, optimizer, device):
    model.train()

    for it, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)
    
        loss = criterion(y_hat, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.argmax(y_hat, dim=1).eq(y).sum()
        n = y.shape[0]

        # metrics per gpu/process
        print(f"Device:\t{x.device}"
              f"\n\tInput: \t{x.shape}"
              f"\n\tLoss:  \t{loss.cpu().item():.5f}"
              f"\n\tAcc:   \t{correct / n:.5f} ({correct}/{n})")

        # synchronize metrics across gpus/processes
        loss = dist.reduce(loss)
        correct = dist.reduce(correct)
        n = dist.reduce(torch.tensor(n).to(device))
        acc = correct / n

        # metrics over all gpus, printed only on the main process
        if dist.is_primary():
            print(f"Total"
                  f" - acc: {acc.cpu().item():.4f} ({correct}/{n})"
                  f" - loss: {loss.cpu().item():.4f}")


if __name__ == "__main__":
    # only run once
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<12}: {}".format(name, val))

    # start different processes, if multi-gpu is available
    # otherwise, it just starts the main_worker once
    dist.launch(main_worker, args)
```

In case the training gets interrupted without freeing the port, run
```
kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')
```
to kill all `multiprocessing.spawn` related processes. 