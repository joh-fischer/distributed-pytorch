import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', default=2, type=int, metavar='N',
                    help='Number of epochs to run (default: 2)')
parser.add_argument('--batch-size', default=16, metavar='N', type=int,
                    help='Mini-batch size (default: 16)')
parser.add_argument('--lr', default=0.001, type=float, metavar='LR',
                    help='Initial learning rate (default: 0.001)')
parser.add_argument('--gpus', default=0, type=int, nargs='+', metavar='GPUS',
                    help='If GPU(s) available, which GPU(s) to use for training.')
args = parser.parse_args()

# setup gpu
args.gpus = args.gpus if isinstance(args.gpus, list) else [args.gpus]
os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in args.gpus])

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp

from torch.nn.parallel import DistributedDataParallel as DDP

from vit.model import ViT
from dataloader import DummyDataset


def setup(rank, world_size):
    """
    process group: K gpus form process group
    backend: gloo, nccl
    rank: within process group each process is identified by its rank
        from 0 to K-1
    world_size: number of processes in the group (ie gpu number K)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def main(rank, world_size):
    print(f"Running basic DDP example on rank {rank}.")
    setup(rank, world_size)
    torch.cuda.set_device(rank)

    model = ViT().to(rank)
    model = DDP(model, device_ids=[rank])

    criterion = nn.CrossEntropyLoss().to(rank)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # data
    data = CIFAR10(128, )

    # create model and move it to GPU with id rank
    model = ViT().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    loss_fn = nn.MSELoss()
    optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

    optimizer.zero_grad()
    outputs = ddp_model(torch.randn(20, 10))
    labels = torch.randn(20, 5).to(rank)
    loss_fn(outputs, labels).backward()
    optimizer.step()

    cleanup()


def run_demo(demo_fn, world_size):
    mp.spawn(demo_fn,
             args=(world_size,),
             nprocs=world_size,
             join=True)


if __name__ == "__main__":
    run_demo(demo_basic, 4)
