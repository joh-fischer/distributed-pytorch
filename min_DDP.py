import os
import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# DDP stuff
from torch.nn.parallel import DistributedDataParallel as DDP
import distributed as dist

from helpers import *


parser = argparse.ArgumentParser(description='PyTorch Multi-GPU Training')
parser.add_argument('--gpu', default=None, type=int, metavar='GPU',
                    help='Specify GPU for single GPU training. If not specified, it runs on all '
                         'CUDA_VISIBLE_DEVICES.')
parser.add_argument('--epochs', default=1, type=int, metavar='N',
                    help='Number of training epochs.')
parser.add_argument('--batch-size', default=8, type=int, metavar='N',
                    help='Batch size.')
parser.add_argument('--update-freq', default=2, type=int, metavar='N',
                    help='Gradient accumulation steps.')

# data
parser.add_argument('--n-classes', default=10, type=int, metavar='N',
                    help='Number of classes for fake dataset.')
parser.add_argument('--data-size', default=32, type=int, metavar='N',
                    help='Size of fake dataset.')
parser.add_argument('--image-size', default=64, type=int, metavar='N',
                    help='Size of input images.')


class DummyDataset(Dataset):
    def __init__(self, length, im_size, n_classes):
        self.len = length
        self.data = torch.randn(length, 3, im_size, im_size)
        self.labels = torch.randint(0, n_classes, (length,),
                                    generator=torch.Generator().manual_seed(0))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.len


class DummyModel(nn.Module):
    def __init__(self, n_classes, in_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=7)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.lin = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = self.avg_pool(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)

        return x


# Main workers ##################
def main_worker(gpu, world_size, args):
    args.gpu = gpu
    
    if args.distributed:
        dist.init_process_group(gpu, world_size)

    """ Data """
    dataset = DummyDataset(args.data_size, args.image_size, args.n_classes)
    sampler = dist.data_sampler(dataset, args.distributed, shuffle=False)
    loader = DataLoader(dataset, batch_size=args.batch_size,
                        shuffle=(sampler is None), sampler=sampler)

    """ Model """
    model = DummyModel(args.n_classes)

    # determine device
    if not torch.cuda.is_available():               # cpu
        device = torch.device('cpu')
    else:                                           # single or multi gpu
        device = torch.device(f'cuda:{args.gpu}')
    model.to(device)

    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])

    """ Optimizer and Loss """
    # optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), 0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    """ Run Epochs """
    for epoch in range(args.epochs):
        if dist.is_primary():
            print(f"------- Epoch {epoch+1}")
        
        if args.distributed:
            sampler.set_epoch(epoch)

        # training
        train(model, loader, criterion, optimizer, device, args.update_freq)

    # kill process group
    dist.cleanup()


def train(model, loader, criterion, optimizer, device, update_freq):
    model.train()
    optimizer.zero_grad()

    acc_meter = dist.AverageMeter()
    loss_meter = dist.AverageMeter()

    for it, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)

        # loss
        loss = criterion(y_hat, y)
        loss = loss / update_freq
        loss.backward()

        loss_meter.update(loss, y.shape[0])

        # accuracy
        correct = torch.argmax(y_hat, dim=1).eq(y).sum()
        n = y.shape[0]
        acc = correct / n
        acc_meter.update(acc, n)

        # metrics per gpu/process
        print(f"Device: {x.device}"
              f"\n\tLoss:  \t{loss.cpu().item():.5f}"
              f"\n\tLMeter:\t{loss_meter.avg:.5f}")

        dist.synchronize()

        if dist.is_primary():
            print(f"Finish iteration {(it % update_freq) + 1} / {update_freq}")
        
        # continue if step is not complete
        if not (it + 1) % update_freq == 0:
            continue
        
        if dist.is_primary():
            print("Update step")
        optimizer.step()
        optimizer.zero_grad()

        # synchronize metrics across gpus/processes
        acc_meter.synchronize_between_processes()
        loss_meter.synchronize_between_processes()


        # metrics over all gpus, printed only on the main process
        if dist.is_primary():
            print(f"Finish complete step"
                  f" - acc: {acc_meter.avg:.4f}"
                  f" - loss: {running_loss:.4f}"
                  f" - loss_meter: {loss_meter.avg:.4f}"
                  "\n------")

        acc_meter.reset()
        loss_meter.reset()


if __name__ == "__main__":
    # only run once
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<12}: {}".format(name, val))

    # start different processes, if multi-gpu is available
    # otherwise, it just starts the main_worker once
    dist.launch(main_worker, args)
    