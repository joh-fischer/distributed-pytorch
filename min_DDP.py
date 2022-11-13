import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"]="6,7"

import argparse

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

# DDP stuff
from torch.nn.parallel import DistributedDataParallel as DDP
import distributed as dist

# own stuff
from hip import HierarchicalPerceiver, ConvHierarchicalPerceiver
from config import HPARAMS_REGISTRY
from utils import Logger

# Parameters ####################
BATCH_SIZE = 8
DATA_SIZE = BATCH_SIZE * 2 + 12
EPOCHS = 2
DUMMY = True
CFG = 'hip16'
#################################

""" kill processes with
kill $(ps aux | grep multiprocessing.spawn | grep -v grep | awk '{print $2}')
"""

parser = argparse.ArgumentParser(description='PyTorch Training')
parser.add_argument('--gpu', default=None, type=int, metavar='GPU',
                    help='If GPU is available, which GPU to use for training.')

# config
if DUMMY:
    IM_SIZE = 128
    CLASSES = 100
else:
    H = HPARAMS_REGISTRY[CFG]
    IM_SIZE = H.im_size
    CLASSES = H.n_classes


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
    def __init__(self, n_classes, im_size, in_channels=3):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, kernel_size=im_size)
        self.lin = nn.Linear(64, n_classes)

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.lin(x)

        return x


def setup_model():
    if DUMMY:
        return DummyModel(CLASSES, IM_SIZE)
    else:
        return ConvHierarchicalPerceiver(H) if H.conv else HierarchicalPerceiver(H)


# Main workers ##################
def main_worker(gpu, world_size, args, dummy):
    args.gpu = gpu
    
    if args.distributed:
        dist.init_process_group(gpu, world_size)


    """ Model """
    model = setup_model()

    # determine device
    if not torch.cuda.is_available():               # cpu
        device = torch.device('cpu')
    # single or multi gpu
    else:                                           # gpu
        device = torch.device(f'cuda:{args.gpu}')
    model.to(device)
    
    if args.distributed:
        model = DDP(model, device_ids=[args.gpu])

    # optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), 0.0001)
    criterion = nn.CrossEntropyLoss().to(device)

    """ Data """
    dataset = DummyDataset(DATA_SIZE, IM_SIZE, CLASSES)
    sampler = dist.data_sampler(dataset, args.distributed, shuffle=True)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=(sampler is None), sampler=sampler)

    """ Logging """
    logger = Logger() if dist.is_primary() else None

    """ Run Epochs """
    for epoch in range(EPOCHS):
        if dist.is_primary():
            print(f"------- Epoch {epoch+1} - rank {gpu}")
            logger.init_epoch(epoch)
        
        if args.distributed:
            sampler.set_epoch(epoch)

        # training
        train(model, loader, criterion, optimizer, device, args, logger)

        # validate

        if args.distributed:
            dist.synchronize()
        
        if logger is not None:
            logger.end_epoch()

    if logger is not None:
        logger.save()

    # kill process group
    dist.cleanup()


def train(model, loader, criterion, optimizer, device, args, logger):
    model.train()

    # if dist.is_primary():
    #     loader = tqdm(loader, desc=f"Rank {dist.get_rank()}")
    for it, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)

        y_hat = model(x)
    
        loss = criterion(y_hat, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        correct = torch.argmax(y_hat, dim=1).eq(y).sum()
        n = y.shape[0]
        acc = correct / n

        # output
        if dist.is_primary():
            print(f"-- Iteration {it}")
        
        # output on every rank
        print(f"Device:\t{x.device}"
              f"\n\tInput: \t{x.shape}"
              f"\n\tLoss:  \t{loss.cpu().item():.5f}"
              f"\n\tAcc:   \t{correct / n:.5f}"
              f"\n\tCorr:  \t{correct}"
              f"\n\tN:     \t{n}"
              f"\n\tlabels:\t{y.tolist()}"
              f"\n\tpred:  \t{torch.argmax(y_hat, dim=1).tolist()}")

        # synchronize all metrics
        loss = dist.reduce(loss)
        correct = dist.reduce(correct)
        n = dist.reduce(torch.tensor(n).to(device))
        acc = correct / n

        # if dist.is_primary():
        #     loader.set_description("hi")

        if dist.is_primary():
            print(f"Main worker"
                  f" - N: {n}"
                  f" - corr: {correct}"
                  f" - acc: {acc.cpu().item():.4f}"
                  f" - loss: {loss.cpu().detach().item():.4f}")

        if logger is not None:
            logger.log_metrics({'loss': loss, 'acc': acc},
                               phase='train', aggregate=True, n=x.shape[0])


def main():         # one process
    args = parser.parse_args()
    for name, val in vars(args).items():
        print("{:<12}: {}".format(name, val))

    dist.launch(main_worker, args, None)


if __name__ == "__main__":
    main()
    