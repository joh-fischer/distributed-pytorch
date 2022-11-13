
import os

os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hip import HierarchicalPerceiver, ConvHierarchicalPerceiver
from config import HPARAMS_REGISTRY


# Parameters
BATCH_SIZE = 256 * torch.cuda.device_count() 
DATA_SIZE = BATCH_SIZE * 4
DUMMY = True
CFG = 'cifar10_learned'

# config
if DUMMY:
    IM_SIZE = 32
    CLASSES = 10
    print("Use dummy model!")
else:
    H = HPARAMS_REGISTRY[CFG]
    IM_SIZE = H.im_size
    CLASSES = H.n_classes
    print(f"Use {CFG} model")



# set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Dummy Dataset
class DummyDataset(Dataset):
    def __init__(self, length, im_size, n_classes):
        self.len = length
        self.data = torch.randn(length, 3, im_size, im_size)
        self.labels = torch.randint(0, n_classes, (length,))

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


class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        print("\t\tmodel:\t", x.shape)
        return self.model(x)


######################################################################

dataset = DummyDataset(DATA_SIZE, IM_SIZE, CLASSES)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

if DUMMY:
    model = DummyModel(CLASSES, IM_SIZE)
else:
    model = ConvHierarchicalPerceiver(H) if H.conv else HierarchicalPerceiver(H)
model = ModelWrapper(model)

# put model to gpu
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = nn.DataParallel(model)
model.to(device)


optimizer = torch.optim.AdamW(model.parameters(), 0.0001)
criterion = nn.CrossEntropyLoss()
for it, (x, y) in enumerate(dataloader):
    print(f"Iteration {it+1}")
    x, y = x.to(device), y.to(device)
    print("\tInput:\t", x.shape)

    y_hat = model(x)
    print("\tOutput:\t", y_hat.shape)

    # compute loss
    loss = criterion(y_hat, y)
    # compute accuracy
    correct = torch.argmax(y_hat, dim=1).eq(y).sum()
    acc = correct / y.shape[0]
    print(f"\tloss: {loss.cpu().item():.5f} - acc: {acc:.5f}")

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
