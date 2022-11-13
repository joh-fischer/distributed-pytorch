import torch
from torch.utils.data import Dataset, DataLoader


class DummyDataset(Dataset):
    def __init__(self, im_size: int = 224, n_samples: int = 20000):
        super().__init__()
        self.im_size = im_size
        self.n_samples = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        im = torch.randn((3, self.im_size, self.im_size))
        label = idx % 10

        return im, label


if __name__ == "__main__":
    data = DummyDataset()
    loader = DataLoader(data, batch_size=32, shuffle=True)
    ims, labels = next(iter(loader))
    print("ims:", ims.shape)
    print("labels:", labels.shape)
