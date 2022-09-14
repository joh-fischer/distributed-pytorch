import os
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data.distributed import DistributedSampler


class CIFAR10:
    def __init__(self, batch_size: int = 16, multi_gpu: bool = False,
                 num_workers: int = 0, pin_memory: bool = False):
        """
        Wrapper to load, preprocess and deprocess CIFAR-10 dataset.
        Args:
            batch_size (int): Batch size, default: 16.
        """
        self.classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        self.n_classes = 10

        self.batch_size = batch_size

        self.mean = [0.491, 0.482, 0.446]
        self.std = [1., 1., 1.]

        self.train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomCrop((32, 32), padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.Normalize(self.mean, self.std)
        ])

        self.val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std)
        ])

        self.train_set = torchvision.datasets.CIFAR10(root='./data', train=True, download=True,
                                                      transform=self.train_transform)
        train_sampler = DistributedSampler(dataset=self.train_set, shuffle=True) if multi_gpu else None
        self.train_loader = torch.utils.data.DataLoader(self.train_set, batch_size=self.batch_size,
                                                        sampler=train_sampler,
                                                        num_workers=num_workers, pin_memory=pin_memory)

        self.val_set = torchvision.datasets.CIFAR10(root='./data', train=False, download=True,
                                                    transform=self.val_transform)
        val_sampler = DistributedSampler(dataset=self.val_set, shuffle=True) if multi_gpu else None
        self.val_loader = torch.utils.data.DataLoader(self.val_set, batch_size=self.batch_size,
                                                      sampler=val_sampler, shuffle=False,
                                                      num_workers=num_workers)

        # invert normalization for tensor to image transform
        self.inv_normalize = transforms.Compose([
            transforms.Normalize(mean=0, std=[1./s for s in self.std]),
            transforms.Normalize(mean=[-m for m in self.mean], std=1.),
            lambda x: x*255
        ])

    @property
    def train(self):
        """ Return training dataloader. """
        return self.train_loader

    @property
    def val(self):
        """ Return validation dataloader. """
        return self.val_loader

    def idx2label(self, idx):
        """ Return class label for given index. """
        return self.classes[idx]

    def prob2label(self, prob_vector):
        """ Return class label with highest confidence. """
        return self.idx2label(torch.argmax(prob_vector).cpu())

    def tensor2img(self, tensor):
        """ Convert torch.Tensor to PIL image. """
        n_channels = tensor.shape[0]

        img = tensor.detach().cpu()
        img = self.inv_normalize(img)

        if n_channels > 1:
            return Image.fromarray(img.permute(1, 2, 0).numpy().astype('uint8')).convert("RGB")
        else:
            return Image.fromarray(img[0].numpy()).convert("L")


if __name__ == "__main__":
    data = CIFAR10(batch_size=16)

    for batch in data.train:
        ims, labels = batch

        print("images")
        print("\t", ims.shape)
        print(f"\t {ims.min()} < {torch.mean(ims)} < {ims.max()}")
        print("labels")
        print("\t", labels)

        ims = ims.detach().cpu()
        ims_grid = torchvision.utils.make_grid(ims)

        pil_ims = data.tensor2img(ims_grid)
        SAVE_DIR = 'datasamples'
        if not os.path.exists(SAVE_DIR):
            os.mkdir(SAVE_DIR)
        pil_ims.save(os.path.join(SAVE_DIR, 'cifar10_sample.png'))

        break