import torch
import kornia.augmentation as KA
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms


class __CIFAR_Customized(Dataset):
    """
        Wrapping torchvision's cifar dataset so that it has the same format as example dataset.
    """

    def __init__(self, root_dir,
                 cifar_class=None,
                 train=True,
                 transform=None,
                 conditional=False,
                 download=True):
        if cifar_class is None:
            raise NotImplementedError("Please call the subclass or specify cifar_class manually")
        self.cifar = cifar_class(root_dir, train=train, download=download)
        stats = ((0.5074,0.4867,0.4411),(0.2011,0.1987,0.2025))
        if train:
            self.transforms = transforms.Compose([
                transforms.RandomCrop(32, padding=4,padding_mode="reflect"),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
                transforms.ToTensor(),
                transforms.Normalize(*stats)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(*stats)
            ])
        self.conditional = conditional
        self.num_classes = max(self.cifar.targets) - min(self.cifar.targets) + 1

    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        # Single indexing or slicing (range of indices)
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, label = self.cifar[idx]
        image = self.transforms(image)
        condition = torch.tensor(label, dtype=torch.long)
        if self.conditional:
            return image, condition  # image shape: (Batch, Channel, Height, Width)
        else:
            return image


class CIFAR10_Customized(__CIFAR_Customized):
    def __init__(self, root_dir, train=True, transforms=None, conditional=False, download=True):
        super().__init__(root_dir, cifar_class=CIFAR10, train=train, transform=transforms, conditional=conditional,
                         download=download)


class CIFAR100_Customized(__CIFAR_Customized):
    def __init__(self, root_dir, train=True, transforms=None, conditional=False, download=True):
        super().__init__(root_dir, cifar_class=CIFAR100, train=train, transform=transforms, conditional=conditional,
                         download=download)
