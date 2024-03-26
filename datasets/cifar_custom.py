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
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
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
        return image, condition  # image shape: (Batch, Channel, Height, Width)


class CIFAR10_Customized(__CIFAR_Customized):
    def __init__(self, root_dir, train=True, transforms=None, conditional=False, download=True):
        super().__init__(root_dir, cifar_class=CIFAR10, train=train, transform=transforms, conditional=conditional,
                         download=download)


class CIFAR100_Customized(__CIFAR_Customized):
    def __init__(self, root_dir, train=True, transforms=None, conditional=False, download=True):
        super().__init__(root_dir, cifar_class=CIFAR100, train=train, transform=transforms, conditional=conditional,
                         download=download)
