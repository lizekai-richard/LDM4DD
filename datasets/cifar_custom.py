import torch
import kornia.augmentation as KA
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms.functional import pil_to_tensor


class __CIFAR_Customized(Dataset):
    '''
        Wrapping torchvision's cifar100 dataset so that it has the same format as example dataset.
    '''
    def __init__(self, root_dir, cifar_class=None, transforms=None, conditional=False, download=True):
        if cifar_class is None:
            raise NotImplementedError("Please call the subclass or specify cifar_class manually")
        self.cifar = cifar_class(root_dir, download=download)
        self.transforms = transforms
        self.conditional = conditional

        # set up transforms
        if self.transforms is not None:
            if self.conditional:
                data_keys = 2 * ['input']
            else:
                data_keys = ['input']

            self.input_T = KA.container.AugmentationSequential(
                *self.transforms,
                data_keys=data_keys,
                same_on_batch=False
            )
    
    def __len__(self):
        return len(self.cifar)

    def __getitem__(self, idx):
        # Single indexing or slicing (range of indices)
        if torch.is_tensor(idx):
            if idx.dim() == 0:
                idx = [idx.item()]
            else:
                idx = idx.tolist()
        
        if isinstance(idx, int):
            idx  = [idx]

        image = torch.cat([pil_to_tensor(self.cifar[i][0]).unsqueeze(0) for i in idx], dim=0)

        if self.conditional:
            c, h, w = image.shape
            slice = int(w / 2)
            condition = image[:, :, slice:]
            image = image[:, :, :slice]
            if self.transforms is not None:
                out = self.input_T(image, condition)
                image = out[0][0]
                condition = out[1][0]
            
            if image.shape[0] == 1:
                # Single image
                return image.squeeze(0), condition # image shape: (Channel, Height, Width)
            else:
                # Batch image
                return image, condition # image shape: (Batch, Channel, Height, Width)
        
        elif self.transforms is not None:
            image = self.input_T(image)[0]
            if image.shape[0] == 1:
                # Single image
                return image.squeeze(0)
            else:
                # Batch image
                return image
        
        else:
            if image.shape[0] == 1:
                # Single image
                return image.squeeze(0)
            else:
                # Batch image
                return image
    
    def get_label(self, idx):
        # Single indexing or slicing (range of indices)
        if torch.is_tensor(idx):
            if idx.dim() == 0:
                idx = idx.item()
            else:
                idx = idx.tolist()

        if isinstance(idx, int):
            return torch.tensor(self.cifar[idx][1]) # dim = 0
        else:
            return torch.tensor([self.cifar[i][1] for i in idx]) # dim = 1

class CIFAR10_Customized(__CIFAR_Customized):
    def __init__(self, root_dir, transforms=None, conditional=False, download=True):
        super().__init__(root_dir, cifar_class=CIFAR10, transforms=transforms, conditional=conditional, download=download)

class CIFAR100_Customized(__CIFAR_Customized):
    def __init__(self, root_dir, transforms=None, conditional=False, download=True):
        super().__init__(root_dir, cifar_class=CIFAR100, transforms=transforms, conditional=conditional, download=download)
        