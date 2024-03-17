import torch
import kornia.augmentation as KA
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms.functional import pil_to_tensor
from torchvision import transforms


class __CIFAR_Customized(Dataset):
    '''
        Wrapping torchvision's cifar dataset so that it has the same format as example dataset.
    '''
    def __init__(self, root_dir, 
                 cifar_class=None, 
                 target_size=None,
                 transforms=None, 
                 conditional=False, 
                 download=True):
        if cifar_class is None:
            raise NotImplementedError("Please call the subclass or specify cifar_class manually")
        self.cifar = cifar_class(root_dir, download=download)
        self.transforms = transforms
        self.conditional = conditional
        self.target_size = target_size

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
            idx = idx.tolist()

        image, label = self.cifar[idx]
        image = pil_to_tensor(image)  / 255

        # Standardize resolution to target size
        if self.target_size:
            resize_transform = transforms.Resize((self.target_size, self.target_size))
            image = resize_transform(image)

        if self.conditional:
            # Not sure how to use labels as condition - specified by input_T?
            condition = label
            if self.transforms is not None:
                # TODO: Check how condition works
                out = self.input_T(image, condition)
                image = out[0][0]
                condition = out[1][0]
                return image, condition # image shape: (Batch, Channel, Height, Width)
        
        elif self.transforms is not None:
            image = self.input_T(image)[0]
            return image
        
        else:
            return image
    
class CIFAR10_Customized(__CIFAR_Customized):
    def __init__(self, root_dir, transforms=None, conditional=False, download=True):
        super().__init__(root_dir, cifar_class=CIFAR10, transforms=transforms, conditional=conditional, download=download)

class CIFAR100_Customized(__CIFAR_Customized):
    def __init__(self, root_dir, transforms=None, conditional=False, download=True):
        super().__init__(root_dir, cifar_class=CIFAR100, transforms=transforms, conditional=conditional, download=download)
