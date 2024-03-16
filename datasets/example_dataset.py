import os
from skimage import io
import torch
from kornia.utils import image_to_tensor
import kornia.augmentation as KA
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR100
from torchvision.transforms.functional import pil_to_tensor

class ExampleImageDataset(Dataset):
    """Dataset returning images in a folder."""

    def __init__(self,
                 root_dir,
                 transforms=None,
                 conditional=False):
        self.root_dir = root_dir
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

            # check files
        supported_formats = ['webp', 'jpg', 'jpeg']
        self.files = [el for el in os.listdir(self.root_dir) if el.split('.')[-1].lower() in supported_formats]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.files[idx])
        image = image_to_tensor(io.imread(img_name)) / 255

        if self.conditional:
            c, h, w = image.shape
            slice = int(w / 2)
            condition = image[:, :, slice:]
            image = image[:, :, :slice]
            if self.transforms is not None:
                out = self.input_T(image, condition)
                image = out[0][0]
                condition = out[1][0]
            return image, condition
        
        elif self.transforms is not None:
            image = self.input_T(image)[0]
            return image
        else:
            return image
