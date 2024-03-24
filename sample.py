import os
import argparse
import random
import torch
import numpy as np
from classifier_free.DDPM import GaussianDiffusion
from classifier_free.UNet import Unet
import torchvision.transforms as T

IDX2CLASS = {
    0: 'airplane',
    1: 'automobile',
    2: 'bird',
    3: 'cat',
    4: 'deer',
    5: 'dog',
    6: 'frog',
    7: 'horse',
    8: 'ship',
    9: 'truck'
}

def manual_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def conditional_sample(args):

    manual_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cuda")

    model = Unet(
        dim=64,
        num_classes=10,
        dim_mults=(1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=args.num_timesteps,  # number of steps
        sampling_timesteps=args.num_sampling_steps
    ).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    diffusion.load_state_dict(checkpoint['model'])

    condition = torch.tensor([int(args.label) for _ in range(args.ipc)], dtype=torch.long).to(device)
    syn_images = diffusion.sample(condition)
    syn_images = syn_images.detach().cpu()

    to_image = T.ToPILImage()
    save_dir = IDX2CLASS[args.label] + "/ipc" + args.ipc + "/"
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(args.ipc):
        image = to_image(syn_images[i])
        image.save(save_dir + str(i) + ".png")


def test():
    transform = T.ToPILImage()
    random_image = torch.randn(4, 3, 32, 32)
    image = transform(random_image)
    image.save("test1.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="/path/to/checkpoint")
    parser.add_argument("--save_dir", type=str, default="/path/to/save/samples")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--label", type=int, default=0)

    args = parser.parse_args()

    conditional_sample(args)

