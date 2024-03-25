import os
import argparse
import random
import torch
import uuid
import numpy as np
from tqdm import tqdm
from classifier_free.DDPM import GaussianDiffusion
from classifier_free.UNet import Unet
import torchvision.transforms as T
# os.environ["CUDA_VISIBLE_DEVICES"]="3"
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
    device = torch.device("cuda:3") if torch.cuda.is_available() else torch.device("cpu")

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

    for label in range(10):
        save_dir = "./CIFAR10" + "/ipc" + str(args.ipc) + "/" + IDX2CLASS[label]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        to_image = T.ToPILImage()

        if args.ipc > 50:
            batch_size = 32
        else:
            batch_size = 4

        num_batches = args.ipc // batch_size

        for i in tqdm(range(num_batches)):
            condition = torch.tensor([int(label) for _ in range(args.ipc)], dtype=torch.long).to(device)
            syn_images = diffusion.sample(condition)
            syn_images = syn_images.detach().cpu()

            for _ in range(batch_size):
                _id = str(uuid.uuid4())
                image = to_image(syn_images[i])
                save_path = os.path.join(save_dir, _id + ".png")
                image.save(save_path)
        
        condition = torch.tensor([int(label) for _ in range(args.ipc - num_batches * batch_size)], dtype=torch.long).to(device)
        syn_images = diffusion.sample(condition)
        syn_images = syn_images.detach().cpu()

        for _ in range(batch_size):
            _id = str(uuid.uuid4())
            image = to_image(syn_images[i])
            save_path = os.path.join(save_dir, _id + ".png")
            image.save(save_path)


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
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--num_sampling_steps", type=int, default=250)

    args = parser.parse_args()

    conditional_sample(args)

