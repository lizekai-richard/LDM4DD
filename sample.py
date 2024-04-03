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

IDX2CLASS = ['apple', 'aquarium_fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'castle', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dolphin', 'elephant', 'flatfish', 'forest', 'fox', 'girl', 'hamster', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn_mower', 'leopard', 'lion', 'lizard', 'lobster', 'man', 'maple_tree', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak_tree', 'orange', 'orchid', 'otter', 'palm_tree', 'pear', 'pickup_truck', 'pine_tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'sweet_pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow_tree', 'wolf', 'woman', 'worm']

def manual_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def conditional_sample(args):
    manual_seed(42)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
    model = Unet(
        dim=64,
        num_classes=100,
        dim_mults=(1, 2, 4, 8)
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=args.num_timesteps,  # number of steps
        sampling_timesteps=args.num_sampling_steps
    ).to(device)

    checkpoint = torch.load(args.ckpt_path, map_location=device)
    print(checkpoint.keys())
    diffusion.load_state_dict(checkpoint)

    for label in range(100):
        save_dir = "./CIFAR100" + "/ipc" + str(args.ipc) + "/" + IDX2CLASS[label]
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        to_image = T.ToPILImage()

        if args.ipc > 50:
            batch_size = 32
        else:
            batch_size = 1

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
                
        # remaining images
        if args.ipc - num_batches * batch_size != 0:
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
    parser.add_argument("--ckpt_path", type=str, default="saved_model/model-zekai.pt")
    parser.add_argument("--ipc", type=int, default=10)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--num_sampling_steps", type=int, default=250)

    args = parser.parse_args()

    conditional_sample(args)