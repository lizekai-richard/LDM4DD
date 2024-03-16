import os
import random
import argparse
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import matplotlib.pyplot as plt
from diffusion.LatentDiffusion import LatentDiffusionConditional
from diffusion.samplers.DDPM import DDPMSampler


def conditional_sample(args):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    num_classes = args.num_classes
    ipc = args.img_per_class

    ldm = LatentDiffusionConditional.load_from_checkpoint(args.ckpt_path).to(device)
    sampler = DDPMSampler(args.num_stpes, ldm.model.num_timesteps)

    transform = T.ToPILImage()
    for c in range(num_classes):
        condition = F.one_hot(torch.ones(ipc) * c, num_classes=num_classes)
        samples = ldm(condition.cuda(), sampler=sampler, verbose=True)
        samples = samples.detach().cpu()

        save_dir = os.path.join(args.save_dir, str(c))
        for i in range(ipc):
            sample = samples[i, :, :, :].permute(1, 2, 0)
            image = transform(sample)
            save_path = os.path.join(save_dir, str(i) + ".png")
            image.save(save_path)

        print("Visualizing Class {}...".format(c))
        for_visualization = random.sample(samples, k=4)
        for idx in range(for_visualization.shape[0]):
            plt.subplot(1, len(for_visualization), idx + 1)
            plt.imshow(for_visualization[idx].permute(1, 2, 0))
            plt.axis('off')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="/path/to/checkpoint")
    parser.add_argument("--save_dir", type=str, default="/path/to/save/samples")
    parser.add_argument("--img_per_class", type=int, default=10)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--num_classes", type=int, default=10)
    args = parser.parse_args()

    conditional_sample(args)

