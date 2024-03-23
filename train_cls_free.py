import os
import argparse
import random
import torch
import numpy as np
from classifier_free.DDPM import GaussianDiffusion
from classifier_free.UNet import Unet
from trainer import Trainer
from datasets.cifar_custom import CIFAR10_Customized, CIFAR100_Customized


def manual_seed(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(args):

    manual_seed(42)

    train_ds = CIFAR100_Customized(args.train_data_path,
                                  train=True,
                                  conditional=True)
    val_ds = CIFAR100_Customized(args.val_data_path,
                                train=False,
                                conditional=True)

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
    )

    trainer = Trainer(
        diffusion,
        train_dataset=train_ds,
        val_dataset=val_ds,
        train_batch_size=args.batch_size,
        train_lr=args.lr,
        train_num_steps=args.num_steps,  # total training steps
        gradient_accumulate_every=2,  # gradient accumulation steps
        ema_decay=0.995,  # exponential moving average decay
        amp=True,  # turn on mixed precision
        calculate_fid=True,  # whether to calculate fid during training
        results_folder=args.save_dir,
        save_and_sample_every=args.save_freq
    )

    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="/path/to/train/data")
    parser.add_argument("--val_data_path", type=str, default="/path/to/validation/data")
    parser.add_argument("--save_dir", type=str, default="/path/to/save/model")
    parser.add_argument("--save_freq", type=int, default=200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--num_steps", type=int, default=100000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--num_sampling_steps", type=int, default=250)
    parser.add_argument("--accelerator", type=str, default="cpu")
    #parser.add_argument("--devices", type=str, default="auto")

    args = parser.parse_args()
    train(args)
