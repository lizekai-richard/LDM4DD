import argparse
import lightning.pytorch as pl
import kornia.augmentation as KA
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from diffusion.LatentDiffusion import LatentDiffusionConditional
from datasets.cifar_custom import CIFAR10_Customized
from utils.EMA import EMA


def train(args):
    print("Building the datasets...")
    train_ds = CIFAR10_Customized(args.train_data_path,
                                  train=True,
                                  conditional=True)
    val_ds = CIFAR10_Customized(args.val_data_path,
                                train=False,
                                conditional=True)
    print("Done!")

    print("Building the latent diffusion model...")
    model = LatentDiffusionConditional(train_dataset=train_ds,
                                       valid_dataset=val_ds,
                                       num_timesteps=args.num_timesteps,
                                       lr=args.lr,
                                       num_warmup_steps=args.num_warmup_steps,
                                       num_epochs=args.num_epochs,
                                       batch_size=args.batch_size)
    print("Done!")
    logger = WandbLogger(project="CS5340")
    # logger = TensorBoardLogger(save_dir="tensorboard_logs/")

    trainer = pl.Trainer(
        max_steps=args.max_steps,
        max_epochs=args.num_epochs,
        callbacks=[
            EMA(0.9999),
            ModelCheckpoint(
                monitor='val_loss',
                dirpath=args.save_path,
                every_n_epochs=args.save_freq,
                save_top_k=args.top_k
            )],
        accelerator="gpu",
        devices="auto",
        logger=logger,
        log_every_n_steps=args.log_steps
    )
    print("Start Training...")
    trainer.fit(model)

    trainer.save_checkpoint(args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data_path", type=str, default="/path/to/train/data")
    parser.add_argument("--val_data_path", type=str, default="/path/to/validation/data")
    parser.add_argument("--save_path", type=str, default="/path/to/save/model")
    parser.add_argument("--save_freq", type=int, default=200)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--max_steps", type=int, default=100000)
    parser.add_argument("--log_steps", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=1000)
    parser.add_argument("--num_warmup_steps", type=int, default=100)
    parser.add_argument("--devices", type=str, default="0,1,2,3")

    args = parser.parse_args()
    train(args)
