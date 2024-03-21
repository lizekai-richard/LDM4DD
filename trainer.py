import torch
import torchvision
from pathlib import Path
from datetime import timedelta
from multiprocessing import cpu_count
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import utils
from ema_pytorch import EMA
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
import os
import numpy as np
import wandb
from einops import rearrange, repeat
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
from torch.nn.functional import adaptive_avg_pool2d
from tqdm.auto import tqdm
from utils.helper_functions import *


class FIDEvaluation:
    def __init__(
            self,
            batch_size,
            dl,
            sampler,
            channels=3,
            accelerator=None,
            stats_dir="./results",
            device="cuda",
            num_classes=10,
            num_fid_samples=50000,
            inception_block_idx=2048,
    ):
        self.batch_size = batch_size
        self.n_samples = num_fid_samples
        self.n_classes = num_classes
        self.device = device
        self.channels = channels
        self.dl = dl
        self.sampler = sampler
        self.stats_dir = stats_dir
        self.print_fn = print if accelerator is None else accelerator.print
        assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
        block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
        self.inception_v3 = InceptionV3([block_idx]).to(device)
        self.dataset_stats_loaded = False

    def calculate_inception_features(self, samples):
        if self.channels == 1:
            samples = repeat(samples, "b 1 ... -> b c ...", c=3)

        self.inception_v3.eval()
        features = self.inception_v3(samples)[0]

        if features.size(2) != 1 or features.size(3) != 1:
            features = adaptive_avg_pool2d(features, output_size=(1, 1))
        features = rearrange(features, "... 1 1 -> ...")
        return features

    def load_or_precalc_dataset_stats(self):
        path = os.path.join(self.stats_dir, "dataset_stats")
        try:
            ckpt = np.load(path + ".npz")
            self.m2, self.s2 = ckpt["m2"], ckpt["s2"]
            self.print_fn("Dataset stats loaded from disk.")
            ckpt.close()
        except OSError:
            num_batches = int(math.ceil(self.n_samples / self.batch_size))
            stacked_real_features = []
            self.print_fn(
                f"Stacking Inception features for {self.n_samples} samples from the real dataset."
            )
            for _ in tqdm(range(num_batches)):
                try:
                    real_samples, _ = next(self.dl)
                except StopIteration:
                    break
                real_samples = real_samples.to(self.device)
                real_features = self.calculate_inception_features(real_samples)
                stacked_real_features.append(real_features)
            stacked_real_features = (
                torch.cat(stacked_real_features, dim=0).cpu().numpy()
            )
            m2 = np.mean(stacked_real_features, axis=0)
            s2 = np.cov(stacked_real_features, rowvar=False)
            np.savez_compressed(path, m2=m2, s2=s2)
            self.print_fn(f"Dataset stats cached to {path}.npz for future use.")
            self.m2, self.s2 = m2, s2
        self.dataset_stats_loaded = True

    @torch.inference_mode()
    def fid_score(self):
        if not self.dataset_stats_loaded:
            self.load_or_precalc_dataset_stats()
        self.sampler.eval()
        batches = num_to_groups(self.n_samples, self.batch_size)
        stacked_fake_features = []
        self.print_fn(
            f"Stacking Inception features for {self.n_samples} generated samples."
        )
        for batch in tqdm(batches):
            fake_labels = torch.randint(0, self.n_classes, (batch,)).to(self.device)
            fake_samples = self.sampler.sample(classes=fake_labels)
            fake_features = self.calculate_inception_features(fake_samples)
            stacked_fake_features.append(fake_features)
        stacked_fake_features = torch.cat(stacked_fake_features, dim=0).cpu().numpy()
        m1 = np.mean(stacked_fake_features, axis=0)
        s1 = np.cov(stacked_fake_features, rowvar=False)

        return calculate_frechet_distance(m1, s1, self.m2, self.s2)


class Trainer(object):
    def __init__(
            self,
            diffusion_model,
            train_dataset,
            val_dataset,
            *,
            train_batch_size=16,
            gradient_accumulate_every=1,
            augment_horizontal_flip=True,
            train_lr=1e-4,
            train_num_steps=100000,
            ema_update_every=10,
            ema_decay=0.995,
            adam_betas=(0.9, 0.99),
            save_and_sample_every=1000,
            log_every=100,
            num_samples=25,
            results_folder='./results',
            amp=False,
            mixed_precision_type='fp16',
            split_batches=True,
            convert_image_to=None,
            calculate_fid=True,
            inception_block_idx=2048,
            max_grad_norm=1.,
            num_fid_samples=50000,
            plot_samples=True,
            save_best_and_latest_only=False,
            project_name="CS5340"
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision=mixed_precision_type if amp else 'no',
            kwargs_handlers=[InitProcessGroupKwargs(timeout=timedelta(seconds=10800))]
        )

        # wandb

        wandb.init(sync_tensorboard=False, project=project_name, job_type="CleanRepo")

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels
        is_ddim_sampling = diffusion_model.is_ddim_sampling

        # default convert_image_to depending on channels

        if not exists(convert_image_to):
            convert_image_to = {1: 'L', 3: 'RGB', 4: 'RGBA'}.get(self.channels)

        # sampling and training hyperparameters

        assert has_int_squareroot(num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.log_every = log_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        assert (train_batch_size * gradient_accumulate_every) >= 16, f'your effective batch size (' \
                                                                     f'train_batch_size x ' \
                                                                     f'gradient_accumulate_every) should ' \
                                                                     f'be at least 16 or above'

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        self.max_grad_norm = max_grad_norm

        # dataset and dataloader

        # self.ds = Dataset(folder, self.image_size, augment_horizontal_flip=augment_horizontal_flip,
        #                   convert_image_to=convert_image_to)
        self.train_ds = train_dataset
        self.val_ds = val_dataset

        assert len(
            self.train_ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

        train_dl = DataLoader(self.train_ds, batch_size=train_batch_size, shuffle=True, pin_memory=True, num_workers=cpu_count())
        val_dl = DataLoader(self.val_ds, batch_size=train_batch_size, shuffle=False, pin_memory=True, num_workers=cpu_count())
        train_dl = self.accelerator.prepare(train_dl)
        val_dl = self.accelerator.prepare(val_dl)
        self.train_dl = cycle(train_dl)
        self.val_dl = cycle(val_dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        self.sch = ReduceLROnPlateau(self.opt, mode='min', factor=0.5, patience=1)
        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

        # FID-score computation

        self.calculate_fid = calculate_fid and self.accelerator.is_main_process

        if self.calculate_fid:
            if not is_ddim_sampling:
                self.accelerator.print(
                    "WARNING: Robust FID computation requires a lot of generated samples and can therefore be very "
                    "time consuming.""Consider using DDIM sampling to save time."
                )
            self.fid_scorer = FIDEvaluation(
                batch_size=self.batch_size,
                dl=self.val_dl,
                sampler=self.ema.ema_model,
                channels=self.channels,
                accelerator=self.accelerator,
                stats_dir=results_folder,
                device=self.device,
                # num_fid_samples=num_fid_samples,
                num_fid_samples=len(self.val_ds),
                num_classes=self.val_ds.num_classes,
                inception_block_idx=inception_block_idx
            )

        if save_best_and_latest_only:
            assert calculate_fid, "`calculate_fid` must be True to provide a means for model evaluation for " \
                                  "`save_best_and_latest_only`."
            self.best_fid = 1e10  # infinite

        self.save_best_and_latest_only = save_best_and_latest_only
        self.plot_samples = plot_samples

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'sch': self.sch.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.sch.load_state_dict(data['sch'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    images, conditions = next(self.train_dl)
                    images = images.to(device)
                    conditions = conditions.to(device)

                    with self.accelerator.autocast():
                        loss = self.model(images, classes=conditions)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if accelerator.is_main_process and (self.step % self.log_every == 0):
                    wandb.log({'Train Loss': total_loss}, step=self.step)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()
                        milestone = self.step // self.save_and_sample_every

                        if self.calculate_fid:
                            fid_score = self.fid_scorer.fid_score()
                            self.sch.step(fid_score)

                            accelerator.print(f'fid_score: {fid_score}')
                            wandb.log({'FID Score': fid_score}, step=self.step)

                        if self.plot_samples:
                            sample_labels = []
                            for c in self.train_ds.num_classes:
                                sample_labels.extend([c] * 10)
                            sample_labels = torch.tensor(sample_labels, dtype=torch.long).to(device)
                            sample_images = self.ema.ema_model.sample(sample_labels)
                            grid = torchvision.utils.make_grid(sample_images, nrow=10, normalize=True, scale_each=True)
                            wandb.log({"Sampled Images": wandb.Image(torch.nan_to_num(grid.detach().cpu()))},
                                      step=self.step)

                        if self.save_best_and_latest_only:
                            if self.best_fid > fid_score:
                                self.best_fid = fid_score
                                self.save("best")
                            self.save("latest")
                        else:
                            self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')
        wandb.finish()
