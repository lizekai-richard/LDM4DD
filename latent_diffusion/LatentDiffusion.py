import lightning.pytorch as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from diffusers.models import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from .DenoisingDiffusion import DenoisingDiffusionProcess, DenoisingDiffusionConditionalProcess


class AutoEncoder(nn.Module):
    def __init__(self,
                 model_type="stabilityai/sd-vae-ft-ema" 
                 # @param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
                 ):
        """
            A wrapper for an AutoEncoder model

            By default, a pretrained AutoencoderKL is used from stabilitai

            A custom AutoEncoder could be trained and used with the same interface.
            Yet, this model works quite well for many tasks out of the box!
        """
        super().__init__()
        self.model = AutoencoderKL.from_pretrained(model_type)

    def forward(self, input):
        return self.model(input).sample

    def encode(self, input, mode=False):
        dist = self.model.encode(input).latent_dist
        if mode:
            return dist.mode()
        else:
            return dist.sample()

    def decode(self, input):
        return self.model.decode(input).sample


class LatentDiffusion(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4,
                 num_epochs=100,
                 num_warmup_steps=100):
        """
            This is a simplified version of Latent Diffusion
        """

        super().__init__()
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_warmup_steps = num_warmup_steps

        self.ae = AutoEncoder()
        with torch.no_grad():
            self.latent_dim = self.ae.encode(torch.ones(1, 3, 256, 256)).shape[1]
        for p in self.ae.parameters():
            p.requires_grad = False
        self.model = DenoisingDiffusionProcess(generated_channels=self.latent_dim,
                                               num_timesteps=num_timesteps)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        # return self.output_T(self.model(*args,**kwargs))
        return self.output_T(self.ae.decode(self.model(*args, **kwargs) / self.latent_scale_factor))

    def input_T(self, input):
        # By default, let the model accept samples in [0,1] range, and transform them automatically
        return (input.clip(0, 1).mul_(2)).sub_(1)

    def output_T(self, input):
        # Inverse transform of model output from [-1,1] to [0,1] range
        return (input.add_(1)).div_(2)

    def training_step(self, batch, batch_idx):

        latents = self.ae.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):

        latents = self.ae.encode(self.input_T(batch)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents)

        self.log('val_loss', loss)

        return loss

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=4)

    def val_dataloader(self):
        if self.valid_dataset is not None:
            return DataLoader(self.valid_dataset,
                              batch_size=self.batch_size,
                              shuffle=False,
                              num_workers=4)
        else:
            return None

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(list(filter(lambda p: p.requires_grad, self.model.parameters())), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=self.num_warmup_steps,
                                                    num_training_steps=len(self.train_dataloader()) * self.num_epochs)
        return [optimizer], [scheduler]


class LatentDiffusionConditional(LatentDiffusion):
    def __init__(self,
                 train_dataset,
                 valid_dataset=None,
                 num_timesteps=1000,
                 latent_scale_factor=0.1,
                 batch_size=1,
                 lr=1e-4,
                 num_epochs=100,
                 num_warmup_steps=100):
        pl.LightningModule.__init__(self)
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.lr = lr
        self.register_buffer('latent_scale_factor', torch.tensor(latent_scale_factor))
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_warmup_steps = num_warmup_steps

        self.ae = AutoEncoder()
        with torch.no_grad():
            self.latent_dim = self.ae.encode(torch.ones(1, 3, 256, 256)).shape[1]

        for p in self.ae.parameters():
            p.requires_grad = False
            
        self.model = DenoisingDiffusionConditionalProcess(generated_channels=self.latent_dim,
                                                          condition_channels=self.latent_dim,
                                                          num_timesteps=num_timesteps)

    @torch.no_grad()
    def forward(self, condition, *args, **kwargs):
        condition_latent = self.ae.encode(self.input_T(condition.to(self.device))).detach() * self.latent_scale_factor

        output_code = self.model(condition_latent, *args, **kwargs) / self.latent_scale_factor

        return self.output_T(self.ae.decode(output_code))

    def training_step(self, batch, batch_idx):
        condition, output = batch

        with torch.no_grad():
            latents = self.ae.encode(self.input_T(output)).detach() * self.latent_scale_factor
            latents_condition = self.ae.encode(self.input_T(condition)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents, latents_condition)

        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch, batch_idx):
        condition, output = batch

        with torch.no_grad():
            latents = self.ae.encode(self.input_T(output)).detach() * self.latent_scale_factor
            latents_condition = self.ae.encode(self.input_T(condition)).detach() * self.latent_scale_factor
        loss = self.model.p_loss(latents, latents_condition)

        self.log('val_loss', loss)

        return loss