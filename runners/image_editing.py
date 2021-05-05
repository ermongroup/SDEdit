import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.optim as optim
import torch.utils.data as data
import torchvision
import torchvision.utils as tvu

from models.diffusion import Model
from models.ema import EMAHelper
from functions.process_data import *

import pdb


def get_beta_schedule(*, beta_start, beta_end, num_diffusion_timesteps):
    betas = np.linspace(beta_start, beta_end,
                            num_diffusion_timesteps, dtype=np.float64)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def extract(a, t, x_shape):
    """Extract coefficients from a based on t and reshape to make it
    broadcastable with x_shape."""
    bs, = t.shape
    assert x_shape[0] == bs
    out = torch.gather(torch.tensor(a, dtype=torch.float, device=t.device), 0, t.long())
    assert out.shape == (bs,)
    out = out.reshape((bs,)+(1,)*(len(x_shape)-1))
    return out


def image_editing_denoising_step_flexible_mask(x, t, *,
                   model,
                   logvar,
                   betas,
                   return_pred_xstart=False):
    """
    Sample from p(x_{t-1} | x_t)
    """
    alphas = 1.0-betas
    alphas_cumprod = alphas.cumprod(dim=0)

    model_output = model(x, t)
    weighted_score = betas / torch.sqrt(1 - alphas_cumprod)
    mean = extract(1/torch.sqrt(alphas), t, x.shape) * (x - extract(weighted_score, t, x.shape)*model_output)

    logvar = extract(logvar, t, x.shape)
    noise = torch.randn_like(x)
    mask = 1-(t == 0).float()
    mask = mask.reshape((x.shape[0],)+(1,)*(len(x.shape)-1))
    sample = mean + mask*torch.exp(0.5*logvar)*noise
    sample = sample.float()
    if return_pred_xstart:
        return sample, pred_xstart
    return sample


class Diffusion(object):
    def __init__(self, args, config, device=None):
        self.args = args
        self.config = config
        if device is None:
            device = torch.device(
                "cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device

        self.model_var_type = config.model.var_type
        betas = get_beta_schedule(
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
        )
        self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]

        alphas = 1.0-betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
        posterior_variance = betas * \
            (1.0-alphas_cumprod_prev) / (1.0-alphas_cumprod)
        if self.model_var_type == "fixedlarge":
            self.logvar = np.log(np.append(posterior_variance[1], betas[1:]))

        elif self.model_var_type == 'fixedsmall':
            self.logvar = np.log(np.maximum(posterior_variance, 1e-20))


    def image_editing_sample(self):
        print("Loading model")
        if self.config.data.dataset == "LSUN":
            if self.config.data.category == "bedroom":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
            elif self.config.data.category == "church_outdoor":
                url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
        elif self.config.data.dataset == "CelebA_HQ":
            url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
        else:
            raise ValueError

        model = Model(self.config)
        ckpt = torch.hub.load_state_dict_from_url(url, map_location=self.device)
        model.load_state_dict(ckpt)
        model.to(self.device)
        model = torch.nn.DataParallel(model)
        print("Model loaded")
        ckpt_id = 0

        download_process_data(path="colab_demo")
        n = self.config.sampling.batch_size
        model.eval()
        print("Start sampling")
        with torch.no_grad():
            name = self.args.npy_name
            [mask, img] = torch.load("colab_demo/{}.pth".format(name))

            mask = mask.to(self.config.device)
            img = img.to(self.config.device)
            img = img.unsqueeze(dim=0)
            img = img.repeat(n, 1, 1, 1)
            x0 = img

            tvu.save_image(x0, os.path.join(self.args.image_folder,
                                            f'original_input.png'))
            x0 = (x0 - 0.5) * 2.

            for it in range(self.args.sample_step):
                e = torch.randn_like(x0)
                total_noise_levels = self.args.t
                a = (1 - self.betas).cumprod(dim=0)
                x = x0 * a[total_noise_levels-1].sqrt() + e * (1.0 - a[total_noise_levels-1]).sqrt()
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                           f'init_{ckpt_id}.png'))

                with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                    for i in reversed(range(total_noise_levels)):
                        t = (torch.ones(n) * i).to(self.device)
                        x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                          logvar=self.logvar,
                                                          betas=self.betas,
                                                          return_pred_xstart=False)
                        x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                        x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                        # added intermediate step vis
                        if (i - 99) % 100 == 0:
                            tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                                                   f'noise_t_{i}_{it}.png'))
                        progress_bar.update(1)

                x0[:, (mask != 1.)] = x[:, (mask != 1.)]
                torch.save(x, os.path.join(self.args.image_folder,
                                           f'samples_{it}.pth'))
                tvu.save_image((x + 1) * 0.5, os.path.join(self.args.image_folder,
                                           f'samples_{it}.png'))
