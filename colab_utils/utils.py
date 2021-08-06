import torch
import warnings
from runners.image_editing import *
from main import dict2namespace
from functions.process_data import *
import os
import yaml
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.append("../")


warnings.filterwarnings("ignore")

device = "cuda"


def get_checkpoint(dataset, category):
    if category == "bedroom":
        url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/bedroom.ckpt"
    elif category == "church_outdoor":
        url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/church_outdoor.ckpt"
    elif dataset == "CelebA_HQ":
        url = "https://image-editing-test-12345.s3-us-west-2.amazonaws.com/checkpoints/celeba_hq.ckpt"
    else:
        raise ValueError
    return url


def get_config(file):
    with open(os.path.join('tutorial_code', 'configs', file), 'r') as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)
    return new_config


def load_model(dataset, category, file):
    url = get_checkpoint(dataset, category)
    config = get_config(file)
    model = Model(config)
    ckpt = torch.hub.load_state_dict_from_url(url, map_location=device)
    model.load_state_dict(ckpt)
    model.to(device)
    model = torch.nn.DataParallel(model)
    model.eval()
    print("Model loaded")

    betas = get_beta_schedule(
        beta_start=config.diffusion.beta_start,
        beta_end=config.diffusion.beta_end,
        num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps
    )
    betas = torch.from_numpy(betas).float()
    num_timesteps = betas.shape[0]

    alphas = 1.0 - betas
    alphas_cumprod = np.cumprod(alphas, axis=0)
    alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])
    posterior_variance = betas * \
        (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    logvar = np.log(np.maximum(posterior_variance, 1e-20))

    return model, betas, num_timesteps, logvar


def imshow(img, title=""):
    img = img.to("cpu")
    img = img.permute(1, 2, 0, 3)
    img = img.reshape(img.shape[0], img.shape[1], -1)
    img = img / 2 + 0.5     # unnormalize
    img = torch.clamp(img, min=0., max=1.)
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def SDEditing(betas, logvar, model, name, sample_step, total_noise_levels, n=4):
    print("Start sampling")

    with torch.no_grad():
        [mask, img] = torch.load("colab_demo/{}.pth".format(name))
        mask = mask.to(device)
        img = img.to(device)
        img = img.unsqueeze(dim=0)
        img = img.repeat(n, 1, 1, 1)
        x0 = img
        x0 = (x0 - 0.5) * 2.
        imshow(x0, title="Initial input")

        for it in range(sample_step):
            e = torch.randn_like(x0)
            a = (1 - betas).cumprod(dim=0).to(device)
            x = x0 * a[total_noise_levels - 1].sqrt() + e * (1.0 - a[total_noise_levels - 1]).sqrt()
            imshow(x, title="Perturb with SDE")

            with tqdm(total=total_noise_levels, desc="Iteration {}".format(it)) as progress_bar:
                for i in reversed(range(total_noise_levels)):
                    t = (torch.ones(n) * i).to(device)
                    x_ = image_editing_denoising_step_flexible_mask(x, t=t, model=model,
                                                                    logvar=logvar,
                                                                    betas=betas)
                    x = x0 * a[i].sqrt() + e * (1.0 - a[i]).sqrt()
                    x[:, (mask != 1.)] = x_[:, (mask != 1.)]
                    # added intermediate step vis
                    if (i - 99) % 100 == 0:
                        imshow(x, title="Iteration {}, t={}".format(it, i))
                    progress_bar.update(1)

            x0[:, (mask != 1.)] = x[:, (mask != 1.)]
            imshow(x)
