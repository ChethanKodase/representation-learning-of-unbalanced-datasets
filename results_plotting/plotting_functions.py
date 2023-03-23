from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from matplotlib.colors import Normalize

import torch
import numpy as np


def get_batch_psnr_ssim_lists(test_samples, reconstructions):
    all_psnr = []
    all_ssim = []
    for i in range(len(test_samples)):
        test_samples = test_samples.cpu().detach()
        reconstructions = reconstructions.cpu().detach()

        test_sample_normal = Normalize()(test_samples[i][0])
        reconstruction_normal = Normalize()(reconstructions[i][0])

        im_psnr = max(psnr(test_sample_normal, reconstruction_normal, data_range=1.), 0)
        im_ssim = max(ssim(test_sample_normal, reconstruction_normal, data_range=1.), 0)

        all_psnr.append(im_psnr)
        all_ssim.append(im_ssim)

    return all_psnr, all_ssim


def get_perturbed_samples(test_samples, proz, no_channels,dx, dy, device):
    perturbed_samples = torch.tensor([])
    for i in range(len(test_samples)):
        orig = test_samples[i].cpu().detach().numpy()

        noise_to_add = np.random.rand(no_channels,dx, dy)*(orig.max()-orig.min())*proz
        perturbed_im = np.add(orig,noise_to_add)
        perturbed_im = torch.tensor(perturbed_im)
        perturbed_samples = torch.cat((perturbed_samples, perturbed_im),0)
    return perturbed_samples.unsqueeze(1).float().to(device)