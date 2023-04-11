import math
import copy
from tqdm import tqdm
from pathlib import Path
from random import random
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torchaudio import transforms

from asteroid.losses.sdr import MultiSrcNegSDR


def melspec_loss_fn(s, s_hat, n_freq):
    
    loss = 0
    sl = s.shape[-1]

    for n in n_freq:

        mel_transform = transforms.MelSpectrogram(
            n_fft=2**n, hop_length=(2**n)//4, 
            win_length=2**n, window_fn=torch.hann_window,n_mels=64,
            normalized=True, center=False, pad_mode=None, power=1).to(s.device)
        
        mel_s = mel_transform(s)
        mel_s_hat = mel_transform(s_hat)
    
    loss += torch.sum(abs(mel_s - mel_s_hat)) + torch.sum((mel_s - mel_s_hat) ** 2)

    loss /= 8*sl
    
    return loss

# for VAE
def prior_loss_fn(mu, logvar, mu_p=0, logvar_p=0):

    # mu_p = torch.zeros(mu.shape).to if mu_p == 0 else mu_p
    # logvar_p = torch.zeros(logvar.shape) if logvar_p == 0 else logvar_p\

    out = 0.5 * torch.mean(mu.pow(2) + logvar.exp() - logvar - 1 )

    # mu_p = torch.tensor(mu_p).to(mu.device)
    # logvar_p = torch.tensor(logvar_p).to(mu.device)
    # a = logvar_p - logvar
    # b = (torch.exp(logvar) + torch.pow((mu - mu_p), 2))/(torch.exp(logvar_p) + 1e-20)
    # out = 0.5 * (a + b -1)

    return torch.mean(out)


class ClippedSDR(nn.Module):
    def __init__(self, clip_value=-30):
        super(ClippedSDR, self).__init__()

        self.snr = MultiSrcNegSDR("sdsdr")
        self.clip_value = float(clip_value)

    def forward(self, est_targets, targets):

        return torch.clamp(self.snr(est_targets, targets), min=self.clip_value)


def cal_sdr(s, s_hat):

    # s, s_hat - (bt, L)
    s = s.cpu()
    s_hat = s_hat.cpu()
    return torch.mean(
        -10 * torch.log10(
        torch.sum((s - s_hat)**2, -1) / torch.sum(s**2, -1))
    )
