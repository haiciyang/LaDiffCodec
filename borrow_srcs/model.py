import math

import torch
from torch import nn

from .unet import UNet
from .diff_loss import DenoiseDiffusion

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class DiffAudioRep(nn.Module):

    def __init__(self):

        super(). __init__()

        self.eps_model = UNet(
            image_channels=1,
            n_channels=64,
            ch_mults=[1, 2, 2, 4],
            is_attn=[False, False, False, True],
        ).to(device)

        # Create [DDPM class](index.html)
        self.diffusion = DenoiseDiffusion(
            eps_model=self.eps_model,
            n_steps=1_000,
            device=device,
        )

    def forward(self, x):  
        
        # ==== Run Latent Diffusion =====
        # x_rep = self.encoder(x)
        
        # x_rep_qtz = None
        # if self.quantization:
        #     quantizedResults = self.quantizer(x_rep, sample_rate=self.frame_rate, bandwidth=self.bandwidth)
        #     x_rep_qtz = quantizedResults.quantized
        #     qtz_loss = quantizedResults.penalty

        # # --- VAE model --- 
        # if self.run_vae:
        #     x_rep, prior_loss = self.vae(x_rep)

        # # --- Diffusion Loss --- 
        # if self.run_diff:
        #     # B, C, L = x_rep.shape
        #     x_rep = x_rep/torch.max(torch.abs(x_rep.squeeze()))
        #     diff_loss = self.diff_process(x_rep, x_rep_qtz) # condition on training or only on sampling
        # else:
        #     x_hat = self.decoder(x_rep)
        #     neg_loss = sdr_loss(x, x_hat).mean()

        # ==== Run Diffusion on time-domain ======

        # print(x.shape)
        # fake()
        x = x.squeeze(1)
        diff_loss = self.diffusion.loss(x) # condition on training or only on sampling

        # ==== Output losses ==== 
        return {'diff_loss': diff_loss}, 0



if __name__ == '__main__':

    dAR = DiffAudioRep()

    x = torch.rand(10, 128, 256)
    l = dAR.diff_loss(x)
    


        




