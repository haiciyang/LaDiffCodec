import math

import torch
from torch import nn

from .quantization import ResidualVectorQuantizer
from .modules import SEANetEncoder, SEANetDecoder, Unet1D, TransformerDDPM, UNet2D
from .losses import GaussianDiffusion1D, prior_loss_fn, sdr_loss, melspec_loss_fn, DenoiseDiffusion

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
class DiffAudioRep(nn.Module):

    def __init__(self, rep_dims=128, emb_dims=128, diff_dims=128, norm: str='weight_norm', causal: bool=True, dilation_base=2, n_residual_layers=1, n_filters=32, lstm=0, quantization=False, bandwidth=6, sample_rate=16000, self_condition=False, seq_length=320, ratios=[8, 5, 4, 2], run_diff=False, run_vae=False, model_type=''):

        super(). __init__()

        self.quantization = quantization
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth

        self.run_diff = run_diff
        self.run_vae = run_vae
        self.model_type = model_type

        self.encoder = SEANetEncoder(channels=1, ratios=ratios,\
            dimension=rep_dims, norm=norm, causal=causal, dilation_base=dilation_base, n_residual_layers=n_residual_layers, n_filters=n_filters, lstm=lstm, kernel_size=7, last_kernel_size=7)
        self.decoder = SEANetDecoder(channels=1, ratios=ratios, \
            dimension=rep_dims, norm=norm, causal=causal, dilation_base=dilation_base, n_residual_layers=n_residual_layers, n_filters=n_filters, lstm=lstm, kernel_size=7, last_kernel_size=7) 
        
        if run_vae:
            self.vae_mu_conv = nn.Conv1d(rep_dims//2, rep_dims, 1)
            self.vae_logvar_conv = nn.Conv1d(rep_dims//2, rep_dims, 1)

        if quantization:
            self.frame_rate = self.sample_rate/self.encoder.hop_length
            n_q = int(1000 * bandwidth // (math.ceil(self.frame_rate) * 10))
            self.quantizer = ResidualVectorQuantizer(dimension=rep_dims, n_q=n_q)

        if run_diff:
            if model_type == 'unet':
                self.diff_model = Unet1D(dim = diff_dims, dim_mults=(1, 2, 2, 4, 4), inp_channels=rep_dims, self_condition=self_condition)
                
            elif model_type == 'transformer':
                self.diff_model = TransformerDDPM(rep_dims = rep_dims,
                                            emb_dims = emb_dims, 
                                            mlp_dims= diff_dims,
                                            num_layers= 6,
                                            num_heads= 8,
                                            num_mlp_layers=2,)
            elif model_type == 'unet2d':
                self.diff_model = UNet2D(
                    inp_channels=1,
                    n_channels=64,
                    ch_mults=[1, 2, 2, 4],
                    is_attn=[False, False, False, True],
                ).to(device)

            else:
                print('Model type undefined')

            if model_type == 'unet2d':
                self.diffusion = DenoiseDiffusion(
                    eps_model=self.diff_model,
                    n_steps=1_000,
                    device=device,
                )
            else:
                self.diffusion = GaussianDiffusion1D(model=self.diff_model, seq_length=seq_length)
            

    def vae(self, rep):
        
        Bt, C, F = rep.shape
        mu = self.vae_mu_conv(rep[:, :C//2, :])
        logvar = self.vae_logvar_conv(rep[:, C//2:, :])

        noise = torch.randn_like(mu)
        rep = mu + torch.exp(logvar) * noise

        prior_loss = prior_loss_fn(mu, logvar)

        return rep, prior_loss


    def forward(self, x):  
        
        # ==== Run Latent Diffusion =====
        x_rep = self.encoder(x)
        
        x_rep_qtz = None
        if self.quantization:
            quantizedResults = self.quantizer(x_rep, sample_rate=self.frame_rate, bandwidth=self.bandwidth)
            x_rep_qtz = quantizedResults.quantized
            qtz_loss = quantizedResults.penalty

        # --- VAE model --- 
        if self.run_vae:
            x_rep, prior_loss = self.vae(x_rep)

        # --- Diffusion Loss --- 
        if self.run_diff:
            # B, C, L = x_rep.shape
            x_rep = x_rep/torch.max(torch.abs(x_rep.squeeze()))
            if self.model_type == 'unet2d':
                # x_rep = x_rep.unsqueeze(1)
                diff_loss = self.diffusion.loss(x_rep) # condition on training or only on sampling
            # diff_loss = self.diff_process(x_rep, x_rep_qtz) # condition on training or only on sampling
            else:
                x_rep = x_rep.squeeze(1)
                # print(x_rep.shape)
                # fake()
                diff_loss, x_self_cond, t = self.diffusion(x_rep) # condition on training or only on sampling
                x_hat = self.decoder(x_self_cond)
        else:
            x_hat = self.decoder(x_rep)
        
        neg_loss = sdr_loss(x, x_hat).mean()
        tot_loss = 0.01 * neg_loss + diff_loss
        # ==== Run Diffusion on time-domain ======

        # diff_loss = self.diff_process(x) # condition on training or only on sampling

        # ==== Output losses ==== 
        if self.run_diff:
            return {'tot_loss': tot_loss, 'diff_loss': diff_loss, 'neg_loss': neg_loss}, x_hat, x_rep, x_self_cond, t
        if self.run_vae:
            tot_loss = 0.1 * prior_loss + neg_loss
            return {'total_loss': tot_loss, 'prior_loss': prior_loss, 'neg_sdr': neg_loss}, x_hat
        else:
            return {'neg_sdr': neg_loss}, x_hat


if __name__ == '__main__':

    dAR = DiffAudioRep()

    x = torch.rand(10, 128, 256)
    l = dAR.diff_loss(x)
    


        




