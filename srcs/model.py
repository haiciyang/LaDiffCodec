import math
import torch
import random

from torch import nn

from .quantization import ResidualVectorQuantizer
from .modules import SEANetEncoder, SEANetDecoder, Unet1D, TransformerDDPM, UNet2D
from .losses import GaussianDiffusion1D, prior_loss_fn, sdr_loss, melspec_loss_fn, DenoiseDiffusion

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def reshape_to_4dim(x):

    if len(x.shape) == 4:
        return x
    elif len(x.shape) == 3:
        return x.unsqueeze(1)
    else:
        raise ValueError('Input has an unexpected shape:', x.shape)


def reshape_to_3dim(x):
    if len(x.shape) == 3:
        return x
    elif len(x.shape) == 4 and x.shape[1] == 1:
        return x.squeeze(1)
    else:
        raise ValueError('Input has an unexpected shape:', x.shape)

class FeatureLearner(nn.Module):

    def __init__(self, quantization=False, target_bandwidths=[1.5, 3, 6, 9, 12], **base_kwargs):
        super(). __init__()

        self.quantization = quantization
        self.sample_rate = base_kwargs['sample_rate'] 
        self.target_bandwidths = target_bandwidths
        
        self.encoder = SEANetEncoder(channels=1, kernel_size=7, last_kernel_size=7, **base_kwargs)
        self.decoder = SEANetDecoder(channels=1, kernel_size=7, last_kernel_size=7, **base_kwargs) 
        
        if quantization:
            print('bandwidth:', target_bandwidths)

            self.frame_rate = self.sample_rate/self.encoder.hop_length
            n_q = int(1000 * target_bandwidths[-1] // (math.ceil(self.frame_rate) * 10)) # Total number of quantizer needed
            self.quantizer = ResidualVectorQuantizer(dimension=base_kwargs['rep_dims'], n_q=n_q)

    def forward(self, x, bandwidth=None):
        
        x_rep = self.encoder(x)
        
        bandwidth = self.target_bandwidths[random.randint(0, len(self.target_bandwidths)-1)] if bandwidth is None else bandwidth
        
        if self.quantization:
            quantizedResults = self.quantizer(x_rep, sample_rate=self.frame_rate, bandwidth=bandwidth)
            x_rep = quantizedResults.quantized
            qtz_loss = quantizedResults.penalty
            
        x_hat = self.decoder(x_rep)
        neg_sdr = sdr_loss(x, x_hat).mean()
        
        if self.quantization:
            tot_loss = neg_sdr + qtz_loss
            return {'tot_loss': tot_loss, 'neg_sdr': neg_sdr, "qtz_loss": qtz_loss}, x_hat
        else:
            return {'neg_sdr': neg_sdr}, x_hat


    def get_feature(self, x, bandwidth=None):
        
        x_rep = self.encoder(x)
        if self.quantization:
            bandwidth = self.target_bandwidths[random.randint(0, len(self.target_bandwidths)-1)] if bandwidth is None else bandwidth
            quantizedResults = self.quantizer(x_rep, sample_rate=self.frame_rate, bandwidth=bandwidth)
            x_rep = quantizedResults.quantized
        
        return x_rep

class DiffAudioRep(nn.Module):
    def __init__(self, quantization=False, self_condition=False, other_cond=False, seq_length=320, ratios=[8],scaling_frame=False, scaling_feature=False, scaling_global=False, scaling_dim=False, sampling_timesteps=None, cond_global=1, cond_channels=128, upsampling_ratios=[5, 4, 2], unet_scale_x = False, unet_scale_cond = True, cond_bandwidth=3, **base_kwargs):

        super(). __init__()

        self.quantization = quantization
        self.cond_bandwidth = cond_bandwidth
        
        ENCODEC_RATIO = [8, 5, 4, 2]

        self.continuous_AE = FeatureLearner(quantization=False, ratios=ratios, **base_kwargs).eval() # Learn discrete features
        self.discrete_AE = FeatureLearner(quantization=True, ratios=ENCODEC_RATIO, **base_kwargs).eval()
        self.continuous_AE.requires_grad_(False)
        self.discrete_AE.requires_grad_(False)

        
        self.scaling_frame = scaling_frame
        self.scaling_feature = scaling_feature
        self.scaling_global = scaling_global
        self.scaling_dim = scaling_dim
        self.cond_global = cond_global
        self.unet_scale_x = unet_scale_x
        
        diff_backbone = Unet1D(dim = base_kwargs['diff_dims'], dim_mults=(1, 2, 2, 4, 4), inp_channels=base_kwargs['rep_dims'], self_condition=self_condition, other_cond=other_cond, scaling_frame=scaling_frame, scaling_feature=scaling_feature, scaling_global=scaling_global, scaling_dim=scaling_dim, cond_global=cond_global, cond_channels=cond_channels, upsampling_ratios=upsampling_ratios, unet_scale_x=unet_scale_x, unet_scale_cond=unet_scale_cond)

        self.diffusion = GaussianDiffusion1D(model=diff_backbone, seq_length=seq_length, sampling_timesteps=sampling_timesteps)              


    def scaling(self, x_rep, global_max=1):

        B, C, L = x_rep.shape
        
        scale = None
        if self.scaling_frame:
            # ---- Scaling for every frames -----
            scale, _ = torch.max(torch.abs(x_rep), 1, keepdim=True)
            x_rep = x_rep / (scale + 1e-20)
        elif self.scaling_feature:
            # --- Scaling for the feature map --- 
            scale, _ = torch.max(torch.abs(x_rep.reshape(B, C * L)), 1, keepdim=True)
            scale = scale.unsqueeze(-1)
            x_rep = x_rep / (scale + 1e-20)
        elif self.scaling_global:
            scale = global_max
            x_rep = x_rep / scale
        elif self.scaling_dim:
            scale, _ = torch.max(torch.abs(x_rep), -1, keepdim=True)
            x_rep = x_rep / scale

        return x_rep, scale
    
    def get_cond(self, x):
        return self.discrete_AE.get_feature(x, bandwidth=self.cond_bandwidth)
    
    def get_rep(self, x):
        x_rep = self.continuous_AE.get_feature(x)
        x_rep, scale = self.scaling(x_rep, global_max=18.0)
        return x_rep, scale
    
    def decode(self, rep):
        return self.continuous_AE.decoder(rep)
        # x_hat = self.discrete_AE.decoder(in_dec) # learn discrete features

    def forward(self, x, t=None):  
        
        # with torch.no_grad():
        #     cond = self.discrete_AE.get_feature(x, bandwidth=self.cond_bandwidth)
        #     x_rep = self.continuous_AE.get_feature(x)  
        #     # x_rep = self.discrete_AE.get_feature(x, bandwidth=12) # learn discrete features
            
        # # if not self.unet_scale_x:
        # x_rep, scale = self.scaling(x_rep, global_max=18.0)
        
        with torch.no_grad():
            cond = self.get_cond(x)
            rep, scale = self.get_rep(x)
            
        rep = reshape_to_3dim(rep)
        diff_loss, predicted_x_start, *other_reps_from_diff = self.diffusion(rep.detach(), cond, t=t) 

        in_dec = predicted_x_start * scale if scale is not None else predicted_x_start
        
        with torch.no_grad():
            x_hat = self.decode(in_dec)
            # x_hat = self.continuous_AE.decoder(in_dec)

        neg_sdr = sdr_loss(x, x_hat).mean()
        
        return {'diff_loss': diff_loss, 'neg_sdr': neg_sdr}, x_hat, rep, predicted_x_start, *other_reps_from_diff, scale

    @torch.no_grad()
    def sample(self, x, seq_length, sample_type='', midway_t = 100, lam = 0.1):
        
        midway_t = 100
        lam = 0.1
        
        self.diffusion.seq_length = seq_length
        
        with torch.no_grad():
            cond = self.get_cond(x)
            _, scale = self.get_rep(x)        
        
        # print(x.shape, cond.shape)
        # fake()
        
        # ------ rep diff ----- 
        
        # sampled_rep = self.diffusion.sample(batch_size=1, condition=cond)
        # x_scale_sample = self.continuous_AE.decoder(sampled_rep * scale)

        # ----- Infilling ----
        infill_img = cond
        for layer in self.diffusion.model.upsampling_layers:
            infill_img = layer(infill_img)
        infill_img = infill_img / torch.max(torch.abs(infill_img.flatten())) + 1e-8
        sample = self.diffusion.infilling(infill_img = infill_img, condition=cond, midway_t=midway_t, lam=lam)
        x_sample_infill = self.continuous_AE.decoder(sample * scale)
        
        # return x_scale_sample, x_sample_infill   
        return  x_sample_infill   
    

if __name__ == '__main__':

    dAR = DiffAudioRep()

    x = torch.rand(10, 128, 256)
    l = dAR.diff_loss(x)
    


        




