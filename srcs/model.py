import math

import torch
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


class DiffAudioRep(nn.Module):

    def __init__(self, rep_dims=128, emb_dims=128, diff_dims=128, norm: str='weight_norm', causal: bool=True, dilation_base=2, n_residual_layers=1, n_filters=32, lstm=0, quantization=False, bandwidth=3, sample_rate=16000, qtz_condition=False, self_condition=False, other_cond=False, seq_length=320, enc_ratios=[8, 5, 4, 2], run_diff=False, run_vae=False, model_type='', scaling_frame=False, scaling_feature=False, scaling_global=False, scaling_dim=False, freeze_ed=False, final_activation=None, sampling_timesteps=None, use_film=False, cond_global=1, cond_channels=128, upsampling_ratios=[5, 4, 2], unet_scale_x = False, unet_scale_cond = True, **kwargs):

        super(). __init__()

        self.quantization = quantization
        self.qtz_condition = qtz_condition
        self.sample_rate = sample_rate
        self.bandwidth = bandwidth

        self.run_diff = run_diff
        self.run_vae = run_vae
        self.model_type = model_type
        self.scaling_frame = scaling_frame
        self.scaling_feature = scaling_feature
        self.scaling_global = scaling_global
        self.scaling_dim = scaling_dim
        self.cond_global = cond_global

        self.encoder = SEANetEncoder(channels=1, ratios=enc_ratios,\
            dimension=rep_dims, norm=norm, causal=causal, dilation_base=dilation_base, n_residual_layers=n_residual_layers, n_filters=n_filters, lstm=lstm, kernel_size=7, last_kernel_size=7, final_activation=final_activation)
        self.decoder = SEANetDecoder(channels=1, ratios=enc_ratios, \
            dimension=rep_dims, norm=norm, causal=causal, dilation_base=dilation_base, n_residual_layers=n_residual_layers, n_filters=n_filters, lstm=lstm, kernel_size=7, last_kernel_size=7) 
        
        if run_vae:
            self.vae_mu_conv = nn.Conv1d(rep_dims//2, rep_dims, 1)
            self.vae_logvar_conv = nn.Conv1d(rep_dims//2, rep_dims, 1)

        if quantization:
            print('bandwidth:', bandwidth)

            self.frame_rate = self.sample_rate/self.encoder.hop_length
            n_q = int(1000 * bandwidth // (math.ceil(self.frame_rate) * 10))
            self.quantizer = ResidualVectorQuantizer(dimension=rep_dims, n_q=n_q)
        
        if freeze_ed:
            self.encoder.eval()
            self.decoder.eval()
        
        # if self.freeze_e_only:
        #     self.encoder.eval()
            
        if run_diff:
            if model_type == 'unet':
                self.diff_model = Unet1D(dim = diff_dims, dim_mults=(1, 2, 2, 4, 4), inp_channels=rep_dims, self_condition=self_condition, qtz_condition=qtz_condition, other_cond=other_cond, use_film=use_film, scaling_frame=scaling_frame, scaling_feature=scaling_feature, scaling_global=scaling_global, scaling_dim=scaling_dim, cond_global=cond_global, cond_channels=cond_channels, upsampling_ratios=upsampling_ratios, unet_scale_x=unet_scale_x, unet_scale_cond=unet_scale_cond)
                
            elif model_type == 'transformer':
                self.diff_model = TransformerDDPM(rep_dims = rep_dims,
                                            emb_dims = emb_dims, 
                                            mlp_dims= diff_dims,
                                            num_layers= 6,
                                            num_heads= 8,
                                            num_mlp_layers=2,
                                            self_condition=self_condition, 
                                            qtz_condition=qtz_condition)
            elif model_type == 'unet2d':
                self.diff_model = UNet2D(
                    inp_channels=1,
                    n_channels=diff_dims,
                    ch_mults=[1, 2, 2, 4],
                    is_attn=[False, False, False, True],
                    self_condition=self_condition, 
                    qtz_condition=qtz_condition
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
                self.diffusion = GaussianDiffusion1D(model=self.diff_model, seq_length=seq_length, sampling_timesteps=sampling_timesteps)
            

    def vae(self, rep):
        
        Bt, C, F = rep.shape
        mu = self.vae_mu_conv(rep[:, :C//2, :])
        logvar = self.vae_logvar_conv(rep[:, C//2:, :])

        noise = torch.randn_like(mu)
        rep = mu + torch.exp(logvar) * noise

        prior_loss = prior_loss_fn(mu, logvar)

        return rep, prior_loss

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


    def forward(self, x, t=None, cond=None):  
        
        # ==== Run Latent Diffusion =====

        # if model_ed is None:
        #     encoder = self.encoder
        #     decoder = self.decoder
        # else:
        #     model_ed.eval()
        #     encoder = model_ed.encoder
        #     decoder = model_ed.decoder
    
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

            B, C, L = x_rep.shape

            x_rep, scale = self.scaling(x_rep, global_max=18.0)
                
            if self.model_type == 'unet2d':
                x_rep = reshape_to_4dim(x_rep)
                diff_loss, predicted_x_start, *other_reps_from_diff = self.diffusion.loss(x_rep, t=t) # condition on training or only on sampling
                in_dec = predicted_x_start.squeeze(1) * scale if scale is not None else predicted_x_start.squeeze(1)
                x_hat = self.decoder(in_dec)
            else:
                x_rep = reshape_to_3dim(x_rep)
                if cond is not None: # Conditions from a different model
                    # cond, _ = self.scaling(cond, global_max=self.cond_global)
                    # cond, _ = self.scaling(cond)
                    diff_loss, predicted_x_start, *other_reps_from_diff = self.diffusion(x_rep.detach(), cond, t=t) 
                elif self.qtz_condition:
                    diff_loss, predicted_x_start, *other_reps_from_diff = self.diffusion(x_rep, x_rep_qtz, t=t) 
                    # condition on training or only on sampling
                else:
                    diff_loss, predicted_x_start, *other_reps_from_diff = self.diffusion(x_rep.detach(), t=t)

                in_dec = predicted_x_start * scale if scale is not None else predicted_x_start
                x_hat = self.decoder(in_dec)

        else:
            in_dec = x_rep_qtz if self.quantization else x_rep
            # # with torch.no_grad():
            x_hat = self.decoder(in_dec)

            # ****** only for encodec_tanh *******
            # qtz_loss.detach()
            # x_hat = self.decoder(x_rep)

        neg_loss = sdr_loss(x, x_hat).mean()
        # neg_loss = sdr_loss(x, x_hat).mean()
        # tot_loss = 0.01 * neg_loss + diff_loss

        # ==== Run Diffusion on time-domain ======

        # diff_loss, x_hat, xt, t = self.diffusion(x, None, t) # condition on training or only on 
        # neg_loss = sdr_loss(x, x_hat.detach()).mean()

        # ==== Output losses ==== 
        if self.run_diff:
            # return {'diff_loss': diff_loss}
            # return {'diff_loss': diff_loss, 'neg_loss': neg_loss}, x_hat, xt, t
            return {'diff_loss': diff_loss, 'neg_loss': neg_loss}, x_hat, x_rep, predicted_x_start, *other_reps_from_diff, x_rep_qtz, scale
            # return {'tot_loss': tot_loss, 'diff_loss': diff_loss, 'neg_loss': neg_loss}, x_hat, x_rep, predicted_x_start, *other_reps_from_diff, x_rep_qtz
        if self.run_vae:
            tot_loss = 0.1 * prior_loss + neg_loss
            return {'total_loss': tot_loss, 'prior_loss': prior_loss, 'neg_sdr': neg_loss}, x_hat
        else:
            # return {'neg_sdr': neg_loss}, x_hat
            if not self.quantization: # and not self.training:
                return {'neg_sdr': neg_loss}, x_hat
            else:
                tot_loss = qtz_loss + neg_loss
                # return {'neg_sdr': neg_loss}, x_hat
                return {'tot_loss': tot_loss, 'qtz_loss': qtz_loss, 'neg_sdr': neg_loss}, x_hat

    def get_cond(self, x):
        
        with torch.no_grad():
            x_rep = self.encoder(x)
            if self.quantization:
                quantizedResults = self.quantizer(x_rep, sample_rate=self.frame_rate, bandwidth=self.bandwidth)
                x_rep = quantizedResults.quantized
        
        return x_rep
    
    def get_scale(self, x):
        
        x_rep = self.encoder(x)
        x_rep, scale = self.scaling(x_rep, global_max=18.0)
        
        return scale


class DiffAudioTime(nn.Module):

    def __init__(self, rep_dims=128, emb_dims=128, diff_dims=128, norm: str='weight_norm', causal: bool=True, dilation_base=2, n_residual_layers=1, n_filters=32, lstm=0, quantization=False, bandwidth=3, sample_rate=16000, qtz_condition=False, self_condition=False, other_cond=False, seq_length=320, enc_ratios=[8, 5, 4, 2], run_diff=False, run_vae=False, model_type='', scaling_frame=False, scaling_feature=False, scaling_global=False, scaling_dim=False, freeze_ed=False, final_activation=None, sampling_timesteps=None, use_film=False, cond_global=1, cond_channels=128, upsampling_ratios=[5, 4, 2], unet_scale_x = False, unet_scale_cond = True,  **kwargs):

        super(). __init__()

        self.sample_rate = sample_rate
        self.bandwidth = bandwidth

        self.model_type = model_type
        

        if run_diff:
            if model_type == 'unet':
                self.diff_model = Unet1D(dim = diff_dims, dim_mults=(1, 2, 2, 4, 4), inp_channels=1, self_condition=self_condition, qtz_condition=qtz_condition, other_cond=other_cond, use_film=use_film, scaling_frame=scaling_frame, scaling_feature=scaling_feature, scaling_global=scaling_global, scaling_dim=scaling_dim, cond_global=cond_global, cond_channels=cond_channels, upsampling_ratios=upsampling_ratios, unet_scale_x=unet_scale_x, unet_scale_cond=unet_scale_cond)
                
            elif model_type == 'transformer':
                self.diff_model = TransformerDDPM(rep_dims = rep_dims,
                                            emb_dims = emb_dims, 
                                            mlp_dims= diff_dims,
                                            num_layers= 6,
                                            num_heads= 8,
                                            num_mlp_layers=2,
                                            self_condition=self_condition, 
                                            qtz_condition=qtz_condition)
            elif model_type == 'unet2d':
                self.diff_model = UNet2D(
                    inp_channels=1,
                    n_channels=diff_dims,
                    ch_mults=[1, 2, 2, 4],
                    is_attn=[False, False, False, True],
                    self_condition=self_condition, 
                    qtz_condition=qtz_condition
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
                self.diffusion = GaussianDiffusion1D(model=self.diff_model, seq_length=seq_length, sampling_timesteps=sampling_timesteps)


    def forward(self, x, t=None, cond=None):  

        # ==== Run Diffusion on time-domain ======
        diff_loss, predicted_x_start, *other_reps_from_diff = self.diffusion(x, cond, t=t) 
        neg_loss = sdr_loss(x, predicted_x_start.detach()).mean()

        return {'diff_loss': diff_loss, 'neg_loss': neg_loss}, predicted_x_start, *other_reps_from_diff

if __name__ == '__main__':

    dAR = DiffAudioRep()

    x = torch.rand(10, 128, 256)
    l = dAR.diff_loss(x)
    


        




