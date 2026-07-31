import json
import math
import random


import torch
from torch import nn
from stable_audio_tools.models import create_model_from_config

from .quantization import ResidualVectorQuantizer
from .modules import SEANetEncoder, SEANetDecoder, Unet1D, TransformerDDPM, UNet2D, AE
from .losses import GaussianDiffusion1D, ShortcutModel, prior_loss_fn, sdr_loss, melspec_loss_fn, DenoiseDiffusion
from .utils import load_from_checkpoint, gaussian_kl_diag

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def exists(x):
    return x is not None

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


# https://github.com/yukara-ikemiya/modified-shortcut-models-pytorch/blob/master/src/trainer.py
class TimestepSampler:
    def __init__(
        self,
        rate_self_consistency: float = 0.25,
        min_dt: float = 0.0078125  # 1/128
    ):
        """
        rate_self_consistency: Propotion of samples for self-consistency term (default: 0.25)
        min_dt: Minimum value of 'dt' (default: 1/128)
        """
        assert 0 <= rate_self_consistency <= 1.0
        self.rate_sc = rate_self_consistency
        self.min_dt = min_dt

    def sample_t(self, num: int, device):
        num_sc = round(num * self.rate_sc)
        num_fm = num - num_sc

        # t for flow-matching term
        t_fm = torch.rand(num_fm, device=device)  # 0 -- 1
        dt_fm = torch.zeros(num_fm, device=device)

        # t/dt for self-consistency term
        t_sc = torch.rand(num_sc, device=device) * (1 - self.min_dt)  # 0 -- 1-min_dt
        max_dt = 1. - t_sc
        dt_sc = self.min_dt + torch.rand(num_sc, device=device) * (max_dt - self.min_dt)  # min_dt -- 1-t

        t = torch.cat([t_sc, t_fm])
        dt = torch.cat([dt_sc, dt_fm])
        assert len(t) == len(dt) == num

        return t, dt, num_sc


class VAE(nn.Module):
    def __init__(self, model_config='config/vae_2458.json', ckpt_path='ckpts/VAE_speech_2458.ckpt'):
        super(). __init__()

        with open(model_config) as f:
            model_config = json.load(f)

        self.model = create_model_from_config(model_config)
        state_dict = torch.load(ckpt_path)['state_dict']
        self.model.load_state_dict(state_dict, strict=True)

    @property
    def compression_rate(self):
        return 320

    @torch.no_grad()
    def encode(self, x):

        assert len(x.shape) == 3
        latents, encoder_info = self.model.encode(x, return_info=True)
        return latents
    @torch.no_grad()
    def decode(self, latents):

        decoded = self.model.decode(latents)
        return decoded
    @torch.no_grad()
    def forward(self, x):
        assert len(x.shape) == 3
        return self.decode(self.encode(x))

class Encodec_AE(nn.Module):
    def __init__(self, ratios=[8], sample_rate=16000, ckpt_path='ckpts/ae_compression_state_dict.bin'):
        super(). __init__()

        # encoder = SEANetEncoder(**model_config)
        # decoder = SEANetDecoder(**model_config) 

        model_config={
            "ratios":ratios, 
            "n_residual_layers": 1,
            "lstm": 2,
            "norm" : 'weight_norm',
            "pad_mode": "constant"
        }
        self.ratios = ratios

        import audiocraft 
        encoder = audiocraft.modules.SEANetEncoder(**model_config)
        decoder = audiocraft.modules.SEANetDecoder(**model_config)

        # frame_rate = kwargs["sample_rate"] // encoder.hop_length
        # renormalize = kwargs.pop("renormalize", False)

        self.model = AE(
            encoder = encoder,
            decoder = decoder,
            frame_rate = 50,  # Hard-coded for now
            renormalize = False, 
            channels = 1,
            sample_rate = sample_rate
        )
        # print(self.model)
        if ratios == [8]: # [8]
            ckpt_path = 'ckpts/ae_compression_state_dict.bin'
        elif ratios == [8, 5, 4, 2]: 
            ckpt_path = 'ckpts/ae8542_compression_state_dict.bin'

        pkt = torch.load(ckpt_path)
        self.model.load_state_dict(pkt['best_state'])
    
    @property
    def compression_rate(self):
        return torch.prod(torch.tensor(self.ratios))

    @torch.no_grad()
    def encode(self, x):
        emb, scale = self.model.encode(x)
        return emb
    @torch.no_grad()
    def decode(self, emb):
        return self.model.decode(emb)
    @torch.no_grad()
    def forward(self, x):
        return self.decode(self.encode(x))
              

class Encodec_official(nn.Module):
    def __init__(self, ckpt_path='ckpts/compression_state_dict.bin'):
        super(). __init__()

        from audiocraft.models import CompressionModel
        self.model = CompressionModel.get_pretrained(ckpt_path)

        self.frame_rate = 50
    
    def forward(self, x):
        pass

    @torch.no_grad()
    def encode(self, x, bandwidth=1.5):

        n_q = int(bandwidth * 1000 / (self.frame_rate * 10)) # 10 is log2(1024)
        self.model.quantizer.n_q = n_q

        emb = self.model.encoder(x)
        q_res = self.model.quantizer(emb, self.frame_rate)

        return q_res.x
    
    @torch.no_grad()
    def decode(self, quantized):
        return self.model.decoder(quantized)


class Encodec(nn.Module):
    def __init__(self, model_config='config/discrete.json', ckpt_path='saved_models/discrete_AE.amlt'): # TODO nearest
        super(). __init__()
        with open(model_config) as f:
            model_config = json.load(f)

        self.model = FeatureLearner(model_config)
        load_from_checkpoint(self.model, ckpt_path)
    
    def forward(self, x, bandwidth=None):
        return self.model(x, bandwidth)

    def encode(self, x, bandwidth=None):
        return self.model.encode(x, bandwidth)
    
    def decode(self, x):
        return self.model.decode(x)


class FeatureLearner(nn.Module):
    # def __init__(self, quantization=False, target_bandwidths=[1.5, 3, 6, 9, 12], **base_kwargs): # TODO nearest
    def __init__(self, model_config): # TODO nearest
        super(). __init__()

        self.quantization = model_config['quantization']
        self.sample_rate = model_config['sample_rate'] 
        self.target_bandwidths = model_config['target_bandwidths']

        self.encoder = SEANetEncoder(**model_config)
        self.decoder = SEANetDecoder(**model_config) 
        
        if self.quantization:
            print('bandwidth:', self.target_bandwidths)

            self.frame_rate = self.sample_rate/self.encoder.hop_length
            n_q = int(1000 * self.target_bandwidths[-1] // (math.ceil(self.frame_rate) * 10)) # Total number of quantizer needed
            self.quantizer = ResidualVectorQuantizer(dimension=model_config['dimension'], n_q=n_q)

    def forward(self, x, bandwidth=None):
        
        x_rep = self.encoder(x)
        
        bandwidth = self.target_bandwidths[random.randint(0, len(self.target_bandwidths)-1)] if bandwidth is None else bandwidth
        
        if self.quantization:
            quantizedResults = self.quantizer(x_rep, sample_rate=self.frame_rate, bandwidth=bandwidth)
            x_rep = quantizedResults.quantized
            qtz_loss = quantizedResults.penalty
            
        x_hat = self.decoder(x_rep)
        neg_sdr = sdr_loss(x, x_hat).mean()

        l_t = torch.mean(torch.abs(x - x_hat))
        l_f = melspec_loss_fn(x, x_hat, range(5,12))
        
        if self.quantization:
            # tot_loss = neg_sdr + qtz_loss
            return {'neg_sdr': neg_sdr, "qtz_loss": qtz_loss, 'l_t': l_t, 'l_f': l_f}, x_hat
        else:
            return {'neg_sdr': neg_sdr, "qtz_loss": torch.tensor(0), 'l_t': l_t, 'l_f': l_f}, x_hat

    def encode(self, x, bandwidth=None):
        
        x_rep = self.encoder(x)
        if self.quantization:
            bandwidth = self.target_bandwidths[random.randint(0, len(self.target_bandwidths)-1)] if bandwidth is None else bandwidth
            quantizedResults = self.quantizer(x_rep, sample_rate=self.frame_rate, bandwidth=bandwidth)
            x_rep = quantizedResults.quantized
        
        return x_rep
    
    def decode(self, x_rep):
        return self.decoder(x_rep)


class DAC(nn.Module):
    def __init__(self, model_type: str = "16khz"):
        super().__init__()
        try:
            import dac.utils
        except ImportError:
            raise RuntimeError("Could not import dac, make sure it is installed, "
                               "please run `pip install descript-audio-codec`")
        self.model = dac.utils.load_model(model_type=model_type)
        self.FRAME_RATE = 50
        self.CARDINALITY = 1024
        # self.n_quantizers = self.total_codebooks
        self.model.eval()

    def forward(self, x: torch.Tensor):
        raise NotImplementedError("Forward and training with DAC not supported.")
    
    def get_num_qtz(self, bandwidth):
        return int(bandwidth * 1000 / math.log2(self.CARDINALITY) / self.FRAME_RATE)

    def encode(self, x: torch.Tensor, bandwidth=3):
        
        n_quantizers = self.get_num_qtz(bandwidth)
        codes = self.model.encode(x)[1] # return shape (bt, 12, L/320); 12 is the total num of codebooks
        codes = codes[:, :n_quantizers]
        latent = self.model.quantizer.from_codes(codes)[0] # the original decode_latent def in DAC
        return latent

    def decode(self, latent):
        return self.model.decode(latent)


class DiffAudioRep(nn.Module):
    def __init__(self, 
                 discrete_type='Encodec', 
                 continuous_type='VAE_2458', 
                 inp_channels=128, 
                 cond_channels=128, 
                 quantization=False, 
                 cond_bandwidth=3, 
                 self_condition=False, 
                 other_cond=False, 
                 seq_length=320, sampling_timesteps=None, 
                 ratios=[8],
                 upsampling_ratios=[5, 4, 2], 
                 scaling_frame=False, scaling_feature=False, scaling_global=False, scaling_dim=False, cond_global=None,
                 unet_scale_x = False, unet_scale_cond = True, 
                 multi_cond=False, 
                 use_shortcut=False, 
                 random_condition = False,
                 **base_kwargs):

        super(). __init__()

        self.quantization = quantization
        self.cond_bandwidth = cond_bandwidth
        self.multi_cond = multi_cond
        self.random_condition = random_condition

        self.use_shortcut = use_shortcut
        self.inp_channels = inp_channels
        
        ENCODEC_RATIO = [8, 5, 4, 2]

        if continuous_type == 'AE':
            # self.continuous_AE = FeatureLearner(quantization=False, ratios=ratios, nearest=continuous_nearest,**base_kwargs)
            self.continuous_AE = Encodec_AE(ratios=ratios)
        elif continuous_type == "VAE_8":
            self.continuous_AE = VAE(model_config='config/vae_8.json', ckpt_path='ckpts/VAE_speech_8.ckpt')
        elif continuous_type == "VAE_2458":
            self.continuous_AE = VAE(model_config='config/vae_2458.json', ckpt_path='ckpts/VAE_speech_2458.ckpt')
        else:
            raise ValueError('Unsupported continuous autoencoder type.')
        
        self.continuous_AE.eval()
        self.continuous_AE.requires_grad_(False)

        other_cond = True
        if discrete_type == 'Encodec':
            # self.discrete_AE = FeatureLearner(quantization=True, ratios=ENCODEC_RATIO, cond_dims=cond_dims, nearest=False, **base_kwargs).eval() # TODO nearest
            # self.discrete_AE = Encodec().eval() # TODO nearest
            self.discrete_AE = Encodec_official().eval()
            self.discrete_AE.requires_grad_(False)
            
        elif discrete_type == "DAC":
            self.discrete_AE = DAC().eval()
            self.discrete_AE.requires_grad_(False)
        elif discrete_type == "":
            self.discrete_AE = None
            other_cond = False
            print('Running unconditional model')
        else:
            raise ValueError('Unsupported discrete autoencoder type.')
        

        self.scaling_frame = scaling_frame
        self.scaling_feature = scaling_feature
        self.scaling_global = scaling_global
        self.scaling_dim = scaling_dim
        self.cond_global = cond_global
        self.unet_scale_x = unet_scale_x
        
        diff_backbone = Unet1D(dim = base_kwargs['diff_dims'], dim_mults=(1, 2, 2, 4, 4), inp_channels=inp_channels, self_condition=self_condition, other_cond=other_cond, scaling_frame=scaling_frame, scaling_feature=scaling_feature, scaling_global=scaling_global, scaling_dim=scaling_dim, cond_global=cond_global, cond_channels=cond_channels, upsampling_ratios=upsampling_ratios, unet_scale_x=unet_scale_x, unet_scale_cond=unet_scale_cond, use_shortcut=use_shortcut)

        if not use_shortcut: # Standard DDPM
            self.diffusion = GaussianDiffusion1D(model=diff_backbone, seq_length=seq_length, sampling_timesteps=sampling_timesteps)    
        elif use_shortcut:
            self.ts_sampler = TimestepSampler(rate_self_consistency=0.25, min_dt=1/128)
            self.diffusion = ShortcutModel(model=diff_backbone)    


    def scaling(self, x_rep, global_max=1):

        B, C, L = x_rep.shape
        
        scale = 1
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
    
    def get_bandwidth_by_step(self, t):

        COND_STEP = [1.5, 3, 4.5, 6, 9, 12]
        SAMPLING_STEP = [475, 378, 326, 291, 246, 218]

        assert len(COND_STEP) == len(SAMPLING_STEP)

        if self.random_condition:
            return COND_STEP[torch.randint(len(COND_STEP), (1,))]

        for i, step in enumerate(SAMPLING_STEP):
            if t >= step:
                break
        return COND_STEP[i]
        
    
    def get_cond(self, x, t=None):

        if self.multi_cond and self.training:
            bandwidth = self.get_bandwidth_by_step(t) # Only use for training time
        else:
            bandwidth = self.cond_bandwidth

        if self.discrete_AE:
            return self.discrete_AE.encode(x, bandwidth=bandwidth)
        else: # Unconditionl model - not a codec
            return None
    
    def get_rep(self, x):
        if self.continuous_AE:
            x_rep = self.continuous_AE.encode(x)
            x_rep, scale = self.scaling(x_rep, global_max=18.0)
            return x_rep, scale
        else:
            return x, 1
    
    def decode(self, rep):
        if self.continuous_AE:
            return self.continuous_AE.decode(rep)
        else: # Model time-domain;
            return rep
        # x_hat = self.discrete_AE.decoder(in_dec) # learn discrete features

    def forward(self, x, t=None):  

        if self.multi_cond:
            t = torch.randint(0, self.diffusion.num_timesteps, (1,), device=device).long()

        with torch.no_grad():
            cond = self.get_cond(x, t)
            rep, scale = self.get_rep(x)
       
        rep = reshape_to_3dim(rep)

        if self.multi_cond and t is not None:
            t = t.expand(x.shape[0],)
        
        if self.use_shortcut:
            t, dt, num_sc = self.ts_sampler.sample_t(x.shape[0], device=x.device)
        else:
            dt = num_sc = None

        diff_loss, predicted_x_start, *other_reps_from_diff = self.diffusion(rep.detach(), t=t, cond=cond, dt=dt, num_self_consistency=num_sc) 

        in_dec = predicted_x_start * scale if scale is not None else predicted_x_start
        
        with torch.no_grad():
            x_hat = self.decode(in_dec)
            # x_hat = self.continuous_AE.decoder(in_dec)

        neg_sdr = sdr_loss(x, x_hat).mean()
        
        return {'diff_loss': diff_loss, 'neg_sdr': neg_sdr}, x_hat, rep, predicted_x_start, *other_reps_from_diff, scale

    @torch.no_grad()
    def run_continuous_ae(self, x):
        """ For debugging purpose mainly; Run time-domain signal through the continuous auto-encoder """
        return self.continuous_AE(x)

    @torch.no_grad()
    def run_discrete_ae(self, x):
        """ For debugging purpose mainly; Run time-domain signal through the discrete auto-encoder """
        return self.discrete_AE(x) 

    @torch.no_grad()
    def sample(self, x, seq_length, sample_type='', midway_t = 100, lam = 0.1, use_midway = False, clip_denoised=True):


        ######## DEBUGGING PURPOSE ##########

        # # # loss, x = self.run_discrete_ae(x) # x.shape - [1, 1, L]
        # loss, x = self.run_continuous_ae(x) # x.shape - [1, 1, L]
        # return x

        #####################################
        
        self.diffusion.seq_length = seq_length

        with torch.no_grad():
            cond = self.get_cond(x)
            # cond = None
            x_rep, scale = self.get_rep(x) #[1, 64, 160]  

        # print(torch.max(cond), torch.min(cond), scale) # (15, -14, 1.7)
        
        # ------ rep diff ----- 
        if not use_midway:
            if not self.use_shortcut:
                sampled_rep, means, vars, pred_noises, x_ts  = self.diffusion.sample(
                    batch_size=1, 
                    condition=cond, 
                    clip_denoised=clip_denoised)

                # print(torch.max(sampled_rep), torch.min(sampled_rep))
                entropy_step = self.compute_entropy(x_rep, means, vars, pred_noises, x_ts)
                x_scale_sample = self.decode(sampled_rep * scale)

                return x_scale_sample, entropy_step
            else:
                sampled_rep = self.diffusion.sample(
                    condition=cond, 
                    dim_in=self.inp_channels)
                x_scale_sample = self.decode(sampled_rep * scale)

                return x_scale_sample, None
                
        # # ----- Infilling ----
        elif use_midway:
            infill_img = cond
            # for layer in self.diffusion.model.upsampling_layers:
            #     infill_img = layer(infill_img)
            infill_img = infill_img / torch.max(torch.abs(infill_img.flatten())) + 1e-8

            sample = self.diffusion.infillin_new(
                infill_img = infill_img, 
                condition=cond, 
                midway_t=midway_t, 
                lam=lam,
                clip_denoised=clip_denoised)
            x_sample_infill = self.decode(sample * scale)
        
        return  x_sample_infill, None # Return a dummy placeholder for entropy steop

    def compute_entropy(self, x_start, means, vars, pred_noises,x_ts):
        
        print('Computing entropy ... ')

        from .calculate_entropy import compute_ddpm_elbo_batch
        ent = []
        sum_all = 0
        for i, t in enumerate(reversed(range(1, self.diffusion.num_timesteps))):

            t = torch.tensor([t,]).to(x_start.device).long()
            x_t = self.diffusion.q_sample(x_start, t)
            # posterior_mean, posterior_variance, _ = self.diffusion.q_posterior(x_start, x_t, t)
            # print(torch.mean(posterior_mean), torch.mean(posterior_variance), torch.mean(means[i]), torch.mean(vars[i]))
            # input_x_t = (x_t + x_ts[i])/2
            kl = compute_ddpm_elbo_batch(
                x0 = x_start, 
                x_t = x_ts[i], # Using the same x_t for both true and predicted posterior
                # x_t_pred = x_ts[i],
                t = t,
                betas = self.diffusion.betas,
                model_output = pred_noises[i],
                mode = "eps",   # "eps" or "mu"
                sigma_special_default = "tilde_beta",
            )
            # print(kl['L_t'], torch.mean(kl['mu_p']), torch.mean(kl['mu_q']), torch.mean((kl['mu_p']-kl['mu_q'])**2), kl['var_q'])
            # print(kl['L_t'], torch.mean((kl['mu_p']-kl['mu_q'])**2), kl['var_q'])
            sum_all += kl['L_t']
            ent.append(kl['L_t'])
        
        # print(sum_all)

        return ent
        

        # ents = []

        # for i in range(len(means)):
            
        #     pred_means = means[i] # torch.Size([1, 64, 160]) 
        #     pred_vars = vars[i]   # torch.Size([1, 1, 1])

        #     t = torch.tensor([self.diffusion.num_timesteps-i-1,]).to(x_start.device)
        #     x_t = self.diffusion.q_sample(x_start, t)
        #     posterior_mean, posterior_variance, _ = self.diffusion.q_posterior(x_start, x_t, t)

        #     # print(torch.mean((pred_means - posterior_mean)**2))
        #     # print(posterior_variance)
        #     entropy = gaussian_kl_diag(posterior_mean, posterior_variance, pred_means, pred_vars)
        #     print(entropy)
        #     ents.append(entropy)
        # # fake()
        
        
        # return ents

    

if __name__ == '__main__':

    # dAR = DiffAudioRep()

    # x = torch.rand(10, 128, 256)
    # l = dAR.diff_loss(x)

    # --- test encodec class ---

    # # codec = Encodec()
    # vae = VAE(model_config='config/vae_8.json', ckpt_path='ckpts/VAE_speech_8.ckpt')
    # import torchaudio
    # audio, sr = torchaudio.load('eval_wavs/1_x.wav')
    # output = vae(audio.unsqueeze(0))

    # torchaudio.save("test_vae.wav", audio, sr)

    # ----- Test Encodec Official ------
    # import torchaudio
    # codec = Encodec_official().to('cuda').eval()
    # data, sr = torchaudio.load('eval_wavs/1_x.wav')
    # data = data.unsqueeze(1).to('cuda')

    # with torch.no_grad():
    #     emb = codec.encode(data)
    #     print(emb.shape)
    #     y = codec.decode(emb)

    # torchaudio.save("test_enc_official.wav", y[0].cpu(), sr)

    # ------ Test condition ------
    t = 0
    COND_STEP = [0.5, 1, 1.5, 3, 4.5, 6, 9, 12]
    SAMPLING_STEP = [632, 534, 475, 378, 326, 291, 246, 218]

    assert len(COND_STEP) == len(SAMPLING_STEP)

    for i, step in enumerate(SAMPLING_STEP):
        if t >= step:
            break
    print(COND_STEP[i])
