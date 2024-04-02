import sys
sys.path.insert(0, '/N/u/hy17/BigRed200/venvs/env_pt12/lib/python3.8/site-packages')

import os
import re
import glob
import time
import argparse
import numpy as np
from tqdm import tqdm
from pesq import pesq
from scipy.io import wavfile
from collections import OrderedDict
from matplotlib import pyplot as plt
# from asteroid.losses.pit_wrapper import PITLossWrapper

import torch
import torchaudio
import torch.nn as nn
from torch import optim
import torch.distributed as dist
from torchaudio import transforms
import torch.multiprocessing as mp
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler

# from ema_pytorch import EMA
from .utils import EMA, save_img, save_plot, save_torch_wav, load_model, nn_parameters
from .model import DiffAudioRep, DiffAudioTime
from .dataset import EnCodec_data
from .dataset_libri import Dataset_Libri
from .losses import sdr_loss


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def apply_mask(rep, ratio=0.5):

    B, C, L = rep.shape
    mask = torch.tensor([1, 0, 0]).repeat(C, (L+2)//3)
    mask = mask[None,:,:L].to(rep.device)
    return rep * mask

def scaling(x_rep, global_max=1):

    B, C, L = x_rep.shape
    
    # scale = None
    # if self.scaling_frame:
    #     # ---- Scaling for every frames -----
    #     scale, _ = torch.max(torch.abs(x_rep), 1, keepdim=True)
    #     x_rep = x_rep / (scale + 1e-20)
    # elif self.scaling_feature:
    #     # print('Scaling on feature map after upsampling.')
    #     # --- Scaling for the feature map --- 
    #     scale, _ = torch.max(torch.abs(x_rep.reshape(B, C * L)), 1, keepdim=True)
    #     scale = scale.unsqueeze(-1)
    #     x_rep = x_rep / (scale + 1e-20)
    # elif self.scaling_global:
    #     # print('Scaling globally after upsampling.')
    #     scale = global_max
    #     x_rep = x_rep / scale
    # elif self.scaling_dim:
    #     scale, _ = torch.max(torch.abs(x_rep), -1, keepdim=True)
    #     x_rep = x_rep / scale
    
    # ---- Condition features only do feature-level scalig ---- 
    scale, _ = torch.max(torch.abs(x_rep.reshape(B, C * L)), 1, keepdim=True)
    scale = scale.unsqueeze(-1)
    x_rep = x_rep / (scale + 1e-20)

    return x_rep, scale

def synthesis(inp_args):
    
    other_cond = True if inp_args.model_for_cond else False
    if inp_args.train_time_diff:
        model = DiffAudioTime(other_cond=other_cond, **vars(inp_args)).to(device)
    else:
        model = DiffAudioRep(other_cond=other_cond, **vars(inp_args)).to(device)

    load_model(model, inp_args.model_path, strict=True)
    model.eval()

    model_for_cond = None
    if inp_args.model_for_cond:
        model_for_cond = DiffAudioRep(rep_dims=inp_args.rep_dims, emb_dims=inp_args.emb_dims, n_residual_layers=inp_args.n_residual_layers, n_filters=inp_args.n_filters, lstm=inp_args.lstm, quantization=inp_args.cond_quantization, bandwidth=inp_args.cond_bandwidth, ratios=inp_args.cond_enc_ratios, final_activation=inp_args.final_activation).to(device) # An autoencoder
        load_model(model_for_cond, inp_args.model_for_cond)
        model_for_cond.eval()

    out_dir = inp_args.output_dir
    
    midway_t = 100
    lam = 0.1
    with torch.no_grad():
        for wav_file in glob.glob(os.path.join(inp_args.input_dir, '*.wav')):
        
            filename = wav_file.split('/')[-1][:-4]
            
            wav, sr = torchaudio.load(wav_file)
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
            wav = wav.unsqueeze(1).to(torch.float).to(device)
            
            model.diffusion.seq_length = int(wav.shape[-1] / inp_args.enc_ratios[0])
            
            cond = None
            if model_for_cond is not None:
                cond = model_for_cond.get_cond(wav)
            
            scale = model.get_scale(wav)
            
            # ------ rep diff ----- 
            sampled_rep = model.diffusion.sample(batch_size=1, condition=cond)
            x_scale_sample = model.decoder(sampled_rep*scale)
            torchaudio.save(os.path.join(out_dir, f'{filename}_{inp_args.cond_bandwidth}kb.wav'), x_scale_sample.squeeze(1).cpu(), 16000)

            # ----- Infilling ----
            infill_img = cond
            if inp_args.upsampling_ratios is not None:
                for layer in model.diff_model.upsampling_layers:
                    infill_img = layer(infill_img)
            infill_img = infill_img / torch.max(torch.abs(infill_img.flatten())) + 1e-8
            sample = model.diffusion.infilling(infill_img = infill_img, condition=cond, midway_t=midway_t, lam=lam)
            x_sample_infil = model.decoder(sample)

            torchaudio.save(os.path.join(out_dir, f'{filename}_{inp_args.cond_bandwidth}kb_infill.wav'), x_sample_infil.squeeze(1).cpu(), 16000)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Encodec_baseline")
    
    # Data related
    # parser.add_argument("--data_path", type=str, default='/data/hy17/dns_pth/*')
    parser.add_argument("--data_folder_path", type=str, default='/data/hy17/librispeech/librispeech')
    parser.add_argument("--n_spks", type=int, default=500)
    parser.add_argument('--seq_len_in_sec', type=float, default=1.8000) 
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--qtzer_path', type=str, default='')
    parser.add_argument('--note', type=str, default='')

    # Encoder and decoder
    parser.add_argument('--rep_dims', type=int, default=128)
    parser.add_argument('--emb_dims', type=int, default=128) # Only when using transformer for the diffusion model
    parser.add_argument('--quantization', dest='quantization', action='store_true')
    parser.add_argument('--bandwidth', type=float, default=3.0)
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--lstm', type=int, default=2)
    parser.add_argument('--n_residual_layers', type=int, default=1)
    parser.add_argument('--enc_ratios', nargs='+', type=int, default=[8])
    parser.add_argument('--final_activation', type=str, default=None)


    parser.add_argument('--run_diff', dest='run_diff', action='store_true')
    parser.add_argument('--run_vae', dest='run_vae', action='store_true')

    # Diff model
    parser.add_argument('--train_time_diff', dest='train_time_diff', action='store_true')

    parser.add_argument('--diff_dims', type=int, default=256)
    parser.add_argument('--qtz_condition', dest='qtz_condition', action='store_true')
    parser.add_argument('--self_condition', dest='self_condition', action='store_true')
    parser.add_argument('--seq_length', type=int, default=16000)
    parser.add_argument('--model_type', type=str, default='unet')  

    parser.add_argument('--scaling_frame', dest='scaling_frame', action='store_true')
    parser.add_argument('--scaling_feature', dest='scaling_feature', action='store_true')
    parser.add_argument('--scaling_global', dest='scaling_global', action='store_true')
    parser.add_argument('--scaling_dim', dest='scaling_dim', action='store_true')

    parser.add_argument('--sampling_timesteps', type=int, default=1000)
    parser.add_argument('--use_film', dest='use_film', action='store_true')

    # Cond model
    parser.add_argument('--model_for_cond', type=str, default='')
    parser.add_argument('--upsampling_ratios', nargs='+', type=int, default=[5,4,2])
    parser.add_argument('--cond_enc_ratios', nargs='+', type=int, default=[8,5,4,2])
    parser.add_argument('--cond_quantization', dest='cond_quantization', action='store_true')
    parser.add_argument('--cond_bandwidth', type=float, default=3.0)
    parser.add_argument('--cond_global', type=float, default=3.0)

    parser.add_argument('--unet_scale_cond', dest='unet_scale_cond', action='store_true')
    parser.add_argument('--unet_scale_x', dest='unet_scale_x', action='store_true')
    
    # Input and output
    parser.add_argument('--input_dir', type=str, default='')
    parser.add_argument('--output_dir', type=str, default='outputs/')
    

    inp_args = parser.parse_args() # Input arguments
    
    synthesis(inp_args)