import sys
sys.path.insert(0, '/N/u/hy17/BigRed200/venvs/env_pt12/lib/python3.8/site-packages')

import os
import re
import glob
import time
import argparse
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile
from collections import OrderedDict
from matplotlib import pyplot as plt
# from asteroid.losses.pit_wrapper import PITLossWrapper

import torch
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
from .model import DiffAudioRep
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

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Encodec_baseline")
    
    # Data related
    parser.add_argument("--output_dir", type=str, default='../saved_models')
    # parser.add_argument("--data_path", type=str, default='/data/hy17/dns_pth/*')
    parser.add_argument("--data_folder_path", type=str, default='/data/hy17/librispeech/librispeech')
    parser.add_argument("--n_spks", type=int, default=500)
    parser.add_argument('--seq_len_p_sec', type=float, default=1.8000) 
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
    parser.add_argument('--enc_ratios', nargs='+', type=int)
    parser.add_argument('--final_activation', type=str, default=None)


    parser.add_argument('--run_diff', dest='run_diff', action='store_true')
    parser.add_argument('--run_vae', dest='run_vae', action='store_true')

    # Diff model
    parser.add_argument('--diff_dims', type=int, default=128)
    parser.add_argument('--qtz_condition', dest='qtz_condition', action='store_true')
    parser.add_argument('--self_condition', dest='self_condition', action='store_true')
    parser.add_argument('--seq_length', type=int, default=16000)
    parser.add_argument('--model_type', type=str, default='transformer')  

    parser.add_argument('--scaling_frame', dest='scaling_frame', action='store_true')
    parser.add_argument('--scaling_feature', dest='scaling_feature', action='store_true')
    parser.add_argument('--scaling_global', dest='scaling_feature', action='store_true')
    parser.add_argument('--scaling_dim', dest='scaling_feature', action='store_true')

    parser.add_argument('--sampling_timesteps', type=int, default=1000)
    parser.add_argument('--use_film', dest='use_film', action='store_true')

    # Cond model
    parser.add_argument('--model_for_cond', type=str, default='')
    parser.add_argument('--cond_enc_ratios', nargs='+', type=int)
    parser.add_argument('--cond_quantization', dest='cond_quantization', action='store_true')
    parser.add_argument('--cond_bandwidth', type=float, default=3.0)
    

    inp_args = parser.parse_args() # Input arguments

    # valid_dataset = EnCodec_data(inp_args.data_path, task = 'valid', seq_len_p_sec = inp_args.seq_len_p_sec, sample_rate=inp_args.sample_rate, multi=False, n_spks = inp_args.n_spks)
    valid_dataset = Dataset_Libri(task = 'eval', seq_len_p_sec = inp_args.seq_len_p_sec, data_folder_path=inp_args.data_folder_path)

    valid_loader = DataLoader(valid_dataset, batch_size=1, pin_memory=True)

    # model = DiffAudioRep(rep_dims=inp_args.rep_dims, diff_dims=inp_args.diff_dims, n_residual_layers=inp_args.n_residual_layers, n_filters=inp_args.n_filters, lstm=inp_args.lstm, quantization=inp_args.quantization, bandwidth=inp_args.bandwidth, sample_rate=inp_args.sample_rate, self_condition=inp_args.self_cond, seq_length=inp_args.seq_length, ratios=inp_args.enc_ratios, run_diff=inp_args.run_diff, run_vae=inp_args.run_vae, model_type=inp_args.model_type).to(device)
    other_cond = True if inp_args.model_for_cond else False
    model = DiffAudioRep(other_cond=other_cond, **vars(inp_args)).to(device)


    # if inp_args.run_diff:
    #     ema = EMA(
    #         model.diffusion,
    #         beta = 0.9999,              # exponential moving average factor
    #         update_after_step = 100,    # only after this number of .update() calls will it start updating
    #         update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
    #     )
    #     ema = load_model(ema, inp_args.model_path[:-15]+'ema_best.amlt', strict=True)
    #     ema.ema_model.eval()
    
    load_model(model, inp_args.model_path, strict=True)
    model.eval()

    model_for_cond = None
    if inp_args.model_for_cond:

        model_for_cond = DiffAudioRep(rep_dims=inp_args.rep_dims, emb_dims=inp_args.emb_dims, n_residual_layers=inp_args.n_residual_layers, n_filters=inp_args.n_filters, lstm=inp_args.lstm, quantization=inp_args.cond_quantization, bandwidth=inp_args.cond_bandwidth, ratios=inp_args.cond_enc_ratios, final_activation=inp_args.final_activation).to(device) # An autoencoder

        load_model(model_for_cond, inp_args.model_for_cond + '/model_best.amlt')
        model_for_cond.eval()

    note = inp_args.model_path.split('/')[-2]

    # Conditioned
    with torch.no_grad():
        for idx, batch in enumerate(valid_loader):

            if idx < 300:
                continue

            # ---- Speaker embeddings and estimated separation learnt from mixture sources ----
            x = batch.unsqueeze(1).to(torch.float).to(device)
            
            t = torch.tensor([100]).to(device)

            cond = None
            if model_for_cond is not None:
                cond = model_for_cond.get_cond(x)

            nums, *reps = model(x, t=t, cond=cond)
            # nums, *reps = model_qtz(x, t)

            # x_rep = model.encoder(x)
            # quantizedResults = model_qtz.quantizer(x_rep, sample_rate=model_qtz.frame_rate, bandwidth=model_qtz.bandwidth)
            # x_rep_qtz = quantizedResults.quantized
            # # x_rep_qtz = apply_mask(x_rep_qtz)
            # x_hat = model_qtz.decoder(x_rep_qtz)

            # # ======

            x_hat, x0, predicted_x0, xt, t, qtz_x0, scale = reps

            img = torch.randn((120, 200))

            # print(scale.squeeze())

            # qtz_x0 = apply_mask(qtz_x0)

            # x_hat = reps[0]

            # print(torch.mean(xt.squeeze(),0).mean())
            # print(torch.std(xt.squeeze(),0).mean())

            # ----- Sampling -----
            # sample = ema.ema_model.sample(batch_size=1, condition=qtz_x0)
            # x_sample = model.decoder(sample)

            sampled_rep = model.diffusion.sample(batch_size=1, condition=cond, clip_denoised=True)
            x_sample = model.decoder(sampled_rep)
            x_scale_sample = model.decoder(sampled_rep*scale)


            # ----- Infilling ----

            # sample = model.diffusion.infilling(condition=sampled_rep, offset=300, lam=0.5)
            # x_sample = model.decoder(sample)
            # diff_fl = sample - x_rep_qtz

            # sample = model.diffusion.halfway_sampling(img=x_rep_qtz, t=400)
            # print(sum(sample-x_rep_qtz))
            # x_sample_half = model.decoder(sample)
            # diff_half = sample - x_rep_qtz


            out_dir = 'outputs/'
            # px = str(t.item())
            px = 'c_true'

            # save_img(x0, name='rep', note=note, out_path = out_dir)
            # save_img(predicted_x0, name=f'pred_t{t[0]}_{px}', note=note, out_path = out_dir)
            # save_img(xt, name=f'xt_t{t[0]}_{px}', note=note, out_path = out_dir)
            # save_img(img, name=f'rand_{px}', note=note, out_path = out_dir)
            # fake()
            # save_img(predicted_x0 * scale, name=f'pred_scaled_t{t[0]}', note=note, out_path = out_dir)

            # save_plot(scale.squeeze(), f'scale', note=note, out_path = out_dir)
            
            save_img(sampled_rep, name=f'sample_{px}', note=note, out_path = out_dir)
            save_img(sampled_rep*scale, name=f'sample_scale_f{px}', note=note, out_path = out_dir)

            # save_plot(x, f'x', note=note, out_path = out_dir)
            # save_plot(x_hat, f'x_hat_t{t[0]}', note=note, out_path = out_dir)
            # save_plot(x_sample, f'x_sample_{px}', note=note, out_path = out_dir)
            # save_plot(x_scale_sample, f'x_scale_sample_{px}', note=note, out_path = out_dir)
            
            # save_torch_wav(x, f'x', note=note, out_path = out_dir)
            # save_torch_wav(x_hat, f'x_hat_{px}', note=note, out_path = out_dir)

            save_torch_wav(x_sample, f'x_sample_{px}', note=note, out_path = out_dir)
            save_torch_wav(x_scale_sample, f'x_scale_sample{px}', note=note, out_path = out_dir)
            
            print(sdr_loss(x, x_sample))
            # print(sdr_loss(x, x_scale_sample))

            break

            # if idx == 2:
            #     break



    # # Unconditioned
    

    # sampled_rep = model.diffusion.sample(batch_size=1)    
    
    # samples = model.decoder(rep)


    # save_torch_wav(samples, name='outwav', note=note)
    # save_plot(samples, name='outwav_plot', note=note)

    # save_img(rep, name='rep_img', note=note)




