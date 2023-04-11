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

from .model import DiffAudioRep
from .dataset import EnCodec_data


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def save_img(rep, name, note):
    *_, h, w = rep.shape
    rep = rep.reshape(h, w).cpu().data.numpy()
    plt.imshow(rep, aspect='auto', origin='lower')
    plt.savefig(f"{name}_{note}.png")
    plt.clf()

def save_plot(x, name, note):
    x = x.squeeze().cpu().data.numpy()
    plt.plot(x/np.max(np.abs(x)))
    plt.savefig(f"{name}_{note}.png")
    plt.clf()

def save_torch_wav(x, name, note):

    x = x.squeeze().cpu().data.numpy()
    # plt.plot(x/np.max(np.abs(x)))
    # plt.savefig(f"{name}_{note}.png")
    # plt.clf()
    wavfile.write(f"eval_wavs/{note}_{name}.wav", 16000, x/np.max(np.abs(x)))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Encodec_baseline")
    
    # Data related
    parser.add_argument("--output_dir", type=str, default='../saved_models')
    parser.add_argument("--data_path", type=str, default='/data/hy17/dns_pth/*')
    parser.add_argument("--n_spks", type=int, default=500)
    parser.add_argument('--seq_len_p_sec', type=float, default=1.8000) 
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--model_path', type=str, default='')
    parser.add_argument('--note', type=str, default='')

    # Encoder and decoder
    parser.add_argument('--rep_dims', type=int, default=128)
    parser.add_argument('--quantization', dest='quantization', action='store_true')
    parser.add_argument('--bandwidth', type=float, default=3.0)
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--lstm', type=int, default=2)
    parser.add_argument('--n_residual_layers', type=int, default=1)
    parser.add_argument('--enc_ratios', nargs='+', type=int)

    parser.add_argument('--run_diff', dest='run_diff', action='store_true')
    parser.add_argument('--run_vae', dest='run_vae', action='store_true')

    # Diff model
    parser.add_argument('--diff_dims', type=int, default=128)
    parser.add_argument('--self_cond', dest='self_cond', action='store_true')
    parser.add_argument('--seq_length', type=int, default=16000)
    parser.add_argument('--model_type', type=str, default='transformer')  

    
    inp_args = parser.parse_args() # Input arguments

    # valid_dataset = EnCodec_data(inp_args.data_path, task = 'valid', seq_len_p_sec = inp_args.seq_len_p_sec, sample_rate=inp_args.sample_rate, multi=False, n_spks = inp_args.n_spks)
    # valid_loader = DataLoader(valid_dataset, batch_size=1, pin_memory=True)

    model = DiffAudioRep(rep_dims=inp_args.rep_dims, diff_dims=inp_args.diff_dims, n_residual_layers=inp_args.n_residual_layers, n_filters=inp_args.n_filters, lstm=inp_args.lstm, quantization=inp_args.quantization, bandwidth=inp_args.bandwidth, sample_rate=inp_args.sample_rate, self_condition=inp_args.self_cond, seq_length=inp_args.seq_length, ratios=inp_args.enc_ratios, run_diff=inp_args.run_diff, run_vae=inp_args.run_vae, model_type=inp_args.model_type).to(device)


    state_dict = torch.load(inp_args.model_path)
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    model.load_state_dict(model_dict, strict=False)

    note = inp_args.model_path.split('/')[-1][:-5]

    # # Conditioned
    # with torch.no_grad():
    #     for batch in valid_loader:

    #         # ---- Speaker embeddings and estimated separation learnt from mixture sources ----
    #         x = batch.unsqueeze(1).to(torch.float).to(device)
            
    #         nums, x_hat = model(x) # nums: [total_loss, diff_loss, rec_loss]

    #         # print(nums)

    #         # print(x_hat.shape)
    #         # fake()

    #         # save_torch_wav(inputs, 'mix', note)
    #         save_torch_wav(x, 'x', note)
    #         save_torch_wav(x_hat, 'x_hat', note)

    #         break


    # Unconditioned
    self.ema.ema_model.eval()
    rep = model.diffusion.sample(batch_size=1)    
    
    samples = model.decoder(rep)


    save_torch_wav(samples[0], name='outwav', note=note)
    save_plot(samples[0], name='outwav_plot', note=note)

    save_img(rep[0], name='rep_img', note=note)




