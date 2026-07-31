# import sys
# sys.path.insert(0, '/N/u/hy17/BigRed200/venvs/env_pt12/lib/python3.8/site-packages')

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

from srcs.utils import EMA, logging, save_checkpoints, load_from_checkpoint, log_params, nn_parameters
from srcs.losses import melspec_loss_fn
from srcs.model import DiffAudioRep, Encodec_official
# from .dataset import EnCodec_data
from srcs.dataset_libri import Dataset_Libri
from srcs.dataset_max import Dataset_Max
from srcs.msstftd import MultiScaleSTFTDiscriminator as MSDisc
from srcs.dacdisc import Discriminator as DACDisc

print('All package loaded.')


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def get_model(inp_args):
    
    model = Encodec_official().to(device)

    return model


def synthesis(inp_args):
    
    model = get_model(inp_args)
    model.eval()

    n_total, n_trainable = nn_parameters(model)
    # print(n_total, n_trainable)
    print(f'Loaded model has {n_total / 1_000_000 :<.2f}M parameters; {n_trainable / 1_000_000 :<.2f} M trainable parameters')
    
    wav_list = glob.glob(os.path.join(inp_args.input_dir, '**/*.wav'), recursive=True) \
    + glob.glob(os.path.join(inp_args.input_dir, '**/*.flac'), recursive=True)

    wav_list = wav_list[:200]
    
    with torch.no_grad():
        for wav_file in tqdm(wav_list):
            
            if 'flac' in wav_file:
                filename = wav_file[len(inp_args.input_dir):][:-5]
            elif 'wav' in wav_file:
                filename = wav_file[len(inp_args.input_dir):][:-4]
            save_path = inp_args.output_dir + filename

            folder = save_path[: -(len(save_path.split('/')[-1])+1)]
            if not os.path.exists(folder):
                os.makedirs(folder)
            
            wav, sr = torchaudio.load(wav_file)

            if sr != 16000:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
            wav = wav.unsqueeze(1).to(torch.float).to(device)
            
            if inp_args.seq_len_p_sec != 0:
                length = int(inp_args.seq_len_p_sec * sr)
                wav = wav[:, :, :length]
                
            length = wav.shape[-1]//5120*5120
            wav = wav[:, :, :length]

            x = model.decode(model.encode(wav, bandwidth=inp_args.bandwidth))
            torchaudio.save(f'{save_path}.wav', x.squeeze(1).cpu(), 16000)
            # fake()


def load_dac(model_type, tag):

    load_path = ''

    if model_type == '44khz':
        model = load_from_checkpoint(tag='latest', model_type=model_type)
    else:
        model = model(load_path)
    model.eval().to('cuda')

    return model


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Encodec_baseline")
    
    # Synthesis
    parser.add_argument('--input_dir', type=str, default='eval_wavs/')
    parser.add_argument('--output_dir', type=str, default='output_wavs/')
    parser.add_argument('--bandwidth', type=float, default=1.5)
    parser.add_argument('--seq_len_p_sec', type=float, default=1.5)
    inp_args = parser.parse_args() # Input arguments

    synthesis(inp_args)