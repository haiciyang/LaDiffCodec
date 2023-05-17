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

from .utils import EMA, logging, save_checkpoints, load_model
from .losses import melspec_loss_fn
from .model import DiffAudioRep
from .dataset import EnCodec_data
from .dataset_libri import Dataset_Libri
from .msstftd import MultiScaleSTFTDiscriminator as MSDisc

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

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

def run_gen_loss(disc, s, s_hat):

    B, C, L = s.shape 
    if C == 2:
        s = s.reshape(B*C, 1, L)
        s_hat = s_hat.reshape(B*C, 1, L)

    s_disc_r, fmap_r = disc(s) # list of the outputs from each discriminator
    s_disc_gen, fmap_gen = disc(s_hat)

    # s_disc_r: [3*[batch_size*[1, 309, 65]]]
    # fmap_r: [3*5*torch.Size([64, 32, 59, 513/257/128..])] 5 conv layers, different stride size
    K = len(fmap_gen)
    L = len(fmap_gen[0])

    l_g = 0
    l_feat = 0

    for d_id in range(len(fmap_r)):

        l_g += 1/K * torch.mean(torch.max(torch.tensor(0), 1-s_disc_gen[d_id])) # Gen loss

        for l_id in range(len(fmap_r[0])):
            l_feat += 1/(K*L) * torch.mean(abs(fmap_r[d_id][l_id] - \
                    fmap_gen[d_id][l_id]))/torch.mean(abs(fmap_r[d_id][l_id]))
    
    del s_disc_r, fmap_r, s_disc_gen, fmap_gen

    return l_g, l_feat

def run_disc_loss(disc, s, s_hat):

    l_d = 0
    
    B, C, L = s.shape 
    if C == 2:
        s = s.reshape(B*C, 1, L)
        s_hat = s_hat.reshape(B*C, 1, L)

    s_disc_r, fmap_r = disc(s) # list of the outputs from each discriminator
    s_disc_gen, fmap_gen = disc(s_hat.detach())

    K = len(fmap_gen)
    L = len(fmap_gen[0])

    for d_id in range(len(fmap_r)):
        l_d += 1/K * torch.mean(torch.max(torch.tensor(0), 1-s_disc_r[d_id]) + torch.max(torch.tensor(0), 1+s_disc_gen[d_id])) # Disc loss

    del s_disc_r, fmap_r, s_disc_gen, fmap_gen

    return l_d


def run_model(model=None, model_for_cond=None, ema=None, disc=None, data_loader=None, optimizer_G=None, optimizer_D=None, use_disc=None, disc_freq=None, debug=None):
    
    tot_nums = dict()
    for idx, batch in enumerate(data_loader):

        x = batch.unsqueeze(1).to(torch.float).to(device)

        cond = None
        if model_for_cond is not None:
            cond = model_for_cond.get_cond(x)
        nums, *rest = model(x, cond=cond) 
        x_hat = rest[0]


        if model.training:
            if use_disc:
                # if len(list(nums.values())) > 1:
                l_orig = list(nums.values())[0]
                l_g, l_feat = run_gen_loss(disc, x, x_hat)

                l_t = torch.mean(torch.abs(x - x_hat))
                l_f = melspec_loss_fn(x, x_hat, range(5,12))

                nums['l_g'] = l_g
                nums['l_feat'] = l_feat
                nums['l_t'] = l_t
                nums['l_f'] = l_f

                optimizer_G.zero_grad()
                g_loss = 0.1 * l_t + l_f + 3 * l_g + 3 * l_feat + 0.1 * l_orig
                g_loss.backward()
                optimizer_G.step()

                # Update Discriminator
                if idx % disc_freq == 0:
                    optimizer_D.zero_grad()
                    l_d = run_disc_loss(disc, x, x_hat)
                    nums['l_d'] = l_d
                    l_d.backward()
                    optimizer_D.step()                    

            else:
                # pass
                optimizer_G.zero_grad()
                loss = list(nums.values())[0]

                loss.backward()
                optimizer_G.step()

            if ema is not None:
                ema.update(copy_back=False)

        for key, value in nums.items():
            tot_nums[key] = tot_nums.get(key, 0) + value.detach().data.cpu()

        # if idx == 0:
        #     tot_nums = nums
        # else:
        #     for key, value in nums.items():
        #         tot_nums[key] = tot_nums[key] + value.detach().data.cpu()
        
        if debug:
            break

    for key, value in tot_nums.items():
        tot_nums[key] = tot_nums[key] / (idx + 1)

    return tot_nums, rest
    # return nums


def get_args():
    
    envvars = [
    "WORLD_SIZE",
    "RANK",
    "LOCAL_RANK",
    "NODE_RANK",
    "NODE_COUNT",
    "HOSTNAME",
    "MASTER_ADDR",
    "MASTER_PORT",
    "NCCL_SOCKET_IFNAME",
    "OMPI_COMM_WORLD_RANK",
    "OMPI_COMM_WORLD_SIZE",
    "OMPI_COMM_WORLD_LOCAL_RANK",
    "AZ_BATCHAI_MPI_MASTER_NODE",
    ]
    args = dict(gpus_per_node=torch.cuda.device_count())
    missing = []
    for var in envvars:
        if var in os.environ:
            args[var] = os.environ.get(var)
            try:
                args[var] = int(args[var])
            except ValueError:
                pass
        else:
            missing.append(var)
    # print(f"II Args: {args}")
    if missing:
        print(f"II Environment variables not set: {', '.join(missing)}.")
    return args


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Encodec_baseline")
    
    # Data related
    parser.add_argument("--output_dir", type=str, default='saved_models')
    # parser.add_argument("--data_folder_path", type=str, default='/data/hy17/dns_pth/*') # for the dsn data
    parser.add_argument("--data_folder_path", type=str, default='/data/hy17/librispeech/librispeech')
    parser.add_argument("--n_spks", type=int, default=500)
    parser.add_argument('--seq_len_p_sec', type=float, default=1.) 
    parser.add_argument('--sample_rate', type=int, default=16000)

    # Training
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4)    
    parser.add_argument('--batch_size', type=int, default=5)  
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--finetune_model', type=str, default='')
    parser.add_argument('--model_for_cond', type=str, default='')
    parser.add_argument('--write_on_every', type=int, default=50)  
    parser.add_argument('--model_type', type=str, default='transformer')  
    parser.add_argument('--freeze_ed', dest='freeze_ed', action='store_true')

    # Encoder and decoder
    parser.add_argument('--rep_dims', type=int, default=128)
    parser.add_argument('--emb_dims', type=int, default=128)
    parser.add_argument('--quantization', dest='quantization', action='store_true')
    parser.add_argument('--bandwidth', type=float, default=3.0)
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--lstm', type=int, default=2)
    parser.add_argument('--n_residual_layers', type=int, default=1)
    parser.add_argument('--enc_ratios', nargs='+', type=int)
    parser.add_argument('--final_activation', type=str, default=None)

    # Diff model
    parser.add_argument('--diff_dims', type=int, default=128)
    parser.add_argument('--qtz_condition', dest='qtz_condition', action='store_true')
    parser.add_argument('--self_condition', dest='self_condition', action='store_true')
    parser.add_argument('--seq_length', type=int, default=800)
    parser.add_argument('--run_diff', dest='run_diff', action='store_true')
    parser.add_argument('--run_vae', dest='run_vae', action='store_true')
    parser.add_argument('--scaling', dest='scaling', action='store_true')

    # Cond model
    parser.add_argument('--cond_enc_ratios', nargs='+', type=int)
    parser.add_argument('--cond_quantization', dest='cond_quantization', action='store_true')
    parser.add_argument('--cond_bandwidth', type=float, default=3.0)
    
    # Dist
    parser.add_argument('--use_disc', dest='use_disc', action='store_true')
    parser.add_argument('--disc_freq', type=int, default=1)

    
    inp_args = parser.parse_args() # Input arguments
    # args = get_args() # Enviornmente arguments

    assert not (inp_args.self_condition and inp_args.qtz_condition) # self_cond and quantization can't be both true.
   
    run_ddp = False #if len(args) == 1 else True
    # if not inp_args.debug:
    #     writer = SummaryWriter(f'../runs/{inp_args.exp_name}')

    if run_ddp:
        master_uri = "tcp://%s:%s" % (args.get("MASTER_ADDR"), args.get("MASTER_PORT"))
        os.environ["NCCL_DEBUG"] = "WARN"
        node_rank = args.get("NODE_RANK")

        os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'

        gpus_per_node = torch.cuda.device_count()
        world_size = args.get("WORLD_SIZE")
        gpu_rank = args.get("LOCAL_RANK")
        # if inp_args.debug:
        node_rank = 0 #tmp
        global_rank = node_rank * gpus_per_node + gpu_rank
        dist.init_process_group(
            backend="nccl", init_method=master_uri, world_size=world_size, rank=global_rank
        )

        # synchronizes all the threads to reach this point before moving on
        dist.barrier()

    # train_dataset = EnCodec_data(inp_args.data_path, task = 'train', seq_len_p_sec = inp_args.seq_len_p_sec, sample_rate=inp_args.sample_rate, multi=False, n_spks = inp_args.n_spks)
    # valid_dataset = EnCodec_data(inp_args.data_path, task = 'valid', seq_len_p_sec = inp_args.seq_len_p_sec, sample_rate=inp_args.sample_rate, multi=False, n_spks = inp_args.n_spks)

    train_dataset = Dataset_Libri(task = 'train', seq_len_p_sec = inp_args.seq_len_p_sec, data_folder_path=inp_args.data_folder_path)
    valid_dataset = Dataset_Libri(task = 'valid', seq_len_p_sec = inp_args.seq_len_p_sec, data_folder_path=inp_args.data_folder_path)


    if run_ddp:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True) 
        valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=True) 
        train_loader = DataLoader(train_dataset, batch_size=inp_args.batch_size, sampler=train_sampler, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=inp_args.batch_size, sampler=valid_sampler, pin_memory=True)
        
        torch.manual_seed(global_rank)  
        torch.cuda.set_device(gpu_rank)
    else:
        train_loader = DataLoader(train_dataset, batch_size=inp_args.batch_size, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=inp_args.batch_size, pin_memory=True)
        gpu_rank = 0
    

    # model = DiffAudioRep(rep_dims=inp_args.rep_dims, emb_dims=inp_args.emb_dims, diff_dims=inp_args.diff_dims,  n_residual_layers=inp_args.n_residual_layers, n_filters=inp_args.n_filters, lstm=inp_args.lstm, quantization=inp_args.quantization, bandwidth=inp_args.bandwidth, sample_rate=inp_args.sample_rate, self_condition=inp_args.self_cond, seq_length=inp_args.seq_length, ratios=inp_args.enc_ratios, run_diff=inp_args.run_diff, run_vae=inp_args.run_vae, model_type=inp_args.model_type, scaling=inp_args.scaling).to(device)

    model = DiffAudioRep(**vars(inp_args)).to(device)
    disc = MSDisc(filters=32).cuda(gpu_rank) if inp_args.use_disc else None

    if inp_args.finetune_model:
        load_model(model, inp_args.finetune_model + '/model_best.amlt', strict=False)
        if inp_args.use_disc:
            load_model(disc, inp_args.finetune_model + '/disc_best.amlt')

    model_for_cond = None
    if inp_args.model_for_cond:

        model_for_cond = DiffAudioRep(rep_dims=inp_args.rep_dims, emb_dims=inp_args.emb_dims, n_residual_layers=inp_args.n_residual_layers, n_filters=inp_args.n_filters, lstm=inp_args.lstm, quantization=inp_args.cond_quantization, bandwidth=inp_args.cond_bandwidth, ratios=inp_args.cond_enc_ratios, final_activation=inp_args.final_activation).to(device) # An autoencoder

        load_model(model_for_cond, inp_args.model_for_cond + '/model_best.amlt')
        model_for_cond.eval()


    if inp_args.run_diff and inp_args.model_type != 'unet2d':
        ema = EMA(
            model.diffusion,
            beta = 0.9999,              # exponential moving average factor
            update_after_step = 100,    # only after this number of .update() calls will it start updating
            update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
        )
    else:
        ema = None

        # model.load_state_dict(torch.load(f'{inp_args.output_dir}/{inp_args.finetune_model}.amlt'))

    if inp_args.run_diff and inp_args.freeze_ed:
        if inp_args.model_type == 'unet2d':
            optimizer_G = optim.Adam(model.diff_model.parameters(), lr=inp_args.lr)
        else:
            optimizer_G = optim.Adam(model.diffusion.parameters(), lr=inp_args.lr)
    else:
        optimizer_G = optim.Adam(model.parameters(), lr=inp_args.lr)
    optimizer_D = optim.Adam(disc.parameters(), lr=3e-4, betas=(0.5, 0.9)) if inp_args.use_disc else None

    if run_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_rank], find_unused_parameters=True)   
        if inp_args.use_disc:
            disc = nn.parallel.DistributedDataParallel(disc, device_ids=[gpu_rank])

    # run = Run.get_context()
    best_loss = torch.tensor(float('inf'))
    write_on_every = 1 if not inp_args.debug else 1
    eval_every_step = 500 

    # ---- Train 2000 steps
    for step in range(50000):
        if step == 0:
            print('Starts training ...')
        if run_ddp:
            train_loader.sampler.set_epoch(step)
        model.train()
        start_time = time.time()
        tr_losses, rest = run_model(model, model_for_cond, ema, disc, train_loader, optimizer_G, optimizer_D, inp_args.use_disc, 
        inp_args.disc_freq, inp_args.debug)
            
        # if step % eval_every_step == 0:
        #     x0 = reps[1]
        #     self_cond = reps[2]
        #     # sample = ema.ema_model.sample(batch_size=1)

        #     for i in range(1):
        #         save_img_on_tr(x0[i], name=f'rep_{step}_{i}th', note=inp_args.exp_name, out_path='outputs/')
        #         save_img_on_tr(self_cond[i], name=f'selfCond_{step}_t{t[i]}_{i}th', note=inp_args.exp_name, out_path='outputs/')

        # save_torch_wav(x[0], 'x', 'tr_sample')
        # save_torch_wav(x_hat[0], 'x_hat', 'tr_sample')
        # save_plot(x[0], 'x', 'tr_sample')
        # save_plot(x_hat[0], 'x_hat', 'tr_sample')
        # fake()

        if step % write_on_every == 0:
            with torch.no_grad():
                model.eval()
                val_losses, *_ = run_model(model=model, model_for_cond=model_for_cond, data_loader=valid_loader, debug=inp_args.debug)
            end_time = time.time()

            # if gpu_rank == 0: # only save model and logs for the main process

            vall = list(val_losses.values())[-1] # negsdr
            
            if inp_args.debug:
                # print(val_losses)
                print([val.item() for val in val_losses.values()])
            else:
                if vall < best_loss:
                    best_loss = vall
                    save_checkpoints(model, ema, disc, inp_args.output_dir, inp_args.exp_name, note='best')
                    if step % 1000 == 0 and step > 0:
                        save_checkpoints(model, ema, disc, inp_args.output_dir, inp_args.exp_name, note=str(step))


                # for key, value in tr_losses.items():
                #     # run.log(key, value.item())
                #     writer.add_scalar('Train/'+key, value.item(), step)
                    
                # for key, value in val_losses.items():
                #     writer.add_scalar('Valid/'+key, value.item(), step)
                
                logging(step, tr_losses, val_losses, end_time-start_time, inp_args.exp_name, best_loss)
                       

