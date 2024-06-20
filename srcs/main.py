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

from .utils import EMA, logging, save_checkpoints, load_from_checkpoint, log_params, nn_parameters
from .losses import melspec_loss_fn
from .model import DiffAudioRep
# from .dataset import EnCodec_data
from .dataset_libri import Dataset_Libri
from .dataset_max import Dataset_Max
from .msstftd import MultiScaleSTFTDiscriminator as MSDisc

print('All package loaded.')


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

    
def get_model(inp_args):
    
    other_cond = True if inp_args.discrete_AE else False
    if inp_args.train_time_diff:
        model = DiffAudioTime(other_cond=other_cond, **vars(inp_args)).to(device)
    else:
        model = DiffAudioRep(other_cond=other_cond, **vars(inp_args)).to(device)
    
    if inp_args.discrete_AE:
        load_from_checkpoint(model.discrete_AE, f'saved_models/{inp_args.discrete_AE}/model_best.amlt', strict=False)
    if inp_args.continuous_AE:
        load_from_checkpoint(model.continuous_AE, f'saved_models/{inp_args.continuous_AE}/model_best.amlt')
    
    if inp_args.load_model:
        model_path = f'saved_models/{inp_args.load_model}/model_best.amlt'
        load_from_checkpoint(model, model_path)
    
    return model.to(device)

def get_disc(inp_args):
    
    if inp_args.use_disc:
        disc = MSDisc(filters=32).cuda(gpu_rank)
        if inp_args.load_model:
            load_from_checkpoint(disc, inp_args.load_model + '/disc_best.amlt')
        return disc.to(device)
    else:
        return None
    
def synthesis(inp_args):
    
    model = get_model(inp_args)
    model.eval()
    
    n_total, n_trainable = nn_parameters(model)
    print(n_total, n_trainable)
    print(f'Loaded model has {n_total / 1000000 :<.2f}M parameters; {n_trainable / 1000000 :<.2f} M trainable parameters')

    out_dir = inp_args.output_dir
    
    # midway_t = 100
    # lam = 0.1
    midway_t = inp_args.midway_t
    lam = inp_args.lam
    
    with torch.no_grad():
        for wav_file in tqdm(glob.glob(os.path.join(inp_args.input_dir, '**/*.wav'), recursive=True)):
        
            # filename = wav_file.split('/')[-1][:-4]
            filename = wav_file[len(inp_args.input_dir):][:-4]
            save_path = inp_args.output_dir + filename

            
            # out_dir_full = inp_args.output_dir + '_full'
            # out_dir_infill = inp_args.output_dir + '_infill'
            # if not os.path.exists(out_dir_infill):
            #     os.mkdir(out_dir_infill)
            # if not os.path.exists(out_dir_full):
            #     os.mkdir(out_dir_full)
            
            # save_path_full = out_dir_full + filename
            # full_folder = save_path_full[: -(len(save_path_full.split('/')[-1])+1)]
            # if not os.path.exists(full_folder):
            #     os.mkdir(full_folder)
                
            # save_path_fill = out_dir_infill + filename
            infill_folder = save_path[: -(len(save_path.split('/')[-1])+1)]
            if not os.path.exists(infill_folder):
                os.makedirs(infill_folder)
            
            # print(wav_file)
            # try:
            wav, sr = torchaudio.load(wav_file)
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=16000)
            wav = wav.unsqueeze(1).to(torch.float).to(device)
            
            if inp_args.seq_len_p_sec != 0:
                length = int(inp_args.seq_len_p_sec * sr)
                wav = wav[:, :, :length]
                
            
            length = wav.shape[-1]//640*640
            wav = wav[:, :, :length]
            seq_length = int(wav.shape[-1] / np.prod(inp_args.ratios))
            
            # x_scale_sample, x_sample_infill = model.sample(wav, seq_length, midway_t, lam)
            x_sample_infill = model.sample(wav, seq_length, midway_t, lam)

            # torchaudio.save(os.path.join(out_dir, f'{filename}_{inp_args.cond_bandwidth}_{inp_args.load_model}_full.wav'), x_scale_sample.squeeze(1).cpu(), 16000)
            # torchaudio.save(os.path.join(out_dir, f'{filename}_{inp_args.cond_bandwidth}_{inp_args.load_model}_infill.wav'), x_sample_infill.squeeze(1).cpu(), 16000)
            
            # torchaudio.save(f'{save_path_full}.wav', x_scale_sample.squeeze(1).cpu(), 16000)
            torchaudio.save(f'{save_path}.wav', x_sample_infill.squeeze(1).cpu(), 16000)
            # except:
            #     pass


def train(inp_args, global_rank):
    
    if not inp_args.debug:
        log_params(vars(inp_args), inp_args.exp_name)
    
    if not inp_args.debug and global_rank == 0:
        if not os.path.exists(f'runs/diff'):
            os.mkdir(f'runs/diff')
        if not os.path.exists(f'runs/diff/{inp_args.exp_name}'):
            os.mkdir(f'runs/diff/{inp_args.exp_name}')
        writer = SummaryWriter(f'runs/diff/{inp_args.exp_name}') 
    else:
        writer = None
    
    if inp_args.dataset == 'libri':
        train_dataset = get_libri_dataset(task='train', seq_len_p_sec=inp_args.seq_len_p_sec, data_proc = inp_args.data_process)
        valid_dataset = get_libri_dataset(task='valid', seq_len_p_sec=inp_args.seq_len_p_sec, data_proc = inp_args.data_process)
    elif inp_args.dataset == 'max':
        train_dataset = get_max_dataset(task='train', seq_len_p_sec=inp_args.seq_len_p_sec, data_proc = inp_args.data_process)
        valid_dataset = get_max_dataset(task='valid', seq_len_p_sec=inp_args.seq_len_p_sec, data_proc = inp_args.data_process)
    else:
        raise ValueError('Invalid dataset type.')
        
    if run_ddp:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True) 
        valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=True) 
        train_loader = DataLoader(train_dataset, batch_size=inp_args.batch_size, sampler=train_sampler, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=inp_args.batch_size, sampler=valid_sampler, pin_memory=True)
        
        torch.manual_seed(global_rank)  
        torch.cuda.set_device(gpu_rank)
    else:
        train_loader = DataLoader(train_dataset, batch_size=inp_args.batch_size, pin_memory=True, num_workers=4)
        valid_loader = DataLoader(valid_dataset, batch_size=inp_args.batch_size, pin_memory=True, num_workers=4)
        gpu_rank = 0


    model = get_model(inp_args)
    disc = get_disc(inp_args)
    n_total, n_trainable = nn_parameters(model)
    print(n_total, n_trainable)
    print(f'Loaded model has {n_total / 1000000 :<.2f}M parameters; {n_trainable / 1000000 :<.2f} M trainable parameters')

    ema = None
    optimizer_G = optim.Adam(model.parameters(), lr=inp_args.lr)
    optimizer_D = optim.Adam(disc.parameters(), lr=3e-4, betas=(0.5, 0.9)) if inp_args.use_disc else None

    if run_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_rank], find_unused_parameters=True)   
        if inp_args.use_disc:
            disc = nn.parallel.DistributedDataParallel(disc, device_ids=[gpu_rank])

    # run = Run.get_context()
    best_loss = torch.tensor(float('inf'))
    save_on_every = 100000 if not inp_args.debug else 1

    step = inp_args.load_step if inp_args.load_step is not None else 0 
    while step < 1000000:
        if step == 0:
            print('Starts training ...')
        if run_ddp:
            train_loader.sampler.set_epoch(step)
            
        model.train()
        start_time = time.time()
        
        tr_losses, step = train_loop(
            model=model, 
            ema=ema, 
            disc=disc, 
            data_loader = train_loader, 
            optimizer_G = optimizer_G, 
            optimizer_D = optimizer_D, 
            use_disc = inp_args.use_disc, 
            disc_freq = inp_args.disc_freq, 
            writer = writer, 
            step = step, 
            debug = inp_args.debug)        

        val_losses = valid_loop(model=model, data_loader=valid_loader, debug=inp_args.debug)
        
        if inp_args.debug:
            print([val.item() for val in val_losses.values()])
        else:
            if global_rank == 0 and writer is not None:
                # --- Write validation scores on TB ---
                for key, value in val_losses.items():
                    writer.add_scalar('Valid/'+key, value.item(), step)

            vall = val_losses['neg_sdr'] # negsdr
            if vall < best_loss:
                best_loss = vall
                save_checkpoints(model, inp_args.save_dir, inp_args.exp_name, ema, disc, note='best')
            if step % save_on_every == 0 and step > 0:
                save_checkpoints(model, inp_args.save_dir, inp_args.exp_name, ema, disc, note=str(step))
            end_time = time.time()
            logging(step, tr_losses, val_losses, end_time-start_time, inp_args.exp_name, best_loss)
                
def train_loop(model=None, ema=None, disc=None, data_loader=None, optimizer_G=None, optimizer_D=None, use_disc=None, disc_freq=None, writer=None, step = None, debug=None):
    
    tot_nums = dict()
    
    for idx, batch in enumerate(data_loader):

        step += 1

        x = batch.unsqueeze(1).to(torch.float).to(device)
        nums, *rest = model(x) 
        x_hat = rest[0]

        if use_disc:
            # if len(list(nums.values())) > 1:
            l_w = nums['qtz_loss']
            l_g, l_feat = run_gen_loss(disc, x, x_hat)

            l_t = torch.mean(torch.abs(x - x_hat))
            l_f = melspec_loss_fn(x, x_hat, range(5,12))

            nums['l_g'] = l_g
            nums['l_feat'] = l_feat
            nums['l_t'] = l_t
            nums['l_f'] = l_f

            optimizer_G.zero_grad()
            g_loss = 0.1 * l_t + l_f + 3 * l_g + 3 * l_feat + 0.1 * l_w
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
        
        # Write to Tensorboard
        if writer is not None:
            for key, value in nums.items():
                writer.add_scalar('Train/'+key, value.item(), step)

        # Add to the a global dict after every batch
        for key, value in nums.items():
            tot_nums[key] = tot_nums.get(key, 0) + value.detach().data.cpu()

        if debug:
            break

    for key, value in tot_nums.items():
        tot_nums[key] = tot_nums[key] / (idx + 1)

    return tot_nums, step
    
@torch.no_grad()
def valid_loop(model, data_loader, debug):
    
    model.eval()
    tot_nums = dict()
    for idx, batch in enumerate(data_loader):
        x = batch.unsqueeze(1).to(torch.float).to(device)
        nums, *_ = model(x) 
    
        # Add to the a global dict after every batch
        for key, value in nums.items():
            tot_nums[key] = tot_nums.get(key, 0) + value.detach().data.cpu()

        if debug:
            break

    for key, value in tot_nums.items():
        tot_nums[key] = tot_nums[key] / (idx + 1)

    return tot_nums

def load_dac(model_type, tag):

    load_path = ''

    if model_type == '44khz':
        model = load_from_checkpoint(tag='latest', model_type=model_type)
    else:
        model = model(load_path)
    model.eval().to('cuda')

    return model

def get_max_dataset(task, seq_len_p_sec, data_proc):
    
    if task == 'train':
        folder_list = [
            # '/data/hy17/librittsR/LibriTTS_R/train-clean-360', # libritts
            '/data/hy17/librispeech/librispeech/train-clean-360', # librispeech
            # '/data/hy17/RAVDESS/', # RAVDESS
            # '/data/hy17/quesst14Database/Audio', # quesst14Database
            '/data/hy17/voxceleb/dev_wav/' # voxceleb
        ]
    elif task == 'valid':
        folder_list = [
            '/data/hy17/voxceleb/val_wav/',
            '/data/hy17/librispeech/librispeech/dev-clean'
        ]
    
    dataset = Dataset_Max(task = task, seq_len_p_sec = seq_len_p_sec, folder_list=folder_list, data_proc = data_proc)
    
    return dataset


def get_libri_dataset(task, seq_len_p_sec, data_proc):
    
    dataset = Dataset_Libri(task = task, seq_len_p_sec = seq_len_p_sec, data_proc = data_proc)
    
    return dataset


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
    parser.add_argument("--save_dir", type=str, default='saved_models')
    # parser.add_argument("--data_folder_path", type=str, default='/data/hy17/dns_pth/*') # for the dsn data
    parser.add_argument("--data_folder_path", type=str, default='/data/hy17/librispeech/librispeech')
    parser.add_argument('--seq_len_p_sec', type=float, default=0.) 
    parser.add_argument('--sample_rate', type=int, default=16000)

    # Training
    parser.add_argument('--train', dest='train', action='store_true')
    parser.add_argument('--synthesis', dest='synthesis', action='store_true')
    
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4)    
    parser.add_argument('--batch_size', type=int, default=20)  
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--load_model', type=str, default='')
    parser.add_argument('--load_step', type=int, default=0)
    parser.add_argument('--write_on_every', type=int, default=50)  
    parser.add_argument('--dataset', type=str, default='libri')
    parser.add_argument('--data_process', type=str, default='norm')
    
    parser.add_argument('--train_time_diff', dest='train_time_diff', action='store_true')

    # Encoder and decoder
    parser.add_argument('--discrete_AE', type=str, default='')
    parser.add_argument('--continuous_AE', type=str, default='')
    
    parser.add_argument('--rep_dims', type=int, default=128)
    parser.add_argument('--emb_dims', type=int, default=128)
    parser.add_argument('--quantization', dest='quantization', action='store_true')
    parser.add_argument('--bandwidth', type=float, default=3.0)
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--lstm', type=int, default=2)
    parser.add_argument('--n_residual_layers', type=int, default=1)
    parser.add_argument('--ratios', nargs='+', type=int, default=[8])
    parser.add_argument('--upsampling_ratios', nargs='+', type=int, default=[5, 4, 2])
    parser.add_argument('--final_activation', type=str, default=None)
    
    # Diff model
    parser.add_argument('--model_type', type=str, default='unet')  
    parser.add_argument('--diff_dims', type=int, default=128)
    parser.add_argument('--qtz_condition', dest='qtz_condition', action='store_true')
    parser.add_argument('--self_condition', dest='self_condition', action='store_true')
    parser.add_argument('--seq_length', type=int, default=800)
    parser.add_argument('--scaling_frame', dest='scaling_frame', action='store_true')
    parser.add_argument('--scaling_feature', dest='scaling_feature', action='store_true')
    parser.add_argument('--scaling_global', dest='scaling_global', action='store_true')
    parser.add_argument('--scaling_dim', dest='scaling_dim', action='store_true')
    parser.add_argument('--use_film', dest='use_film', action='store_true')
    parser.add_argument('--unet_scale_cond', dest='unet_scale_cond', action='store_true')
    parser.add_argument('--unet_scale_x', dest='unet_scale_x', action='store_true')

    # Cond model
    # parser.add_argument('--cond_quantization', dest='cond_quantization', action='store_true')
    parser.add_argument('--target_bandwidths', nargs='+', type=float, default=[1.5,3,6,12])
    parser.add_argument('--cond_bandwidth', type=float, default=None)
    # parser.add_argument('--cond_global', type=float, default=1)
    
    # Dist
    parser.add_argument('--use_disc', dest='use_disc', action='store_true')
    parser.add_argument('--disc_freq', type=int, default=1)
    
    # Synthesis
    parser.add_argument('--midway_t', type=int, default=100)
    parser.add_argument('--lam', type=float, default=0.1)
    parser.add_argument('--input_dir', type=str, default='eval_wavs/')
    parser.add_argument('--output_dir', type=str, default='output_wavs/')
    
    inp_args = parser.parse_args() # Input arguments
    # args = get_args() # Enviornmente arguments

    assert not (inp_args.self_condition and inp_args.qtz_condition) # self_cond and quantization can't be both true.
    assert not (inp_args.synthesis and inp_args.train) 
    assert inp_args.synthesis or inp_args.train
    
    # inp_args.seq_length = inp_args.seq_len_p_sec * inp_args.sample_rate / np.prod(inp_args.ratios)
    
    run_ddp = False #if len(args) == 1 else True

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
    else:
        global_rank = 0

    if inp_args.train:
        train(inp_args, global_rank)
    if inp_args.synthesis:
        synthesis(inp_args)