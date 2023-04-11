import os
import re
import glob
import time
import argparse
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
# from matplotlib import pyplot as plt
# from asteroid.losses.pit_wrapper import PITLossWrapper

import torch
import torchvision
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
from labml import lab, tracker, experiment, monit

from .model import DiffAudioRep
# from .dataset import EnCodec_data

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

def logging(step, tr_loss_dict, val_loss_dict, time, exp_name, vall):

    result_path = 'logs/'+ exp_name +'.txt'

    tr_loss_rec = ' | '.join([ f"tr_{key}: {value:.3f}" for key, value in tr_loss_dict.items()])
    val_loss_rec = ' | '.join([ f"val_{key}: {value:.3f}" for key, value in val_loss_dict.items()])

    records = f'Step: {step} | {tr_loss_rec} | {val_loss_rec} | Best: {vall:.3f} | Duration: {time:.1f} \n'
    
    with open(result_path, 'a+') as file:
        file.write(records)
        file.flush()

class MNISTDataset(torchvision.datasets.MNIST):
    """
    ### MNIST dataset
    """

    def __init__(self, image_size=32, train=True):
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(),
        ])

        super().__init__(str(lab.get_data_path()), train=train, download=True, transform=transform)

    def __getitem__(self, item):
        return super().__getitem__(item)[0]



def run_model(model=None, data_loader=None, optimizer=None, debug=None):
    
    tot_nums = dict()
    for idx, batch in enumerate(data_loader):

        x = batch.unsqueeze(1).to(torch.float).to(device)
        nums, x_hat = model(x) 
        
        if model.training:
            loss = list(nums.values())[0]
            loss.backward()
            optimizer.step()

            # if ema is not None:
            #     ema.update(copy_back=True)

        if idx == 0:
            tot_nums = nums
        else:
            for key, value in nums.items():
                tot_nums[key] += value
        
        if debug:
            break

    for key, value in tot_nums.items():
        tot_nums[key] /= (idx + 1)

    return tot_nums


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Encodec_baseline")
    
    # Data related
    parser.add_argument("--output_dir", type=str, default='saved_models')
    parser.add_argument("--data_path", type=str, default='/data/hy17/dns_pth/*')
    parser.add_argument("--n_spks", type=int, default=500)
    parser.add_argument('--seq_len_p_sec', type=float, default=1.) 
    parser.add_argument('--sample_rate', type=int, default=16000)

    # Training
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--lr', type=float, default=5e-4)    
    parser.add_argument('--batch_size', type=int, default=5)  
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--finetune_model', type=str, default='')
    parser.add_argument('--write_on_every', type=int, default=50)  
    parser.add_argument('--model_type', type=str, default='transformer')  


    # Encoder and decoder
    parser.add_argument('--rep_dims', type=int, default=128)
    parser.add_argument('--emb_dims', type=int, default=128)
    parser.add_argument('--quantization', dest='quantization', action='store_true')
    parser.add_argument('--bandwidth', type=float, default=3.0)
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--lstm', type=int, default=2)
    parser.add_argument('--n_residual_layers', type=int, default=1)
    parser.add_argument('--enc_ratios', nargs='+', type=int)

    # Diff model
    parser.add_argument('--diff_dims', type=int, default=128)
    parser.add_argument('--self_cond', dest='self_cond', action='store_true')

    parser.add_argument('--run_diff', dest='run_diff', action='store_true')
    parser.add_argument('--run_vae', dest='run_vae', action='store_true')
    
    # Dist
    parser.add_argument('--use_disc', dest='use_disc', action='store_true')
    parser.add_argument('--disc_freq', type=int, default=1)

    
    inp_args = parser.parse_args() # Input arguments

    # train_dataset = EnCodec_data(inp_args.data_path, task = 'train', seq_len_p_sec = inp_args.seq_len_p_sec, sample_rate=inp_args.sample_rate, multi=False, n_spks = inp_args.n_spks)
    # valid_dataset = EnCodec_data(inp_args.data_path, task = 'valid', seq_len_p_sec = inp_args.seq_len_p_sec, sample_rate=inp_args.sample_rate, multi=False, n_spks = inp_args.n_spks)
    train_dataset = MNISTDataset(train=True)
    valid_dataset = MNISTDataset(train=False)
    train_loader = DataLoader(train_dataset, batch_size=inp_args.batch_size, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=inp_args.batch_size, pin_memory=True)

    model = DiffAudioRep().to(device)
    

    # if inp_args.run_diff:
    #     ema = EMA(
    #         model.diff_process,
    #         beta = 0.9999,              # exponential moving average factor
    #         update_after_step = 100,    # only after this number of .update() calls will it start updating
    #         update_every = 10,          # how often to actually update, to save on compute (updates every 10th .update() call)
    #     )
    # else:
    #     ema = None

    if inp_args.finetune_model:

        state_dict = torch.load(inp_args.finetune_model)
        model_dict = OrderedDict()
        pattern = re.compile('module.')
        for k,v in state_dict.items():
            if re.search("module", k):
                model_dict[re.sub(pattern, '', k)] = v
            else:
                model_dict = state_dict
        model.load_state_dict(model_dict, strict=False)


    if inp_args.run_diff:
        optimizer = optim.Adam(model.diff_process.parameters(), lr=inp_args.lr)
    else:
        optimizer = optim.Adam(model.parameters(), lr=inp_args.lr)

    # run = Run.get_context()
    best_loss = torch.tensor(float('inf'))
    write_on_every = 20 if not inp_args.debug else 1
    

    # ---- Train 2000 steps
    for step in range(200000):
        if step == 0:
            print('Starts training ...')
        model.train()
        start_time = time.time()
        tr_losses = run_model(model, train_loader, optimizer, inp_args.debug)

        if step % write_on_every == 0:
            with torch.no_grad():
                model.eval()
                val_losses = run_model(model=model, data_loader=valid_loader, debug=inp_args.debug)
            end_time = time.time()

            # if gpu_rank == 0: # only save model and logs for the main process

            vall = list(val_losses.values())[-1] # negsdr
            
            if inp_args.debug:
                # print(val_losses)
                print([val.item() for val in val_losses.values()])
            else:

                if vall < best_loss:
                    best_loss = vall
                    if not inp_args.debug:
                        torch.save(model.state_dict(), f'{inp_args.output_dir}/{inp_args.exp_name}.amlt')
                        if inp_args.use_disc:
                            torch.save(disc.state_dict(), f'{inp_args.output_dir}/{inp_args.exp_name}_disc.amlt')

                # for key, value in tr_losses.items():
                #     # run.log(key, value.item())
                #     writer.add_scalar('Train/'+key, value.item(), step)
                    
                # for key, value in val_losses.items():
                #     writer.add_scalar('Valid/'+key, value.item(), step)
                
                logging(step, tr_losses, val_losses, end_time-start_time, inp_args.exp_name, best_loss)

                # writer.flush()
                        
    # Tear down the process group
    dist.destroy_process_group()

    

