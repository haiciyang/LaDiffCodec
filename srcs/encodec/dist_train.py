import os
import gc
import json
import time
# import librosa
import argparse
import numpy as np
# from matplotlib import pyplot as plt

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

# from azureml.core.run import Run

from .dataset_stable import EnCodec_data
from .model import EncodecModel
from .msstftd import MultiScaleSTFTDiscriminator as MSDisc
from .balancer import Balancer

def cal_sdr(s, s_hat):

    if len(s.shape) == 3:
        B, C, L = s.shape
        s = s.reshape(B*C, L)
    if len(s_hat.shape) == 3:
        B, C, L = s_hat.shape
        s_hat = s_hat.reshape(B*C, L)

     # s, s_hat - (bt, L)
    return torch.mean(
        -10 * torch.log10(
        torch.sum((s - s_hat)**2, -1) / torch.sum(s**2, -1))
    )


def melspec_loss(s, s_hat, gpu_rank, n_freq):
    
    loss = 0
    sl = s.shape[-1]

    for n in n_freq:

        mel_transform = transforms.MelSpectrogram(
            n_fft=2**n, hop_length=(2**n)//4, 
            win_length=2**n, window_fn=torch.hann_window,n_mels=64,
            normalized=True, center=False, pad_mode=None, power=1).cuda(gpu_rank)
        
        mel_s = mel_transform(s)
        mel_s_hat = mel_transform(s_hat)
    
    loss += torch.mean(abs(mel_s - mel_s_hat)) + torch.mean((mel_s - mel_s_hat) ** 2)

    loss /= 8*sl
    
    return loss

def train(model, disc, train_loader, bandwidth, optimizer_G, optimizer_D, gpu_rank, use_disc, disc_freq, debug):

    # ---------- Run model ------------
    g_loss, d_loss, t_loss, f_loss, w_loss, feat_loss = 0,0,0,0,0,0

    model.train()
    for idx, batch in enumerate(train_loader):
        
        s = batch[0]
        # s shape (64, 1, 16000)
        s = s.unsqueeze(1).to(torch.float).cuda(gpu_rank)

        emb = model.encoder(s) # [64, 128, 50]
        frame_rate = 16000 // model.encoder.hop_length
        quantizedResult = model.quantizer(emb, sample_rate=frame_rate, bandwidth=bandwidth) # !!! should be frame_rate - 50
            # Resutls contain - quantized, codes, bw, penalty=torch.mean(commit_loss))
        qtz_emb = quantizedResult.quantized
        s_hat = model.decoder(qtz_emb) #(64, 1, 16000)
        
        # --- Update Generator ---
        optimizer_G.zero_grad()

         # ---- VQ Commitment loss l_w -----
        l_w = quantizedResult.penalty # commitment loss

        l_t = torch.mean(torch.abs(s - s_hat))
        l_f = melspec_loss(s, s_hat, gpu_rank, range(5,12))

        t_loss += l_t.detach().data.cpu()
        f_loss += l_f.detach().data.cpu()
        w_loss += l_w.detach().data.cpu()

        # ---- Discriminator l_d, l_g, l_feat -----

        if use_disc:
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

            g_loss += l_g.detach().data.cpu()
            feat_loss += l_feat.detach().data.cpu()

            l_w.backward(retain_graph=True)
            losses = {'l_t': l_t, 'l_f': l_f, 'l_g': l_g, 'l_feat': l_feat}
            balancer = Balancer(weights={'l_t': 0.1, 'l_f': 1, 'l_g': 3, 'l_feat': 3}, rescale_grads=True)
            balancer.backward(losses, s_hat)

            # loss = 0.1 * l_t + l_f + 3 * l_g + 3 * l_feat + l_w
            # loss.backward()

            optimizer_G.step()

             # Update Discriminator
            if idx % disc_freq == 0:
                optimizer_D.zero_grad()

                l_d = 0
                s_disc_r, fmap_r = disc(s) # list of the outputs from each discriminator
                s_disc_gen, fmap_gen = disc(s_hat.detach())
            
                for d_id in range(len(fmap_r)):
                    l_d += 1/K * torch.mean(torch.max(torch.tensor(0), 1-s_disc_r[d_id]) + torch.max(torch.tensor(0), 1+s_disc_gen[d_id])) # Disc loss

                l_d.backward()
                optimizer_D.step()

                d_loss += l_d.detach().data.cpu()

            # dist.barrier()
            losses = {'l_g': g_loss/len(train_loader), 'l_d': d_loss/len(train_loader)*disc_freq, 'l_t': t_loss/len(train_loader), 'l_f': f_loss/len(train_loader), 'l_w': w_loss/len(train_loader), 'l_feat': feat_loss/len(train_loader)}

        else:
            loss = 0.1 * l_t + l_f + l_w
            loss.backward()
            optimizer_G.step()
            # dist.barrier()

            losses = {'l_t': t_loss/len(train_loader), 'l_f': f_loss/len(train_loader), 'l_w': w_loss/len(train_loader)}
        if debug:
            break
    return losses


def valid(model, valid_loader, bandwidth, gpu_rank, debug):

    # ---------- Run model ------------
    t_loss, sdr_tt = 0, 0
    
    model.eval()

    for batch in valid_loader:

        # s shape (64, 1, 16000)
        s = batch[0]
        s = s.unsqueeze(1).to(torch.float).cuda(gpu_rank)

        emb = model.encoder(s) # [64, 128, 50]
        frame_rate = 16000 // model.encoder.hop_length
        quantizedResult = model.quantizer(emb, sample_rate=frame_rate, bandwidth=bandwidth) # !!! should be frame_rate - 50
            # Resutls contain - quantized, codes, bw, penalty=torch.mean(commit_loss))
        qtz_emb = quantizedResult.quantized
        s_hat = model.decoder(qtz_emb) #(64, 1, 16000)

        # ------ Reconstruction loss l_t, l_f --------
        l_t = torch.mean(torch.abs(s - s_hat))
        t_loss += l_t.detach().data.cpu()

        sdr = cal_sdr(s, s_hat)
        sdr_tt += sdr.detach().data.cpu()

        if debug:
            break

    losses = {'val_l_t': t_loss/len(valid_loader), 'sdr_tt': sdr_tt/len(valid_loader)}

    return losses


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
    # if missing:
    #     print(f"II Environment variables not set: {', '.join(missing)}.")
    return args


if __name__ == '__main__':
        
    parser = argparse.ArgumentParser(description="Encodec_baseline")
    parser.add_argument("--output_dir", type=str, default='/home/hy17/Projects/encodec/saved_models')
    # parser.add_argument("--data_path", type=str, default=os.environ.get("AMLT_DATA_DIR", ".")+'/*')
    parser.add_argument("--data_path", type=str, default='/data/hy17/dns_pth/*')
    # parser.add_argument("--data_path", type=str, defaulst='/home/v-haiciyang/data/haici/dns_pth/*')
    parser.add_argument("--disc_freq", type=int, default=1)
    parser.add_argument('--multi', dest='multi', action='store_true')
    parser.add_argument('--use_disc', dest='use_disc', action='store_true')
    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--bandwidth', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--exp_name', type=str, default='')
    parser.add_argument('--finetune_model', type=str, default='')
    
    inp_args = parser.parse_args()

    args = get_args()

    run_ddp = False if len(args) == 1 else True
    if not inp_args.debug:
        writer = SummaryWriter(f'runs/{inp_args.exp_name}')

    if run_ddp:
        master_uri = "tcp://%s:%s" % (args.get("MASTER_ADDR"), args.get("MASTER_PORT"))
        os.environ["NCCL_DEBUG"] = "WARN"
        node_rank = args.get("NODE_RANK")

        gpus_per_node = torch.cuda.device_count()
        world_size = args.get("WORLD_SIZE")
        gpu_rank = args.get("LOCAL_RANK")
        if inp_args.debug:
            node_rank = 0
        global_rank = node_rank * gpus_per_node + gpu_rank
        dist.init_process_group(
            backend="nccl", init_method=master_uri, world_size=world_size, rank=global_rank
        )

    # synchronizes all the threads to reach this point before moving on
    # dist.barrier() 

    train_dataset = EnCodec_data(inp_args.data_path, task = 'train', seq_len_p_sec = 1, sample_rate=16000, multi=inp_args.multi)
    valid_dataset = EnCodec_data(inp_args.data_path, task = 'valid', seq_len_p_sec = 1, sample_rate=16000, multi=inp_args.multi)
    if run_ddp:
        train_sampler = DistributedSampler(dataset=train_dataset, shuffle=True) 
        valid_sampler = DistributedSampler(dataset=valid_dataset, shuffle=True) 
        train_loader = DataLoader(train_dataset, batch_size=8, sampler=train_sampler, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=8, sampler=valid_sampler, pin_memory=True)
        torch.manual_seed(global_rank)
        torch.cuda.set_device(gpu_rank)    
    else:
        train_loader = DataLoader(train_dataset, batch_size=inp_args.batch_size, pin_memory=True)
        valid_loader = DataLoader(valid_dataset, batch_size=inp_args.batch_size, pin_memory=True)
        gpu_rank = 0
    # get new model
    model = EncodecModel._get_model(
                   target_bandwidths = [1.5, 3, 6], 
                   sample_rate = 16000,  # 24_000
                   channels  = 1,
                   causal  = True,
                   model_norm  = 'weight_norm',
                   audio_normalize  = False,
                   segment = None, # tp.Optional[float]
                   name = 'unset').cuda(gpu_rank)
    
    model.set_target_bandwidth(inp_args.bandwidth)

    if finetune_model:
        model.load_state_dict(torch.load(inp_args.finetune_model))

    disc = MSDisc(filters=32).cuda(gpu_rank)

    optimizer_G = optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(disc.parameters(), lr=3e-4, betas=(0.5, 0.9))

    if run_ddp:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu_rank])    
        disc = nn.parallel.DistributedDataParallel(disc, device_ids=[gpu_rank])
        
    loss = torch.tensor(float('-inf'))
    # ---- Train 2000 epochs
    for epoch in range(2000):
        
        if run_ddp:
            train_loader.sampler.set_epoch(epoch)

        tr_losses = train(model, disc, train_loader, inp_args.bandwidth, optimizer_G, 
        optimizer_D, gpu_rank, inp_args.use_disc, inp_args.disc_freq, inp_args.debug)
        with torch.no_grad():
            val_losses = valid(model, valid_loader, inp_args.bandwidth, gpu_rank, inp_args.debug)

        if gpu_rank == 0: # only save model and logs for the main process

            if inp_args.debug:
                print(list(val_losses.values()))
            else:
                for key, value in tr_losses.items():
                    writer.add_scalar('Train/'+key, value.item(), epoch)
                for key, value in val_losses.items():
                    writer.add_scalar('Valid/'+key, value.item(), epoch)

                writer.flush()

                vall = list(val_losses.values())[-1] # sdr
                if vall > loss:
                    loss = vall
                    # print(epoch, 'save_model')
                    if not inp_args.debug:
                        torch.save(model.state_dict(), f'{inp_args.output_dir}/{inp_args.exp_name}.amlt')
                        torch.save(disc.state_dict(), f'{inp_args.output_dir}/{inp_args.exp_name}_disc.amlt')
    
    # Tear down the process group
    # dist.destroy_process_group()
# 
    

    



