import os
import gc
import json
import time
import glob
# import librosa
import argparse
import numpy as np
# from matplotlib import pyplot as plt

import torch
import torch.nn as nn
from torch import optim
from torchaudio import transforms
from torch.distributions.normal import Normal
from torch.utils.data import Dataset, DataLoader
# from torch.utils.tensorboard import SummaryWriter

from .dataset import EnCodec_data
from .model import EncodecModel
from .msstftd import MultiScaleSTFTDiscriminator as MSDisc
from .balancer import Balancer

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")


def melspec_loss(s, s_hat):
    
    loss = 0
    
    n=10

    mel_transform = transforms.MelSpectrogram(
        n_fft=2**n, hop_length=(2**n)//4, 
        win_length=2**n, window_fn=torch.hann_window,n_mels=64,
        normalized=True, center=False, pad_mode=None, power=1).to(device)
    
    mel_s = mel_transform(s)
    mel_s_hat = mel_transform(s_hat)
    
    loss += torch.sum(abs(mel_s - mel_s_hat)) + torch.sum((mel_s - mel_s_hat) ** 2)

    loss /= torch.sum(abs(s))
    
    return loss

def train(model, disc, train_loader, optimizer_G, optimizer_D):



        # model = models[model_name]()
        # model.set_target_bandwidth(bw)
        # audio_suffix = model_name.split('_')[1][:3]
        # wav, sr = torchaudio.load(f"test_{audio_suffix}.wav")
        # wav = wav[:, :model.sample_rate * 2]
        # wav_in = wav.unsqueeze(0)
        # wav_dec = model(wav_in)[0]

    # ---------- Run model ------------
    g_loss, d_loss, t_loss, f_loss, w_loss, feat_loss = 0,0,0,0,0,0
    
    for s in train_loader:


        # s shape (64, 1, 16000)
        s = s.unsqueeze(1).to(torch.float).to(device)

        emb = model.encoder(s) # [64, 128, 50]
        quantizedResult = model.quantizer(emb, sample_rate=16000) 
            # Resutls contain - quantized, codes, bw, penalty=torch.mean(commit_loss))
        qtz_emb = quantizedResult.quantized
        s_hat = model.decoder(qtz_emb) #(64, 1, 16000)

        # ---- VQ Commitment loss l_w -----
        l_w = quantizedResult.penalty # commitment loss

        # ---- Discriminator l_d, l_g, l_feat -----
        s_disc_r, fmap_r = disc(s) # list of the outputs from each discriminator
        s_disc_gen, fmap_gen = disc(s_hat)
        # s_disc_r: [3*[64*[]]]
        # fmap_r: [3*5*torch.Size([64, 32, 59, 513/257/128..])] 5 conv layers, different stride size

        l_g = 0
        l_d = 0
        l_feat = 0

        # --- Update Generator ---
        optimizer_G.zero_grad()
        # ---- VQ Commitment loss l_w -----
        l_w = quantizedResult.penalty # commitment loss

        
        for d_id in range(len(fmap_r)):

            l_g += torch.sum(torch.max(torch.tensor(0), 1-s_disc_gen[d_id])) # Gen loss
            # l_g += lgs

            for l_id in range(len(fmap_r[0])):
                
                l_feat += torch.sum(abs(fmap_r[d_id][l_id] - \
                        fmap_gen[d_id][l_id]))/torch.mean(abs(fmap_r[d_id][l_id]))

        # ------ Reconstruction loss l_t, l_s --------
        l_t = torch.sum(torch.abs(s - s_hat))
        l_f = melspec_loss(s, s_hat)


        # ------ Commitement loss and Balancer for other losses ----
        l_w.backward(retain_graph=True)

        losses = {'l_t': l_t, 'l_f': l_f, 'l_g': l_g, 'l_feat': l_feat}
        balancer = Balancer(weights={'l_t': 0.1, 'l_f': 1, 'l_g': 3, 'l_feat': 3}, rescale_grads=False)
        balancer.backward(losses, s_hat)
        
        optimizer_G.step()


        # Update
        optimizer_D.zero_grad()
        l_d = 0
        # s_disc_r, fmap_r = disc(s) # list of the outputs from each discriminator
        # s_disc_gen, fmap_gen = disc(s_hat)
       
        for d_id in range(len(fmap_r)):
            l_d += torch.sum(torch.max(torch.tensor(0), 1-s_disc_r[d_id]) + torch.max(torch.tensor(0), 1+s_disc_gen[d_id].detach())) # Disc loss
            # l_d += lds
        l_d.backward()

        optimizer_D.step()

        g_loss += l_g.detach().data.cpu()
        d_loss += l_d.detach().data.cpu()
        t_loss += l_t.detach().data.cpu()
        f_loss += l_f.detach().data.cpu()
        w_loss += l_w.detach().data.cpu()
        feat_loss += l_feat.detach().data.cpu()
        
        for key, value in reduced_loss_dict.items():
            self.run.log(key, value.item())

        break


    return g_loss/len(train_loader), d_loss/len(train_loader), t_loss/len(train_loader), f_loss/len(train_loader), w_loss/len(train_loader), feat_loss/len(train_loader)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Encodec_baseline")
    parser.add_argument("--output_dir", type=str, default=os.getenv("AMLT_OUTPUT_DIR", "/tmp"))
    parser.add_argument("--data_path", type=str, default=os.environ.get("AMLT_DATA_DIR", ".")+'/read_speech_16k/*')

    args = parser.parse_args()


    # SW = SummaryWriter(os.environ.gfet("AMLT_OUTPUT_DIR", "."), flush_secs=30)

    train_dataset = EnCodec_data(args.data_path)

    # test_dataset = EnCodec_data()

    train_loader = DataLoader(train_dataset, batch_size=5, \
        shuffle=True, num_workers=4, pin_memory=True)
    # test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=4, pin_memory=True)

    # # get pretrained model
    # model = EncodecModel.encodec_model_24khz()

    # get new model
    model = EncodecModel._get_model(
                   target_bandwidths = [1.5], 
                   sample_rate = 16000,  # 24_000
                   channels  = 1,
                   causal  = True,
                   model_norm  = 'weight_norm',
                   audio_normalize  = False,
                   segment = None, # tp.Optional[float]
                   name = 'unset').to(device)

    disc = MSDisc(filters=32).to(device)

    optimizer_G = optim.Adam(model.parameters(), lr=3e-4, betas=(0.5, 0.9))
    optimizer_D = optim.Adam(disc.parameters(), lr=3e-4, betas=(0.5, 0.9))

    # ---- Train 2000 epochs
    for epoch in range(2000):

        losses = train(model, disc, train_loader, optimizer_G, optimizer_D)
        print(losses)

        print(args.output_dir + "/result.txt")

        with open(args.output_dir + "/result.txt", 'a+') as file:
            file.write('epoch {}: {}\n'.format(epoch, losses))
            file.flush()

        # if epoch%20 == 0:
        torch.save(model.state_dict(), args.output_dir + "/epoch{}_model.amlt".format(str(epoch)))
        print('model saved at: '+ args.output_dir + "/result.txt")

        break

        if epoch == 3:
            break
        # eva_loss = eval(model, disc, )

    # --- log ----


    


    

    



