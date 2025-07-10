import os
import re
import copy
import torch
import numpy as np
from torch import nn
from scipy.io import wavfile
from collections import OrderedDict
from matplotlib import pyplot as plt


def nn_parameters(model):
    
    # pp=0
    # for p in list(model.parameters()):
    #     nn=1
    #     for s in list(p.size()):
    #         nn = nn*s
    #     pp += nn
    
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    nontrainable = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    
    total = trainable + nontrainable
    # trainable = 
    # print(list(model.parameters())[0].requires_grad)
    
    return total, trainable


def save_img(rep, name, note, out_path=''):

    if out_path:
        directory = out_path + note
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        save_path = f"{directory}/{name}.png"
    else:
        save_path = f"{note}_{name}.png"

    *_, h, w = rep.shape
    rep = rep.reshape(h, w).cpu().data.numpy()
    plt.imshow(rep, aspect='auto', origin='lower')
    plt.colorbar()
    plt.savefig(save_path)
    plt.clf()

def save_plot(x, name, note, out_path=''):

    if out_path:
        directory = out_path + note
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        save_path = f"{directory}/{name}.png"
    else:
        save_path = f"{note}_{name}.png"

    x = x.squeeze().cpu().data.numpy()
    plt.plot(x/np.max(np.abs(x)))
    plt.savefig(save_path)
    plt.clf()

def save_torch_wav(x, name, note, out_path=''):

    if out_path:
        directory = out_path + note
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        save_path = f"{directory}/{name}.wav"
    else:
        save_path = f"{note}_{name}.wav"
 
    x = x.squeeze().cpu().data.numpy()
    print(f'Saved at {save_path}')
    wavfile.write(save_path, 16000, x/np.max(np.abs(x)))

def save_checkpoints(model, output_dir, exp_name, ema=None, disc=None, note=''):

    directory = f'{output_dir}/{exp_name}'
    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(model.state_dict(), f'{output_dir}/{exp_name}/model_{note}.amlt')
    if ema is not None:
        torch.save(ema.state_dict(), f'{output_dir}/{exp_name}/ema_{note}.amlt')
    if disc is not None:
        torch.save(disc.state_dict(), f'{output_dir}/{exp_name}/disc_{note}.amlt')


def load_from_checkpoint(model, model_path, strict=True):

    state_dict = torch.load(model_path)
    model_dict = OrderedDict()
    pattern = re.compile('module.')
    for k,v in state_dict.items():
        if re.search("module", k):
            model_dict[re.sub(pattern, '', k)] = v
        else:
            model_dict = state_dict
    model.load_state_dict(model_dict, strict=strict)

    # return model 

def logging(step, tr_loss_dict, val_loss_dict, time, exp_name, vall):

    if not os.path.exists('logs/'):
        os.mkdir('logs/')
        
    result_path = 'logs/'+ exp_name +'.txt'

    tr_loss_rec = ' | '.join([ f"tr_{key}: {value:.3f}" for key, value in tr_loss_dict.items()])
    val_loss_rec = ' | '.join([ f"val_{key}: {value:.3f}" for key, value in val_loss_dict.items()])

    records = f'Step: {step} | {tr_loss_rec} | {val_loss_rec} | Best: {vall:.3f} | Duration: {time:.1f} \n'
    
    with open(result_path, 'a+') as file:
        file.write(records)
        file.flush()


def log_params(params_dict, exp_name):

    if not os.path.exists('logs/'):
        os.mkdir('logs/')
        
    result_path = 'logs/'+ exp_name +'.txt'

    with open(result_path, 'a+') as file:
        for key, value in params_dict.items():
            print(key, value)
            file.write('%s %s\n'%(key, value))
            file.flush()


def exists(val):
    return val is not None

def clamp(value, min_value = None, max_value = None):
    assert exists(min_value) or exists(max_value)
    if exists(min_value):
        value = max(value, min_value)

    if exists(max_value):
        value = min(value, max_value)

    return value

""" START - utils.py functions from the original encodec repo """

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""Various utilities."""

from hashlib import sha256
from pathlib import Path
import typing as tp

import torch
import torchaudio


def _linear_overlap_add(frames: tp.List[torch.Tensor], stride: int):
    # Generic overlap add, with linear fade-in/fade-out, supporting complex scenario
    # e.g., more than 2 frames per position.
    # The core idea is to use a weight function that is a triangle,
    # with a maximum value at the middle of the segment.
    # We use this weighting when summing the frames, and divide by the sum of weights
    # for each positions at the end. Thus:
    #   - if a frame is the only one to cover a position, the weighting is a no-op.
    #   - if 2 frames cover a position:
    #          ...  ...
    #         /   \/   \
    #        /    /\    \
    #            S  T       , i.e. S offset of second frame starts, T end of first frame.
    # Then the weight function for each one is: (t - S), (T - t), with `t` a given offset.
    # After the final normalization, the weight of the second frame at position `t` is
    # (t - S) / (t - S + (T - t)) = (t - S) / (T - S), which is exactly what we want.
    #
    #   - if more than 2 frames overlap at a given point, we hope that by induction
    #      something sensible happens.
    assert len(frames)
    device = frames[0].device
    dtype = frames[0].dtype
    shape = frames[0].shape[:-1]
    total_size = stride * (len(frames) - 1) + frames[-1].shape[-1]

    frame_length = frames[0].shape[-1]
    t = torch.linspace(0, 1, frame_length + 2, device=device, dtype=dtype)[1: -1]
    weight = 0.5 - (t - 0.5).abs()

    sum_weight = torch.zeros(total_size, device=device, dtype=dtype)
    out = torch.zeros(*shape, total_size, device=device, dtype=dtype)
    offset: int = 0

    for frame in frames:
        frame_length = frame.shape[-1]
        out[..., offset:offset + frame_length] += weight[:frame_length] * frame
        sum_weight[offset:offset + frame_length] += weight[:frame_length]
        offset += stride
    assert sum_weight.min() > 0
    return out / sum_weight


def _get_checkpoint_url(root_url: str, checkpoint: str):
    if not root_url.endswith('/'):
        root_url += '/'
    return root_url + checkpoint


def _check_checksum(path: Path, checksum: str):
    sha = sha256()
    with open(path, 'rb') as file:
        while True:
            buf = file.read(2**20)
            if not buf:
                break
            sha.update(buf)
    actual_checksum = sha.hexdigest()[:len(checksum)]
    if actual_checksum != checksum:
        raise RuntimeError(f'Invalid checksum for file {path}, '
                           f'expected {checksum} but got {actual_checksum}')


def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.shape[0] in [1, 2], "Audio must be mono or stereo."
    if target_channels == 1:
        wav = wav.mean(0, keepdim=True)
    elif target_channels == 2:
        *shape, _, length = wav.shape
        wav = wav.expand(*shape, target_channels, length)
    elif wav.shape[0] == 1:
        wav = wav.expand(target_channels, -1)
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav


def save_audio(wav: torch.Tensor, path: tp.Union[Path, str],
               sample_rate: int, rescale: bool = False):
    limit = 0.99
    mx = wav.abs().max()
    if rescale:
        wav = wav * min(limit / mx, 1)
    else:
        wav = wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


""" END - utils.py functions from the original encodec repo """



class EMA(nn.Module):
    """
    Implements exponential moving average shadowing for your model.
    Utilizes an inverse decay schedule to manage longer term training runs.
    By adjusting the power, you can control how fast EMA will ramp up to your specified beta.
    @crowsonkb's notes on EMA Warmup:
    If gamma=1 and power=1, implements a simple average. gamma=1, power=2/3 are
    good values for models you plan to train for a million or more steps (reaches decay
    factor 0.999 at 31.6K steps, 0.9999 at 1M steps), gamma=1, power=3/4 for models
    you plan to train for less (reaches decay factor 0.999 at 10K steps, 0.9999 at
    215.4k steps).
    Args:
        inv_gamma (float): Inverse multiplicative factor of EMA warmup. Default: 1.
        power (float): Exponential factor of EMA warmup. Default: 1.
        min_value (float): The minimum EMA decay rate. Default: 0.
    """
    def __init__(
        self,
        model,
        ema_model = None,           # if your model has lazylinears or other types of non-deepcopyable modules, you can pass in your own ema model
        beta = 0.9999,
        update_after_step = 100,
        update_every = 10,
        inv_gamma = 1.0,
        power = 2 / 3,
        min_value = 0.0,
        param_or_buffer_names_no_ema = set(),
        ignore_names = set(),
        ignore_startswith_names = set(),
        include_online_model = True  # set this to False if you do not wish for the online model to be saved along with the ema model (managed externally)
    ):
        super().__init__()
        self.beta = beta

        # whether to include the online model within the module tree, so that state_dict also saves it

        self.include_online_model = include_online_model

        if include_online_model:
            self.online_model = model
        else:
            self.online_model = [model] # hack

        # ema model

        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = copy.deepcopy(model)
            except:
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        self.ema_model.requires_grad_(False)

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if param.dtype == torch.float}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if buffer.dtype == torch.float}

        self.update_every = update_every
        self.update_after_step = update_after_step

        self.inv_gamma = inv_gamma
        self.power = power
        self.min_value = min_value

        assert isinstance(param_or_buffer_names_no_ema, (set, list))
        self.param_or_buffer_names_no_ema = param_or_buffer_names_no_ema # parameter or buffer

        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

        self.register_buffer('initted', torch.Tensor([False]))
        self.register_buffer('step', torch.tensor([0]))

    @property
    def model(self):
        return self.online_model if self.include_online_model else self.online_model[0]
    
    def restore_ema_model_device(self):
        device = self.initted.device
        self.ema_model.to(device)

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name not in self.parameter_names:
                continue
            yield name, param

    def get_buffers_iter(self, model):
        for name, buffer in model.named_parameters():
            if name not in self.buffer_names:
                continue
            yield name, buffer

    def copy_params_from_model_to_ema(self):
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_model), self.get_params_iter(self.model)):
            ma_params.data.copy_(current_params.data)

        for (_, ma_buffers), (_, current_buffers) in zip(self.get_buffers_iter(self.ema_model), self.get_buffers_iter(self.model)):
            ma_buffers.data.copy_(current_buffers.data)

    def get_current_decay(self):
        epoch = clamp(self.step.item() - self.update_after_step - 1, min_value = 0.)
        value = 1 - (1 + epoch / self.inv_gamma) ** - self.power

        if epoch <= 0:
            return 0.

        return clamp(value, min_value = self.min_value, max_value = self.beta)

    def update(self, copy_back=False):
        step = self.step.item()
        self.step += 1

        if (step % self.update_every) != 0:
            return

        if step <= self.update_after_step:
            self.copy_params_from_model_to_ema()
            return

        if not self.initted.item():
            self.copy_params_from_model_to_ema()
            self.initted.data.copy_(torch.Tensor([True]))

        self.update_moving_average(self.ema_model, self.model, copy_back)
        

    @torch.no_grad()
    def update_moving_average(self, ma_model, current_model, copy_back):
        current_decay = self.get_current_decay()

        for (name, current_params), (_, ma_params) in zip(self.get_params_iter(current_model), self.get_params_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_params.data.copy_(current_params.data)
                continue

            ma_params.data.lerp_(current_params.data, 1. - current_decay)
            if copy_back:
                current_params.data.copy_(ma_params.data)

        for (name, current_buffer), (_, ma_buffer) in zip(self.get_buffers_iter(current_model), self.get_buffers_iter(ma_model)):
            if name in self.ignore_names:
                continue

            if any([name.startswith(prefix) for prefix in self.ignore_startswith_names]):
                continue

            if name in self.param_or_buffer_names_no_ema:
                ma_buffer.data.copy_(current_buffer.data)
                continue

            ma_buffer.data.lerp_(current_buffer.data, 1. - current_decay)
            if copy_back:
                current_buffer.data.copy_(ma_buffer.data)


    def __call__(self, *args, **kwargs):
        return self.ema_model(*args, **kwargs)
