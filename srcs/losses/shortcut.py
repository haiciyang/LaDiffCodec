"""
Copyright (C) 2025 Yukara Ikemiya
-----------------------------------------------------
Shortcut Models.
https://github.com/yukara-ikemiya/modified-shortcut-models-pytorch/tree/master/src/shortcut_models
"""

import typing as tp

import torch
from torch import nn
from torch.nn import functional as F

def exists(x):
    return x is not None


class ShortcutModel(nn.Module):
    """
    Shortcut model class for label-conditioned image generation
    """

    def __init__(
        self,
        model,
        seq_length = 160
    ):
        super().__init__()

        self.model = model
        self.input_shape = None
        self._seq_length = seq_length
        self.num_timesteps=128

    @property
    def seq_length(self):
        return self._seq_length

    @seq_length.setter
    def seq_length(self, value):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Sequence length must be a non-negative int.")
        self._seq_length = value

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.LongTensor,
        t: torch.Tensor,
        dt: torch.Tensor,
        num_self_consistency: int,
    ):
        """
        x1: ground-truth data (e.g. image)
        """
        bs, D, L = x.shape
        assert len(cond) == len(t) == len(dt) == bs and torch.all(t + dt <= 1.0)
        assert num_self_consistency < bs

        x0 = torch.randn_like(x)  # noise
        x_t = (1 - t[:, None, None]) * x0 + t[:, None, None] * x  # eq.(1)
        v_t = x - x0

        if num_self_consistency > 0:
            x_t_sc = x_t[:num_self_consistency]
            t_sc = t[:num_self_consistency]
            dt_half = dt[:num_self_consistency] * 0.5
            conds_sc = cond[:num_self_consistency]
            # calculate targets for self-consistency term (eq.(5))
            with torch.no_grad():
                v1_sc = self.model(x_t_sc, t_sc, conds_sc, dt_half)
                v2_sc = self.model(x_t_sc + dt_half[:, None, None] * v1_sc, t_sc + dt_half, conds_sc, dt_half)

            v_t_sc = (v1_sc + v2_sc) / 2.

        # dt = 0.0 -> naive flow-matching
        dt[num_self_consistency:] = 0.

        # forward
        v_out = self.model(x_t, t, cond, dt)

        # output = {}

        # flow-matching loss (eq.(5))
        loss = F.mse_loss(v_t[num_self_consistency:], v_out[num_self_consistency:])
        # output['loss_fm'] = loss.detach()

        # self-consistency loss (eq.(5))
        if num_self_consistency > 0:
            loss_sc = F.mse_loss(v_t_sc, v_out[:num_self_consistency])
            loss += loss_sc
            # output['loss_sc'] = loss_sc.detach()

        # output['loss'] = loss
        # print(v_out.shape)
        # print(dt.shape)
        # print(num_self_consistency, v_out[:num_self_consistency].shape, dt[:num_self_consistency].shape)

        out = v_out.clone()
        out[:num_self_consistency] = out[:num_self_consistency] * dt[:num_self_consistency][:, None, None].detach()

        x_out = out + x0
        x_out.detach()
        return loss, x_out, None, None

    @torch.no_grad()
    def sample(
        self,
        condition,
        dim_in,
        n_step: tp.Optional[int] = None,
        dt_list: tp.Optional[tp.List[int]] = None,
        disable_shortcut: bool = False,
        **kwargs
    ):
        # assert exists(n_step) or exists(dt_list)
        if n_step is None:
            n_step = self.num_timesteps
        device = condition.device
        num_sample = len(condition)

        if exists(n_step):
            dt_list = [1. / n_step] * n_step

        assert sum(dt_list) <= 1 + 1e-6

        # initial noise
        x = torch.randn(num_sample, dim_in, self._seq_length, device=device)

        # sample
        t_cur = torch.zeros(num_sample, device=device)
        for dt_val in dt_list:
            dt = torch.full((num_sample,), dt_val, device=device)

            # predict
            if disable_shortcut:
                dt_in = torch.zeros_like(dt)
            else:
                dt_in = dt

            vel = self.model(x, t_cur, condition, dt_in)

            # update
            x += vel * dt[:, None, None]

            t_cur = t_cur + dt

        return x