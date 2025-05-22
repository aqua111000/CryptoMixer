from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from .utils import TemporalPositionalEncoding
import math
from torch.nn.parameter import Parameter


class AdaptiveU1(nn.Module):

    def __init__(self, 
                 input_channels,
                 activation_fn = F.relu):
        super().__init__()

        self.linear_weight = nn.Linear(input_channels, 1)
        self.activation_fn = activation_fn

    def forward(self, x, mask = None):

        mix_weight = self.activation_fn(self.linear_weight(x))[..., 0]  # batch * seq * p
        
        if mask is not None:
            mask = mask.transpose(1, 2)[:, :, :-1]  # batch, seq, p-1
            mask = torch.cat([mask, torch.ones(mask.shape[0], mask.shape[1], 1).to(mask)], dim = -1)  # batch, seq, p
            mix_weight.masked_fill_(mask==0, -1e9)  

        mix_weight = F.softmax(mix_weight, dim = -1) # batch * seq * p
        return mix_weight.unsqueeze(2)  # batch * seq * 1 * p

class TimeBatchNorm2d(nn.BatchNorm1d):

    def __init__(self, normalized_shape):

        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

    def forward(self, x: Tensor) -> Tensor:

        x = x.reshape(x.shape[0], -1, 1)
        x = super().forward(x)
        x = x.reshape(x.shape[0], self.num_time_steps, self.num_channels)

        return x


class UserTimeBatchNorm2d(nn.BatchNorm1d):

    def __init__(self, normalized_shape):

        num_time_steps, num_channels = normalized_shape
        super().__init__(num_channels * num_time_steps)
        self.num_time_steps = num_time_steps
        self.num_channels = num_channels

    def forward(self, x):

        # b, p, s, d
        b, p, s, d = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1, p)
        x = super().forward(x)
        x = x.reshape(b, s, d, p).permute(0, 3, 1, 2)

        return x


class MarketInfoMixer(nn.Module):

    def __init__(
        self,
        sequence_length,
        input_channels,
        output_channels,
        ff_dim,
        activation_fn = F.relu,
        dropout_rate = 0.05,
        norm_type = TimeBatchNorm2d,
    ):
        super().__init__()

        self.norm = norm_type((sequence_length, input_channels))

        self.activation_fn = activation_fn
        self.fc1 = nn.Linear(input_channels, ff_dim)
        self.fc2 = nn.Linear(ff_dim, output_channels)

        self.projection = (
            nn.Linear(input_channels, output_channels)
            if input_channels != output_channels
            else nn.Identity()
        )

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x_proj = self.projection(x)

        x = self.norm(x)

        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)  

        x = x_proj + self.dropout(x) 

        return x


class ConditionalMarketInfoMixer(nn.Module):

    def __init__(
        self,
        sequence_length,
        input_channels,
        output_channels,
        static_channels,
        ff_dim,
        activation_fn = F.relu,
        dropout_rate = 0.05,
        norm_type = nn.LayerNorm,
        use_market_mixing = True
    ):
        super().__init__()

        self.use_market_mixing = use_market_mixing

        if use_market_mixing:
        
            self.conditional_transfer = nn.Linear(static_channels, output_channels)
            self.market_mixing = MarketInfoMixer(
                sequence_length,
                input_channels + output_channels,
                output_channels,
                ff_dim,
                activation_fn,
                dropout_rate,
                norm_type=norm_type,
            )

        else:

            self.transform = nn.Linear(input_channels, output_channels)
            print('not use conditional_market_info_mixer')

    def forward(self, x, x_t):

        if self.use_market_mixing:
            conditional_x_t = self.conditional_transfer(x_t) 
            conditional_x_t = conditional_x_t.unsqueeze(1).unsqueeze(2).repeat(1, x.shape[1], x.shape[2], 1) 
            return self.market_mixing(torch.cat([x, conditional_x_t], dim=-1))
        else:
            return self.transform(x)


class TimeInfoMixing(nn.Module):

    def __init__(
        self,
        sequence_length,
        input_channels,
        activation_fn = F.relu,
        dropout_rate = 0.05,
        norm_type = TimeBatchNorm2d,
    ):
        super().__init__()

        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.fc1 = nn.Linear(sequence_length, sequence_length)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):

        x_temp = x.permute(0, 2, 1)
        x_temp = self.activation_fn(self.fc1(x_temp))
        x_res = x_temp.permute(0, 2, 1)

        return self.norm(x + x_res)   


class UserInfoMixing(nn.Module):

    def __init__(
        self,
        sequence_length,
        input_channels,
        activation_fn = F.relu,
        dropout_rate = 0.05,
        norm_type = TimeBatchNorm2d,
        use_interp = False,
    ):
        super().__init__()
        self.norm = norm_type((sequence_length, input_channels))
        self.activation_fn = activation_fn
        self.adaptiveU1 = AdaptiveU1(input_channels * 2, activation_fn = activation_fn)
        self.U2 = nn.Linear(input_channels * 2, input_channels)
        self.p_pad = nn.Parameter(torch.randn(input_channels * 2))
        self.use_interp = use_interp
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, trx_mask = None, pad_mask = None):

        # data preparation
        x = x.permute(0, 2, 1, 3) # b, t, p, d
        x_v = x[:, :, :-1]
        x_u = x[:, :, -1:].repeat(1, 1, x_v.shape[2], 1)
        x_v = torch.cat([x_v, x_u], dim = -1)
        pad = self.p_pad.reshape(1,1,1,-1).repeat(x_v.shape[0], x_v.shape[1], 1, 1)
        x_v_pad = torch.cat([x_v, pad], dim = -2)  # batch * seq * p * dim
        if pad_mask is not None:
            mask = pad_mask.unsqueeze(-1).repeat(1,1,trx_mask.shape[2])
        else:
            mask = None

        # compute U1 and mixing
        U1 = self.adaptiveU1(x_v_pad, mask = mask)
        U1 = self.dropout(U1)
        mix_out = self.U2(self.activation_fn(U1 @ x_v_pad)) # batch * seq * 1 * dim
        return self.norm(mix_out[:, :, 0] + x[:, :, -1])

class TwoStreamFusionMixing(nn.Module):

    '''
    CryptoMixer designs a two-stream fusion mixer to capture both time-over-user and user-over-time trading behavior patterns. 
    '''
    
    def __init__(
        self,
        sequence_length,
        input_channels,
        output_channels,
        ff_dim,
        activation_fn = F.relu,
        dropout_rate = 0.1,
        norm_type = nn.LayerNorm,
        use_user_mixing = True,
        use_time_mixing = True,
        use_interp = False,
    ):
        super().__init__()

        self.use_time_mixing = use_time_mixing
        self.use_user_mixing = use_user_mixing
        
        if use_time_mixing:
            self.time_info_mixing = TimeInfoMixing(
                sequence_length,
                input_channels,
                activation_fn,
                dropout_rate,
                norm_type=norm_type,
            )
        else:
            print('not use time mixing')

        if use_user_mixing:
            self.user_info_mixing = UserInfoMixing(
                sequence_length,
                input_channels,
                activation_fn,
                dropout_rate,
                norm_type=norm_type,
                use_interp = use_interp,
            )
        else:
            print('not use user mixing')

    def forward(self, x, trx_mask = None, pad_mask = None):

        if self.use_time_mixing:
            time_x = self.time_info_mixing(x[:, -1]) 
        else:
            time_x = x[:, -1]
        
        if self.use_user_mixing:
            user_x = self.user_info_mixing(x, trx_mask = trx_mask, pad_mask = pad_mask)  #batch * seq * dim
            x = time_x + user_x
        else:
            x = time_x

        return x


class MarketStateEstimation(nn.Module):
    '''
    an attention based market information interpolation 
    method. This method aggregates users’ transactions
    from other time points to the missing time points.
    '''
    
    def __init__(
        self,
        tpe_dim = 16,
        use_interp = True
    ):
        super().__init__()
        self.use_interp = use_interp
        if use_interp:
            self.alpha = nn.Linear(1,1)
            self.tpe = TemporalPositionalEncoding(tpe_dim)
            print('use interp')
        
    def attention(self, tpe, x, mask=None):
        # tpe [batch, seq, dim]
        # mask [batch, p, seq]
        # x [batch, p, seq, inputs]
        d_k = tpe.size(-1)   # [batch, seq, dim]
        p = x.size(1)
        seq = tpe.size(-2)
        scores = torch.matmul(tpe, tpe.transpose(-2, -1)) \
                 / math.sqrt(d_k)   # [batch, seq, seq]
        scores = scores.unsqueeze(1).repeat(1, p, 1, 1)  # [batch, p, seq, seq]
        if mask is not None:
            scores = scores.masked_fill((mask.unsqueeze(-2).repeat(1, 1, seq, 1) + torch.eye(seq).unsqueeze(0).unsqueeze(1).to(mask)) < 0.5, -1e9)  # mask [batch, p, 1, seq]
        p_attn = F.softmax(scores, dim = -1)  # [batch, p, seq, seq]
        return p_attn @ x   # [batch, p, seq, inputs]

    def max_min_norm_t(self, t):
        t_min = torch.min(t, dim = 1)[0].unsqueeze(-1) # b, seq
        t_max = torch.max(t, dim = 1)[0].unsqueeze(-1) # b, seq
        return (t - t_min) / (t_max - t_min)
    
    def forward(self, t, x, mask = None):

        if self.use_interp:
            # mask b*p*seq
            t = self.max_min_norm_t(t)
            t = (F.elu(self.alpha.weight * 1000, 1, False) + 1) * t
            tpe = self.tpe(t) # b, seq, dim
            inter_x = self.attention(tpe, x, mask = mask)
            mask = mask.unsqueeze(-1)
            x = x * mask + inter_x * (1 - mask)
        else:
            x = x

        return x
