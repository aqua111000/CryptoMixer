import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import numpy as np
import torch.nn.functional as F
from .layers import MarketInfoMixer, TimeBatchNorm2d, ConditionalMarketInfoMixer, TwoStreamFusionMixing, UserTimeBatchNorm2d, MarketStateEstimation
from .data import CollateFunc

class CryptoMxier(nn.Module):
    def __init__(self, 
                 input_xtime_index, 
                 input_x_index, 
                 input_xntime_index, 
                 input_x_n_index, 
                 input_x_action_index,
                 input_x_market_info_index, 
                 hidden_dim = 64, 
                 time_dim = 64,
                 output_dim = 2,  
                 dropout = 0.0,
                 market_info_dim = 23,
                 sequence_length = 64,
                 ff_dim = 64,
                 activation_fn = "gelu",
                 use_market_mixing = True,
                 use_user_mixing = True,
                 use_time_mixing = True,
                 use_interp = True,
                 data_y_mode = 'trx_or_not',
                 y_col = [1],
                 ):
        super().__init__()

        activation_fn = getattr(F, activation_fn)
        
        self.input_x_action_index = input_x_action_index
        self.input_x_market_info_index = input_x_market_info_index
        self.tinterp = MarketStateEstimation(16, use_interp = use_interp)
        
        self.conditional_market_info_mixer = ConditionalMarketInfoMixer(
            sequence_length,
            len(input_x_index),
            hidden_dim,
            len(input_x_n_index),
            ff_dim,
            activation_fn,
            dropout_rate = dropout,
            norm_type = UserTimeBatchNorm2d,
            use_market_mixing = use_market_mixing
        )

        self.two_stream_fusion_mixing = TwoStreamFusionMixing(
            sequence_length,
            hidden_dim,
            hidden_dim,
            ff_dim,
            activation_fn = activation_fn,
            dropout_rate = dropout,
            norm_type = TimeBatchNorm2d,
            use_user_mixing = use_user_mixing,
            use_time_mixing = use_time_mixing,
            use_interp = use_interp,
        )
        
        self.temporal_projection = nn.Linear(sequence_length, 1)
        self.xn_projection = nn.Linear(len(input_x_n_index), hidden_dim)

        if data_y_mode != 'trx_vol':
            self.final_nn = nn.Linear(hidden_dim * 2, output_dim)
            self.activation_fn = activation_fn
        else:
            self.final_nn = nn.Linear(hidden_dim * 2, hidden_dim)
            self.activation_fn = activation_fn
            self.final_final_nn = nn.Linear(hidden_dim, len(y_col))
            
    
        self.input_x_index = input_x_index
        self.input_xtime_index = input_xtime_index
        self.input_x_n_index = input_x_n_index
        self.input_xntime_index = input_xntime_index
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.use_interp = use_interp
        self.data_y_mode = data_y_mode
        
    def forward(self, x, trx_mask, pad_mask, xn):

        batch, p, seq, _ = x.shape

        x_time = x[:, -1, :, self.input_xtime_index[0]]  # b, s
        xn_time = xn[:, self.input_xntime_index[0]]
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)
        x = x[:, :, :, self.input_x_index]

        x = self.tinterp(x_time, x, mask = trx_mask)
        x = self.conditional_market_info_mixer(x, xn)
        x = self.two_stream_fusion_mixing(x, trx_mask = trx_mask, pad_mask = pad_mask)
        
        x = self.temporal_projection(x.permute(0, 2, 1))[:,:,0]
        x = torch.cat([x, self.xn_projection(xn)], dim = -1)
        if self.data_y_mode != 'trx_vol':
            x = self.activation_fn(self.final_nn(x))
        else:
            x = self.final_final_nn(self.activation_fn(self.final_nn(x)))

        return x
        

