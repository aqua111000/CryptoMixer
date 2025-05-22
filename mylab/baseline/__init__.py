import torch.nn as nn
from torch.nn.parameter import Parameter
import torch
import numpy as np
import torch.nn.functional as F
from .data import CollateFunc
from .time2vec import T2V, PositionalEncoding, TemporalPositionalEncoding
from .model import TransformerEncoder
from .lstm import LSTM
from .gru import GRU
from .node import NODE
from .mtand.model import enc_mtan_classif_activity
from .timesnet.model import TimesNet
from .resnet.model import RestNet18
from .deeplob.model import deeplob
from .cmt.model import CMT
from .swintrf.model import SwinTransformer
from .cas_vit.model import rcvit_xs
from .tsmixer.model import TSMixer
from .stockmixer.model import StockMixer
from .gru_d.model import GRUD, FadeDelta
from .transformers import iTransformer, Informer
from .FEDformer import FEDformer
from .PatchTST import PatchTST


class FinalMLP(nn.Module):
    def __init__(self, hidden_dim, final_hidden, output_dim):
        super().__init__()
        self.nn1 = nn.Linear(hidden_dim, final_hidden)
        self.nn2 = nn.Linear(final_hidden, output_dim)

    def forward(self, x):
        x = F.leaky_relu(self.nn1(x))
        x = self.nn2(x)
        return x


class BaselineClassifier(nn.Module):
    def __init__(self, tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 32, output_dim = 2, final_hidden = 128, market_info_dim = 23):
        super().__init__()
        
        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)
        self.market_info_nn = nn.Linear(market_info_dim, hidden_dim)
        
        self.tsmodel = tsmodel
        
        self.finalmlp = FinalMLP(hidden_dim * 2, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, xn):

        x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)

        x = self.init_x_nn(x)     #(batch, seq, dim)
        xn = self.init_xn_nn(xn)  #(batch, 1, dim)
        market_info = self.market_info_nn(market_info)  #(batch, seq, dim)

        x = market_info + x

        need_encoded_x  = self.tsmodel(x)   #x2(batch, seq, dim * 2)
        
        final = torch.cat([need_encoded_x, xn], dim = 1) #(batch, dim * 4)
        out = self.finalmlp(final)

        return out

class LSTMClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 64, num_layers = 2, dropout = 0.0, market_info_dim = 23):
        super().__init__()
        
        tsmodel = LSTM(hidden_dim, hidden_dim, num_layers = num_layers, batch_first = True)
        self._model_ = BaselineClassifier(tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = hidden_dim, output_dim = output_dim, final_hidden = final_hidden, market_info_dim = market_info_dim)

    def forward(self, x, market_info, xn):
        out = self._model_(x, market_info, xn)
        return out


class ATTClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 64, num_layers = 2, dropout = 0.0, market_info_dim = 23):
        super().__init__()
        
        tsmodel = TransformerEncoder(hidden_dim, 8, num_layers = num_layers)
        self._model_ = BaselineClassifier(tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = hidden_dim, output_dim = output_dim, final_hidden = final_hidden, market_info_dim = market_info_dim)

    def forward(self, x, market_info, xn):
        out = self._model_(x, market_info, xn)
        return out


class GRUClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 32, output_dim = 2, final_hidden = 64, num_layers = 2, dropout = 0.0, market_info_dim = 23):
        super().__init__()
        
        tsmodel = GRU(hidden_dim, hidden_dim, num_layers = num_layers, batch_first = True)
        self._model_ = BaselineClassifier(tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = hidden_dim, output_dim = output_dim, final_hidden = final_hidden, market_info_dim = market_info_dim)

    def forward(self, x, market_info, xn):
        out = self._model_(x, market_info, xn)
        return out


class SMambaClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, num_layers = 2, dropout = 0.0, market_info_dim = 23):
        super().__init__()
        
        tsmodel = SMamba(hidden_dim, hidden_dim, num_layers = num_layers, dropout = dropout)
        self._model_ = BaselineClassifier(tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = hidden_dim, output_dim = output_dim, final_hidden = final_hidden, market_info_dim = market_info_dim)

    def forward(self, x, market_info, xn):
        out = self._model_(x, market_info, xn)
        return out


class NODEClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 32, output_dim = 2, final_hidden = 128, market_info_dim = 23, dropout = 0.0):
        super().__init__()
        
        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)
        self.market_info_nn = nn.Linear(market_info_dim, hidden_dim)
        
        self.node = NODE(hidden_dim, hidden_dim, dropout_p = dropout)
        
        self.finalmlp = FinalMLP(hidden_dim * 2, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, xn):

        x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)
        xt = x[:, :, self.input_xtime_index[0]]  #(batch, seq)
        xnt = xn[:, self.input_xntime_index]  #(batch, 1)
        t = torch.cat([xt, xnt], dim = 1)  #(batch, seq + 1)
        delta_t = t[:, 1:] - t[:, :-1]  #(batch, seq)

        x = self.init_x_nn(x)     #(batch, seq, dim)
        xn = self.init_xn_nn(xn)  #(batch, 1, dim)
        market_info = self.market_info_nn(market_info)  #(batch, seq, dim)

        x = market_info + x
        
        need_encoded_x = self.node.integrate(delta_t, x, xn)   #x2(batch, seq, dim * 2)
        
        final = torch.cat([need_encoded_x, xn], dim = 1) #(batch, dim * 4)
        out = self.finalmlp(final)

        return out


class MtandClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 32, output_dim = 2, final_hidden = 128, market_info_dim = 23, dropout = 0.0):
        super().__init__()
        
        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)
        self.market_info_nn = nn.Linear(market_info_dim, hidden_dim)
        
        self.mtand = enc_mtan_classif_activity(hidden_dim, nhidden=hidden_dim, embed_time=hidden_dim, num_heads=1, dropout = dropout)
        
        self.finalmlp = FinalMLP(hidden_dim * 2, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, xn):

        x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)
        xt = x[:, :, self.input_xtime_index[0]]  #(batch, seq)
        xnt = xn[:, self.input_xntime_index]  #(batch, 1)
        t = torch.cat([xt, xnt], dim = 1)  #(batch, seq + 1)

        x = self.init_x_nn(x)     #(batch, seq, dim)
        xn = self.init_xn_nn(xn)  #(batch, 1, dim)
        market_info = self.market_info_nn(market_info)  #(batch, seq, dim)

        x = market_info + x
        
        need_encoded_x = self.mtand(x, t)   #x2(batch, seq, dim * 2)
        
        final = torch.cat([need_encoded_x, xn], dim = 1) #(batch, dim * 4)
        out = self.finalmlp(final)

        return out


class TimesNetClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, market_info_dim = 23, dropout = 0.0):
        super().__init__()
        
        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)
        self.market_info_nn = nn.Linear(market_info_dim, hidden_dim)
        
        self.tsnet =TimesNet(enc_in = hidden_dim, d_model = hidden_dim, dropout = dropout)
        
        self.finalmlp = FinalMLP(hidden_dim * 2, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, xn):

        x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)
        xt = x[:, :, self.input_xtime_index[0]]  #(batch, seq)
        xnt = xn[:, self.input_xntime_index]  #(batch, 1)
        t = xt  #(batch, seq + 1)

        x = self.init_x_nn(x)     #(batch, seq, dim)
        xn = self.init_xn_nn(xn)  #(batch, 1, dim)
        market_info = self.market_info_nn(market_info)  #(batch, seq, dim)

        x = market_info + x
        
        need_encoded_x = self.tsnet(x, t)   #x2(batch, seq, dim * 2)
        
        final = torch.cat([need_encoded_x, xn], dim = 1) #(batch, dim * 4)
        out = self.finalmlp(final)

        return out


class ResNetClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 32, output_dim = 2, final_hidden = 128, dropout = 0.0, market_info_dim = 23):
        super().__init__()
        
        tsmodel = RestNet18()
        self._model_ = BaselineClassifier(tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = hidden_dim, output_dim = output_dim, final_hidden = final_hidden, market_info_dim = market_info_dim)

    def forward(self, x, market_info, xn):
        out = self._model_(x, market_info, xn)
        return out


class DeepLOBClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 32, output_dim = 2, final_hidden = 128, dropout = 0.0, market_info_dim = 23):
        super().__init__()
        
        tsmodel = deeplob(hidden_dim)
        self._model_ = BaselineClassifier(tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = hidden_dim, output_dim = output_dim, final_hidden = final_hidden, market_info_dim = market_info_dim)

    def forward(self, x, market_info, xn):
        out = self._model_(x, market_info, xn)
        return out


class CMTClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, market_info_dim = 23, dropout = 0.0):
        super().__init__()
        
        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)
        self.market_info_nn = nn.Linear(market_info_dim, hidden_dim)
        
        self.cmt = CMT()
        
        self.finalmlp = FinalMLP(hidden_dim + 96, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, xn):

        x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)

        x = self.init_x_nn(x)     #(batch, seq, dim)
        xn = self.init_xn_nn(xn)  #(batch, 1, dim)
        market_info = self.market_info_nn(market_info)  #(batch, seq, dim)

        x = market_info + x
        
        need_encoded_x = self.cmt(x.unsqueeze(1))   #x2(batch, seq, dim * 2)
        
        final = torch.cat([need_encoded_x, xn], dim = 1) #(batch, dim * 4)
        out = self.finalmlp(final)

        return out


class CasVitClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, market_info_dim = 23, dropout = 0.0):
        super().__init__()
        
        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)
        self.market_info_nn = nn.Linear(market_info_dim, hidden_dim)
        
        self.cmt = rcvit_xs()
        
        self.finalmlp = FinalMLP(hidden_dim * 2, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, xn):

        x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)

        x = self.init_x_nn(x)     #(batch, seq, dim)
        xn = self.init_xn_nn(xn)  #(batch, 1, dim)
        market_info = self.market_info_nn(market_info)  #(batch, seq, dim)

        x = market_info + x
        
        need_encoded_x = self.cmt(x.unsqueeze(1))   #x2(batch, seq, dim * 2)
        
        final = torch.cat([need_encoded_x, xn], dim = 1) #(batch, dim * 4)
        out = self.finalmlp(final)

        return out


class SwinTransformerClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, market_info_dim = 23, dropout = 0.0):
        super().__init__()
        
        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)
        self.market_info_nn = nn.Linear(market_info_dim, hidden_dim)
        
        self.swintrf = SwinTransformer(input_dim = hidden_dim, use_t2v = False, patch_size=8, window_size=8, in_chans=1, embed_dim=96)
        
        self.finalmlp = FinalMLP(hidden_dim + 96, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, xn):

        x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)

        x = self.init_x_nn(x)     #(batch, seq, dim)
        xn = self.init_xn_nn(xn)  #(batch, 1, dim)
        market_info = self.market_info_nn(market_info)  #(batch, seq, dim)

        x = market_info + x
        
        need_encoded_x = self.swintrf(x.unsqueeze(1))   #x2(batch, seq, dim * 2)
        
        final = torch.cat([need_encoded_x, xn], dim = 1) #(batch, dim * 4)
        out = self.finalmlp(final)

        return out


class TSMixerClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, dropout = 0.0, market_info_dim = 23):
        super().__init__()
        
        tsmodel = TSMixer(64, 1, hidden_dim, hidden_dim, dropout_rate = dropout)
        self._model_ = BaselineClassifier(tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = hidden_dim, output_dim = output_dim, final_hidden = final_hidden, market_info_dim = market_info_dim)

    def forward(self, x, market_info, xn):
        out = self._model_(x, market_info, xn)
        return out


class StockMixerClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, market_info_dim = 23, dropout = 0.0):
        super().__init__()
        
        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)
        self.market_info_nn = nn.Linear(market_info_dim, hidden_dim)
        
        self.stockmixer = StockMixer(hidden_dim)
        
        self.finalmlp = FinalMLP(hidden_dim * 2, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, trx_mask, pad_mask, xn):

        x = x[:, :, :, self.input_x_index]  #(batch, seq, dim1)
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)

        x = self.init_x_nn(x)     #(batch, seq, dim)
        xn = self.init_xn_nn(xn)  #(batch, 1, dim)
        
        need_encoded_x = self.stockmixer(x)   #x2(batch, seq, dim * 2)
        
        final = torch.cat([need_encoded_x, xn], dim = 1) #(batch, dim * 4)
        out = self.finalmlp(final)

        return out


class GRUDClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, market_info_dim = 23, dropout = 0.0, device = 'cuda:0'):
        super().__init__()
        
        self.init_x_nn = nn.Linear(len(input_x_index), hidden_dim)
        self.init_xn_nn = nn.Linear(len(input_x_n_index), hidden_dim)
        self.market_info_nn = nn.Linear(market_info_dim, hidden_dim)
        
        self.grud = GRUD(hidden_dim, hidden_dim, np.zeros(hidden_dim), device = device, output_last = True)
        self.fadedelta = FadeDelta(hidden_dim)
        self.finalmlp = FinalMLP(hidden_dim * 2, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, delta_t, xn_delta_t, mask, xn):

        #print(x.shape, market_info.shape, delta_t.shape, xn_delta_t.shape, mask.shape, xn.shape)

        x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        xn = xn[:, self.input_x_n_index]  #(batch, dim2)

        x = self.init_x_nn(x)     #(batch, seq, dim)
        xn = self.init_xn_nn(xn)  #(batch, 1, dim)
        market_info = self.market_info_nn(market_info)  #(batch, seq, dim)

        mask = mask.unsqueeze(-1).repeat(1, 1, x.shape[-1])
        x = x + market_info

        last_encoded_x = self.grud(x, mask, delta_t)
        now_encoded_x = self.fadedelta(last_encoded_x, xn_delta_t)

        final = torch.cat([now_encoded_x, xn], dim = 1) #(batch, dim * 4)
        out = self.finalmlp(final)

        return out


class iTransformerClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 64, num_layers = 2, dropout = 0.0, market_info_dim = 23):
        super().__init__()
        
        tsmodel = iTransformer()
        self._model_ = BaselineClassifier(tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = hidden_dim, output_dim = output_dim, final_hidden = final_hidden, market_info_dim = market_info_dim)

    def forward(self, x, market_info, xn):
        out = self._model_(x, market_info, xn)
        return out


class InformerClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, market_info_dim = 23):
        super().__init__()

        tsmodel = Informer(len(input_x_index)+market_info_dim, len(input_x_n_index))
        self.tsmodel = tsmodel
        
        self.finalmlp = FinalMLP(hidden_dim, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, xn):

        enc_x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        enc_x = torch.cat([enc_x, market_info], dim = -1)
        dec_x = torch.cat([x[:, -16:], xn.unsqueeze(1)], dim = 1)
        dec_x = dec_x[:, :, self.input_x_n_index]  #(batch, dim2)

        final = self.tsmodel(enc_x, dec_x)   #x2(batch, seq, dim * 2)

        return final


class FEDformerClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 128, market_info_dim = 23):
        super().__init__()

        tsmodel = FEDformer(len(input_x_index)+market_info_dim, len(input_x_n_index))
        self.tsmodel = tsmodel
        
        self.finalmlp = FinalMLP(hidden_dim, final_hidden, output_dim)

        self.input_xtime_index = input_xtime_index
        self.input_xntime_index = input_xntime_index
        self.input_x_index = input_x_index
        self.input_x_n_index = input_x_n_index

        self.output_dim = output_dim

    def forward(self, x, market_info, xn):

        enc_x = x[:, :, self.input_x_index]  #(batch, seq, dim1)
        enc_x = torch.cat([enc_x, market_info], dim = -1)
        dec_x = torch.cat([x[:, -32:], xn.unsqueeze(1)], dim = 1)
        dec_x = dec_x[:, :, self.input_x_n_index]  #(batch, dim2)

        final = self.tsmodel(enc_x, dec_x)   #x2(batch, seq, dim * 2)

        return final

class PatchTSTClassifier(nn.Module):
    def __init__(self, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = 64, output_dim = 2, final_hidden = 64, num_layers = 2, dropout = 0.0, market_info_dim = 23):
        super().__init__()
        
        tsmodel = PatchTST(64)
        self._model_ = BaselineClassifier(tsmodel, input_xtime_index, input_x_index, input_xntime_index, input_x_n_index, hidden_dim = hidden_dim, output_dim = output_dim, final_hidden = final_hidden, market_info_dim = market_info_dim)

    def forward(self, x, market_info, xn):
        out = self._model_(x, market_info, xn)
        return out