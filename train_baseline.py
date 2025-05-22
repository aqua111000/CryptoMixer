import os
import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from tqdm import tqdm
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
from mylab.utils import setup_seed, EarlyStopAndSave
from mylab.trainer import train, evaluate
from mylab.loss import focal_loss
from mylab.data import get_train_test_data
import pickle
import torchcde
import torchmetrics
import torchvision
import json
from sklearn import metrics
from mylab.lr_scheduler import CyclicLRWithRestarts
from mylab.utils import LossFigure
np.set_printoptions(threshold=4096)
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.benchmark = True

base_domain = ['block_num', 'value0', 'value1', 'gas_price', 'gas_used', 'balance0', 'balance1', 'last_balance0', 'last_balance1',]

base_time_domain = ['block_num']
base_maket_info_domain = ['last_balance0', 'last_balance1',]
base_action_info_domain = ['value0', 'value1', 'gas_price', 'gas_used']

self_info_domain = ['token0_trx_amount_in_sum', 'token1_trx_amount_in_sum', 'token0_trx_amount_out_sum', 'token1_trx_amount_out_sum', 'trx_num_in_sum', 'trx_num_out_sum']
dex_price_domain = ['max_dex_token0_price', 'min_dex_token0_price', 'mean_dex_token0_price', 'max_dex_token1_price', 'min_dex_token1_price', 'mean_dex_token1_price',]
exchange_rate_domain = ['exchage_rate',]
dex_price_volatility_domain = [f'token0_dex_price_volatility{i}' for i in range(7)] + [f'token1_dex_price_volatility{i}' for i in range(7)]
circle_ar_domain = ['one_reward_for_token0', 'one_reward_for_token1', 'max_ex_for_token0', 'max_ex_for_token1', 'ifa2b_best_ex_for_token0', 'ifa2b_best_ex_for_token1', 'ifb2a_best_ex_for_token0', 'ifb2a_best_ex_for_token1',]
other_user_inform_domain = [f'other_inform{i}' for i in range(13)]
cex_price_domain = [f'token0_cex_price_volatility{i}' for i in range(7)] + [f'token0_cex_price_last{i}block' for i in range(1, 45+1)]

domain_dict = {
    'base_time_domain': base_time_domain,
    'base_maket_info_domain': base_maket_info_domain,
    'base_action_info_domain': base_action_info_domain,
    'self_info_domain': self_info_domain,
    'dex_price_domain': dex_price_domain,
    'exchange_rate_domain': exchange_rate_domain,
    'dex_price_volatility_domain': dex_price_volatility_domain,
    'circle_ar_domain': circle_ar_domain,
    'other_user_inform_domain': other_user_inform_domain,
    'cex_price_domain': cex_price_domain
}

all_col = base_domain + self_info_domain + dex_price_domain + exchange_rate_domain + dex_price_volatility_domain + circle_ar_domain + other_user_inform_domain + cex_price_domain

class args:
    def __init__(self):
        self.device = 'cuda:2'
        self.early_stop = True
        self.save_path = 'model_log'
        self.data_root_dir = '../..'
        self.result_root_dir = 'result'
        self.data_to_cuda = False
        self.figure_path = 'loss_fig'
        self.loss_figure = LossFigure(self.figure_path)
        self.now_figure_path = None
        
        self.batch_size = 256
        self.epochs = 1000
        self.lr = 3e-3
        self.seed = 1
        self.weight_decay = 1e-3
        
        self.classify = True
        self.distribute = False
    
        self.data_name = 'user_transaction_data'
        self.market_info_data_name = 'additional_market_info'

        self.negetive_sample_rate = 0.5
        self.split_rate = [8,1,1]
        
        self.random_sample = False
        self.data_y_mode = 'trx_or_not'  # trx_direction  trx_vol
        self.y_col = [1]
        self.circle_ar_inform = False
        self.other_inform = False
        self.binance_price_inform = False
        self.use_account_feature = False
        self.use_drop_data = True
        self.drop_max_len = 100
        self.data_split_by = 'data'
        self.range_len = 900 # 900
        self.get_data_by_seq_len = True
        self.hidden_dim = 64
        
        self.user_embedding = True
        self.user_similarity_type = 1
        
        self.num_layers = 3

        # cru setting
        self.enc_var_activation = 'square'
        self.dec_var_activation = 'exp'
        self.trans_var_activation = 'relu'
        self.t_sensitive_trans_net = False
        self.num_basis = 15
        self.f_cru = False
        self.orthogonal = True
        self.rkn = False
        self.trans_net_hidden_units = []
        self.trans_net_hidden_activation = 'tanh'
        self.bandwidth = 3
        self.trans_covar = 0.1
        
        self.dropout_rate = 0.0

        # parameter for sim mask model
        self.use_co_trx_sim_mask = False
        self.sim_alpha = 0.1
        self.use_sim_rate = True
        self.sim_mask_thredhold = 0.1
        #self.sim_mask_before_softmax = True
        
        self.iteract = True
        self.use_matnd = True
        self.iteract_module = 'pooling'
        self.use_t2v = True

        self.use_market_mixing = True
        self.use_user_mixing = True
        self.use_time_mixing = True
        self.use_interp = True
        
        # parameter for feature enhance model

        self.model = 'cryptomixer'   

        self.domain_dict = domain_dict
        self.x_col = all_col
        self.xn_col = self.x_col
        self.x_domain = [
                         'base_time_domain',
                         'self_info_domain',
                         'base_action_info_domain',
                         'base_maket_info_domain',
                         'dex_price_domain', 
                         'exchange_rate_domain', 
                         'dex_price_volatility_domain']
        self.other_x_domain = [
                         'base_time_domain',
                         'self_info_domain',
                         'base_action_info_domain',
                         'base_maket_info_domain',
                         'dex_price_domain', 
                         'exchange_rate_domain', 
                         'dex_price_volatility_domain']
        self.x_action_domain = [
                         'base_time_domain',
                         'self_info_domain',
                         'base_action_info_domain']
        self.x_market_info_domain = [
                         'base_time_domain',
                         'base_maket_info_domain', 
                         'dex_price_domain', 
                         'exchange_rate_domain', 
                         'dex_price_volatility_domain']
        self.xn_domain = [
                          'base_time_domain',
                          'base_maket_info_domain', 
                          'dex_price_domain', 
                          'exchange_rate_domain', 
                          'dex_price_volatility_domain'#
        ]

        self.x_time_domain = ['base_time_domain']
        self.other_x_time_domain = ['base_time_domain']
        self.xn_time_domain = ['base_time_domain']

        self.std_zeros_index = []

    @property
    def day_mode_data(self):
        if 'day' in self.data_name:
            return True 
        else:
            return False

    @property
    def result_table_dir(self):
        return self.result_root_dir + '/' + self.model + '.csv'

    @property
    def output_dim(self):
        if self.data_y_mode == 'both':
            return 3
        elif self.data_y_mode == 'trx_vol':
            return len(self.y_col)
        else:
            return 2

    @property
    def seq_len(self):
        return 64

    def get_x_xn_domain(self):
        x_domain = self.x_domain
        xn_domain = self.xn_domain
        return x_domain, xn_domain

    @property
    def input_xtime_index(self):
        return [self.x_col.index(j) for i in self.x_time_domain for j in self.domain_dict[i] if self.x_col.index(j) not in self.std_zeros_index]

    @property
    def input_x_index(self):
    
        x_domain, xn_domain = self.get_x_xn_domain()
        return [self.x_col.index(j) for i in x_domain for j in self.domain_dict[i] if self.x_col.index(j) not in self.std_zeros_index]

    @property
    def input_xntime_index(self):
        return [self.xn_col.index(j) for i in self.xn_time_domain for j in self.domain_dict[i] if self.xn_col.index(j) not in self.std_zeros_index]

    @property
    def input_x_n_index(self):
        x_domain, xn_domain = self.get_x_xn_domain()
        return [self.xn_col.index(j) for i in xn_domain for j in self.domain_dict[i] if self.xn_col.index(j) not in self.std_zeros_index]

    @property
    def input_other_xtime_index(self):
        return [self.x_col.index(j) for i in self.other_x_time_domain for j in self.domain_dict[i] if self.x_col.index(j) not in self.std_zeros_index]

    @property
    def input_other_x_index(self):
        return [self.x_col.index(j) for i in self.other_x_domain for j in self.domain_dict[i] if self.x_col.index(j) not in self.std_zeros_index]

    @property
    def input_x_action_index(self):
        return [self.x_col.index(j) for i in self.x_action_domain for j in self.domain_dict[i] if self.x_col.index(j) not in self.std_zeros_index]

    @property
    def input_x_market_info_index(self):
        return [self.x_col.index(j) for i in self.x_market_info_domain for j in self.domain_dict[i] if self.x_col.index(j) not in self.std_zeros_index]

    @property
    def model_type(self):
        if self.model == 'stockmixer' or self.model == 'cryptomixer':
            return 'cryptomixer'
        elif self.model == 'grud':
            return 'grud'
        else:
            return 'baseline'

    @property
    def task_type(self):
        if self.data_y_mode == 'trx_vol':
            return 'regression'
        else:
            return 'classification'

args = args()

def get_output_file_name():
    all_name = [i[:-5] for i in os.listdir('result')]
    count = 1
    while 1:
        if str(count) in all_name:
            count+=1
        else:
            break
    return f'{count}'

def update_result_table(modelname, result):
    if os.path.exists(args.result_table_dir):
        result_table = pd.read_csv(args.result_table_dir)
    else:
        result_table = pd.DataFrame(columns=['model', 
                                             'batchsize', 
                                             'epochs', 
                                             'lr', 
                                             'seed',  
                                             'weight_decay', 
                                             'num_layers',
                                             'use_sim_rate',
                                             'drop_max_len',
                                             'dataname', 
                                             'y_mode',
                                             'figure_path'] + sorted(list(result.keys())))
    result_table.loc[len(result_table)] = {
        'model': modelname, 
        'batchsize': args.batch_size, 
        'epochs': args.epochs, 
        'lr': args.lr, 
        'seed': args.seed, 
        'weight_decay': args.weight_decay, 
        'num_layers': args.num_layers,
        'use_sim_rate': args.use_sim_rate,
        'drop_max_len': args.drop_max_len,
        'dataname': args.data_name, 
        'y_mode': args.data_y_mode,
        'figure_path': args.now_figure_path,
        'val_precision': result['val_precision'],
        'val_recall': result['val_recall'],
        'val_f1Score': result['val_f1Score'],
        'val_acc': result['val_acc'],
        'val_roauc': result['val_roauc'],
        'val_prauc': result['val_prauc'],
        'val_loss': result['val_loss'],
    }
    result_table.to_csv(args.result_table_dir, encoding='utf-8',index=False)

def get_data_from_origin():
    
    # get main data
    data_name = args.data_name# '>100day_day_weth2usdc_train_user_data_argument'# '>100day_day_train_user_data_argument'
    with open(f'{args.data_root_dir}/data/{data_name}.dat', 'rb') as f:
        user_data = pickle.load(f)

    with open(f'{args.data_root_dir}/data/{args.market_info_data_name}.dat', 'rb') as f:
        market_info = pickle.load(f)
    market_info_array = np.stack(list(market_info.values()))
    market_info_mean = np.mean(market_info_array, axis = 0)
    market_info_std = np.std(market_info_array, axis = 0)

    market_info_non_zero_index = np.where(market_info_std!=0)[0]
    market_info_mean = market_info_mean[market_info_non_zero_index]
    market_info_std = market_info_std[market_info_non_zero_index]

    for i in market_info:
        market_info[i] = (market_info[i][market_info_non_zero_index] - market_info_mean) / market_info_std
            
    # get mean and std
    if not os.path.exists(f'{args.data_root_dir}/data/mean.npy'):
        mean = None
        num = None
        for i in tqdm(user_data):
            if mean is None:
                value = np.sum(user_data[i]['data'], axis = 0)
                if (abs(value)==np.inf).any():
                    print(i)
                    raise ValueError()
                mean = value
                num = len(user_data[i]['data'])
            else:
                value = np.sum(user_data[i]['data'], axis = 0)
                if (abs(value)==np.inf).any():
                    print(i)
                    raise ValueError()
                mean += value
                num += len(user_data[i]['data'])
        mean = mean / num
        std = None
        for i in tqdm(user_data):
            if std is None:
                value = np.sum((user_data[i]['data'] - mean)**2, axis = 0)
                if not np.isnan(value).any():
                    std = value
            else:
                value = np.sum((user_data[i]['data'] - mean)**2, axis = 0)
                if not np.isnan(value).any():
                    std += value
                else:
                    print(value)
                    print(user_data[i]['data'] - mean)
                    raise ValueError()
        std = std / num
        std = np.sqrt(std)
        with open(f'{args.data_root_dir}/data/mean.npy', 'wb') as f:
            pickle.dump(mean, f)
        with open(f'{args.data_root_dir}/data/std.npy', 'wb') as f:
            pickle.dump(std, f)
    with open(f'{args.data_root_dir}/data/mean.npy', 'rb') as f:
        mean = pickle.load(f)
    with open(f'{args.data_root_dir}/data/std.npy', 'rb') as f:
        std = pickle.load(f)
        
    std_zeros_index = list(np.where(std==0)[0])
    args.std_zeros_index = std_zeros_index
    
    print(len(user_data))
    
    start_time = 1e20
    end_time = 0
    
    for i in tqdm(user_data):
        for j in user_data[i]['data']:
            if j[0] < start_time:
                start_time = j[0]
            if j[0] > end_time:
                end_time = j[0]
    
    #get all_data
    setup_seed(1)
    args.mean = mean

    train_sutd, val_sutd, test_sutd, std_zeros_index = get_train_test_data(user_data, mean, std, start_time, end_time, args, market_info)
    
    args.std_zeros_index = std_zeros_index
    
    if args.model == 'cryptomixer' or args.model == 'stockmixer':
        from mylab.cryptomixer import CollateFunc
    elif args.model == 'grud':
        from mylab.baseline.gru_d import CollateFunc
    else:
        from mylab.baseline import CollateFunc
        
    train_dataloader = DataLoader(train_sutd, batch_size = args.batch_size, shuffle = True, collate_fn=CollateFunc(args), num_workers = 4)
    val_dataloader = DataLoader(val_sutd, batch_size = args.batch_size, shuffle = True, collate_fn=CollateFunc(args), num_workers = 4)
    test_dataloader = DataLoader(test_sutd, batch_size = args.batch_size, shuffle = False, collate_fn=CollateFunc(args), num_workers = 4)

    return train_dataloader, val_dataloader, test_dataloader

def get_data():

    train_dataloader, val_dataloader, test_dataloader = get_data_from_origin()

    return train_dataloader, val_dataloader, test_dataloader
    

def get_model():
    setup_seed(args.seed)

    if args.model == 'grud':
        from mylab.baseline import GRUDClassifier
        model = GRUDClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               device = args.device,
                               output_dim = args.output_dim)
    elif args.model == 'lstm':
        from mylab.baseline import LSTMClassifier
        model = LSTMClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim
                               )
    elif args.model == 'gru':
        from mylab.baseline import GRUClassifier
        model = GRUClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'smamba':
        from mylab.baseline import SMambaClassifier
        model = SMambaClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'att':
        from mylab.baseline import ATTClassifier
        model = ATTClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'mtand':
        from mylab.baseline import MtandClassifier
        model = MtandClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'deeplob':
        from mylab.baseline import DeepLOBClassifier
        model = DeepLOBClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'stockmixer':
        from mylab.baseline import StockMixerClassifier
        model = StockMixerClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'cmt':
        from mylab.baseline import CMTClassifier
        model = CMTClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'casvit':
        from mylab.baseline import CasVitClassifier
        model = CasVitClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'swintrf':
        from mylab.baseline import SwinTransformerClassifier
        model = SwinTransformerClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'tsmixer':
        from mylab.baseline import TSMixerClassifier
        model = TSMixerClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'resnet':
        from mylab.baseline import ResNetClassifier
        model = ResNetClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'tsnet':
        from mylab.baseline import TimesNetClassifier
        model = TimesNetClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'cru':
        from mylab.baseline import CRUClassifier
        model = CRUClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               args = args,
                               output_dim = args.output_dim)
    elif args.model == 'node':
        from mylab.baseline import NODEClassifier
        model = NODEClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim)
    elif args.model == 'swintrf':
        from mylab.swintrf_good import mTANDClassifier
        model = mTANDClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index, 
                               args.input_x_action_index,
                               args.input_x_market_info_index,
                               input_other_x_index = args.input_other_xtime_index,
                               input_other_xtime_index = args.input_other_xtime_index,
                               output_dim = args.output_dim, 
                               interact = args.iteract,
                               use_matnd = args.use_matnd,
                               dropout = args.dropout_rate,
                               model = 'swintrf',
                               iteract_module = args.iteract_module,
                               use_t2v = args.use_t2v)
    elif args.model == 'iTransformer':
        from mylab.baseline import iTransformerClassifier
        model = iTransformerClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               dropout = args.dropout_rate,
                               output_dim = args.output_dim
                               )
    elif args.model == 'Informer':
        from mylab.baseline import InformerClassifier
        model = InformerClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               output_dim = args.output_dim
                               )
    elif args.model == 'FEDformer':
        from mylab.baseline import FEDformerClassifier
        model = FEDformerClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               output_dim = args.output_dim
                               )
    elif args.model == 'PatchTST':
        from mylab.baseline import PatchTSTClassifier
        model = PatchTSTClassifier(args.input_xtime_index, 
                               args.input_x_index, 
                               args.input_xntime_index, 
                               args.input_x_n_index,
                               output_dim = args.output_dim
                               )
    elif args.model == 'cryptomixer':
        from mylab.cryptomixer import CryptoMxier
        model = CryptoMxier(args.input_xtime_index, 
                            args.input_x_index, 
                            args.input_xntime_index, 
                            args.input_x_n_index, 
                            args.input_x_action_index,
                            args.input_x_market_info_index,
                            hidden_dim = args.hidden_dim,
                            output_dim = args.output_dim, 
                            dropout = args.dropout_rate,
                            use_market_mixing = args.use_market_mixing,
                            use_user_mixing = args.use_user_mixing,
                            use_time_mixing = args.use_time_mixing,
                            use_interp = args.use_interp)
        
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    early_stop_s = EarlyStopAndSave('val_loss', 'min', 20, model, args.save_path, 'model')
    lr_s = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 60, eta_min=1e-5)
    setup_seed(args.seed)
    if args.data_y_mode in ['trx_direction', 'trx_or_not']:
        loss_fn = nn.CrossEntropyLoss()
        metrics = {
            'precision': torchmetrics.classification.MulticlassPrecision(2, average='macro').to(args.device),
            'recall': torchmetrics.classification.MulticlassRecall(2, average='macro').to(args.device),
            'f1Score': torchmetrics.classification.MulticlassF1Score(2, average='macro').to(args.device),
            'roauc': torchmetrics.classification.MulticlassAUROC(2, average='macro').to(args.device),
            'prauc': torchmetrics.classification.MulticlassAveragePrecision(2, average='macro').to(args.device),
            'acc': torchmetrics.classification.MulticlassAccuracy(2, average='macro').to(args.device),
        }
    elif args.data_y_mode == 'both':
        loss_fn = nn.CrossEntropyLoss()
        metrics = {
            'precision': torchmetrics.classification.MulticlassPrecision(3, average='macro').to(args.device),
            'recall': torchmetrics.classification.MulticlassRecall(3, average='macro').to(args.device),
            'f1Score': torchmetrics.classification.MulticlassF1Score(3, average='macro').to(args.device),
            'roauc': torchmetrics.classification.MulticlassAUROC(3, average='macro').to(args.device),
            'prauc': torchmetrics.classification.MulticlassAveragePrecision(3).to(args.device),
        }
    elif args.data_y_mode == 'trx_vol':
        loss_fn = torch.nn.MSELoss()
        metrics = {}

    return model, loss_fn, opt, early_stop_s, lr_s, metrics


def train_model(modelname, debug = False, print_alpha = False):

    print('modelname', modelname)
    print('start')
    args.loss_figure.add_model(modelname)
    train_dataloader, val_dataloader, test_dataloader = get_data()
    model, loss_fn, opt, early_stop_s, lr_s, metrics = get_model()
    result = train(model, 
      train_dataloader, 
      opt, 
      loss_fn, 
      args, 
      val_dataloader = val_dataloader, 
      test_dataloader = test_dataloader, 
      lr_s = lr_s, 
      early_stop_s = early_stop_s, 
      metrics = metrics,
      debug = debug,
      loss_figure = args.loss_figure,
      print_alpha = print_alpha
    )
    
    if not debug:
        update_result_table(modelname, result)
        return result


if __name__ == '__main__':

    ########################################
    '''
    args.model = 'gru'    
    for args.data_y_mode in ['trx_vol']:
        train_model('gru')
    '''
    args.model = 'PatchTST'    
    for args.data_y_mode in ['trx_or_not']:
        train_model('PatchTST')
    '''
    args.model = 'FEDformer'    
    for args.data_y_mode in ['trx_or_not']:
        train_model('FEDformer')
    '''
    '''
    args.model = 'iTransformer'    
    for args.data_y_mode in ['trx_or_not']:
        train_model('iTransformer')
        
    args.model = 'lstm'    
    for args.data_y_mode in ['trx_or_not']:
        train_model('lstm')

    args.model = 'deeplob'
    for args.data_y_mode in ['trx_or_not']:
        train_model('deeplob')
    
    args.model = 'resnet'
    for args.data_y_mode in ['trx_or_not']:
        train_model('resnet')

    args.model = 'tsnet'
    for args.data_y_mode in ['trx_or_not']:
        train_model('tsnet')

    args.model = 'casvit'
    for args.data_y_mode in ['trx_or_not']:
        train_model('casvit')
    
    args.model = 'cmt'
    for args.data_y_mode in ['trx_or_not']:
        train_model('cmt')
    '''
    ########################################
    '''
    args.model = 'node'
    for args.data_y_mode in ['trx_or_not']:
        train_model('node')
    '''
    '''
    args.model = 'grud'
    for args.data_y_mode in ['trx_or_not']:
        train_model('grud')
    
    ########################################

    args.model = 'tsmixer'
    for args.data_y_mode in ['trx_or_not']:
        train_model('tsmixer')

    args.model = 'stockmixer'
    for args.data_y_mode in ['trx_or_not']:
        train_model('stockmixer')
    '''
    ########################################

    args.loss_figure.draw_all()


