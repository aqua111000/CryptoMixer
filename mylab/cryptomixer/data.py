from torch.utils.data import Dataset
import numpy as np
import os
import torch
import torch.nn as nn
import random
import pandas as pd
from tqdm import tqdm


class CollateFunc:
    def __init__(self, args):
        self.data_y_mode = args.data_y_mode
        self.y_col = args.y_col

    def __call__(self, batch):

        # multiuser_x, xn, now_y

        multiuser_x = [i[0] for i in batch]
        now_xn = [i[1] for i in batch]   
        y = [i[2] for i in batch]
    
        max_p = max([i.shape[0] for i in multiuser_x])

        pad_mask = [
            np.pad(np.ones([i.shape[0],], dtype = 'int'), 
                   ((max_p - i.shape[0], 0),), 
                   'constant', 
                   constant_values=(0))
            for i in multiuser_x]
        
        multiuser_x = [np.pad(i, 
                           ((max_p - i.shape[0], 0), (0, 0), (0, 0)), 
                           'constant',  
                            constant_values=(0)) for i in multiuser_x]

        multiuser_x = np.stack(multiuser_x, axis = 0)
        multiuser_x_data = multiuser_x[:, :, :, :-2]
        multiuser_x_mask = multiuser_x[:, :, :, -1]
        xn = np.stack(now_xn, axis = 0)
        if self.data_y_mode == 'trx_vol':
            y = xn[:, self.y_col]
        else:
            y = np.stack(y, axis = 0)
        pad_mask = np.stack(pad_mask, axis = 0)
        
        x = torch.FloatTensor(multiuser_x)
        trx_mask = torch.IntTensor(multiuser_x_mask)
        pad_mask = torch.IntTensor(pad_mask)
        xn = torch.FloatTensor(xn)
        y = torch.FloatTensor(y)

        return  x, trx_mask, pad_mask, xn, y
