from transformers.models.esm.modeling_esm import *
from collections import OrderedDict
import torch
from torch import nn
from typing import Type, Any, Callable, Union, List, Optional
from torch import Tensor
import os
import numpy as np
from torch.utils.data import Dataset
import inspect
from torch import Tensor
import torch.nn.functional as F
from typing import Type, Any, Callable, Union, List, Optional
from scipy.ndimage import convolve


class Complex_Dataset(Dataset):
    def __init__(self, df,struc_label=-1,max_len=1024):

        df = df[df['seq'].apply(len) <= max_len]
        # df = df[df['seq'].apply(len) >= 70]
        self.df = df
        self.df.reset_index(drop=True, inplace=True)
        self.struc_label = struc_label
        print(f'len of dataset: {len(self.df)}')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        seq = row['seq']
        structure_label = self.struc_label 
        seq = seq.upper().replace('U', 'T')

        return seq,  structure_label 


class SS_predictor(nn.Module):
    def __init__(self, extractor, model_config, is_freeze=False,dropout=0.3): # nub_family is fam + 1
        super(SS_predictor, self).__init__()

        self.freeze = is_freeze
        self.extractor = extractor
        feat_num = model_config.hidden_size

        if self.freeze:
            for param in self.extractor.parameters():
                param.requires_grad = False

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(feat_num, feat_num)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(feat_num,3)

    def forward(self, data_dict):
        input_ids, attention_mask = data_dict['input_ids'], data_dict['attention_mask']
        output = self.extractor(input_ids, attn_mask=attention_mask)
        pool_feat = output[1]
        pool_feat = torch.mean(pool_feat,dim=1)
        output = self.dropout(pool_feat)
        output = self.fc(output)
        output = self.relu(output)
        output1 = self.fc2(output)

        return output1

