import warnings
warnings.filterwarnings("ignore")
import os
import sys 
sys.path.insert(0, sys.path[0]+"/../")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,  random_split,ConcatDataset
import random
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from sklearn.metrics import (
    roc_auc_score, 
)
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from utils.lm import get_extractor,get_model_args
from utils.predictor import Complex_Dataset, SS_predictor
from transformers import EsmTokenizer
import torch.nn.functional as F

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
set_seed(42)


def collate_fn(batch):
    seqs, stru_lbs= zip(*batch)
    max_len = max([len(seq)+2 for seq in seqs])

    if isinstance(tokenizer, EsmTokenizer):
        data_dict = tokenizer.batch_encode_plus(seqs,
                                                padding='max_length',
                                                max_length=max_len,
                                                truncation=True,
                                                return_tensors='pt')

    data_dict['structure_label'] = torch.FloatTensor(stru_lbs).unsqueeze(-1)

    return data_dict


def main(args) :

    kwargs_handlers = [DistributedDataParallelKwargs(find_unused_parameters=True)]
    accelerator = Accelerator(kwargs_handlers=kwargs_handlers)

    model_config = get_model_args(args.model_scale.split('_')[0], 'encoder', tokenizer.vocab_size)
    model = SS_predictor(extractor, model_config, is_freeze=args.is_freeze)

    if args.trained_weight != None :
        pretrained_dict = torch.load(args.trained_weight+'/pytorch_model.bin')
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)    

    df_test_true = pd.read_csv('./data/benchmark/benchmark_positive.csv')
    df_test_true = df_test_true[df_test_true['type'] == 'test'].reset_index(drop=True)
    df_test_false = pd.read_csv('./data/benchmark/benchmark_mrna.csv') 
    df_test_false = df_test_false[df_test_false['type'] == 'test'].reset_index(drop=True)
    test_dataset1 = Complex_Dataset(df_test_true,struc_label=1)
    test_dataset2 = Complex_Dataset(df_test_false,struc_label=0)


    test_dataset = ConcatDataset([test_dataset1,test_dataset2])
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              collate_fn=collate_fn)

    model = accelerator.prepare(model)

    all_outputs = []
    all_labels = []

    if accelerator.is_main_process:
        w = 0.6
        model.eval()
        with torch.no_grad():

            all_outputs = []
            all_labels = []
            for data_dict in test_loader:
                for key in data_dict:
                    data_dict[key] = data_dict[key].to(accelerator.device)
                
                output = model(data_dict)
                probs = torch.softmax(output, dim=1)
                all_outputs.extend(probs.cpu().numpy())
                all_labels.extend(data_dict['structure_label'].cpu().numpy())

            all_outputs = np.array(all_outputs)
            all_labels = np.array(all_labels)
            combined_scores = all_outputs[:, 1] + w * all_outputs[:, 2]
            rank_auroc = roc_auc_score(all_labels, combined_scores)
            print(f'Rank AUROC: {rank_auroc}')

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_scale', type=str, default='lx_clip')
    parser.add_argument('--trained_weight',type=str)
    parser.add_argument('--model_type', type=str, default='encoder')
    parser.add_argument('--tok_mode', type=str, default='char')
    parser.add_argument('--is_freeze', type=bool, default=False)

    args = parser.parse_args()

    extractor, tokenizer = get_extractor(args)

    main(args)
