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


    pred_fam = [1, 28,   29]

    df_1 = pd.read_csv('./data/family/5srrna.csv')
    df_28 = pd.read_csv('./data/family/group_i_intron.csv')
    df_29 = pd.read_csv('./data/family/circularly_permuted_group_ii_intron.csv')

    df = pd.concat([df_1, df_28, df_29], axis=0) 

    test_loaders = []
    for fam in pred_fam :
        df_1 = df[(df['family'] == fam) & (df['label'] == 1)].reset_index(drop=True)
        test_dataset1 = Complex_Dataset(df_1,struc_label=1)
        df_2 = df[(df['family'] == fam) & (df['label'] == 2)].reset_index(drop=True)
        test_dataset2 = Complex_Dataset(df_2,struc_label=2)
        df_0 = df[(df['family'] == fam) & (df['label'] == 0)].reset_index(drop=True)
        test_dataset3 = Complex_Dataset(df_0,struc_label=0)
        test_dataset = ConcatDataset([test_dataset1,test_dataset2,test_dataset3])
        test_loaders.append(DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                                  collate_fn=collate_fn))


    model = accelerator.prepare(model)

    all_outputs = []
    all_labels = []

    if accelerator.is_main_process:
        w = 0.6
        model.eval()
        with torch.no_grad():
            with open(f'benchmark_rank_result.csv','w') as f :
                f.write('family,ranking_score,label\n')
                for family, test_loader in zip(pred_fam,test_loaders):
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
                    ranking_score = all_outputs[:, 1] + w * all_outputs[:, 2]
                    for i in range(len(ranking_score)):
                        f.write(f'{family},{ranking_score[i]},{all_labels[i][0]}\n')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    

    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--model_scale', type=str, default='lx_clip')
    parser.add_argument('--trained_weight',type=str)
    parser.add_argument('--model_type', type=str, default='encoder')
    parser.add_argument('--tok_mode', type=str, default='char')
    parser.add_argument('--is_freeze', type=bool, default=False)

    
    args = parser.parse_args()

    extractor, tokenizer = get_extractor(args)

    main(args)
