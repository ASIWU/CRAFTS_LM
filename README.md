# CRAFTS: Contrastive RNA Learning for Structure Screening

## Overview

CRAFTS (Contrastive RNA learning For sTructure Screening) is a deep learning tool designed for retrieving the likelihood of RNA sequences forming tertiary structures within their respective RNA families. Utilizing a pretrained language model based on contrastive learning, CRAFTS aims to extract relevant structural features from RNA sequences, which could be used in complementary to comparative genomics analysis to find sequences of functional relevance and higher likelihood of forming tertiary structures within each RNA family prior to endeavor in structure determination and functional exploration. This potentially contributes to the continuous growth of not only the number of RNA structures but also RNA structure-function relationships. 

## Installation

Please follow the steps below to install CRAFTS:

1. Clone the repository:

    ```bash
    git clone https://github.com/ASIWU/crafts.git
    cd crafts
    ```

2. Create and activate a Conda environment, then install the required dependencies:

    ```bash
    conda create -n crafts python=3.10
    conda activate crafts
    pip install torch==2.0.1
    pip install -r requirements.txt
    ```

3. Download the pretrained model weights:

    Please download the pretrained model weights from the following link: [Google Drive](https://drive.google.com/drive/folders/1UZx6D5haM1bf7UEHs89j37LehazlmBUP?usp=sharing).

    After downloading, place the weight file in the `./checkpoint` directory.

## Inference

### Reproducing Testing Results

To reproduce the testing results shown in Figure S13C of the paper, run the following command:

```bash
# If you do not have a GPU device, you can use the ./config/cpu.yaml configuration
accelerate launch --config_file ./config/single_gpu.yaml ./test/benchmark_test.py --trained_weight ./checkpoint
```

### Reproducing Ranking Results
To reproduce the ranking results shown in Figure S13D of the paper, run the following command:

```bash
# benchmark_rank for 5SrRNA, Group I intron and CP Group II intron
accelerate launch --config_file ./config/single_gpu.yaml ./test/benchmark_rank.py --trained_weight ./checkpoint
# infer_rank for OLE, GOLLD, ARRPOF and ROOL
accelerate launch --config_file ./config/single_gpu.yaml ./test/infer_rank.py --trained_weight ./checkpoint
```
The results will be saved as `benchmark_rank_result.csv` and `infer_rank_result.csv`. 

## Dataset

All the sequences for testing and ranking can be found in the `./data` directory. 
