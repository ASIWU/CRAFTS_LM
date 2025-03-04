import torch
from crafts.llama2 import load_model, load_model_infer
from transformers import EsmTokenizer, EsmModel
from crafts.llama2_t5 import load_encoder_model
from crafts.llama2 import Transformer
from crafts.llama2 import ModelArgs , MoEArgs

def get_extractor(args):

    extractor = load_model_infer()

    if args.tok_mode == 'char' :
        tokenizer = EsmTokenizer.from_pretrained("./utils/vocab_esm_mars.txt")
    elif args.tok_mode == 'MarsTok' :
        from crafts.dataset.mars_new.tokenizer import MarsTokenizer
        tokenizer = MarsTokenizer()
    else :
        raise NotImplemented
    return extractor,tokenizer

def get_model_args(model_size, model_type, vocab_size, pretraine_mode='MLM', dropout = 0.0):
    """Get model args for a given model size and transformer type"""

    multiple_of = 32

    if model_size == "s":
        dim, n_layers, n_heads = 288,6,6
    elif model_size == "m":
        dim, n_layers, n_heads = 512,8,8
    elif model_size == "l":
        dim, n_layers, n_heads = 768,12,12
    elif model_size == "lx":
        dim, n_layers, n_heads = (768+288),12,12
    elif model_size == "lxx":
        dim, n_layers, n_heads = (1280),33,20
    elif model_size == "lxxx":
        dim, n_layers, n_heads = int(1280*1.5), 33, 20
    elif model_size == "es":
        dim, n_layers, n_heads = 320,6,20
    elif model_size == "eg":
        dim, n_layers, n_heads = 480,12,20
    elif model_size == "elx":
        dim, n_layers, n_heads = 640,30,20
    else:
        raise ValueError("Unknown model size")

    # model init
    model_args = dict(
        hidden_size=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        dropout=dropout,
        is_decoder=True if model_type == 'decoder' else False,
        pretrain_mode=pretraine_mode,
    )   # start with model_args from command line

    class MyObject:
        def __init__(self, dictionary):
            for key, value in dictionary.items():
                setattr(self, key, value)

    model_config = MyObject(model_args)

    return model_config

