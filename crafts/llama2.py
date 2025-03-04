import os
import math
import inspect
from dataclasses import dataclass
from typing import Optional, Tuple
import torch
import torch.nn.functional as F
from torch import nn
from crafts.utils.attn_utils import get_extended_attention_mask
from torch.utils.checkpoint import checkpoint
import random

# # use flash attention or a manual implementation?
# FlashAttentionAvailable = False
FlashAttentionAvailable = True if hasattr(torch.nn.functional, 'scaled_dot_product_attention') else False
if not FlashAttentionAvailable:
    print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")

from lightning_utilities.core.imports import RequirementCache
FlashAttention2Available = RequirementCache("flash-attn>=2.0.0.post1")
if FlashAttention2Available:
    from flash_attn import flash_attn_func

attn_map = []

@dataclass
class MoEArgs:
    # -------- moe expert configs --------
    num_experts: int = 8
    num_selects: int = 2
    size_experts: int = None
    dropout: float = 0.
    # -------- moe gate configs --------
    gate_type: str = "TopKBalancedNoisyGate"
    gate_network: str = "mlp"
    gate_use_softmax: bool = True
    gate_use_balance: bool = True
    gate_balance_loss_weight: float = 1e-2
    gate_add_noise: bool = True
    gate_noise_epsilon: float = 1e-2
    # -------- moe calculator configs --------
    calculator_type: str = "UniversalCalculator"
    multiply_gate_scores: bool = True
    score_scale_factor: float = 1.0
    add_weight_norm: bool = False
    drop_tokens: bool = True
    dropped_padding: str = "zero"
    capacity_factor: float = 1.25

@dataclass
class ModelArgs:
    dim: int = 288
    n_layers: int = 6
    n_heads: int = 6
    n_kv_heads: int = 6
    vocab_size: int = 21   # defined later by tokenizer
    mask_token_id: int = None
    multiple_of: int = 32  # make SwiGLU hidden layer size multiple of large power of 2
    norm_eps: float = 1e-5
    max_seq_len: int = 10240
    dropout: float = 0.15
    layer_dropout: float = 0.0
    is_decoder: bool = True
    pretrain_mode: str = 'MLM'  # MLM|GLM|MLM+GLM|
    n_dec_layers: int = 1 # this is for encoder-decoder
    ckpt_inter: int = None
    moe_args: MoEArgs = None


def get_model_args(model_size, model_type, vocab_size, pretraine_mode='MLM', dropout = 0.0):
    """Get model args for a given model size and transformer type"""

    multiple_of = 32

    if model_size == "s":
        dim, n_layers, n_heads = 288,6,6
    elif model_size == "m":
        dim, n_layers, n_heads = 512,8,8
    elif model_size == "g":
        dim, n_layers, n_heads = 512,12,16
    elif model_size == "l":
        dim, n_layers, n_heads = 768,12,12
    elif model_size == "lx":
        dim, n_layers, n_heads = 1056,12,12
    elif model_size == "lxx":
        dim, n_layers, n_heads = 1280,33,20
    elif model_size == "lxxx":
        dim, n_layers, n_heads = 1920,33,20
    elif model_size == "es":
        dim, n_layers, n_heads = 320,6,20
    elif model_size == "eg":
        dim, n_layers, n_heads = 480,12,20
    elif model_size == "elx":
        dim, n_layers, n_heads = 640,30,20
    else:
        raise ValueError(f"Unknown model size {model_size}")

    # model init
    model_args = dict(
        dim=dim,
        n_layers=n_layers,
        n_heads=n_heads,
        n_kv_heads=n_heads,
        vocab_size=vocab_size,
        multiple_of=multiple_of,
        dropout=dropout,
        is_decoder=True if model_type == 'decoder' else False,
        pretrain_mode=pretraine_mode,
    )   # start with model_args from command line

    return model_args


def init_model(init_from, model_args, out_dir, device = 'cpu'):

    checkpoint = None
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    if init_from == "scratch":
        # init a new model from scratch
        print("Initializing a new model from scratch")
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        iter_num = 0
        best_val_loss = 1e9

    elif init_from == "resume":
        print(f"Resuming training from {out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(out_dir, "ckpt.pt")
        checkpoint = torch.load(ckpt_path, map_location=device)
        checkpoint_model_args = checkpoint["model_args"]
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ["dim", "n_layers", "n_heads", "n_kv_heads", "vocab_size", "multiple_of", "max_seq_len"]:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = ModelArgs(**model_args)
        model = Transformer(gptconf)
        state_dict = checkpoint["model"]
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = "_orig_mod."
        for k, v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix) :]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint["iter_num"]
        best_val_loss = checkpoint["best_val_loss"]

    else:
        raise NotImplementedError

    return model, iter_num, best_val_loss, checkpoint

def load_model_infer() :
    model_args = {'dim': 1056, 'n_layers': 12, 'n_heads': 12, 'n_kv_heads': 12, 'vocab_size': 20, 'multiple_of': 32, 'dropout': 0.15, 'is_decoder': False}
    gptconf = ModelArgs(**model_args)

    model = Transformer(gptconf)
    return model

def load_model(ckpt_path, device = 'cpu', dropout = None, layer_dropout = None, from_scratch = False,
               strict=True):
    # init from a model saved in a specific directory
    print(f"loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model_args = {}
    for k in ["dim",
              "n_layers",
              "n_heads",
              "n_kv_heads",
              "vocab_size",
              "multiple_of",
              "max_seq_len",
              'dropout',
              'layer_dropout',
              'is_decoder',
              ]:

        if k in checkpoint['model_args']:
            model_args[k] = checkpoint['model_args'][k]
    gptconf = ModelArgs(**model_args)

    if dropout is not None:
        gptconf.dropout = dropout
    if layer_dropout is not None:
        gptconf.layer_dropout = layer_dropout

    model = Transformer(gptconf)

    if from_scratch:
        print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')
        return model

    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')
    return model

def load_model_(model, ckpt_path, device = 'cpu'):
    print(f"loading model from {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict, strict=False)
    print(f'number of parameters: {sum(p.numel() for p in model.parameters())}')
    return model

class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cos = torch.cos(freqs)  # real part
    freqs_sin = torch.sin(freqs)  # imaginary part
    return freqs_cos, freqs_sin

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)

def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    freqs_cos: torch.Tensor,
    freqs_sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:

    # reshape xq and xk to match the complex representation
    xq_r, xq_i = xq.float().reshape(xq.shape[:-1] + (-1, 2)).unbind(-1)
    xk_r, xk_i = xk.float().reshape(xk.shape[:-1] + (-1, 2)).unbind(-1)

    # reshape freqs_cos and freqs_sin for broadcasting
    freqs_cos = reshape_for_broadcast(freqs_cos, xq_r)
    freqs_sin = reshape_for_broadcast(freqs_sin, xq_r)

    # apply rotation using real numbers
    xq_out_r = xq_r * freqs_cos - xq_i * freqs_sin
    xq_out_i = xq_r * freqs_sin + xq_i * freqs_cos
    xk_out_r = xk_r * freqs_cos - xk_i * freqs_sin
    xk_out_i = xk_r * freqs_sin + xk_i * freqs_cos

    # flatten last two dimensions
    xq_out = torch.stack([xq_out_r, xq_out_i], dim=-1).flatten(3)
    xk_out = torch.stack([xk_out_r, xk_out_i], dim=-1).flatten(3)

    return xq_out.type_as(xq), xk_out.type_as(xk)

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_kv_heads = args.n_heads if args.n_kv_heads is None else args.n_kv_heads
        model_parallel_size = 1
        self.n_local_heads = args.n_heads // model_parallel_size
        self.n_local_kv_heads = self.n_kv_heads // model_parallel_size
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = args.dim // args.n_heads
        self.wq = nn.Linear(args.dim, args.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wv = nn.Linear(args.dim, self.n_kv_heads * self.head_dim, bias=False)
        self.wo = nn.Linear(args.n_heads * self.head_dim, args.dim, bias=False)
        self.attn_dropout = nn.Dropout(args.dropout)
        self.resid_dropout = nn.Dropout(args.dropout)
        self.dropout = args.dropout


    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor,
        freqs_cos: torch.Tensor,
        freqs_sin: torch.Tensor,
        kv_cache = None,
        kv_state = None,
        is_causal = False,
    ):
        bsz, seqlen, _ = x.shape

        # QKV
        xq = self.wq(x)

        seqlen_kv = seqlen
        if kv_state is not None:
            xk, xv = self.wk(kv_state), self.wv(kv_state)
            seqlen_kv = kv_state.shape[1]
        else:
            xk, xv = self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        xk = xk.view(bsz, seqlen_kv, self.n_local_kv_heads, self.head_dim)
        xv = xv.view(bsz, seqlen_kv, self.n_local_kv_heads, self.head_dim)

        if kv_state is None:
            # RoPE relative positional embeddings
            xq, xk = apply_rotary_emb(xq, xk, freqs_cos, freqs_sin)

        # grouped multiquery attention: expand out keys and values
        xk = repeat_kv(xk, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)
        xv = repeat_kv(xv, self.n_rep)  # (bs, seqlen, n_local_heads, head_dim)

        # flash implementation
        if FlashAttention2Available and attn_mask is None:
            # flash attention2 official implementation
            output = flash_attn_func(xq, xk, xv, dropout_p=self.dropout if self.training else 0.0, causal=is_causal)

        elif FlashAttentionAvailable:
            # make heads into a batch dimension
            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            # we prefer usd pytorch2.0 interface
            output = torch.nn.functional.scaled_dot_product_attention(
                xq, xk, xv, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0, is_causal=is_causal)

        else:
            # make heads into a batch dimension
            xq = xq.transpose(1, 2)  # (bs, n_local_heads, seqlen, head_dim)
            xk = xk.transpose(1, 2)
            xv = xv.transpose(1, 2)

            # manual implementation
            scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)

            if attn_mask is not None:
                scores = scores + attn_mask

            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            scores = self.attn_dropout(scores)
            output = torch.matmul(scores, xv)  # (bs, n_local_heads, seqlen, head_dim)

            global attn_map
            attn_map.append(scores)

        # restore time as batch dimension and concat heads
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)

        # final projection into the residual stream
        output = self.wo(output)
        output = self.resid_dropout(output)

        return output, kv_cache


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int, dropout: float):
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.dim = args.dim
        self.head_dim = args.dim // args.n_heads
        self.attention = Attention(args) # this is self attention

        self.args = args
        self.cross_attention_norm = None
        self.cross_attention = None

        if args.is_decoder:
            self.cross_attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
            self.cross_attention = Attention(args)

        if args.moe_args is None:
            self.feed_forward = FeedForward(
                dim=args.dim,
                hidden_dim=4 * args.dim,
                multiple_of=args.multiple_of,
                dropout=args.dropout,
            )


        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)



    def forward(self, x, attn_mask, freqs_cos, freqs_sin, kv_cache = None, kv_state = None, is_casual = False):

        ax, kv_cache = self.attention.forward(self.attention_norm(x), attn_mask, freqs_cos, freqs_sin, kv_cache=kv_cache,
                                              is_causal=is_casual)
        h = x + ax

        if self.cross_attention is not None:
            assert kv_state is not None
            # attention mask is none for cross attention
            ax, _ = self.cross_attention.forward(self.cross_attention_norm(h), None, freqs_cos, freqs_sin,
                                                 kv_state=kv_state, is_causal=False,)
            h = h + ax

        ffn_out = self.feed_forward.forward(self.ffn_norm(h))


        out = h + ffn_out
        return out, kv_cache



class Transformer(nn.Module):
    last_loss: Optional[torch.Tensor]

    def __init__(self, params: ModelArgs):
        super().__init__()

        self.is_kv_cache = False
        self.is_decoder = params.is_decoder
        self.pretrain_mode = params.pretrain_mode

        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.n_layers

        self.tok_embeddings = nn.Embedding(params.vocab_size, params.dim)
        self.dropout = nn.Dropout(params.dropout)
        self.layers = torch.nn.ModuleList()

        for layer_id in range(params.n_layers):
            self.layers.append(TransformerBlock(layer_id, params))

        self.norm = RMSNorm(params.dim, eps=params.norm_eps)
        self.output = nn.Linear(params.dim, params.vocab_size, bias=False)

        # share the unembedding parameters with the embedding parameters
        self.tok_embeddings.weight = self.output.weight # https://paperswithcode.com/method/weight-tying

        # some useful precompute for the RoPE relative positional embeddings
        freqs_cos, freqs_sin = precompute_freqs_cis(self.params.dim // self.params.n_heads, self.params.max_seq_len)
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('w3.weight') or pn.endswith('wo.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * params.n_layers))

        # Initialize attribute for the loss of the last forward call. This will be set if the forward is called with a targets tensor.
        self.last_loss = None
        self.last_losses = {}

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    # input_ids: Optional[torch.LongTensor] = None,
    # attention_mask: Optional[torch.FloatTensor] = None,
    # decoder_input_ids: Optional[torch.Tensor] = None,
    # decoder_attention_mask: Optional[torch.BoolTensor] = None,
    def forward(self,
                tokens: torch.Tensor,
                attn_mask: torch.Tensor,
                decoder_input_ids: Optional[torch.Tensor] = None,
                decoder_attention_mask: Optional[torch.BoolTensor] = None,
                labels: Optional[torch.Tensor] = None,
                kv_caches = None,
                ) -> (torch.Tensor, torch.Tensor):
        '''
        :param tokens: (bsz, seqlen) LongTensor of input token indices
        :param attn_mask: (bsz, seqlen) padding mask for attention
        '''
        #! change by wuhao @ 20250115, for captum
        # _bsz, seqlen = tokens.shape
        _bsz, seqlen = tokens.shape[0],tokens.shape[1]

        h = self.tok_embeddings(tokens)

        # print(tokens.dtype, h.dtype)

        if kv_caches is None:
            # Initialize a list to hold layer-specific kv_caches
            kv_caches = [None] * self.n_layers

        if attn_mask is not None:
            h = (h * attn_mask.unsqueeze(-1)).to(h.dtype)

        h = self.dropout(h)

        freqs_cos = self.freqs_cos[:seqlen]
        freqs_sin = self.freqs_sin[:seqlen]

        # extend attention mask to get the score
        if attn_mask is not None:
            extended_attn_mask = get_extended_attention_mask(attn_mask,(_bsz, seqlen), h.device, h.dtype,
                                                             is_decoder=self.is_decoder)
        else:
            extended_attn_mask = None

        # for every ckpt_inter transformer layer, we do checkpoint
        ckpt_inter = 1 if self.params.ckpt_inter else 999

        balance_loss_all = 0.

        is_casual = self.is_decoder

        for c, layer in enumerate(self.layers):
            h_pre = h

            # Use the cache from the list
            current_cache = kv_caches[c] if self.is_kv_cache else None

            if (c+1) % ckpt_inter == 0:
                out = checkpoint(layer, h, extended_attn_mask, freqs_cos, freqs_sin, current_cache, is_casual,
                                 use_reentrant=False)
            else:
                out = layer(h, extended_attn_mask, freqs_cos, freqs_sin, current_cache, is_casual)

            if self.params.moe_args is None:
                h, new_cache = out
            else:
                h, new_cache, balance_loss = out
                balance_loss_all += balance_loss

            # Update the cache in the list
            kv_caches[c] = new_cache

            if self.training and random.random() < self.params.layer_dropout:
                # dirty imp for layer dropout
                h = h_pre + 0.0 * h

        h = self.norm(h)

        if labels is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.output(h)

            if self.pretrain_mode == 'GLM':
                # print("decoder GLM loss")
                logits = logits[attn_mask == 1]
                labels = labels[attn_mask == 1]
                self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
                self.last_losses['ce_loss'] = self.last_loss.item()

            elif self.pretrain_mode == 'MLM':
                # For Now, if self.is_decoder is False, then we are doing encoder-only pretraining
                # we only apply loss to the <mask> tokens
                # print("encoder MLM loss")
                logits = logits[tokens == self.params.mask_token_id]
                labels = labels[tokens == self.params.mask_token_id]
                self.last_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
                self.last_losses['ce_loss'] = self.last_loss.item()

            if balance_loss_all > 0:
                self.last_losses['balance_loss'] = balance_loss_all.item()
                self.last_loss += balance_loss_all

        else:
            # inference-time mini-optimization: only forward the output on the very last position
            logits = self.output(h[:, [-1], :]) # note: using list [-1] to preserve the time dim
            self.last_loss = None
            balance_loss_all = None

        if self.is_kv_cache:
            return logits, h, kv_caches
        else:
            return logits, h

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = sum(p.numel() for p in self.parameters())
        cfg = self.params
        L, H, Q, T = cfg.n_layers, cfg.n_heads, cfg.dim//cfg.n_heads, cfg.max_seq_len
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.inference_mode()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """

        assert self.is_decoder, "generation only supported for decoder models"

        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits = self(idx_cond, torch.ones_like(idx_cond).to(idx.device))[0]
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    @torch.inference_mode()
    def generate_kv_cache(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        Also note this is a super inefficient version of sampling with no key/value cache.
        """

        assert self.is_decoder, "generation only supported for decoder models"

        kv_cache = None
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.params.max_seq_len else idx[:, -self.params.max_seq_len:]
            # forward the model to get the logits for the index in the sequence
            logits, _, kv_cache = self(idx_cond, torch.ones_like(idx_cond).to(idx.device), kv_cache=kv_cache)
            logits = logits[:, -1, :] # crop to just the final time step
            if temperature == 0.0:
                # "sample" the single most likely index
                _, idx_next = torch.topk(logits, k=1, dim=-1)
            else:
                # pluck the logits at the final step and scale by desired temperature
                logits = logits / temperature
                # optionally crop the logits to only the top k options
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float('Inf')
                # apply softmax to convert logits to (normalized) probabilities
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx