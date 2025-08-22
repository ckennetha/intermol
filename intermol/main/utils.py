"""
This code is adapted from: https://github.com/etowahadams/interprot/blob/main/interprot/utils.py,
as it appears in 'From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models' (Adams et al. 2025),
https://doi.org/10.1101/2025.02.06.636901
"""

import torch
import torch.nn.functional as F
import numpy as np
import polars as pl

from collections import OrderedDict
from transformers import AutoModelForMaskedLM, AutoTokenizer

# load model
@torch.no_grad()
def load_model(
    model: torch.nn.Module, ckpt_pth: str, mode: str='eval'
) -> torch.nn.Module:
    weights = torch.load(ckpt_pth, map_location='cpu', weights_only=True)
    if not isinstance(weights, OrderedDict):
        weights = weights['state_dict']
    
    if '.' in next(iter(weights.keys())):
        weights = {k.split('.', 1)[1]: v for k, v in weights.items()}

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for name, param in model.named_parameters():
        if name in weights:
            param.copy_(weights[name].to(device))
        else:
            print(f'Warning: {name} not found in ckpt.')

    if mode == 'eval':
        model.eval()
        return model
    elif mode == 'train':
        return model
    else:
        raise ValueError(
            f'{mode} mode is not supported. Currently supported modes: '
            '"eval" (default) and "train".'
        )

# load Hugging Face model
def load_hf_model(
    base_pth: str, tokenizer_only: bool=False
) -> tuple: 
    tokenizer = AutoTokenizer.from_pretrained(
        base_pth, trust_remote_code=True
    )
    if tokenizer_only:
        return tokenizer
    
    model = AutoModelForMaskedLM.from_pretrained(
        base_pth, deterministic_eval=True, trust_remote_code=True
        )
    for param in model.parameters():
        param.requires_grad = False

    return tokenizer, model

# cross entropy difference
def diff_cross_entropy(
    ori_logits: torch.Tensor,
    recons_logits: torch.Tensor,
    tokens: torch.Tensor
) -> float:
    ori_logits = ori_logits.view(-1, ori_logits.shape[-1])
    recons_logits = recons_logits.view(-1, ori_logits.shape[-1])
    tokens = tokens.view(-1)

    ori_loss = F.cross_entropy(ori_logits, tokens).mean().item()
    recons_loss = F.cross_entropy(recons_logits, tokens).mean().item()
    return recons_loss - ori_loss

# train-val-test split
def train_val_test_split(
    df: pl.DataFrame, train_size: float=0.80, seed: int=None
) -> tuple[pl.Series, pl.Series, pl.Series]:
    if seed:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    is_train = pl.Series(
        rng.choice(
            [True, False],
            size=len(df),
            replace=True,
            p=[train_size, 1 - train_size]
            )
        )
    train_set, val_test_set = df.filter(is_train), df.filter(~is_train)

    is_val = pl.Series(
        rng.choice(
            [True, False],
            size=len(val_test_set),
            replace=True,
            p=[0.5, 0.5]
            )
        )
    val_set, test_set = val_test_set.filter(is_val), val_test_set.filter(~is_val)

    return train_set, val_set, test_set