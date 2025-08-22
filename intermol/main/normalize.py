"""
This code is adapted from:
    1) https://github.com/ElanaPearl/InterPLM/blob/main/interplm/sae/normalize.py,
       as it appears in 'InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders' (Simon & Zou 2025),
       https://doi.org/10.1101/2024.11.14.623630
"""

import os
import math
import torch
import random
import argparse
import numpy as np
import polars as pl

from tqdm import tqdm
from typing import Optional

from sae import SparseAutoencoder
from utils import load_model, load_hf_model

@torch.no_grad()
def calc_feats_stats(
    sae: torch.nn.Module,
    mol_embs: torch.Tensor
) -> torch.Tensor:
    acts, _, _ = sae.encode(mol_embs)
    max_acts, _ = torch.max(acts, dim=0)
    return max_acts

@torch.no_grad()
def normalize(
    sae: torch.nn.Module, max_per_feat: torch.Tensor
) -> torch.nn.Module:
    sae.w_enc.div_(max_per_feat.unsqueeze(0))
    sae.w_enc[sae.w_enc.isinf()] = 0
    
    sae.b_enc.div_(max_per_feat)
    sae.b_enc[sae.b_enc.isinf()] = 0

    sae.w_dec.mul_(max_per_feat.unsqueeze(1))
    return sae

@torch.no_grad()
def normalize_sae(
    dataset: list[str],
    sae_exp_f: int,
    sae_k: int,
    sae_ckpt_pth: str,
    layer_idx: int,
    chunk_size: int=1000,
    base: str='ibm/MoLFormer-XL-both-10pct',
    outdir_pth: Optional[str]=None
) -> None:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # init model
    print("Loading base model...")
    tokenizer, base_model = load_hf_model(base)
    base_model.to(device)
    
    print("Loading SAE model...")
    sae = SparseAutoencoder(
        exp_f=sae_exp_f,
        k=sae_k
    )
    sae = load_model(sae, sae_ckpt_pth)
    sae.to(device)

    # validate outdir_pth
    if outdir_pth:
        if not os.path.exists(outdir_pth):
            os.mkdir(outdir_pth)
    else:
        outdir_pth = '.'
        print('Warning: outdir_pth is not provided. All outputs will be saved in the current directory.')

    n_samples = len(dataset)
    n_chunks = math.ceil(n_samples / chunk_size)
    n_features = sae.hidden_dim

    # process chunks
    n_tokens = 0
    max_per_feat = torch.zeros(n_chunks, n_features, device=device)
    for smi_start in tqdm(range(0, n_samples, chunk_size), desc='Process chunks...'):
        smi_end = smi_start + chunk_size

        smi_c = dataset[smi_start:smi_end]
        embs_c = []
        for s in tqdm(smi_c, desc='Process SMILES...', leave=False):
            enc = tokenizer(s, return_tensors='pt').to(device)
            outs = (
                base_model(**enc, output_hidden_states=True)
                .hidden_states[layer_idx]
                .squeeze()
                )
            embs_c.append(outs)

        mol_embs = torch.vstack(embs_c)
        max_acts = calc_feats_stats(sae, mol_embs)

        max_per_feat[int(smi_start / chunk_size), :] = max_acts
        n_tokens += mol_embs.shape[0]

        del embs_c, mol_embs, max_acts
        torch.cuda.empty_cache()

    # aggregate chunks
    max_per_feat, _ = torch.max(max_per_feat, dim=0)

    # save as npy
    out_fn = os.path.join(
        outdir_pth,
        f'norm-sae_max_per_feature_{n_samples}s_{n_tokens}t.npy'
        )
    np.save(out_fn, max_per_feat.cpu().numpy())
    print(f'Max activations per feature saved to {out_fn}.')

    print("Normalized SAE model...")
    sae_normalized = normalize(sae, max_per_feat)
    
    out_sae_fn = os.path.join(
        outdir_pth,
        f'norm-sae_state-dict_{n_samples}s_{n_tokens}t.pt'
        )
    torch.save(sae_normalized.state_dict(), out_sae_fn)
    print(f"Normalized model saved to {out_sae_fn}")

def main():
    def dsz_type(value):
        try:
            value = float(value)
            if 0.0 < value <= 1.0:
                return value
            else:
                return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f'Invalid data_size value: {value}. Must be float [0.0 (excl.), 1.0] or integer.'
            )

    parser = argparse.ArgumentParser(
        description='''
        Normalize SAE model features based on maximum activation values, adjusting the model
        weights to maintain the same reconstructions while ensuring that the maximum
        activation value for each feature is 1 across the provided dataset
        (InterPLM, Simon & Zou 2025).
        '''
        )

    parser.add_argument('--dataset', type=str, required=True, help='Path to SMILES dataset. Supported ext.: PARQUET.')
    parser.add_argument('--col_sele', type=str, required=True, help='Column name for SMILES data.')
    parser.add_argument('--sae_exp_f', type=int, required=True, help='SAE expansion factor.')
    parser.add_argument('--sae_k', type=int, required=True, help='SAE K in top-K activations calculation.')
    parser.add_argument('--sae_ckpt', type=str, required=True, help='Path to SAE checkpoint.')
    parser.add_argument('--layer_idx', type=int, required=True, help='Index of investigated base model layer.')
    parser.add_argument('--data_size', type=dsz_type, default=0.1, help='Size of dataset used to normalize the SAE.')
    parser.add_argument('--chunk_size', type=int, default=1024, help='Chunk size to process per iteration. Default: 1024.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for random sampling. Default: 42.')
    parser.add_argument('--outdir', type=str, help='Output directory. Default: current directory.')
    
    args = parser.parse_args()

    # set up data
    random.seed(args.seed)
    data = pl.read_parquet(args.dataset)
    smiles = data[args.col_sele].to_list()
    random.shuffle(smiles)

    if isinstance(args.data_size, float):
        samples = smiles[:math.ceil(args.data_size * len(smiles)):]
    else:
        samples = smiles[:args.data_size]

    # normalize sae
    normalize_sae(
        dataset=samples,
        sae_exp_f=args.sae_exp_f,
        sae_k=args.sae_k,
        sae_ckpt_pth=args.sae_ckpt,
        layer_idx=args.layer_idx,
        chunk_size=args.chunk_size,
        outdir_pth=args.outdir
    )

    # save samples
    out_fn_samples = os.path.join(
        args.outdir, f'samples_s{args.seed}_{len(samples)}.txt'
    )
    with open(out_fn_samples, 'w') as h:
        h.writelines('\n'.join(samples))


if __name__ == '__main__':
    main()