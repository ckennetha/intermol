import os
import argparse
import h5py
import math
import torch
import numpy as np

from scipy.sparse import csc_matrix
from tqdm import tqdm
from typing import Callable

from ..main.inference import SAEInferenceModule

def get_clean_acts(smiles: str, sae_module: Callable) -> torch.Tensor:
    _, sae_acts = sae_module.get_all(smi=smiles)
    return sae_acts.squeeze()[1:-1, :]

def run(
    samples: list[str], sae_module: Callable, chunk_size: int=1024,
    outdir_pth: str='.', out_prefix: str=None
) -> None:
    n_samples = len(samples)
    n_features = sae_module.sae.hidden_dim
    n_chunks = math.ceil(n_samples / chunk_size)

    out_raw = h5py.File(
        os.path.join(outdir_pth, f'{out_prefix + "_" if out_prefix else ""}raw_acts.h5'), 'w'
    )

    out_raw.attrs['dtype'] = 'csc_matrix'
    out_raw.attrs['num_features'] = n_features
    out_raw.attrs['num_samples'] = n_samples
    out_raw.attrs['num_chunks'] = n_chunks
    
    pbar = tqdm(total=n_samples)
    for c, ns in enumerate(range(0, n_samples, chunk_size)):
        pbar.set_description(f'Process chunk {c+1}...')
        
        ns_end = min(ns + chunk_size, n_samples)
        n_sample = ns_end - ns

        g = out_raw.require_group(f'c{c}_s{n_sample}')

        chunk_token = [len(sae_module.tokenize(samples[ns_c])) for ns_c in range(ns, ns_end)]
        g.create_dataset('molptr', data=chunk_token, dtype=np.int32, compression='lzf')
        
        coll_data, coll_indices, coll_indptr = [], [], []
        for ns_c in range(ns, ns_end):
            sae_acts = get_clean_acts(samples[ns_c])
            sae_acts = csc_matrix(sae_acts.cpu().numpy())

            coll_data.append(sae_acts.data)
            coll_indices.append(sae_acts.indices)
            coll_indptr.append(sae_acts.indptr)

            pbar.update(1)

        g.create_dataset(
            'data', data=np.concatenate(coll_data), dtype=np.float32, compression='lzf'
        )
        g.create_dataset(
            'indices', data=np.concatenate(coll_indices), dtype=np.int32, compression='lzf'
        )
        g.create_dataset(
            'indptr', data=np.concatenate(coll_indptr), dtype=np.int32, compression='lzf'
        )

        for _, obj in g.items():
            if isinstance(obj, h5py.Dataset):
                obj.id.flush()

    out_raw.close()


def main():
    parser = argparse.ArgumentParser(
        description='Compute activations for a given set of molecules and save their sparse represntation in HDF5 format.'
    )
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to Smiles dataset. Supported ext.: .txt.')
    parser.add_argument('--sae_exp_f', type=int,  required=True, help='SAE expansion factor.')
    parser.add_argument('--sae_k', type=int,  required=True, help='SAE K in top-K activations calculation.')
    parser.add_argument('--sae_ckpt', type=str,  required=True, help='Path to SAE checkpoint.')
    parser.add_argument('--layer_idx', type=int,  required=True, help='Index of investigated base model layer.')
    parser.add_argument('--chunk_size', type=int, default=10240, help='Chunk size to process per iteration. Default: 10240.')
    parser.add_argument('--out_prefix', type=str, default=None, help='Output file prefix. Default: None.')
    parser.add_argument('--outdir', type=str, help='Output directory. Default: current directory.')

    args = parser.parse_args()

    # set up dataset
    print('Set up dataset...')
    with open(args.dataset, 'r') as h:
        smiles = [smi.rstrip('\n') for smi in h.readlines()]

    # init tokenizer and models
    print('Initialize tokenizer and models...')
    SAEInference = SAEInferenceModule(
        sae_weight=args.sae_ckpt, sae_exp_f=args.sae_exp_f, sae_k=args.sae_k,
        layer_idx=args.layer_idx, base='ibm/MoLFormer-XL-both-10pct'
    )

    run(
        samples=smiles,
        sae_module=SAEInference,
        chunk_size=args.chunk_size,
        outdir_pth=args.outdir,
        out_prefix=args.out_prefix
    )


if __name__ == '__main__':
    main()