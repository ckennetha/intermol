import argparse
import os, re
import h5py
import json
import numpy as np

from collections import Counter
from msgpack import Packer
from tqdm import tqdm
from typing import Optional

from ..main.utils import load_hf_model
from .label_utils import map_atom_token_idx

def describe(
    dataset_pth: str, acts_h5_pth: str, output_fp_dataset: bool=False,
    threshold: Optional[float]=0.0, out_prefix: Optional[str]=None, outdir_pth: Optional[str]=None
) -> None:
    # validate outdir_pth
    if outdir_pth:
        if not os.path.exists(outdir_pth):
            os.mkdir(outdir_pth)
    else:
        outdir_pth = '.'
        print("Warning: outdir_pth is not provided. "
                "All outputs will be saved in the current directory.")

    # dataset
    with open(dataset_pth, 'r') as h:
        samples = [sample.rstrip('\n') for sample in h.readlines()]

    # precomputed activations
    with h5py.File(acts_h5_pth, 'r') as acts_f:
        # sanity check
        n_samples = len(samples)
        if n_samples != acts_f.attrs["num_samples"]:
            raise ValueError(f"Shape Mismatch: Total samples ({n_samples}) does not match "
                                f"the precomputed activations ({acts_f.attrs['num_samples']}).")
        
        chunks = sorted(
            [c_name for c_name in acts_f.keys()], key=lambda gn: int(re.search(r'\d+', gn).group())
        )
        n_chunks = acts_f.attrs["num_chunks"]
        n_features = acts_f.attrs["num_features"]

        # output
        counts = {str(f): {
            "activatingMolecule": 0,
            "tokenType": Counter(), #Atom, Bond, Branches, Rings, or Disconnections
            "activatingToken": Counter(), #Tokens
            "patternActivation": Counter() #Single or Multiple
        } for f in list(range(n_features))}
        
        if output_fp_dataset:
            packer = Packer()
            out_fp = open(
                os.path.join(outdir_pth, f"{out_prefix + '_' if out_prefix else ''}{'thr' + str(threshold) + '_' if threshold > 0.0 else ''}fp_dataset.msgpack"),
                "wb"
            )

        # tokenizer
        tokenizer = load_hf_model("ibm-research/MoLFormer-XL-both-10pct", tokenizer_only=True)

        curr_smi = 0
        pbar = tqdm(total=n_chunks)
        for cn in chunks:
            g = acts_f[cn]
            n_molptr = len(g['molptr'])
            indptr = g['indptr'][:]
            indices = g['indices'][:]
            data = g['data'][:]

            if output_fp_dataset:
                fps = {}

            curr_indptr = 0
            curr_data = 0
            for _ in tqdm(range(n_molptr), leave=False):
                smi = samples[curr_smi]
                tokens = tokenizer.tokenize(smi)

                end_indptr = curr_indptr + n_features + 1
                mol_indptr = indptr[curr_indptr:end_indptr]
                
                end_data = curr_data + mol_indptr[-1]
                mol_indices = indices[curr_data:end_data]
                if threshold > 0.0:
                    mol_data = data[curr_data:end_data]
                    mol_mask = mol_data >= threshold

                    # thresh indices
                    mol_indices = mol_indices[mol_mask]

                    # thresh indptr
                    col_count = np.diff(mol_indptr)
                    cols = np.repeat(np.arange(n_features), col_count)
                    mol_cols = cols[mol_mask]
                    mol_counts = np.bincount(mol_cols, minlength=n_features)
                    
                    mol_indptr  = np.empty(n_features+1, dtype=int)
                    mol_indptr[0] = 0
                    np.cumsum(mol_counts, out=mol_indptr[1:])

                if output_fp_dataset:
                    mapper = map_atom_token_idx(tokens)
                    inv_mapper = {tok_idx: at_idx for at_idx, tok_idx in mapper.items()}
                    atom_idxs = []

                nz_feats = np.where(np.diff(mol_indptr) > 0)[0]
                for nzf in nz_feats:
                    f = str(nzf)
                    nz_indices = mol_indices[mol_indptr[nzf]:mol_indptr[nzf+1]]
                    
                    counts[f]["activatingMolecule"] += 1
                    
                    for tok_idx in nz_indices:
                        counts[f]["activatingToken"][tokens[tok_idx]] += 1
                    
                    if len(nz_indices) == 1:
                        counts[f]["patternActivation"]["single"] += 1
                    else:
                        counts[f]["patternActivation"]["multiple"] += 1

                    if output_fp_dataset:
                        atom_idx = [inv_mapper[tok_idx] for tok_idx in nz_indices if tok_idx in inv_mapper]
                        atom_idxs.append(atom_idx)

                if output_fp_dataset and atom_idxs:
                    fps[smi] = {"features": nz_feats.tolist(), "atom_idxs": atom_idxs}

                curr_indptr = end_indptr
                curr_data = end_data
                curr_smi += 1

            if output_fp_dataset:
                out_fp.write(packer.pack(fps))

            pbar.update()
    
    if output_fp_dataset:
        out_fp.close()

    # counts to dict
    for f in tqdm(counts, desc="Generating feature statistics..."):
        for token, n_tokens in counts[f]["activatingToken"].items():
            if token in {"(", ")"}:
                counts[f]["tokenType"]["branch"] += n_tokens
            elif re.fullmatch(r'[\-=#\\\/]', token):
                counts[f]["tokenType"]["bond"] += n_tokens
            elif token == ".":
                counts[f]["tokenType"]["discon"] += n_tokens
            elif re.fullmatch(r'(\%[0-9]{2}|[0-9])', token):
                counts[f]["tokenType"]["ring"] += n_tokens
            else:
                counts[f]["tokenType"]["atom"] += n_tokens

        for meta in counts[f]:
            if meta == "activatingMolecule":
                continue

            counts[f][meta] = dict(counts[f][meta])

    with open(os.path.join(
        outdir_pth, f"{out_prefix + '_' if out_prefix else ''}{'thr' + str(threshold) + '_' if threshold > 0.0 else ''}stats.json"
    ), "w") as h:
        json.dump(counts, h)


def main():
    parser = argparse.ArgumentParser(
        description='''
            Generate a description and statistics on activation patterns across given samples and their
            procomputed activations in JSON format. The output includes tokenType, activatingToken, patternActivation,
            and activatingMolecule. Additionally, if `output_fp_dataset` is set, a dataset
            for ConceptFromFingerprint will also be generated.
        '''
        )
    
    parser.add_argument('--dataset', type=str, required=True, help='Path to Smiles dataset. Supported ext.: .txt.')
    parser.add_argument('--activations', type=str, default=None, help='Path to precomputed dataset activations. '
                        'Supported ext.: .h5.')
    parser.add_argument('--output_fp_dataset', action='store_true', help='If set, also process data to generate a '
                        'dataset for ConceptFromFingerprint. Default: False.')
    parser.add_argument('--act_thresh', type=float, default=0.0, help='Minimum activation of selected tokens. Default: 0.0.')
    parser.add_argument('--out_prefix', type=str, default=None, help='Output file prefix. Default: None.')
    parser.add_argument('--outdir', type=str, help='Output directory. Default: current directory.')

    args = parser.parse_args()

    describe(
        dataset_pth=args.dataset, acts_h5_pth=args.activations, output_fp_dataset=args.output_fp_dataset,
        threshold=args.act_thresh, out_prefix=args.out_prefix, outdir_pth=args.outdir
    )


if __name__ == '__main__':
    main()