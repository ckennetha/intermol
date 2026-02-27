import os
import json
import math
import tempfile

import h5py
import numpy as np
import polars as pl

from typing import Optional
from tqdm.auto import tqdm
from numba import njit, prange

from intermol.main.utils import load_model_from_HF
from intermol.interp.molecular_concepts import (
    _RX_ATOM, _RX_BOND, _RX_RING, _RX_BRANCH
)
from intermol.interp.utils import h5_chunk_sorter, fast_spearmanr

# nonzero density => num. of activated tokens / num. of tokens
def get_latent_nonzero_density(acts_h5_pth: str) -> pl.DataFrame:
    with h5py.File(acts_h5_pth, 'r') as h5f:
        chunks = h5_chunk_sorter(list(h5f.keys()))
        n_features = h5f.attrs["num_features"]
        n_samples = h5f.attrs["num_samples"]

        try:
            with tempfile.NamedTemporaryFile(delete=False) as ntf:
                tmp_name = ntf.name
                d_arr = np.memmap(
                    tmp_name,
                    mode='w+',
                    shape=(n_samples, n_features),
                    dtype=np.float32
                )

            curr_smi = 0
            for c in tqdm(chunks, desc="Computing latent density...", leave=False):
                g = h5f[c]

                molptr = g['molptr'][:]
                indptr = g['indptr'][:]

                curr_indptr = 0
                for nt in molptr:
                    e_indptr = curr_indptr + n_features + 1
                    mol_indptr = indptr[curr_indptr:e_indptr]

                    nz_cs = np.diff(mol_indptr)
                    nz_fs = np.flatnonzero(nz_cs)

                    nz_d = nz_cs / nt
                    d_arr[curr_smi, nz_fs] = nz_d[nz_fs]

                    curr_indptr = e_indptr
                    curr_smi += 1

            d_stats = []
            for f in tqdm(range(n_features), desc="Generating density statistics..."):
                col = d_arr[:, f]
                nz_col = col[np.flatnonzero(col)]
                d_stats.append({
                    "feature": f,
                    "n": len(nz_col),
                    "min": np.min(nz_col).item(),
                    "q1": np.quantile(nz_col, q=0.25).item(),
                    "median": np.median(nz_col).item(),
                    "q3": np.quantile(nz_col, q=0.75).item(),
                    "max": np.max(nz_col).item(),
                    "mean": np.mean(nz_col).item(),
                    "std": np.std(nz_col).item()
                })

        finally:
            d_arr._mmap.close()
            del d_arr
            os.unlink(tmp_name)

    return pl.DataFrame(d_stats)

# token preference => latent token activation preference
def get_latent_token_preference(
    acts_h5_pth: str,
    data_pth: str,
    model_name: str = 'ibm/MoLFormer-XL-both-10pct',
    inverse: bool = False, # if True, count non-activated tokens
    gtc_pth: Optional[str] = None # JSON file path for dataset total token count
) -> pl.DataFrame:
    # parse dataset
    with open(data_pth, 'r') as h:
        smiles = [smi.rstrip("\n") for smi in h.readlines()]

    # init tokenizer
    tokenizer = load_model_from_HF(model_name, tokenizer_only=True)

    with h5py.File(acts_h5_pth, 'r') as h5f:
        chunks = h5_chunk_sorter(list(h5f.keys()))
        n_features = h5f.attrs['num_features']
        n_samples = h5f.attrs['num_samples']

        # init counters
        ctr_mol = np.zeros(n_features, dtype=np.uint32)
        ctr_tk = np.zeros((n_features, tokenizer.vocab_size), dtype=np.uint32)

        curr_smi = 0
        with tqdm(total=n_samples, leave=False) as pbar:
            pbar.set_description("Count activated tokens...")
            for c in chunks:
                g = h5f[c]

                molptr = g['molptr'][:]
                indptr = g['indptr'][:]
                indices = g['indices'][:]

                curr_indptr = 0
                curr_indices = 0
                for nt in molptr:
                    smi = smiles[curr_smi]
                    tk_ids = np.array(
                        tokenizer.encode(smi, add_special_tokeens=False),
                        dtype=np.uint32
                    )

                    e_indptr = curr_indptr + n_features + 1
                    mol_indptr = indptr[curr_indptr:e_indptr]

                    e_indices = curr_indices + mol_indptr[-1]
                    mol_indices = indices[curr_indices:e_indices]

                    if inverse:
                        fast_counter_inverse(
                            ctr_mol, ctr_tk, tk_ids,
                            mol_indptr, mol_indices,
                            nt, n_features
                        )
                    else:
                        fast_counter(
                            ctr_mol, ctr_tk, tk_ids,
                            mol_indptr, mol_indices
                        )

                    curr_indptr = e_indptr
                    curr_indices = e_indices
                    curr_smi += 1

                    pbar.update()

    # gtc
    if gtc_pth is not None:
        with open(gtc_pth, 'r') as h:
            gtc = json.load(h)

    # token dominance
    ## func util
    def measure_ic(prop: list[float]):
        H, H_max = -sum(p * math.log(p) for p in prop), math.log(len(prop))
        U = 1 - (H / H_max)
        D = sum(p ** 2 for p in prop)
        return H, H_max, U, D

    # per-feature measures
    measures = []
    for f in tqdm(range(n_features), desc="Measuring token preference..."):
        tk_ids_f = ctr_tk[f, :]

        nz_tk_ids = np.nonzero(tk_ids_f)[0]
        nz_tks = tokenizer.convert_ids_to_tokens(nz_tk_ids)

        nz_cs = tk_ids_f[nz_tk_ids]
        nz_idx_argsort = np.argsort(-nz_cs)

        # token-level
        nz_tks = [nz_tks[i] for i in nz_idx_argsort]
        nz_cs = nz_cs[nz_idx_argsort].tolist()

        total = sum(nz_cs)
        prop = [cnt / total for cnt in nz_cs]

        _, _, U, D = measure_ic(prop)

        # group-level
        ctr_grp = {}
        for tk, tk_cnt in zip(nz_tks, nz_cs):
            if _RX_ATOM.search(tk):
                grp = "Atom"
            elif _RX_BRANCH.fullmatch(tk):
                grp = "Branch"
            elif _RX_RING.fullmatch(tk):
                grp = "Ring"
            elif _RX_BOND.fullmatch(tk):
                grp = "Bond"
            else:
                grp = "Disconnection"
            ctr_grp[grp] = ctr_grp.get(grp, 0) + tk_cnt
        ctr_grp = dict(sorted(
            ctr_grp.items(), key=lambda x: x[1], reverse=True
        ))

        grp_total = sum(ctr_grp.values())
        grp_prop = [grp_cnt / grp_total for grp_cnt in ctr_grp.values()]

        _, _, gU, gD = measure_ic(grp_prop)

        # gtc
        gtc_ratios = None
        if gtc_pth is not None:
            gtc_ratios = [tk_cnt / gtc[tk] for tk, tk_cnt in zip(nz_tks, nz_cs)]

        measures.append({
            "feature": f,
            "n_Mol": ctr_mol[f],
            "token": nz_tks,
            "token_Count": nz_cs,
            "token_Global_Ratio": gtc_ratios,
            "token_Simpson_D": D,
            "token_Pielou_E_r": U,
            "group": list(ctr_grp.keys()),
            "group_Count": list(ctr_grp.values()),
            "group_Simpson_D": gD,
            "group_Pielou_E_r": gU
        })

    return pl.DataFrame(measures)

# position latents
def get_positional_latents(
    acts_h5_pth: str, num_tokens_eval_threshold: int = 3
) -> pl.DataFrame:
    with h5py.File(acts_h5_pth, 'r') as h5f:
        chunks = h5_chunk_sorter(list(h5f.keys()))
        n_features = h5f.attrs['num_features']
        n_samples = h5f.attrs['num_samples']

        # init outres
        outres = {f: {"len": [], "coef": []} for f in range(n_features)}

        curr_smi = 0
        with tqdm(total=n_samples, leave=False) as pbar:
            for c in chunks:
                g = h5f[c]

                molptr = g['molptr'][:]
                indptr = g['indptr'][:]
                indices = g['indices'][:]
                data = g['data'][:]

                curr_indptr = 0
                curr_indices = 0
                for nt in molptr:
                    e_indptr = curr_indptr + n_features + 1
                    mol_indptr = indptr[curr_indptr:e_indptr]

                    e_indices = curr_indices + mol_indptr[-1]
                    mol_indices = indices[curr_indices:e_indices]
                    mol_data = data[curr_indices:e_indices]

                    eval_fs = np.flatnonzero(
                        np.diff(mol_indptr) >= num_tokens_eval_threshold
                    )

                    ss = mol_indptr[eval_fs]
                    es = mol_indptr[eval_fs+1]
                    for f, s, e in zip(eval_fs, ss, es):
                        nz_indices = mol_indices[s:e]
                        n_nz_indices = nz_indices.size

                        nz_data = mol_data[s:e]
                        r_pos = nz_indices / nt
                        corr = fast_spearmanr(r_pos, nz_data)

                        outres[f]['len'].append(n_nz_indices)
                        outres[f]['sample_idx'].append(curr_smi)
                        outres[f]['coef'].append(corr)

                    pbar.update()

                curr_indptr = e_indptr
                curr_indices = e_indices
                curr_smi += 1

    corr_stats = []
    for f, out in tqdm(outres.items(), desc="Computing correlation coefficient..."):
        clip_corr = np.clip(out['coef'], -0.99999, 0.99999)
        corr_mean = np.tanh(np.arctanh(clip_corr).mean()).item()

        n_samples = len(out['len'])
        eval_tk_len = np.array(out['len'])
        corr_stats.append({
            "feature": f,
            "n": n_samples,
            "length_Min": eval_tk_len.min(),
            "length_Median": np.median(eval_tk_len),
            "length_Max": eval_tk_len.max(),
            "length_Mean": eval_tk_len.mean(),
            "corr_Mean": corr_mean
        })

    return pl.DataFrame(corr_stats)


# Numba-based helpers
## token_preference
@njit(parallel=True)
def fast_counter(
    ctr_mol: np.ndarray, ctr_tk: np.ndarray,
    tk_ids: np.ndarray,
    mol_indptr: np.ndarray, mol_indices: np.ndarray
):
    nz_fs = np.flatnonzero(np.diff(mol_indptr))
    for i_f in prange(len(nz_fs)):
        f = nz_fs[i_f]
        ctr_mol[f] += 1

        s = mol_indptr[f]
        e = mol_indptr[f+1]
        for tk_i in mol_indices[s:e]:
            ctr_tk[f, tk_ids[tk_i]] += 1

@njit(parallel=True)
def fast_counter_inverse(
    ctr_mol: np.ndarray, ctr_tk: np.ndarray,
    tk_ids: np.ndarray,
    mol_indptr: np.ndarray, mol_indices: np.ndarray,
    nt: int, n_features: int
):
    for f in prange(n_features):
        s = mol_indptr[f]
        e = mol_indptr[f+1]
        d = e - s

        num_zeros = nt - d
        if num_zeros > 0:
            ctr_mol[f] += 1

            is_zero = np.ones(nt, dtype=np.bool_)
            for tk_i in range(s, e):
                is_zero[mol_indices[tk_i]] = False

            for t_i in range(nt):
                if is_zero[t_i]:
                    ctr_tk[f, tk_ids[t_i]] += 1
