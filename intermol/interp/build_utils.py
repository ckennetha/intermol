import h5py
import math
import numpy as np
import polars as pl
from tqdm.auto import tqdm

from intermol.interp.molecular_concepts import BatchLabelFromSmarts
from intermol.interp.utils import h5_chunk_sorter

# a map of activation bins for each SAE latent across activation ranges
def build_bin_map(
    acts_h5_pth: str,
    outfn_pth: str, # default output dtype: np.uint16; available bits index: 0-14
    act_bins: list[tuple[float, float]] #[(lower-bound [incl.], upper-bound [excl.])]
) -> None:
    with h5py.File(acts_h5_pth, 'r') as h5f:
        n_samples = h5f.attrs['num_samples']
        n_features = h5f.attrs['num_features']
        chunks = h5_chunk_sorter(list(h5f.keys()))

        # init output
        binmap = np.memmap(
            outfn_pth,
            dtype=np.uint16,
            mode='w+',
            shape=(n_samples, n_features)
        )

        last_smi = 0
        curr_smi = 0
        for c in tqdm(chunks, desc="Processing per chunk...", leave=False):
            g = h5f[c]

            molptr = g['molptr'][:]
            indptr = g['indptr'][:]
            data = g['data'][:]

            c_bins = np.zeros((len(molptr), n_features), dtype=np.uint16)

            curr_indptr = 0
            curr_data = 0
            for i_m, _ in enumerate(molptr):
                e_indptr = curr_indptr + n_features + 1
                mol_indptr = indptr[curr_indptr:e_indptr]

                e_data = curr_data + mol_indptr[-1]
                mol_data = data[curr_data:e_data]

                nz_cs = np.diff(mol_indptr)
                nz_col_idxs = np.repeat(
                    np.arange(n_features, dtype=np.int64), nz_cs
                )

                bins = np.zeros(n_features, dtype=np.uint16)
                for i_b, (lb, ub) in enumerate(act_bins):
                    mask = (mol_data >= lb) & (mol_data < ub)
                    if mask.any():
                        hit = np.bincount(
                            nz_col_idxs[mask], minlength=n_features
                        ).astype(bool)
                        bins[hit] |= (1 << i_b)
                c_bins[i_m] = bins

                curr_indptr = e_indptr
                curr_data = e_data
                curr_smi += 1

            binmap[last_smi:curr_smi] = c_bins
            binmap.flush()

            last_smi = curr_smi

    print(f"Bin map has been successfully written to {outfn_pth}!")

# a dataset for evaluating latents with molecular concepts
def build_eval_data(
    data_pth: str,
    outfn_pth: str,
    label_df: pl.DataFrame,
    desc_colnm: str,
    concept_colnm: str,
    prefilter_smarts: bool = True,
    use_smiles_indices: bool = False,
    batch_size: int = 8192,
    n_threads: int = 1
) -> None:
    # parse dataset
    with open(data_pth, 'r') as h:
        smiles_map = {
            smi_i: smi.rstrip('\n') for smi_i, smi in enumerate(h.readlines())
        }
        smiles = list(smiles_map.keys())

    print("Building label map...")
    label_map = dict(zip(
        label_df[desc_colnm].to_list(), label_df[concept_colnm].to_list()
    ))

    print("Initializing labeler...")
    labeler = BatchLabelFromSmarts(label_map, prefilter_smarts=prefilter_smarts)
    nb = math.ceil(len(smiles) / batch_size)

    outs = []
    for nb in tqdm(range(nb), desc="Processing per batch...", leave=False):
        s = nb * batch_size
        e = s + batch_size
        b_smiles = smiles[s:e]

        labels = labeler.batch_label(smiles=b_smiles, n_threads=n_threads)
        flat_labels = []
        for smi, label in labels.items():
            smi_str = smi if use_smiles_indices else smiles_map[smi]
            for desc, tk_idxs in label.items():
                flat_labels.append({
                    "smiles": smi_str, "desc": desc, "token_idxs": tk_idxs
                })

        outs.append(pl.DataFrame(flat_labels))

    print(f"Saving output to {outfn_pth}...")
    pl.concat(outs).sink_parquet(outfn_pth)
    print("Eval data saved successfully!")
