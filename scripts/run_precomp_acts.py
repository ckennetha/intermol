import math
import click
import h5py
import warnings
import numpy as np

from datetime import datetime
from pathlib import Path
from scipy.sparse import csc_matrix
from tqdm import tqdm
from typing import Optional

from intermol.main.inference import SAEInferenceConfig, SAEWithBaseModel

def precomp_acts(
    samples: list[str],
    sae_config: list[SAEInferenceConfig],
    sae_module: SAEWithBaseModel,
    outdir_path: Optional[list[str]] = None,
    chunk_size: int = 1024,
    out_prefix: str = None
) -> None:
    if outdir_path is None:
        outdir_path = ['.'] * len(sae_config)
        warnings.warn(
            "No --outdir-paths specified. " \
            "All outputs will be written to current directory."
        )
    elif len(outdir_path) != len(sae_config):
        raise ValueError(
            "Inconsistent number of SAE configs and outdir paths. " \
            f"Got: {len(sae_config)} and {len(outdir_path)}."
        )

    if out_prefix is None:
        out_prefix = datetime.now().strftime("%Y%m%d%H%M")

    n_samples = len(samples)
    n_chunks = math.ceil(n_samples / chunk_size)

    h5_fs = {}
    for cfg, out_pth in zip(sae_config, outdir_path):
        out_pth = Path(out_pth)
        out_pth.mkdir(parents=True, exist_ok=True)

        h5f_pth = out_pth / f"{out_prefix}_acts.h5"
        h5f = h5py.File(h5f_pth, "w")
        h5f.attrs['num_chunks'] = n_chunks
        h5f.attrs['num_features'] = cfg.hidden_dim
        h5f.attrs['num_samples'] = n_samples
        h5f.attrs['type'] = 'csc_matrix'
        h5_fs[cfg.key] = h5f

    try:
        with tqdm(total=n_samples) as pbar:
            for c, cs_s in enumerate(range(0, n_samples, chunk_size)):
                pbar.set_description(f'Process chunk {c+1}/{n_chunks}...')
                cs_e = min(cs_s + chunk_size, n_samples)
                chunk_samples = samples[cs_s:cs_e]
                n_cs = len(chunk_samples)

                colls = {
                    cfg.key: {
                        "chunk_tokens": np.zeros(n_cs, dtype=np.int16),
                        "coll_data": [],
                        "coll_indices": [],
                        "coll_indptr": []
                    } for cfg in sae_config
                }

                for i_cs, cs_smi in enumerate(chunk_samples):
                    out = sae_module.encode_multi(cs_smi, sae_config)

                    for cfg in sae_config:
                        _, sae_acts = out[cfg.key]
                        sae_acts = sae_acts.squeeze()[1:-1, :].cpu().numpy()
                        sae_acts_sp = csc_matrix(sae_acts)

                        coll = colls[cfg.key]
                        coll["chunk_tokens"][i_cs] = sae_acts.shape[0]
                        coll["coll_data"].append(sae_acts_sp.data)
                        coll["coll_indices"].append(sae_acts_sp.indices)
                        coll["coll_indptr"].append(sae_acts_sp.indptr)

                    pbar.update()

                for cfg in sae_config:
                    coll = colls[cfg.key]

                    g = h5_fs[cfg.key].create_group(f'c{c}')
                    g.attrs['num_chunk_samples'] = n_cs

                    # molptr => token count per mol within a chunk
                    g.create_dataset('molptr', data=coll["chunk_tokens"], compression='lzf')

                    g.create_dataset(
                        'data',
                        data=np.concatenate(coll["coll_data"]),
                        dtype=np.float32,
                        compression='lzf'
                    )
                    g.create_dataset(
                        'indices',
                        data=np.concatenate(coll["coll_indices"]),
                        dtype=np.int16,
                        compression='lzf'
                    )
                    g.create_dataset(
                        'indptr',
                        data=np.concatenate(coll["coll_indptr"]),
                        dtype=np.int32,
                        compression='lzf'
                    )

                    for _, obj in g.items():
                        if isinstance(obj, h5py.Dataset):
                            obj.id.flush()

    finally:
        for h5f in h5_fs.values():
            h5f.close()


# main
@click.command()
@click.option(
    "--data-path", type=click.Path(exists=True), required=True,
    help="Path to .txt or one-column .smi file"
)

@click.option(
    "--layer", type=int, multiple=True, required=True,
    help="Base model layer (repeat per SAE)"
)
@click.option(
    "--hidden-dim", type=int, multiple=True, required=True,
    help="SAE latent dimension (repeat per SAE)"
)
@click.option(
    "--k", type=int, multiple=True, required=True,
    help="Number of top-k latents used in the SAE (repeat per SAE)"
)
@click.option(
    "--sae-ckpt-path", type=click.Path(exists=True), multiple=True, required=True,
    help="Path to trained SAE checkpoint (repeat per SAE)"
)
@click.option(
    "--outdir-path", type=click.Path(file_okay=False), multiple=True, default=None,
    help='Output directory (repeat per SAE)'
)

@click.option(
    "--chunk-size", type=int, default=8192,
    help="Number of samples per chunk. Default: 8192"
)
@click.option(
    "--out-prefix", type=str, default=None,
    help='Output file prefix. Default: current timestamp (e.g. 202503101430).'
)
@click.option(
    "--model-name", type=str, required=True, help="Hugging Face model name"
)
@click.option(
    "--use-molformer", is_flag=True, default=False,
    help="Enable MoLFormer-specific setting"
)
@click.option(
    "--device", type=click.Choice(['auto', 'cpu', 'cuda']), default='auto',
    help='Inference device. Default: auto.'
)
def main(**cli_kwargs):
    # sanity check
    per_sae_args = ['layer', 'hidden_dim', 'k', 'sae_ckpt_path']
    lengths = {arg: len(cli_kwargs[arg]) for arg in per_sae_args}
    if len(set(lengths.values())) != 1:
        raise click.UsageError(
            f"--layer, --hidden-dim, --k, --sae-ckpt-path, --outdir-path "
            f"must all be repeated the same number of times. Got: {lengths}"
        )

    # parse dataset
    print("Parse dataset...")
    with open(cli_kwargs['data_path'], 'r') as h:
        smiles = [smi.rstrip('\n') for smi in h.readlines()]

    # init module
    print("Initialize SAE inference module...")
    sae_config = [
        SAEInferenceConfig(layer, hidden_dim, k, ckpt)
        for layer, hidden_dim, k, ckpt in zip(
            cli_kwargs['layer'],
            cli_kwargs['hidden_dim'],
            cli_kwargs['k'],
            cli_kwargs['sae_ckpt_path']
        )
    ]

    sae_module = SAEWithBaseModel(
        sae_config,
        cli_kwargs['model_name'],
        cli_kwargs['use_molformer'],
        cli_kwargs['device']
    )

    # run `precomp_acts`
    precomp_acts(
        smiles,
        sae_config,
        sae_module,
        list(cli_kwargs['outdir_path']) or None,
        cli_kwargs['chunk_size'],
        cli_kwargs['out_prefix']
    )

if __name__ == '__main__':
    main()
