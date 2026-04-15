import math
import click
import h5py
import numpy as np

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from scipy.sparse import csc_matrix

from intermol.main.inference import (
    SAEInferenceConfig, SAEInferenceModule, SAEWithBaseModel
)

def precomp_acts(
    samples: list[str],
    sae_module: SAEInferenceModule,
    chunk_size: int = 1024,
    outdir_pth: str = ".",
    out_prefix: str = None
) -> None:
    outdir_pth = Path(outdir_pth)

    n_samples = len(samples)
    n_features = sae_module.sae.hidden_dim
    n_chunks = math.ceil(n_samples / chunk_size)

    if out_prefix is None:
        out_prefix = datetime.now().strftime("%Y%m%d%H%M")
    h5f_pth = outdir_pth / f'{out_prefix}_acts.h5'
    with h5py.File(h5f_pth, 'w') as h5f:
        # attrs
        h5f.attrs['num_chunks'] = n_chunks
        h5f.attrs['num_features'] = n_features
        h5f.attrs['num_samples'] = n_samples
        h5f.attrs['type'] = 'csc_matrix'

        with tqdm(total=n_samples) as pbar:
            for c, cs_s in enumerate(range(0, n_samples, chunk_size)):
                pbar.set_description(f'Process chunk {c+1}...')
                cs_e = min(cs_s + chunk_size, n_samples)
                n_cs = cs_e - cs_s

                g = h5f.create_group(f'c{c}')
                g.attrs['num_chunk_samples'] = n_cs

                chunk_samples = samples[cs_s:cs_e]
                chunk_tokens = np.zeros(shape=n_cs, dtype=np.uint16)
                coll_data, coll_indices, coll_indptr = [], [], []

                for i_cs, cs in enumerate(chunk_samples):
                    chunk_tokens[i_cs] = len(sae_module.tokenize(cs))

                    _, acts = sae_module.encode(cs)
                    acts = csc_matrix(acts.squeeze()[1:-1, :].cpu().numpy())

                    coll_data.append(acts.data)
                    coll_indices.append(acts.indices)
                    coll_indptr.append(acts.indptr)

                    pbar.update()

                # molptr => token count per mol within a chunk
                g.create_dataset(
                    'molptr', data=chunk_tokens, dtype=np.uint16, compression='lzf'
                )

                g.create_dataset(
                    'data',
                    data=np.concatenate(coll_data),
                    dtype=np.float32,
                    compression='lzf'
                )
                g.create_dataset(
                    'indices',
                    data=np.concatenate(coll_indices),
                    dtype=np.uint16,
                    compression='lzf'
                )
                g.create_dataset(
                    'indptr',
                    data=np.concatenate(coll_indptr),
                    dtype=np.uint32,
                    compression='lzf'
                )

                for _, obj in g.items():
                    if isinstance(obj, h5py.Dataset):
                        obj.id.flush()


# main
@click.command()
@click.option(
    "--data-pth", type=click.Path(exists=True), required=True,
    help="Path to .txt or one-column .smi file"
)
@click.option(
    "--hidden-dim", type=int, required=True, help="SAE latent dimension"
)
@click.option(
    "--k", type=int, required=True, help="Number of top-k latents used in the SAE"
)
@click.option(
    "--sae-ckpt-pth", type=click.Path(exists=True), required=True,
    help="Path to trained SAE checkpoint"
)
@click.option("--layer", type=int, required=True, help="Base model layer")
@click.option(
    "--chunk-size", type=int, default=8192,
    help="Number of samples per chunk. Default: 8192"
)
@click.option(
    "--outdir-pth", type=click.Path(file_okay=False), default='.',
    help='Output directory. Default: current directory.'
)
@click.option(
    "--out-prefix", type=str, default=None,
    help='Output file prefix. Default: current timestamp (e.g. 202503101430).'
)
@click.option(
    "--device", type=click.Choice(['auto', 'cpu', 'cuda']), default='auto',
    help='Inference device. Default: auto.'
)
def main(**cli_kwargs):
    # parse dataset
    print("Parse dataset...")
    with open(cli_kwargs['data_pth'], 'r') as h:
        smiles = [smi.rstrip('\n') for smi in h.readlines()]

    # init module
    print("Initialize SAE inference module...")
    sae_config = SAEInferenceConfig(
        cli_kwargs['layer'],
        cli_kwargs['hidden_dim'],
        cli_kwargs['k'],
        cli_kwargs['sae_ckpt_pth']
    )
    sae_module = SAEWithBaseModel(sae_config, device_name=cli_kwargs['device'])

    # run `precomp_acts`
    precomp_acts(
        smiles, sae_module,
        chunk_size=cli_kwargs['chunk_size'],
        outdir_pth=cli_kwargs['outdir_pth'],
        out_prefix=cli_kwargs['out_prefix']
    )

if __name__ == '__main__':
    main()
