"""
This code is adapted from: https://github.com/ElanaPearl/InterPLM/blob/main/interplm/sae/normalize.py,
as described in Simon & Zou (2025), 'InterPLM: Discovering Interpretable Features in Protein Language Models via Sparse Autoencoders'
https://doi.org/10.1101/2024.11.14.623630
"""

import math
import click
import torch
import numpy as np

from pathlib import Path
from tqdm import tqdm

from intermol.main.sae import SparseAutoencoder
from intermol.main.utils import load_model_from_file, load_model_from_HF

@torch.no_grad()
def normalize(sae: torch.nn.Module, max_per_feat: torch.Tensor) -> torch.nn.Module:
    sae.w_enc.div_(max_per_feat.unsqueeze(0))
    sae.w_enc[sae.w_enc.isinf()] = 0

    sae.b_enc.div_(max_per_feat)
    sae.b_enc[sae.b_enc.isinf()] = 0

    sae.w_dec.mul_(max_per_feat.unsqueeze(1))
    return sae

@torch.no_grad()
def normalize_sae(
    data: list[str],
    layer: int,
    hidden_dim: int,
    k: int,
    weights_path: str,
    chunk_size: int = 1024,
    model_name: str = 'ibm/MoLFormer-XL-both-10pct',
    outdir_path: str = ".",
    device_name: str = "auto"
) -> None:
    # device
    if device_name == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_name)

    # models
    print("Loading base model...")
    tokenizer, base_model = load_model_from_HF(model_name)
    base_model.to(device)

    print("Loading SAE model...")
    sae = SparseAutoencoder(hidden_dim, k)
    sae = load_model_from_file(sae, weights_path)
    sae.to(device)

    # outdir_pth
    outdir_pth = Path(outdir_path)
    outdir_pth.mkdir(exist_ok=True)

    # process chunks
    n_samples = len(data)
    n_chunks = math.ceil(n_samples / chunk_size)
    n_features = sae.hidden_dim

    n_tokens = 0
    max_per_feat = torch.zeros(n_chunks, n_features, device=device)
    for smi_start in tqdm(range(0, n_samples, chunk_size), desc='Process chunks...'):
        smi_end = smi_start + chunk_size

        smi_c = data[smi_start:smi_end]
        enc_c = tokenizer(smi_c, padding=True, return_tensors='pt').to(device)
        base_acts_c = (
            base_model(**enc_c, output_hidden_states=True)
                .hidden_states[layer]
                .squeeze(0)
        )

        _, _, base_fs = base_acts_c.shape
        base_acts = base_acts_c.view(-1, base_fs)
        att_mask = enc_c['attention_mask'].view(-1).unsqueeze(-1)

        acts, _, _ = sae.encode(base_acts)
        acts_masked = torch.where(
            att_mask.bool(), acts,
            torch.tensor(float('-inf'), device=acts.device, dtype=acts.dtype)
        )
        max_acts, _ = torch.max(acts_masked, dim=0)

        max_per_feat[smi_start // chunk_size, :] = max_acts
        n_tokens += base_acts.shape[0]

        del enc_c, base_acts_c, base_acts, acts, max_acts

    # aggregate chunks
    max_per_feat, _ = torch.max(max_per_feat, dim=0)

    # save as npy
    out_max_fn = outdir_pth / f'norm-sae_max-per-feature_{n_samples}s_{n_tokens}t.npy'
    np.save(out_max_fn, max_per_feat.cpu().numpy())
    print(f'Max activations per feature saved to {out_max_fn}.')

    print("Normalized SAE model...")
    norm_sae = normalize(sae, max_per_feat)
    out_sae_fn = outdir_pth / f'norm-sae_state-dict_{n_samples}s_{n_tokens}t.pt'
    torch.save(norm_sae.state_dict(), out_sae_fn)
    print(f"Normalized model saved to {out_sae_fn}")


# main
@click.command()
@click.option(
    "--data-path", type=click.Path(exists=True), required=True,
    help="Path to dataset.txt"
)

@click.option("--layer", type=int, required=True, help="Layer of the base model")
@click.option(
    "--hidden-dim", type=int, required=True, help="Latent dimension of the SAE"
)
@click.option(
    "--k", type=int, required=True, help="Number of top-k latents used in the SAE"
)
@click.option(
    "--sae-ckpt-path", type=click.Path(exists=True), required=True,
    help="Path to a trained model checkpoint"
)

@click.option(
    "--chunk-size", type=int, default=1024, help="Number of samples per chunk"
)
@click.option(
    "--model-name", type=str, default='ibm/MoLFormer-XL-both-10pct',
    help="Hugging Face model name"
)
@click.option(
    "--outdir-path", type=click.Path(file_okay=False), default='.',
    help='Output directory'
)
@click.option(
    "--device", type=click.Choice(['auto', 'cpu', 'cuda']), default='auto',
    help='Device for inference'
)
def main(**cli_kwargs):
    with open(cli_kwargs["data_path"], "r") as h:
        smiles = [smi.rstrip("\n") for smi in h.readlines()]

    # normalize SAE
    normalize_sae(
        smiles,
        cli_kwargs["layer"],
        cli_kwargs["hidden_dim"],
        cli_kwargs["k"],
        cli_kwargs["sae_ckpt_path"],
        cli_kwargs["chunk_size"],
        cli_kwargs["model_name"],
        cli_kwargs["outdir_path"],
        cli_kwargs["device"]
    )


if __name__ == '__main__':
    main()
