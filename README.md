# InterMol
InterMol is an open-source mechanistic interpretability toolbox that leverages sparse autoencoders (SAEs) to interpret chemical language models (cLMs). By default, InterMol uses [MolFormer-XL](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct) as its underlying cLM. Our interactive visualizer of the discovered features is available at [intermol.co](https://www.intermol.co/#/).

Learn more about the implementation details and findings by reading our [preprint]().

## Project Structure
```
.
├── intermol/
│   ├── main/       # Core SAE pipeline: training, normalization, and inference
│   └── interp/     # Interpretability tools: latent profiling, concept generation, labelling, and SMILES variant generation
├── notebooks/      # Jupyter notebooks for visualization and analysis
├── scripts/        # CLI interpretability utils
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```

## Getting Started
### Installation
**Prerequisites:** Python `>= 3.9`

```bash
# Package only
pip install git+https://github.com/ckennetha/intermol.git

# Clone the repository (recommended for local development)
git clone https://github.com/ckennetha/intermol.git
cd intermol
pip install -e .
```

### Pretrained SAEs
We provide SAE weights trained on top of the open-source version of MolFormer-XL at layers 1, 3, 6, 9, and 12. MolFormer-XL weights are fetched on-the-fly from HuggingFace, while SAE weights must be downloaded separately. To extract SAE activations using a pretrained model:

```python
from intermol.main.inference import SAEInferenceModule

SMILES = "c1ccccc1"

sae = SAEInferenceModule(
    hidden_dim=3072,
    k=128,
    sae_pth="norm-MOL-1-3072-128.pt", # normalized SAE weights
    layer_idx=1 # MolFormer-XL layer
)

mf_acts, sae_acts = sae.encode_both(SMILES)
# mf_acts: MolFormer-XL hidden states, sae_acts: SAE activations
```

### Bulk Activation Extraction
To extract SAE activations in bulk, we provide a command line interface (`run-precomp-acts`) that efficiently stores the results as decomposed CSC sparse matrix format using `h5py`:

```
usage: run-precomp-acts [-h] --data-pth DATA_PTH --hidden-dim HIDDEN_DIM --k K
                        --sae-ckpt-pth SAE_CKPT_PTH --layer LAYER
                        [--chunk-size CHUNK_SIZE] [--outdir-pth OUTDIR_PTH]
                        [--out-prefix OUT_PREFIX] [--device {auto,cpu,cuda}]

options:
    -h, --help                    show this help message and exit
    --data-pth DATA_PTH           Path to .txt or one-column .smi file
    --hidden-dim HIDDEN_DIM       SAE latent dimension
    --k K                         Number of top-k SAE latents
    --sae-ckpt-pth SAE_CKPT_PTH   Path to trained SAE checkpoint
    --layer LAYER                 MolFormer-XL layer
    --chunk-size CHUNK_SIZE       Number of samples per chunk. Default: 8192
    --outdir-pth OUTDIR_PTH       Output directory. Default: current directory
    --out-prefix OUT_PREFIX       Output filename prefix. Default: current timestamp
    --device {auto,cpu,cuda}      Inference device. Default: auto
```

### Concept Evaluation
For evaluating association between specific SAE latents and atom-level molecular concepts, we use a two-step approach: first filtering latents by standardized mean difference (SMD) using the `--is-prefilter` flag, then concept presence classification with binarized activations, evaluated by F1 score. We provide `run-eval-concepts` with Numba-accelerated computation:

```
usage: run-eval-concepts [-h] --data-pth DATA_PTH --acts-h5-pth ACTS_H5_PTH
                         --label-pth LABEL_PTH --outdir-pth OUTDIR_PTH
                         --outfn OUTFN --sample-colname SAMPLE_COLNAME
                         --concept-colname CONCEPT_COLNAME
                         --label-colname LABEL_COLNAME
                         --index-colname INDEX_COLNAME
                         [--fpc-pth FPC_PTH] [--desc-colname DESC_COLNAME]
                         [--thresholds THRESHOLDS] [--use-pooling]
                         [--is-prefilter] [--batch-size BATCH_SIZE]
                         [--score-threshold SCORE_THRESHOLD] [--k K]
                         [--is-sampling] [--fraction-sampling FRACTION_SAMPLING]
                         [--seed-sampling SEED_SAMPLING]

options:
    -h, --help                          show this help message and exit

    path options:
    --data-pth DATA_PTH                 Path to input .parquet file
    --acts-h5-pth ACTS_H5_PTH           Path to precomputed activations .h5 file
    --label-pth LABEL_PTH               Path to concept label .tsv file
    --outdir-pth OUTDIR_PTH             Output directory
    --outfn OUTFN                       Output filename (without extension)
    --fpc-pth FPC_PTH                   Path to prefiltering output. If not provided,
                                        concepts are evaluated across all SAE latents
    column name options:
    --sample-colname SAMPLE_COLNAME     Column name for samples
    --concept-colname CONCEPT_COLNAME   Column name for concepts (must match in
                                        both data and label files)
    --label-colname LABEL_COLNAME       Column name for labels
    --index-colname INDEX_COLNAME       Column name for concept indices in label file
    --desc-colname DESC_COLNAME         Column name for concept descriptions (optional)

    evaluation options:
    --thresholds THRESHOLDS             Thresholds for evaluation (pass multiple).
                                        Default: 0
    --use-pooling                       Use pooling-based evaluation for concepts
                                        spanning multiple tokens
    --is-prefilter                      Run SAE latent prefiltering with SMD
                                        instead of full evaluation
    --batch-size BATCH_SIZE             Batch size for evaluation. Default: 65536

    post-prefiltering options:
    --score-threshold SCORE_THRESHOLD   Minimum SMD score threshold. Default: 0
    --k K                               Top-k features per concept. Default: 64

    sampling options:
    --is-sampling                       Enable molecule sampling
    --fraction-sampling FRACTION_SAMPLING
                                        Fraction of data to sample. Default: 0.20
    --seed-sampling SEED_SAMPLING       Random seed for sampling. Default: 42
```

#### Prefiltering Latents
Below is an example of prefiltering single-token concepts with a sampled dataset. For multi-token concepts, simply set the `--use-pooling` flag.
```bash
run-eval-concepts \
    --data-pth valid_dataset.parquet \
    --acts-h5-pth valid_acts.h5 \
    --label-pth SMARTS_ignChi_f10.tsv \
    --outdir-pth ../results_layer1/ \
    --outfn valid_smd \
    --sample-colname smiles \
    --concept-colname concept \
    --index-colname id \
    --label-colname token_idxs \
    --is-prefilter \
    --is-sampling \
    --fraction-sampling 0.2 \
    --seed-sampling 42
```

#### SAE Latent-Concept Association
To run association analysis on single-token concepts, remove the `--is-prefilter` flag. If `--fpc-pth` is supplied with prefiltering output, set `--score-threshold` to filter by SMD score and `--k` for the number of top-k latents. As before, set `--use-pooling` for multi-token concepts.
```bash
run-eval-concepts \
    --data-pth valid_dataset.parquet \
    --acts-h5-pth valid_acts.h5 \
    --label-pth SMARTS_ignChi_f10.tsv \
    --fpc-pth valid_smd.tsv \
    --outdir-pth ../results_layer1/ \
    --outfn valid_eval \
    --sample-colname smiles \
    --concept-colname concept \
    --index-colname id \
    --label-colname token_idxs \
    --thresholds 0 0.1 0.2 0.35 0.5 0.6 0.8 \
    --score-threshold 0 \
    --k 64
```

## Notebooks
1. [`activation_dist.ipynb`](notebooks/activation_dist.ipynb) &mdash; Visualizing activation distribution of a specific SAE latent. Supports coloring by specific tokens and SMARTS patterns.

2. [`logit_lens.ipynb`](notebooks/logit_lens.ipynb) &mdash; Visualizing how MolFormer-XL predicts masked tokens and how predictions evolve across layers, showing top predicted tokens at each layer. Supports latent ablation to assess the causal effect of specific SAE latent(s).

3. [`probe_latents.ipynb`](notebooks/probe_latents.ipynb) &mdash; Extracting SAE activations and other chemical features into a `.h5` file for interpretable linear probing experiments.

## License
This project is licensed under the [MIT License](LICENSE).