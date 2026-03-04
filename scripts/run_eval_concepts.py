import click
import polars as pl

from collections import defaultdict
from pathlib import Path
from typing import Optional

from intermol.interp.eval_utils import calculate_smd, ConceptEvaluator

# config
pl.Config.set_engine_affinity(engine="streaming")

# only supports the Parquet format from 'intermol.interp.build_utils.build_eval_data'
def prep_data(
    data_pth: str,
    sample_colname: str,
    concept_colname: str,
    label_colname: str,
    mapping: dict[str, int],
    batch_size: int = 65536,
    is_sampling: bool = False,
    fraction_sampling: Optional[float] = 0.20,
    seed_sampling: Optional[int] = 42,
    concepts_from_fpc: Optional[list[int]] = None
) -> dict[int, dict[int, list[int]]]:
    # parse data_df
    data_df = pl.scan_parquet(data_pth)

    # map each concept to its indices based on the label
    # and ensures alignment with the correct concept
    data_df = data_df.with_columns(
        pl.col(concept_colname)
        .replace_strict(mapping, return_dtype=pl.UInt16)
        .alias("concept_idx")
    ).sort(
        [sample_colname, "concept_idx"],
        descending=[False, False]
    )

    # when the fpc is built and 'concepts_from_fpc' is provided
    if concepts_from_fpc is not None:
        data_df = data_df.filter(pl.col("concept_idx").is_in(concepts_from_fpc))

    # sampling
    if is_sampling:
        sampled = (
            data_df
            .select(pl.col(sample_colname))
            .unique().collect()
            .sample(
                fraction=fraction_sampling,
                with_replacement=False,
                shuffle=True,
                seed=seed_sampling
            )
        )
        data_df = data_df.join(sampled.lazy(), on=sample_colname, how="inner")

    # pack data
    print("Packing data...")
    p_data = defaultdict(dict)
    for b in data_df.collect_batches(chunk_size=batch_size, maintain_order=False):
        for r in b.iter_rows(named=True):
            p_data[r[sample_colname]][r["concept_idx"]] = r[label_colname]
    return dict(p_data)

# only supports csv and its derivative formats
def prep_labels(
    label_pth: str,
    index_colname: str,
    concept_colname: str,
    desc_colname: Optional[str] = None
) -> tuple[dict[str, int], dict[int, str] | None]:
    # parse label_df
    label_df = pl.read_csv(label_pth, separator="\t")
    index_list = label_df[index_colname].to_list()

    # build mappings
    ## concept to index
    concept_to_idx_map = dict(zip(label_df[concept_colname].to_list(), index_list))
    ## description to index
    desc_to_idx_map_r = None
    if desc_colname is not None:
        desc_to_idx_map_r = dict(zip(index_list, label_df[desc_colname].to_list()))
    return concept_to_idx_map, desc_to_idx_map_r

# build fpc map from 'calculate_smd' output; only supports tsv
def prep_fpc(
    fpc_pth: str, score_threshold: float = 0, k: int = 64
) -> tuple[list[int], dict[int, list[int]], pl.DataFrame]:
    # parse fpc_df
    fpc_df = pl.read_csv(fpc_pth, separator="\t")
    fpc_df = (
        fpc_df
        .filter(
            (pl.col("smd").is_not_nan()) &
            (pl.col("smd") > score_threshold)
        )
        .select(
            pl.all().top_k_by("smd", k=k)
            .over("conceptIdx", mapping_strategy="explode")
        )
        .sort(
            ["conceptIdx", "featureIdx"],
            descending=[False, False]
        )
        .group_by("conceptIdx").agg(pl.col("featureIdx"))
    )

    concepts = fpc_df["conceptIdx"].to_list()
    fpc_map = dict(zip(concepts, fpc_df["featureIdx"].to_list()))
    fpc_lookup = (
        fpc_df
        .explode("featureIdx")
        .with_columns(
            (pl.cum_count("conceptIdx").over("conceptIdx") - 1)
            .cast(pl.UInt32)
            .alias("feature_key")
        )
        .rename({"featureIdx": "mapped_feature"})
    )
    return concepts, fpc_map, fpc_lookup


# main
@click.command()

# paths
@click.option(
    "--data-pth", required=True, type=click.Path(exists=True),
    help="Path to input parquet data."
)
@click.option(
    "--acts-h5-pth", required=True, type=click.Path(exists=True),
    help="Path to precomputed activations h5 file."
)
@click.option(
    "--label-pth", required=True, type=click.Path(exists=True),
    help="Path to concept label tsv."
)
@click.option(
    "--outdir-pth", required=True, type=click.Path(file_okay=False),
    help="Output directory."
)
@click.option(
    "--outfn", required=True, type=str, help="Output filename (without extension)."
)
@click.option(
    "--fpc-pth", required=False, type=click.Path(exists=True), default=None,
    help="Path to 'calculate_smd' or prefiltering output (optional). " \
    "If not provided, concepts will be evaluated across all latents in the SAE."
)

# column names
@click.option(
    "--sample-colname", required=True, type=str, help="Column name for samples."
)
@click.option(
    "--concept-colname", required=True, type=str,
    help="Column name for concepts." \
    "The name must be the same in both the data and label files."
)
@click.option(
    "--label-colname", required=True, type=str, help="Column name for labels."
)
@click.option(
    "--index-colname", required=True, type=str,
    help="Column name for concept indices in label file."
)
@click.option(
    "--desc-colname", required=False, type=str, default=None,
    help="Column name for concept descriptions (optional)."
)

# eval options
@click.option(
    "--thresholds", multiple=True, default=(0.0,), type=float,
    help="Thresholds for evaluation (pass multiple)."
)
@click.option(
    "--use-pooling", is_flag=True, default=False,
    help="Use pooling-based evaluation; suitable for concepts spanning multiple tokens."
)
@click.option(
    "--is-prefilter", is_flag=True, default=False,
    help="Run SAE latent prefiltering with SMD instead of full evaluation."
)
@click.option(
    "--batch-size", default=65536, type=int,
    help="Batch size for data packing and evaluation."
)

# fpc options (if provided)
@click.option(
    "--score-threshold", default=0.0, type=float,
    help="Minimum SMD score threshold for fpc."
)
@click.option(
    "--k", default=64, type=int, help="Top-k features per concept."
)

# sampling options
@click.option(
    "--is-sampling", is_flag=True, default=False, help="Enable molecule sampling."
)
@click.option(
    "--fraction-sampling", default=0.20, type=float,
    help="Fraction of data to sample."
)
@click.option("--seed-sampling", default=42, type=int, help="Random seed for sampling.")

def main(**cli_kwargs):
    # kwargs
    fpc_pth = cli_kwargs["fpc_pth"]
    concept_colname = cli_kwargs["concept_colname"]
    is_prefilter = cli_kwargs["is_prefilter"]

    # sanity check
    if is_prefilter and fpc_pth is not None:
        raise click.UsageError("--fpc-pth is not used in prefilter mode.")

    # build labels
    concept_to_idx_map, desc_to_idx_map_r = prep_labels(
        label_pth = cli_kwargs["label_pth"],
        index_colname = cli_kwargs["index_colname"],
        concept_colname = concept_colname,
        desc_colname = cli_kwargs["desc_colname"]
    )
    n_concepts = len(concept_to_idx_map) # use total number of available concepts
    concept_to_idx_map_r = {v: k for k, v in concept_to_idx_map.items()}

    # build fpc if provided
    concepts_from_fpc, fpc_map, fpc_lookup = None, None, None
    if fpc_pth is not None:
        concepts_from_fpc, fpc_map, fpc_lookup = prep_fpc(
            fpc_pth=fpc_pth,
            score_threshold = cli_kwargs["score_threshold"],
            k = cli_kwargs["k"]
        )

    # build data
    bsz = cli_kwargs["batch_size"]
    data = prep_data(
        data_pth = cli_kwargs["data_pth"],
        sample_colname = cli_kwargs["sample_colname"],
        concept_colname = concept_colname,
        label_colname = cli_kwargs["label_colname"],
        mapping = concept_to_idx_map,
        batch_size = bsz,
        is_sampling = cli_kwargs["is_sampling"],
        fraction_sampling = cli_kwargs["fraction_sampling"],
        seed_sampling = cli_kwargs["seed_sampling"],
        concepts_from_fpc = concepts_from_fpc
    )

    # init prefilter or eval
    acts_h5_pth = cli_kwargs["acts_h5_pth"]
    use_pooling = cli_kwargs["use_pooling"]

    if is_prefilter:
        out = calculate_smd(
            acts_h5_pth=acts_h5_pth,
            samples=data,
            n_concepts=n_concepts,
            use_pooling=use_pooling
        )

        # build out_df
        out_df = pl.DataFrame(out)

    else:
        thresholds = cli_kwargs["thresholds"]
        n_fpc = None if fpc_map is None else len(fpc_map)

        evaluator = ConceptEvaluator(acts_h5_pth=acts_h5_pth, batch_size=bsz)
        if use_pooling:
            out = evaluator.eval_substructure(
                samples=data, n_concepts=n_concepts, thresholds=thresholds,
                fpc_map=fpc_map, n_fpc=n_fpc
            )
        else:
            out = evaluator.eval(
                samples=data, n_concepts=n_concepts, thresholds=thresholds,
                fpc_map=fpc_map, n_fpc=n_fpc
            )

        # build out_df
        out_df = pl.DataFrame(out).with_columns(pl.col(pl.Float64).round(4))

        # remap with fpc if provided
        if fpc_lookup is not None:
            out_df = (
                out_df
                .join(
                    fpc_lookup,
                    left_on=["conceptIdx", "featureIdx"],
                    right_on=["conceptIdx", "feature_key"],
                    how="left"
                ).with_columns(
                    pl.col("mapped_feature").alias("featureIdx")
                ).drop("mapped_feature")
            )

    # append description if 'cli_kwargs["desc_colname"]' is provided
    ext_cols = [] if desc_to_idx_map_r is None else [
        pl.col("conceptIdx").replace_strict(desc_to_idx_map_r).alias("description")
    ]

    out_df = (
        out_df.with_columns(
            pl.col("conceptIdx").replace_strict(concept_to_idx_map_r).alias("concept"),
            *ext_cols
        ).sort("featureIdx")
    )

    # write output to tsv
    outfn_pth = Path(cli_kwargs["outdir_pth"]) / (cli_kwargs["outfn"] + ".tsv")
    out_df.lazy().sink_csv(outfn_pth, separator="\t")
    print(f"Saved successfully to {outfn_pth}.")


if __name__ == '__main__':
    main()
