import json
import time

import click
import wandb
import pytorch_lightning as ptl

from dataclasses import dataclass, fields, MISSING
from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from intermol.main.data_utils import MolDataModule
from intermol.main.sae_module import SAEModule

# dataclass
@dataclass
class Config:
    data_path: str
    col_name: str
    layer: int
    hidden_dim: int
    k: int
    model_name: str = 'ibm/MoLFormer-XL-both-10pct'
    model_dim: int = 768
    batch_size: int = 128
    train_size: float = 0.8
    dead_steps_threshold: int = 5000
    ckpt_path: str = None
    num_epochs: int = 1
    opt_lr: float = 2e-4
    opt_wd: float = 1e-2
    seed: int = None

    def __post_init__(self):
        if self.seed is None:
            self.seed = int(time.time())

# build_config
def build_config(config, **cli_kwargs) -> Config:
    if config:
        with open(config, 'r') as h:
            return Config(**json.load(h)) # cli_kwargs are ignored
    else:
        OPTS = {f.name for f in fields(Config) if f.default is not MISSING}
        cfg = {}
        for k, v in cli_kwargs.items():
            if (v is None) and (k not in OPTS):
                raise click.UsageError(
                    f"Missing required option: '--{k.replace('_', '-')}'. "
                    f"Provide all required args via CLI or use --config."
                )
            if v is not None:
                cfg[k] = v
        return Config(**cfg)


# main
@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True),
    default=None,
    help="Path to config.json"
)

# cfg: data
@click.option(
    "--data-path", type=str, default=None,
    help="Path to dataset.parquet"
)
@click.option(
    "--col-name", type=str, default=None,
    help="Column name containing SMILES strings"
)
@click.option(
    "--batch-size", type=int, default=None,
    help="Number of samples per batch"
)
@click.option(
    "--train-size", type=click.FloatRange(0, 1), default=None,
    help="Fraction of data used for training"
)

# cfg: SAE
@click.option("--layer", type=int, default=None, help="Layer of the base model")
@click.option(
    "--hidden-dim", type=int, default=None, help="Latent dimension of the SAE"
)
@click.option(
    "--k", type=int, default=None, help="Number of top-k latents used in the SAE"
)
@click.option(
    "--model-name", type=str, default='ibm/MoLFormer-XL-both-10pct',
    help="Hugging Face model name"
)
@click.option("--model-dim", type=int, default=768, help="Base model hidden dimension")
@click.option(
    "--dead-steps-threshold", type=int, default=None,
    help="Step threshold for tracking dead latents"
)
@click.option(
    "--ckpt-path", type=str, default=None,
    help="Path to a trained model checkpoint used to resume training"
)

# cfg: run
@click.option("--num-epochs", type=int, default=None, help="Number of training epochs")
@click.option("--opt-lr", type=float, default=None, help="Learning rate")
@click.option("--opt-wd", type=float, default=None, help="Weight decay")
@click.option("--seed", type=int, default=None, help="Random seed")
def main(config, **cli_kwargs):
    # build config
    cfg = build_config(config, **cli_kwargs)

    # seed
    ptl.seed_everything(cfg.seed)

    # output
    output_dir = Path(
        f"../results_layer{cfg.layer}_dim{cfg.hidden_dim}_k{cfg.k}_{cfg.seed}"
    )
    output_dir.mkdir(exist_ok=True)

    # logger
    run_name = (
        f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_"
        f"layer{cfg.layer}_dim{cfg.hidden_dim}_k{cfg.k}"
    )
    wandb_logger = WandbLogger(
        project='intermol',
        name=run_name,
        save_dir=output_dir / 'wandb'
    )

    # data
    data = MolDataModule(
        cfg.data_path,
        cfg.col_name,
        cfg.batch_size,
        cfg.train_size,
        cfg.seed
    )

    # models
    model = SAEModule(
        cfg.layer,
        cfg.hidden_dim,
        cfg.k,
        cfg.opt_lr,
        cfg.opt_wd,
        cfg.model_name,
        cfg.model_dim,
        cfg.batch_size,
        cfg.dead_steps_threshold
    )
    wandb_logger.watch(model, log='all')

    ckpt_callback = ModelCheckpoint(
        dirpath=output_dir / 'ckpts',
        filename=run_name + '-{step}-{avg_mse_loss:.2f}',
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True
    )

    # train
    trainer = ptl.Trainer(
        max_epochs=cfg.num_epochs,
        accelerator='gpu',
        devices=[0],
        strategy='auto',
        logger=wandb_logger,
        log_every_n_steps=10,
        val_check_interval=100,
        limit_val_batches=10,
        callbacks=[ckpt_callback],
        gradient_clip_val=1.0,
    )

    if cfg.ckpt_path is not None:
        trainer.fit(model, data, ckpt_path=cfg.ckpt_path)
    else:
        trainer.fit(model, data)

    try:
        for checkpoint in (output_dir / "ckpts").glob("*.ckpt"):
            wandb.log_artifact(checkpoint, type="model")
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()
