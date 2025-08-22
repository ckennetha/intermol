import os
import wandb
import pytorch_lightning as ptl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from datetime import datetime
from glob import glob

from data_utils import MolDataModule
from sae_module import SAEModule
from config import *

def main():
    ptl.seed_everything(SEED)

    output_dir = (
        f"results_layer{BASE_HOOK_POS}_exp{EXP_F}_k{K}_{SEED}"
    )
    if not os.path.exists(f'../{output_dir}'):
        os.mkdir(f'../{output_dir}')

    run_name = (
        f"molformer_{datetime.now().strftime('%Y%m%d-%H%M%S')}_"
        f"layer{BASE_HOOK_POS}_exp{EXP_F}_k{K}"
    )
    wandb_logger = WandbLogger(
        project='intermol',
        name=run_name,
        save_dir=os.path.join(f'../{output_dir}', 'wandb')
    )

    data = MolDataModule(
        data_pth=DATA_PATH,
        col_sele=COL_SELE,
        batch_size=BATCH_SIZE,
        train_size=TRAIN_SIZE
    )
    model = SAEModule(
        exp_f=EXP_F,
        k=K,
        batch_size=BATCH_SIZE,
        base_hook_pos=BASE_HOOK_POS,
        dead_steps_thresh=DEAD_STEPS_THRESH
    )
    wandb_logger.watch(model, log='all')

    ckpt_callback = ModelCheckpoint(
        dirpath=os.path.join(f'../{output_dir}', 'ckpts'),
        filename=run_name + '-{step}-{avg_mse_loss:.2f}',
        save_top_k=3,
        monitor="train_loss",
        mode="min",
        save_last=True
    )

    trainer = ptl.Trainer(
        max_epochs=EPOCHS,
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

    if WEIGHTS_PATH != None:
        trainer.fit(model, data, ckpt_path=WEIGHTS_PATH)
    else:
        trainer.fit(model, data)

    #trainer.test(model, data)

    try:
        for checkpoint in glob(os.path.join(f'../{output_dir}', "ckpts", "*.ckpt")):
            wandb.log_artifact(checkpoint, type="model")
    finally:
        wandb.finish()


if __name__ == '__main__':
    main()