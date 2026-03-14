"""
This code is adapted from: https://github.com/etowahadams/interprot/blob/main/interprot/utils.py,
as described in Adams et al. (2025), 'From Mechanistic Interpretability to Mechanistic Biology:
Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models'
https://doi.org/10.1101/2025.02.06.636901
"""

import torch
import pytorch_lightning as ptl

from intermol.main.sae import SparseAutoencoder, loss_fn, delta_ce
from intermol.main.utils import load_model_from_HF

class SAEModule(ptl.LightningModule):
    def __init__(
        self,
        hidden_dim: int, k: int,
        model_hook_pos: int,
        lr: float, wd: float,
        model_name: str = 'ibm/MoLFormer-XL-both-10pct',
        model_dim: int = 768,
        batch_size: int = 128,
        dead_steps_threshold: int = 5000
    ):
        super().__init__()
        self.save_hyperparameters()

        # load models
        self.tokenizer, self.base_model = load_model_from_HF(model_name)
        self.sae = SparseAutoencoder(
            hidden_dim, k,
            model_dim,
            batch_size,
            dead_steps_threshold
        )

        self.lr = lr
        self.wd = wd
        self.model_hook_pos = model_hook_pos

        self.val_res = []

    def forward(self, X):
        return self.sae(X)

    def training_step(self, batch, batch_idx):
        mols = self.tokenizer(
            batch['smi'], padding=True, return_tensors='pt'
        ).to(self.device)
        bsz = len(mols)

        _, _, acts = self._base_encode(mols, self.model_hook_pos)
        recons, aux_k, num_dead = self(acts)
        mse_loss, aux_k_loss = loss_fn(acts, recons, aux_k)
        loss = mse_loss + aux_k_loss

        self.log(
            'train_loss',
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=bsz
        )
        self.log_dict(
            {
                'train_mse_loss': mse_loss,
                'train_aux_k_loss': aux_k_loss,
                'num_dead_neurons': num_dead
            },
            on_step=True,
            on_epoch=True,
            logger=True,
            batch_size=bsz
        )

        return loss

    def validation_step(self, batch, batch_idx):
        mols = batch['smi']
        bsz = len(mols)

        diff_ces = torch.zeros(bsz, device=self.device)
        mse_losses = torch.zeros(bsz, device=self.device)
        for i, mol in enumerate(mols):
            mol_enc = self.tokenizer(mol, return_tensors='pt').to(self.device)
            tokens, ori_logits, acts = self._base_encode(
                mol_enc, self.model_hook_pos
            )

            recons = self.sae.forward_val(acts)
            mse_loss, _ = loss_fn(acts, recons, None)
            mse_losses[i] = mse_loss

            recons_logits = self._base_modify(mol_enc, recons, self.model_hook_pos)
            diff_ce = delta_ce(ori_logits, recons_logits, tokens)
            diff_ces[i] = diff_ce

        val_r = {
            'mse_loss': mse_losses.mean(),
            'diff_ce': diff_ces.mean()
        }
        self.val_res.append(val_r)

        return val_r

    def on_validation_epoch_end(self):
        avg_mse_loss = torch.stack([r["mse_loss"] for r in self.val_results]).mean()
        avg_diff_ce = torch.stack([r["diff_ce"] for r in self.val_results]).mean()

        self.log_dict(
            {
                'val_mse_loss': avg_mse_loss,
                'val_diff_ce': avg_diff_ce,
            },
            on_epoch=True,
            prog_bar=True,
            logger=True
        )

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.wd,
        )

    def on_after_backward(self):
        self.sae.norm_weights()
        self.sae.norm_grad()

    @torch.no_grad()
    def _base_encode(self, enc, layer_idx: int):
        outs = self.base_model(**enc, output_hidden_states=True)
        acts = outs.hidden_states[layer_idx]
        return enc['input_ids'], outs.logits, acts

    @torch.no_grad()
    def _base_modify(self, enc, acts, layer_idx: int):
        def hook_fn(module, input, output):
            return acts

        target_layer = (
            self.base_model.molformer.encoder.layer[self.layer_idx - 1].output
        )
        hook = target_layer.register_forward_hook(hook_fn)

        outs = self.base_model(**enc)
        hook.remove()

        return outs.logits
