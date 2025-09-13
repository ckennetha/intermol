"""
This code is adapted from https://github.com/etowahadams/interprot/blob/main/interprot/sae_model.py,
as it appears in 'From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models' (Adams et al. 2025),
https://doi.org/10.1101/2025.02.06.636901
"""

import torch
import pytorch_lightning as ptl

from .sae import SparseAutoencoder, loss_fn
from .utils import load_hf_model, diff_cross_entropy

class SAEModule(ptl.LightningModule):
    def __init__(
        self,
        exp_f: int,
        k: int,
        base_hook_pos: int,
        lr: float,
        wd: float,
        base: str = 'ibm/MoLFormer-XL-both-10pct',
        base_dim: int = 768,
        batch_size: int = 128,
        dead_steps_thresh: int = 5000
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.tokenizer, self.base = load_hf_model(base)
        self.sae = SparseAutoencoder(
            exp_f, k, batch_size, base_dim, dead_steps_thresh
        )

        self.lr = lr
        self.wd = wd
        
        self.base_hook_pos = base_hook_pos

        self.val_results = []

    def forward(self, X):
        return self.sae(X)

    def training_step(self, batch, batch_idx):
        mols = self.tokenizer(batch['smi'], padding=True, return_tensors='pt').to(self.device)
        batch_size = len(mols)

        _, _, acts = self.get_base_out(mols, self.base_hook_pos)
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
            batch_size=batch_size
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
            batch_size=batch_size
        )

        return loss

    def validation_step(self, batch, batch_idx):
        mols = batch['smi']
        batch_size = len(mols)

        diff_ces = torch.zeros(batch_size, device=self.device)
        mse_losses = torch.zeros(batch_size, device=self.device)
        for i, mol in enumerate(mols):
            mol_enc = self.tokenizer(mol, return_tensors='pt').to(self.device)
            tokens, ori_logits, acts = self.get_base_out(mol_enc, self.base_hook_pos)
            
            recons = self.sae.forward_val(acts)
            mse_loss, _ = loss_fn(acts, recons, None)
            mse_losses[i] = mse_loss

            recons_logits = self._modify_base_acts(mol_enc, recons, self.base_hook_pos)
            diff_ce = diff_cross_entropy(ori_logits, recons_logits, tokens)
            diff_ces[i] = diff_ce

        val_result = {
            'mse_loss': mse_losses.mean(),
            'diff_ce': diff_ces.mean()
        }
        self.val_results.append(val_result)

        return val_result

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
    def get_base_out(self, enc, layer_idx: int):
        outs = self.base(**enc, output_hidden_states=True)
        acts = outs.hidden_states[layer_idx]
        return enc['input_ids'], outs.logits, acts
    
    @torch.no_grad()
    def _modify_base_acts(self, enc, acts, layer_idx: int):
        def hook_fn(module, input, output):
            return acts
        
        acts_to_modify = self.base.molformer.encoder.layer[layer_idx - 1].output
        hook = acts_to_modify.register_forward_hook(hook_fn)
        
        outs = self.base(**enc)
        hook.remove()
        
        return outs.logits