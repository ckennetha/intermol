"""
This code is adapted from:
    1) https://github.com/etowahadams/interprot/blob/main/interprot/sae_model.py,
       as it appears in 'From Mechanistic Interpretability to Mechanistic Biology: Training, Evaluating, and Interpreting Sparse Autoencoders on Protein Language Models' (Adams et al. 2025),
       https://doi.org/10.1101/2025.02.06.636901
    2) https://github.com/tylercosgrove/sparse-autoencoder-mistral7b/blob/main/sae.py,
       based on 'Scaling and evaluating sparse autoencoders' (Gao et al. 2024),
       https://doi.org/10.48550/arXiv.2406.04093
"""

import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from typing import Optional

class SparseAutoencoder(nn.Module):
    def __init__(
        self,
        exp_f: int,
        k: int,
        base_dim: int=768,
        batch_size: int=128,
        dead_steps_thresh: int=5000
    ):
        super().__init__()

        self.hidden_dim = base_dim * exp_f
        self.k = k
        self.dead_steps_thresh = dead_steps_thresh / batch_size

        self.w_enc = nn.Parameter(torch.empty(base_dim, self.hidden_dim))
        self.w_dec = nn.Parameter(torch.empty(self.hidden_dim, base_dim))

        self.b_enc = nn.Parameter(torch.zeros(self.hidden_dim))
        self.b_pre = nn.Parameter(torch.zeros(base_dim))

        nn.init.kaiming_uniform_(self.w_enc, a=math.sqrt(5))
        self.w_dec.data = self.w_enc.data.T.clone()
        self.w_dec.data /= self.w_dec.data.norm(dim=0)

        self.register_buffer("stats_last_nonzero", torch.zeros(self.hidden_dim, dtype=torch.long))

    def topK_activation(self, x: torch.Tensor, k: int) -> torch.Tensor:
        topk = torch.topk(x, k=k, dim=-1, sorted=False)
        values = F.relu(topk.values)
        result = torch.zeros_like(x)
        result.scatter_(-1, topk.indices, values)
        return result

    def LN(
        self, x: torch.Tensor, eps: float=1e-5
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu = x.mean(dim=-1, keepdim=True)
        x = x - mu
        std = x.std(dim=-1, keepdim=True)
        x = x / (std + eps)
        return x, mu, std

    def auxk_mask_fn(self) -> torch.Tensor:
        dead_mask = self.stats_last_nonzero > self.dead_steps_thresh
        return dead_mask

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, mu, std = self.LN(x)
        x = x - self.b_pre

        pre_acts = x @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, k=self.k) # (batch_size, embed_dim, hidden_dim)

        self.stats_last_nonzero *= (latents == 0).all(dim=(0, 1)).long()
        self.stats_last_nonzero += 1

        dead_mask = self.auxk_mask_fn()
        num_dead = dead_mask.sum().item()

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu

        if num_dead > 0:
            k_aux = min(x.shape[-1] // 2, num_dead)

            auxk_latents = torch.where(dead_mask[None], pre_acts, -torch.inf)
            auxk_acts = self.topK_activation(auxk_latents, k=k_aux)

            auxk = auxk_acts @ self.w_dec + self.b_pre
            auxk = auxk * std + mu
        else:
            auxk = None

        return recons, auxk, num_dead

    @torch.no_grad()
    def forward_val(self, x: torch.Tensor) -> torch.Tensor:
        x, mu, std = self.LN(x)
        x = x - self.b_pre
        pre_acts = x @ self.w_enc + self.b_enc
        latents = self.topK_activation(pre_acts, self.k)

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons

    @torch.no_grad()
    def norm_weights(self) -> None:
        self.w_dec.data /= self.w_dec.data.norm(dim=0)

    @torch.no_grad()
    def norm_grad(self) -> None:
        dot_products = torch.sum(self.w_dec.data * self.w_dec.grad, dim=0)
        self.w_dec.grad.sub_(self.w_dec.data * dot_products.unsqueeze(0))

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x, mu, std = self.LN(x)
        x = x - self.b_pre
        acts = x @ self.w_enc + self.b_enc
        return acts, mu, std

    @torch.no_grad()
    def decode(self, acts: torch.Tensor, mu: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        latents = self.topK_activation(acts, self.k)

        recons = latents @ self.w_dec + self.b_pre
        recons = recons * std + mu
        return recons
    
    @torch.no_grad()
    def get_latents(self, x: torch.Tensor) -> torch.Tensor:
        acts, _, _ = self.encode(x)
        latents = self.topK_activation(acts, self.k)
        return latents


def loss_fn(
    x: torch.Tensor, recons: torch.Tensor, auxk: Optional[torch.Tensor]=None
) -> tuple[torch.Tensor, torch.Tensor]:
    mse_scale = 1
    auxk_coeff = 1.0 / 32.0

    mse_loss = mse_scale * F.mse_loss(recons, x)
    if auxk is not None:
        auxk_loss = auxk_coeff * F.mse_loss(auxk, x - recons).nan_to_num(0)
    else:
        auxk_loss = torch.tensor(0.0)
    return mse_loss, auxk_loss