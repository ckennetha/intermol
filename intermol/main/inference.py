import torch

from .sae import SparseAutoencoder
from .utils import load_model, load_hf_model

class SAEInferenceModule():
    def __init__(
        self, sae_weight: str, sae_exp_f: int, sae_k: int,
        layer_idx: int, base: str='ibm/MoLFormer-XL-both-10pct'
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.tokenizer, self.base_model = load_hf_model(base)
        self.base_model.to(self.device)

        self.sae = SparseAutoencoder(exp_f=sae_exp_f, k=sae_k)
        self.sae = load_model(self.sae, sae_weight).to(self.device)
        self.layer_idx = layer_idx

    def tokenize(self, smi: str) -> list[str]:
        return self.tokenizer.tokenize(smi)
    
    def tokenize_to_tensor(self, smi: str):
        return self.tokenizer(smi, return_tensors="pt").to(self.device)
    
    def get_all(self, smi: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenize_to_tensor(smi)
        out_base = self.get_base_out(enc)

        acts_base = out_base.hidden_states[self.layer_idx]
        acts_sae = self.sae.get_latents(acts_base)
        return out_base, acts_sae
    
    def get_steered(
        self, smi: str, factor: int, latent_idx: int, return_baseline: bool=False
    ) -> tuple:
        enc = self.tokenize_to_tensor(smi)
        out_base = self.get_base_out(enc)

        acts_base = out_base.hidden_states[self.layer_idx]
        recons = self.steer(acts_base, latent_idx, factor)
        sae_logits = self._modify_base_acts(enc, recons, self.layer_idx)

        if return_baseline:
            bl_recons = self.steer(acts_base, latent_idx)
            bl_sae_logits = self._modify_base_acts(enc, bl_recons, self.layer_idx)
            return sae_logits, bl_sae_logits
        else:
            return sae_logits

    def steer(self, acts_base, latent_idx: int, factor: int=1):
        acts, mu, std = self.sae.encode(acts_base)
        acts[:, :, latent_idx] *= factor
        recons = self.sae.decode(acts, mu, std)
        return recons

    @torch.no_grad()
    def get_base_out(self, enc):
        return self.base_model(**enc, output_hidden_states=True)
    
    @torch.no_grad()
    def _modify_base_acts(self, enc, acts):
        def hook_fn(module, input, output):
            return acts
        
        acts_to_modify = self.base_model.molformer.encoder.layer[self.layer_idx - 1].output
        hook = acts_to_modify.register_forward_hook(hook_fn)
        
        modify_out_base = self.base_model(**enc)
        hook.remove()
        
        return modify_out_base.logits