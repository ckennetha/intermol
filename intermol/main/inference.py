import torch

from intermol.main.sae import SparseAutoencoder
from intermol.main.utils import load_model_from_file, load_model_from_HF

class SAEInferenceModule():
    def __init__(
        self,
        hidden_dim: int,
        k: int,
        sae_pth: str,
        layer_idx: int,
        model_name: str = 'ibm/MoLFormer-XL-both-10pct',
        device_name: str = 'auto'
    ):
        # device
        if device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_name)

        # models
        self.tokenizer, self.base_model = load_model_from_HF(model_name)
        self.base_model.to(self.device)

        self.sae = SparseAutoencoder(hidden_dim, k)
        self.sae = load_model_from_file(self.sae, sae_pth)
        self.sae.to(self.device)

        self.k = k
        self.layer_idx = layer_idx

    def tokenize(self, smi: str) -> list[str]:
        return self.tokenizer.tokenize(smi)

    def tokenize_to_tensor(self, smi: str) -> torch.Tensor:
        return self.tokenizer(smi, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def encode_both(self, smi: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenize_to_tensor(smi)
        base_outs = self.base_model(**enc, output_hidden_states=True)

        base_acts = base_outs.hidden_states[self.layer_idx]
        acts = self.sae.encode_latents(base_acts)
        return base_acts, acts

    @torch.no_grad()
    def ablate_latents(
        self,
        smi: str,
        f: int | torch.IntTensor,
        value: float,
        do_scale: bool = False
    ) -> tuple[torch.Tensor, tuple]:
        enc = self.tokenize_to_tensor(smi)
        base_outs = self.base_model(**enc, output_hidden_states=True)
        base_acts = base_outs.hidden_states[self.layer_idx]

        mod_recons = self._sae_modify(base_acts, f, value, do_scale)
        mod_outs = self._base_modify(enc, mod_recons)
        return mod_outs.logits, mod_outs.hidden_states

    @torch.no_grad()
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        x_norm = self.base_model.molformer.LayerNorm(x)
        logits = self.base_model.lm_head(x_norm)
        return logits

    @torch.no_grad()
    def _sae_modify(
        self,
        base_acts: torch.Tensor,
        f: int | torch.IntTensor,
        value: float,
        do_scale: bool = False
    ) -> torch.Tensor:
        acts, mu, std = self.sae.encode(base_acts)
        acts = self.sae.topK_activation(acts, k=self.k)

        # modify
        if do_scale:
            acts[:, :, f] *= value
        else:
            acts[:, :, f] = value

        mod_recons = self.sae.decode(acts, mu, std)
        return mod_recons

    @torch.no_grad()
    def _base_modify(self, enc, acts):
        def hook_fn(module, input, output):
            return acts

        # modify
        target_layer = (
            self.base_model.molformer.encoder.layer[self.layer_idx - 1].output
        )
        hook = target_layer.register_forward_hook(hook_fn)

        outs = self.base_model(**enc, output_hidden_states=True)
        hook.remove()

        return outs
