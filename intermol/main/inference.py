import torch

from dataclasses import dataclass
from typing import Optional

from intermol.main.sae import SparseAutoencoder
from intermol.main.utils import load_model_from_file, load_model_from_HF

# dataclass
@dataclass
class SAEInferenceConfig:
    layer_idx: int
    hidden_dim: int
    k: int
    sae_pth: str

# core
class SAEInferenceModule():
    def __init__(
        self,
        config: SAEInferenceConfig | list[SAEInferenceConfig],
        device_name: str = 'auto'
    ):
        # device
        if device_name == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device_name)

        # init SAE
        if isinstance(config, SAEInferenceConfig):
            config = [config]

        ## store config for convenience
        if len(config) == 1:
            self.layer_idx = config[0].layer_idx
            self.hidden_dim = config[0].hidden_dim
            self.k = config[0].k

        self.sae: dict[tuple[int, int, int], SparseAutoencoder] = {}
        for cfg in config:
            k = (cfg.layer_idx, cfg.hidden_dim, cfg.k)
            sae = SparseAutoencoder(hidden_dim=k[1], k=k[2])
            sae = load_model_from_file(sae, cfg.sae_pth)
            sae.to(self.device)
            self.sae[k] = sae

    def _resolve_config(
        self,
        layer_idx: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        k: Optional[int] = None
    ) -> tuple[int, int, int]:
        if any(x is None for x in (layer_idx, hidden_dim, k)):
            if not hasattr(self, 'layer_idx'):
                raise AttributeError(
                    "layer_idx, hidden_dim, and k must be provided explicitly "
                    "when more than one SAEInferenceConfig is loaded."
                )
        return (
            layer_idx if layer_idx is not None else self.layer_idx,
            hidden_dim if hidden_dim is not None else self.hidden_dim,
            k if k is not None else self.k
        )

    def _get_sae(self, key: tuple[int, int, int]) -> SparseAutoencoder:
        return self.sae[key]

    @torch.no_grad()
    def get_activations(
        self,
        base_acts: torch.Tensor,
        layer_idx: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        k: Optional[int] = None
    ) -> torch.Tensor:
        key = self._resolve_config(layer_idx, hidden_dim, k)
        sae = self._get_sae(key)
        return sae.encode_latents(base_acts)

class SAEWithBaseModel(SAEInferenceModule):
    def __init__(
        self,
        config: SAEInferenceConfig | list[SAEInferenceConfig],
        device_name: str = 'auto',
        model_name: str = 'ibm/MoLFormer-XL-both-10pct'
    ):
        super().__init__(config, device_name)

        # load base model
        self.tokenizer, self.base_model = load_model_from_HF(model_name)
        self.base_model.to(self.device)

    def tokenize(self, smi: str) -> list[str]:
        return self.tokenizer.tokenize(smi)

    def tokenize_to_tensor(self, smi: str) -> torch.Tensor:
        return self.tokenizer(smi, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def get_hidden_states(self, smi: str) -> torch.Tensor:
        enc = self.tokenize_to_tensor(smi)
        hs = self.base_model(**enc, output_hidden_states=True).hidden_states
        return hs

    def encode(
        self,
        smi: str,
        layer_idx: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        k: Optional[int] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._resolve_config(layer_idx, hidden_dim, k)
        base_acts = self.get_hidden_states(smi)[key[0]]
        acts = self._get_sae(key).encode_latents(base_acts)
        return base_acts, acts

    def encode_multi(
        self, smi: str, configs: list[SAEInferenceConfig]
    ) -> dict[tuple[int, int, int], tuple[torch.Tensor, torch.Tensor]]:
        hs = self.get_hidden_states(smi)
        output = {}
        for cfg in configs:
            key = (cfg.layer_idx, cfg.hidden_dim, cfg.k)
            base_acts = hs[cfg.layer_idx]
            acts = self.get_activations(base_acts, *key)
            output[key] = (base_acts, acts)
        return output
