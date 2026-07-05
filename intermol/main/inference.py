import torch

from dataclasses import dataclass
from typing import Optional

from intermol.main.sae import SparseAutoencoder
from intermol.main.utils import load_model_from_file, load_model_from_HF

# dataclass
@dataclass
class SAEInferenceConfig:
    layer: int
    hidden_dim: int
    k: int
    weights_path: str

    @property
    def key(self) -> str:
        return f"{self.layer}-{self.hidden_dim}-{self.k}"

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
            self._default_key = config[0].key

        self.sae: dict[str, SparseAutoencoder] = {}
        for cfg in config:
            sae = self._build_sae(cfg)
            sae = load_model_from_file(sae, cfg.weights_path)
            sae.to(self.device)
            self.sae[cfg.key] = sae

    def _resolve_config(self, key: Optional[str] = None) -> str:
        if key is None:
            if not hasattr(self, '_default_key'):
                raise ValueError(
                    "key must be provided explicitly "
                    "when more than one SAEInferenceConfig is loaded."
                )
            return self._default_key
        return key

    def _build_sae(self, cfg: SAEInferenceConfig) -> SparseAutoencoder:
        return SparseAutoencoder(hidden_dim=cfg.hidden_dim, k=cfg.k)

    def _get_sae(self, key: str) -> SparseAutoencoder:
        if key not in self.sae:
            raise KeyError(
                f"No SAE loaded for key {key}. "
                f"Loaded keys: {*self.sae,}"
            )
        return self.sae[key]

    @torch.no_grad()
    def get_activations(
        self, base_acts: torch.Tensor, key: Optional[str] = None
    ) -> torch.Tensor:
        sae = self._get_sae(self._resolve_config(key))
        return sae.encode_latents(base_acts)

class SAEWithBaseModel(SAEInferenceModule):
    def __init__(
        self,
        config: SAEInferenceConfig | list[SAEInferenceConfig],
        model_name: str,
        use_molformer: bool = False,
        device_name: str = 'auto'
    ):
        self.tokenizer, self.base_model = load_model_from_HF(model_name, use_molformer)

        super().__init__(config, device_name)
        self.base_model.to(self.device)

        # auto-locate encoder layers
        self._encoder_layers = self._resolve_encoder_layers()

    def _resolve_encoder_layers(self):
        for child in self.base_model.children():
            if hasattr(child, 'encoder') and hasattr(child.encoder, 'layer'):
                return child.encoder.layer
        raise ValueError("Could not auto-detect encoder layers.")

    def _build_sae(self, cfg: SAEInferenceConfig) -> SparseAutoencoder:
        return SparseAutoencoder(
            hidden_dim=cfg.hidden_dim,
            k=cfg.k,
            model_dim=self.base_model.config.hidden_size
        )

    def tokenize(self, smi: str) -> list[str]:
        return self.tokenizer.tokenize(smi)

    def tokenize_to_tensor(self, smi: str):
        return self.tokenizer(smi, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def get_hidden_states(self, smi: str) -> tuple[torch.Tensor, ...]:
        enc = self.tokenize_to_tensor(smi)
        return self.base_model(**enc, output_hidden_states=True).hidden_states

    def encode(
        self, smi: str, key: Optional[str] = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._resolve_config(key)
        layer = int(key.split("-")[0])

        base_acts = self.get_hidden_states(smi)[layer]
        sae_acts = self._get_sae(key).encode_latents(base_acts)
        return base_acts, sae_acts

    def encode_multi(
        self, smi: str, configs: list[SAEInferenceConfig]
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        hs = self.get_hidden_states(smi)
        output = {}
        for cfg in configs:
            base_acts = hs[cfg.layer]
            sae_acts = self.get_activations(base_acts, cfg.key)
            output[cfg.key] = (base_acts, sae_acts)
        return output
