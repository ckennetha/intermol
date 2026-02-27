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
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device_name)

        # models
        self.tokenizer, self.base_model = load_model_from_HF(model_name)
        self.base_model.to(device)

        self.sae = SparseAutoencoder(hidden_dim, k)
        self.sae = load_model_from_file(self.sae, sae_pth)
        self.sae.to(device)

        self.layer_idx = layer_idx

    def tokenize(self, smi: str) -> list[str]:
        return self.tokenizer.tokenize(smi)

    def tokenize_to_tensor(self, smi: str) -> torch.Tensor:
        return self.tokenizer(smi, return_tensors="pt").to(self.device)

    @torch.no_grad()
    def encode_both(self, smi: str) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.tokenize_to_tensor(smi)
        base_acts = self.base_model(**enc, output_hidden_states=True)

        base_act = base_acts.hidden_states[self.layer_idx]
        acts = self.sae.encode_latents(base_act)
        return base_acts, acts
