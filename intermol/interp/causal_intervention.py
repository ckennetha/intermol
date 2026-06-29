import torch
from dataclasses import dataclass
from typing import Optional

from intermol.main.inference import SAEWithBaseModel

# dataclass
@dataclass
class AblationConfig:
    layer: int
    hidden_dim: int
    k: int
    f: int | torch.IntTensor
    value: float | torch.Tensor
    do_scale: bool = False

    @property
    def key(self) -> str:
        return f"{self.layer}-{self.hidden_dim}-{self.k}"

# core
class LatentAblationModule(SAEWithBaseModel):
    def ablate(
        self,
        smi: str,
        config: Optional[AblationConfig] = None,
        key: Optional[str] = None,
        f: Optional[int | torch.IntTensor] = None,
        value: Optional[float | torch.Tensor] = None,
        do_scale: bool = False
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        if config is None:
            if f is None or value is None:
                raise ValueError("must provide either `config` or both `f` and `value`")
            key = self._resolve_config(key)
            layer, hidden_dim, k = key.split("-")
            config = AblationConfig(
                layer=int(layer), hidden_dim=int(hidden_dim), k=int(k),
                f=f, value=value, do_scale=do_scale
            )
        logits, hs, _ = self._ablate_forward(smi, [config])
        return logits, hs

    def ablate_crosslayer(
        self, smi: str, configs: list[AblationConfig]
    )  -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[int, torch.Tensor]]:
        return self._ablate_forward(smi, configs)

    def _make_hook(
        self, config: AblationConfig, sae_acts_store: dict[int, torch.Tensor]
    ):
        sae = self._get_sae(config.key)
        def hook_fn(module, input, output):
            acts, mu, std = sae.encode(output)
            acts = sae.topK_activation(acts, config.k)
            if config.do_scale:
                acts[:, :, config.f] *= config.value
            else:
                acts[:, :, config.f] = config.value
            sae_acts_store[config.layer] = acts
            return sae.decode(acts, mu, std)
        return hook_fn

    @torch.no_grad()
    def _ablate_forward(
        self, smi: str, configs: list[AblationConfig]
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], dict[int, torch.Tensor]]:
        enc = self.tokenize_to_tensor(smi)
        mod_sae = {}
        hooks = []

        for cfg in configs:
            target = self.base_model.molformer.encoder.layer[cfg.layer - 1].output
            hooks.append(target.register_forward_hook(self._make_hook(cfg, mod_sae)))

        try:
            mod_out = self.base_model(**enc, output_hidden_states=True)
        finally:
            for h in hooks:
                h.remove()

        return mod_out.logits, mod_out.hidden_states, mod_sae
