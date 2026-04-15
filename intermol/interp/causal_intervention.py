import torch
from typing import Optional

from intermol.main.inference import SAEWithBaseModel

class LatentAblationModule(SAEWithBaseModel):
    @torch.no_grad()
    def ablate(
        self,
        smi: str,
        f: int | torch.IntTensor,
        value: float,
        do_scale: bool = False,
        layer_idx: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        k: Optional[int] = None
    ) -> tuple[torch.Tensor, tuple]:
        key = self._resolve_config(layer_idx, hidden_dim, k)

        enc = self.tokenize_to_tensor(smi)
        hs = self.base_model(**enc, output_hidden_states=True).hidden_states
        base_acts = hs[key[0]]

        mod_recons = self._sae_modify(base_acts, key, f, value, do_scale)
        mod_out = self._base_modify(key[0], enc, mod_recons)
        return mod_out.logits, mod_out.hidden_states

    @torch.no_grad()
    def _sae_modify(
        self,
        base_acts: torch.Tensor,
        key: tuple[int, int, int],
        f: int | torch.IntTensor,
        value: float,
        do_scale: bool = False
    ):
        sae = self._get_sae(key)
        acts, mu, std = sae.encode(base_acts)
        acts = sae.topK_activation(acts, key[2])

        # modify
        if do_scale:
            acts[:, :, f] *= value
        else:
            acts[:, :, f] = value
        mod_recons = sae.decode(acts, mu, std)
        return mod_recons

    @torch.no_grad()
    def _base_modify(self, layer_idx: int, enc, acts):
        def hook_fn(module, input, output):
            return acts

        target_layer = (
            self.base_model.molformer.encoder.layer[layer_idx - 1].output
        )
        hook = target_layer.register_forward_hook(hook_fn)

        out = self.base_model(**enc, output_hidden_states=True)
        hook.remove()

        return out

class HeadAblationModule(SAEWithBaseModel):
    @torch.no_grad()
    def ablate(
        self,
        smi: str,
        head: int | list[int],
        value: float,
        do_scale: bool = False,
        layer_idx: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        k: Optional[int] = None
    ) -> torch.Tensor:
        key = self._resolve_config(layer_idx, hidden_dim, k)
        enc = self.tokenize_to_tensor(smi)
        out = self._head_modify(enc, key[0], head, value, do_scale)

        mod_base_acts = out.hidden_states[key[0]]
        mod_acts = self._get_sae(key).encode_latents(mod_base_acts)
        return mod_acts

    @torch.no_grad()
    def _head_modify(
        self,
        enc,
        layer_idx: int,
        head: int | list[int],
        value: float,
        do_scale: bool = False
    ):
        def hook_fn(module, input, output):
            ctx = output[0].clone()
            b, seq, hidden_dim = ctx.shape
            num_heads = self.base_model.config.num_attention_heads
            head_dim  = hidden_dim // num_heads

            ctx = ctx.view(b, seq, num_heads, head_dim)
            if do_scale:
                ctx[:, :, head, :] *= value
            else:
                ctx[:, :, head, :] = value
            ctx = ctx.view(b, seq, hidden_dim)
            return (ctx,)

        # register hook
        target_layer = (
            self.base_model.molformer.encoder.layer[layer_idx - 1].attention.self
        )
        hook = target_layer.register_forward_hook(hook_fn)

        out = self.base_model(**enc, output_hidden_states=True)
        hook.remove()

        return out
