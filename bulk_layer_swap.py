"""
Faz 3 — Gerçek KV'siz katman swap (BulkTriggerDecoderLayerV2)
=============================================================
Son N Llama katmanında self-attention kaldırılır; BulkState + GeneratorBlock kullanılır.
Erken katmanlar KV-cache; kuyruk katmanları sabit BulkState (KV yok).

Opsiyonel sliding_window: erken katman KV'sini son W token ile sınırlar → uzun prompt RAM demo.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from bulk_state import BulkStateTensors
from bulk_trigger_v2 import BulkTriggerDecoderLayerV2


def _create_causal_mask_compat(
    config,
    hidden: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    cache_position: torch.Tensor,
    past_key_values: Any,
    position_ids: torch.Tensor,
) -> Any:
    """transformers 4.x (input_embeds) ve 5.9+ (inputs_embeds, cache_position yok) uyumu."""
    from transformers.models.llama.modeling_llama import create_causal_mask
    import inspect

    sig = inspect.signature(create_causal_mask)
    params = sig.parameters
    kwargs: dict[str, Any] = {
        "config": config,
        "attention_mask": attention_mask,
        "past_key_values": past_key_values,
        "position_ids": position_ids,
    }
    if "inputs_embeds" in params:
        kwargs["inputs_embeds"] = hidden
    elif "input_embeds" in params:
        kwargs["input_embeds"] = hidden
    else:
        raise RuntimeError("create_causal_mask: embed kwarg bulunamadı")
    if "cache_position" in params:
        kwargs["cache_position"] = cache_position
    return create_causal_mask(**kwargs)


class CompactBulkCore(nn.Module):
    """
    H=2048 → d_bulk=256 bottleneck — ~3M param/katman (554M yerine).
    Mac 24GB için güvenli eğitim.
    """

    def __init__(
        self,
        hidden_size: int,
        d_bulk: int = 256,
        k_short: int = 8,
        medium_interval: int = 16,
        long_interval: int = 128,
        trigger_stride: int = 4,
        adaptive_trigger: bool = False,
        surprise_threshold: float = 1.0,
        n_heads: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.d_bulk = d_bulk
        self.k_short = k_short
        self.proj_in = nn.Linear(hidden_size, d_bulk, bias=False)
        self.proj_out = nn.Linear(d_bulk, hidden_size, bias=False)
        self.bulk = BulkTriggerDecoderLayerV2(
            d_model=d_bulk,
            n_heads=n_heads,
            k_short=k_short,
            medium_interval=medium_interval,
            long_interval=long_interval,
            trigger_stride=trigger_stride,
            adaptive_trigger=adaptive_trigger,
            surprise_threshold=surprise_threshold,
            dropout=0.0,
        )
        self.gate = nn.Parameter(torch.tensor(0.05))

    @property
    def d_state(self) -> int:
        return self.d_bulk

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.bulk(self.proj_in(x))
        return x + self.gate.tanh() * self.proj_out(h)

    def forward_with_state(self, x: torch.Tensor) -> tuple[torch.Tensor, BulkStateTensors]:
        h, state = self.bulk.forward_with_state(self.proj_in(x))
        return x + self.gate.tanh() * self.proj_out(h), state

    def forward_step(
        self,
        x: torch.Tensor,
        window: torch.Tensor,
        state: BulkStateTensors,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        h, state = self.bulk.forward_step(self.proj_in(x), self.proj_in(window), state)
        return x + self.gate.tanh() * self.proj_out(h), state


def _bulk_d_state(bulk: nn.Module, fallback: int) -> int:
    return getattr(bulk, "d_state", getattr(bulk, "d_bulk", fallback))


@dataclass
class KVFreeCache:
    """Erken katmanlar için KV; kuyruk katmanları için BulkState."""

    early_past: Any = None
    bulk_states: list[BulkStateTensors] = field(default_factory=list)
    tail_hidden_hist: deque = field(default_factory=deque)
    pos: int = 0


class BulkTriggerLlamaLayer(nn.Module):
    """LlamaDecoderLayer — self_attn yerine Bulk (full veya compact)."""

    def __init__(self, llama_layer: nn.Module, bulk_layer: nn.Module):
        super().__init__()
        self.input_layernorm = llama_layer.input_layernorm
        self.bulk = bulk_layer
        self.post_attention_layernorm = llama_layer.post_attention_layernorm
        self.mlp = llama_layer.mlp
        self.k_short = bulk_layer.k_short
        self._compact = isinstance(bulk_layer, CompactBulkCore)

    def _layer_dtype(self) -> torch.dtype:
        return self.mlp.gate_proj.weight.dtype

    def _to_layer(self, x: torch.Tensor) -> torch.Tensor:
        dt = self._layer_dtype()
        return x if x.dtype == dt else x.to(dtype=dt)

    def _bulk_attn(self, residual: torch.Tensor, h_ln: torch.Tensor) -> torch.Tensor:
        wdt = self._layer_dtype()
        if self._compact:
            res_f, h_f = residual.float(), h_ln.float()
            h_out = self.bulk(h_f)
            return (res_f + (h_out - h_f)).to(wdt)
        return (residual + self.bulk(h_ln.float())).to(wdt)

    def _bulk_attn_state(
        self, residual: torch.Tensor, h_ln: torch.Tensor,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        wdt = self._layer_dtype()
        if self._compact:
            res_f, h_f = residual.float(), h_ln.float()
            h_out, state = self.bulk.forward_with_state(h_f)
            return (res_f + (h_out - h_f)).to(wdt), state
        h_out, state = self.bulk.forward_with_state(h_ln.float())
        return (residual + h_out).to(wdt), state

    def _bulk_attn_step(
        self, residual: torch.Tensor, h_ln: torch.Tensor,
        window: torch.Tensor, state: BulkStateTensors,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        wdt = self._layer_dtype()
        if self._compact:
            res_f, h_f = residual.float(), h_ln.float()
            w_f = window.float()
            h_out, state = self.bulk.forward_step(h_f, w_f, state)
            return (res_f + (h_out - h_f)).to(wdt), state
        h_out, state = self.bulk.forward_step(h_ln.float(), window.float(), state)
        return (residual + h_out).to(wdt), state

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> tuple[torch.Tensor, ...]:
        del kwargs
        hidden_states = self._to_layer(hidden_states)
        residual = hidden_states
        h_ln = self.input_layernorm(hidden_states)
        h = self._bulk_attn(residual, h_ln)
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return (residual + h,)

    def forward_with_state(
        self, hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        hidden_states = self._to_layer(hidden_states)
        residual = hidden_states
        h_ln = self.input_layernorm(hidden_states)
        h, state = self._bulk_attn_state(residual, h_ln)
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h, state

    def forward_step(
        self,
        hidden_states: torch.Tensor,
        window: torch.Tensor,
        state: BulkStateTensors,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        hidden_states = self._to_layer(hidden_states)
        window = self._to_layer(window)
        residual = hidden_states
        h_ln = self.input_layernorm(hidden_states)
        h, state = self._bulk_attn_step(residual, h_ln, window, state)
        residual = h
        h = self.post_attention_layernorm(h)
        h = self.mlp(h)
        return residual + h, state


def install_bulk_v2_swap(
    base_model: nn.Module,
    n_swap: int,
    k_short: int = 8,
    medium_interval: int = 16,
    long_interval: int = 128,
    trigger_stride: int = 4,
    adaptive_trigger: bool = False,
    surprise_threshold: float = 1.0,
    n_heads: int = 4,
    compact: bool = True,
    d_bulk: int = 256,
) -> tuple[list[BulkTriggerLlamaLayer], int, int]:
    """Son n_swap katmanı BulkTriggerLlamaLayer ile değiştir."""
    layers = base_model.model.layers
    cfg = base_model.config
    H = cfg.hidden_size
    n_swap = min(n_swap, len(layers))
    n_early = len(layers) - n_swap
    bulk_layers: list[BulkTriggerLlamaLayer] = []

    for i in range(n_early, len(layers)):
        llama_layer = layers[i]
        if compact:
            bulk_core: nn.Module = CompactBulkCore(
                H, d_bulk=d_bulk, k_short=k_short,
                medium_interval=medium_interval, long_interval=long_interval,
                trigger_stride=trigger_stride, adaptive_trigger=adaptive_trigger,
                surprise_threshold=surprise_threshold, n_heads=n_heads,
            )
        else:
            bulk_core = BulkTriggerDecoderLayerV2(
                d_model=H, n_heads=n_heads, k_short=k_short,
                medium_interval=medium_interval, long_interval=long_interval,
                trigger_stride=trigger_stride, adaptive_trigger=adaptive_trigger,
                surprise_threshold=surprise_threshold, dropout=0.0,
            )
        swapped = BulkTriggerLlamaLayer(llama_layer, bulk_core)
        bulk_layers.append(swapped)
        layers[i] = swapped

    return bulk_layers, n_early, n_swap


def _build_window(hist: deque, h_t: torch.Tensor, k_short: int) -> torch.Tensor:
    """Son k hidden → [B, k_short, H]. h_t: [B, 1, H]."""
    cur = h_t.squeeze(1)
    items = list(hist)[-(k_short - 1) :] + [cur]
    if len(items) < k_short:
        pad = [items[0]] * (k_short - len(items))
        items = pad + items
    return torch.stack(items[-k_short:], dim=1)


def truncate_past_key_values(past: Any, max_len: int) -> Any:
    """Erken katman KV'sini son max_len token ile sınırla."""
    if past is None or max_len <= 0:
        return past
    if hasattr(past, "crop"):
        past.crop(max_len)
        return past
    if hasattr(past, "key_cache"):
        for i in range(len(past.key_cache)):
            if past.key_cache[i].size(-2) > max_len:
                past.key_cache[i] = past.key_cache[i][..., -max_len:, :]
                past.value_cache[i] = past.value_cache[i][..., -max_len:, :]
        return past
    truncated = []
    for layer_past in past:
        if layer_past is None:
            truncated.append(None)
            continue
        k, v = layer_past
        if k.size(-2) > max_len:
            k = k[..., -max_len:, :]
            v = v[..., -max_len:, :]
        truncated.append((k, v))
    return tuple(truncated)


class TinyLlamaKVFreeTail(nn.Module):
    """
    TinyLlama + son N katman BulkTriggerDecoderLayerV2 (KV'siz kuyruk).
    Erken katmanlar standart KV; kuyruk BulkState ile O(1) bellek.
    """

    def __init__(
        self,
        base_model: nn.Module,
        n_swap_layers: int = 4,
        k_short: int = 8,
        medium_interval: int = 16,
        long_interval: int = 128,
        trigger_stride: int = 4,
        adaptive_trigger: bool = False,
        surprise_threshold: float = 1.0,
        sliding_window: int = 512,
        freeze_base: bool = True,
        compact: bool = True,
        d_bulk: int = 256,
    ):
        super().__init__()
        self.base = base_model
        self.k_short = k_short
        self.sliding_window = sliding_window
        self.freeze_base = freeze_base
        self.compact = compact
        self.d_bulk = d_bulk

        n_heads = max(1, getattr(base_model.config, "num_attention_heads", 32) // 8)
        self.bulk_layers, self.n_early, self.n_swap = install_bulk_v2_swap(
            base_model, n_swap_layers, k_short, medium_interval, long_interval,
            trigger_stride, adaptive_trigger, surprise_threshold, n_heads,
            compact=compact, d_bulk=d_bulk,
        )
        self._bulk_module_list = nn.ModuleList(self.bulk_layers)
        self._cache: Optional[KVFreeCache] = None

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False
        for layer in self.bulk_layers:
            for p in layer.bulk.parameters():
                p.requires_grad = True

    @property
    def config(self):
        return self.base.config

    def trainable_parameters(self):
        for layer in self.bulk_layers:
            yield from layer.bulk.parameters()

    def reset_cache(self, batch: int, device, dtype):
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            dtype = torch.float32
        H = self.config.hidden_size
        states = [
            BulkStateTensors.zeros(batch, _bulk_d_state(layer.bulk, H), device, dtype)
            for layer in self.bulk_layers
        ]
        self._cache = KVFreeCache(bulk_states=states, tail_hidden_hist=deque(), pos=0)

    def _base_dtype(self) -> torch.dtype:
        return next(self.base.parameters()).dtype

    def _bulk_dtype(self) -> torch.dtype:
        """BulkState/compact swap fp32 — erken katman base dtype (fp16 cuda, fp32 mps)."""
        return torch.float32

    def _to_base(self, x: torch.Tensor) -> torch.Tensor:
        dt = self._base_dtype()
        return x if x.dtype == dt else x.to(dtype=dt)

    def _to_bulk(self, x: torch.Tensor) -> torch.Tensor:
        return x.float() if x.dtype != torch.float32 else x

    def _from_bulk(self, x: torch.Tensor) -> torch.Tensor:
        return self._to_base(x)

    def _embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        m = self.base.model
        return m.embed_tokens(input_ids)

    def _run_early(
        self,
        hidden: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        past_key_values: Any,
        use_cache: bool,
        cache_position: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Any]:
        from transformers.cache_utils import DynamicCache

        model = self.base.model
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=model.config)

        if cache_position is None:
            past_seen = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen,
                past_seen + hidden.shape[1],
                device=hidden.device,
            )

        position_ids = cache_position.unsqueeze(0)
        causal_mask = _create_causal_mask_compat(
            model.config,
            hidden,
            attention_mask,
            cache_position,
            past_key_values,
            position_ids,
        )
        position_embeddings = model.rotary_emb(hidden, position_ids)

        for layer in model.layers[: self.n_early]:
            hidden = layer(
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                use_cache=use_cache,
            )

        return hidden, past_key_values

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        device = input_ids.device
        self.reset_cache(B, device, self._bulk_dtype())

        hidden = self._to_base(self._embed(input_ids))
        cache = self._cache
        assert cache is not None

        hidden, cache.early_past = self._run_early(
            hidden, attention_mask, None, use_cache=True,
        )

        if self.sliding_window > 0:
            cache.early_past = truncate_past_key_values(cache.early_past, self.sliding_window)

        for t in range(T):
            cache.tail_hidden_hist.append(hidden[:, t, :].detach())

        h_bulk = self._to_bulk(hidden)
        for idx, bulk_layer in enumerate(self.bulk_layers):
            h_bulk, cache.bulk_states[idx] = bulk_layer.forward_with_state(h_bulk)
        hidden = h_bulk

        cache.pos = T
        normed = self.base.model.norm(hidden)
        return self.base.lm_head(normed)[:, -1, :]

    @torch.inference_mode()
    def decode_step(self, token_id: torch.Tensor) -> torch.Tensor:
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        cache = self._cache
        assert cache is not None

        device = token_id.device
        cache_position = torch.tensor([cache.pos], device=device)

        hidden = self._to_base(self._embed(token_id))
        hidden, cache.early_past = self._run_early(
            hidden, None, cache.early_past, use_cache=True, cache_position=cache_position,
        )

        if self.sliding_window > 0:
            cache.early_past = truncate_past_key_values(cache.early_past, self.sliding_window)

        window = _build_window(cache.tail_hidden_hist, hidden, self.k_short)
        cache.tail_hidden_hist.append(hidden.squeeze(1).detach())

        h_bulk = self._to_bulk(hidden)
        w_bulk = self._to_bulk(window)
        for idx, bulk_layer in enumerate(self.bulk_layers):
            h_bulk, cache.bulk_states[idx] = bulk_layer.forward_step(
                h_bulk, w_bulk, cache.bulk_states[idx],
            )
        hidden = h_bulk

        cache.pos += 1
        normed = self.base.model.norm(hidden)
        return self.base.lm_head(normed)[:, -1, :]

    @torch.inference_mode()
    def generate_cached(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        logits = self.prefill(input_ids)
        next_tok = logits.argmax(-1, keepdim=True)
        tokens = [next_tok]
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_tok.squeeze(1))
            next_tok = logits.argmax(-1, keepdim=True)
            tokens.append(next_tok)
        return torch.cat(tokens, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ):
        import torch.nn.functional as F
        from bulk_hybrid import HybridOutput

        with torch.no_grad():
            hidden = self._to_base(self._embed(input_ids))
            hidden, _ = self._run_early(hidden, attention_mask, None, use_cache=False)

        h_bulk = self._to_bulk(hidden)
        for bulk_layer in self.bulk_layers:
            h_bulk = bulk_layer(h_bulk)[0]
        hidden = self._from_bulk(h_bulk)

        hidden = self.base.model.norm(hidden)
        logits = self.base.lm_head(hidden)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return HybridOutput(logits=logits, loss=loss)


def estimate_early_kv_bytes(
    seq_len: int,
    n_early_layers: int,
    d_model: int,
    batch: int = 1,
    dtype_bytes: int = 2,
) -> int:
    return 2 * n_early_layers * batch * seq_len * d_model * dtype_bytes


def estimate_bulk_tail_bytes(
    n_swap_layers: int,
    d_model: int,
    batch: int = 1,
    dtype_bytes: int = 4,
    d_bulk: int | None = None,
) -> int:
    d = d_bulk if d_bulk is not None else d_model
    return n_swap_layers * batch * 3 * d * dtype_bytes


def save_bulk_swap_checkpoint(model: TinyLlamaKVFreeTail, path: str) -> None:
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "n_swap": model.n_swap,
            "n_early": model.n_early,
            "sliding_window": model.sliding_window,
            "compact": model.compact,
            "d_bulk": model.d_bulk,
            "bulk_layers": [layer.bulk.state_dict() for layer in model.bulk_layers],
        },
        p,
    )


def load_bulk_swap_checkpoint(model: TinyLlamaKVFreeTail, path: str, device) -> None:
    from pathlib import Path
    ckpt = torch.load(Path(path), map_location=device, weights_only=True)
    if ckpt.get("compact") is False and model.compact:
        raise RuntimeError(
            "Checkpoint full-mod, model compact-mod. --full ile model oluştur."
        )
    if ckpt.get("compact") is True and not model.compact:
        raise RuntimeError(
            "Checkpoint compact-mod. compact=True ile model oluştur."
        )
    states = ckpt.get("bulk_layers", [])
    if not states:
        return
    if len(states) != len(model.bulk_layers):
        raise RuntimeError(
            f"Katman sayısı uyuşmuyor: ckpt={len(states)} model={len(model.bulk_layers)}"
        )
    for layer, state in zip(model.bulk_layers, ckpt.get("bulk_layers", [])):
        layer.bulk.load_state_dict(state)


def count_swap_params(model: TinyLlamaKVFreeTail) -> int:
    return sum(p.numel() for p in model.trainable_parameters())