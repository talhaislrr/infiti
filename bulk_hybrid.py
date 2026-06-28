"""
BulkTrigger Hibrit LLM — TinyLlama + BulkMemoryAdapter + Katman Swap
====================================================================
"""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bulk_state import BulkStateManager, BulkStateTensors, embedding_surprise
from bulk_memory_utils import token_loss_surprise


@dataclass
class HybridOutput:
    logits: torch.Tensor
    loss: Optional[torch.Tensor] = None


class BulkMemoryAdapter(nn.Module):
    """Hidden state üzerine BulkState gated residual enjekte eder (batch path)."""

    def __init__(
        self,
        hidden_size: int,
        k_short: int = 8,
        medium_interval: int = 16,
        long_interval: int = 128,
        trigger_stride: int = 4,
        adaptive_trigger: bool = False,
        surprise_threshold: float = 1.0,
    ):
        super().__init__()
        self.k_short = k_short
        self.bulk_mgr = BulkStateManager(
            hidden_size,
            k_short=k_short,
            medium_interval=medium_interval,
            long_interval=long_interval,
            adaptive_trigger=adaptive_trigger,
            surprise_threshold=surprise_threshold,
        )
        self.trigger_stride = trigger_stride
        self.adaptive_trigger = adaptive_trigger
        self.inject = nn.Linear(hidden_size * 3, hidden_size)
        self.gate = nn.Parameter(torch.tensor(0.05))

    def _windows(self, hidden: torch.Tensor) -> torch.Tensor:
        B, T, H = hidden.shape
        padded = F.pad(hidden, (0, 0, self.k_short - 1, 0))
        w = padded.unfold(1, self.k_short, 1)
        return w.permute(0, 1, 3, 2).contiguous()

    def forward(
        self,
        hidden: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        windows = self._windows(hidden)
        bulk_kv = self.bulk_mgr.evolve_sequence(
            windows, trigger_stride=self.trigger_stride, surprise=surprise,
        )
        B, T, _, H = bulk_kv.shape
        ctx = self.inject(bulk_kv.reshape(B, T, 3 * H))
        return hidden + self.gate.tanh() * ctx

    def forward_with_state(
        self,
        hidden: torch.Tensor,
        surprise: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        """Batch forward + final BulkState (hızlı prefill)."""
        windows = self._windows(hidden)
        bulk_kv, state = self.bulk_mgr.evolve_sequence_with_state(
            windows, trigger_stride=self.trigger_stride, surprise=surprise,
        )
        B, T, _, H = bulk_kv.shape
        ctx = self.inject(bulk_kv.reshape(B, T, 3 * H))
        return hidden + self.gate.tanh() * ctx, state

    def forward_step(
        self,
        h_t: torch.Tensor,
        window: torch.Tensor,
        state: BulkStateTensors,
    ) -> tuple[torch.Tensor, BulkStateTensors]:
        state = self.bulk_mgr.update(window, state)
        kv = state.as_kv()
        ctx = self.inject(kv.reshape(kv.size(0), -1)).unsqueeze(1)
        return h_t + self.gate.tanh() * ctx, state


class BulkLlamaSwapLayer(nn.Module):
    """Llama decoder katmanı + BulkMemoryAdapter (Faz 3 katman swap)."""

    def __init__(self, llama_layer: nn.Module, bulk_adapter: BulkMemoryAdapter):
        super().__init__()
        self.llama_layer = llama_layer
        self.bulk = bulk_adapter

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_value: Any = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        out = self.llama_layer(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
        if isinstance(out, tuple):
            h = out[0]
            rest = out[1:]
        else:
            h, rest = out, ()

        h_bulk = self.bulk(h)
        if rest:
            return (h_bulk,) + rest
        return h_bulk


class TinyLlamaWithBulk(nn.Module):
    """TinyLlama + BulkMemoryAdapter — base frozen, adapter trainable."""

    def __init__(
        self,
        base_model: nn.Module,
        k_short: int = 8,
        medium_interval: int = 16,
        long_interval: int = 128,
        trigger_stride: int = 4,
        adaptive_trigger: bool = False,
        surprise_threshold: float = 1.0,
        freeze_base: bool = True,
        n_swap_layers: int = 0,
    ):
        super().__init__()
        self.base = base_model
        self.freeze_base = freeze_base
        H = base_model.config.hidden_size
        self.adapter = BulkMemoryAdapter(
            H, k_short, medium_interval, long_interval,
            trigger_stride, adaptive_trigger, surprise_threshold,
        )
        self.k_short = k_short
        self.n_swap_layers = n_swap_layers
        self._state: BulkStateTensors | None = None
        self._past_key_values: Any = None
        self._hidden_hist: deque[torch.Tensor] = deque()
        self._swap_adapters = nn.ModuleList()

        if n_swap_layers > 0:
            self._install_swap_layers(n_swap_layers, H, k_short, medium_interval,
                                      long_interval, trigger_stride, adaptive_trigger,
                                      surprise_threshold)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad = False

    def _install_swap_layers(
        self, n: int, H: int, k_short: int, medium_interval: int, long_interval: int,
        trigger_stride: int, adaptive: bool, surprise_threshold: float,
    ) -> None:
        layers = self.base.model.layers
        n = min(n, len(layers))
        start = len(layers) - n
        for i in range(start, len(layers)):
            bulk = BulkMemoryAdapter(
                H, k_short, medium_interval, long_interval,
                trigger_stride, adaptive, surprise_threshold,
            )
            self._swap_adapters.append(bulk)
            layers[i] = BulkLlamaSwapLayer(layers[i], bulk)
        self.n_swap_layers = n

    def trainable_parameters(self):
        params = list(self.adapter.parameters())
        for b in self._swap_adapters:
            params += list(b.parameters())
        return iter(params)

    def reset_states(self, batch: int, device, dtype):
        H = self.base.config.hidden_size
        if dtype not in (torch.float32, torch.float16, torch.bfloat16):
            dtype = torch.float32
        self._state = BulkStateTensors.zeros(batch, H, device, dtype)

    def reset_generation_cache(self, batch: int, device, dtype):
        self.reset_states(batch, device, dtype)
        self._past_key_values = None
        self._hidden_hist.clear()

    def _float_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden.float() if hidden.dtype != torch.float32 else hidden

    def _build_hidden_window(self, h_t: torch.Tensor) -> torch.Tensor:
        cur = h_t.squeeze(1)
        hist = list(self._hidden_hist)[-(self.k_short - 1) :] + [cur]
        if len(hist) < self.k_short:
            pad = [hist[0]] * (self.k_short - len(hist))
            hist = pad + hist
        return torch.stack(hist[-self.k_short :], dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> HybridOutput:
        ctx = torch.no_grad() if self.freeze_base else nullcontext()
        with ctx:
            inner = self.base.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs,
            )
        hidden = inner.last_hidden_state
        if self.freeze_base:
            hidden = hidden.detach()

        if self.n_swap_layers > 0:
            h_out = hidden
        else:
            surprise = None
            if self.adapter.adaptive_trigger:
                surprise = embedding_surprise(hidden)
                if labels is not None:
                    with torch.no_grad():
                        base_logits = self.base.lm_head(hidden)
                        loss_sur = token_loss_surprise(base_logits, labels)
                    surprise = torch.maximum(surprise, loss_sur)
            h_out = self.adapter(hidden, surprise=surprise)

        logits = self.base.lm_head(h_out)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                labels.reshape(-1),
                ignore_index=-100,
            )
        return HybridOutput(logits=logits, loss=loss)

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        fast: bool = True,
    ) -> torch.Tensor:
        B = input_ids.size(0)
        self.reset_generation_cache(B, input_ids.device, torch.float32)

        outputs = self.base.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
        )
        hidden = self._float_hidden(outputs.last_hidden_state)
        self._past_key_values = outputs.past_key_values

        if self.n_swap_layers > 0:
            return self.base.lm_head(hidden)[:, -1, :]

        if fast:
            h_out, state = self.adapter.forward_with_state(hidden)
            self._state = state
            for t in range(hidden.size(1)):
                self._hidden_hist.append(hidden[:, t, :].detach())
            return self.base.lm_head(h_out)[:, -1, :]

        state = self._state
        assert state is not None
        h_out_last = None
        for t in range(hidden.size(1)):
            h_t = hidden[:, t : t + 1]
            window = self._build_hidden_window(h_t)
            h_out, state = self.adapter.forward_step(h_t, window, state)
            self._hidden_hist.append(hidden[:, t, :].detach())
            h_out_last = h_out
        self._state = state
        assert h_out_last is not None
        return self.base.lm_head(h_out_last)[:, -1, :]

    @torch.inference_mode()
    def decode_step(self, token_id: torch.Tensor) -> torch.Tensor:
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)

        outputs = self.base.model(
            input_ids=token_id,
            past_key_values=self._past_key_values,
            use_cache=True,
        )
        hidden = self._float_hidden(outputs.last_hidden_state[:, -1:, :])
        self._past_key_values = outputs.past_key_values

        if self.n_swap_layers > 0:
            return self.base.lm_head(hidden)[:, -1, :]

        window = self._build_hidden_window(hidden)
        assert self._state is not None
        h_out, self._state = self.adapter.forward_step(hidden, window, self._state)
        self._hidden_hist.append(hidden.squeeze(1).detach())
        return self.base.lm_head(h_out)[:, -1, :]

    @torch.inference_mode()
    def generate_cached(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        logits = self.prefill(input_ids, fast=True)
        next_tok = logits.argmax(-1, keepdim=True)
        tokens = [next_tok]
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_tok.squeeze(1))
            next_tok = logits.argmax(-1, keepdim=True)
            tokens.append(next_tok)
        return torch.cat(tokens, dim=1)

    @torch.inference_mode()
    def generate_text(
        self,
        tokenizer,
        prompt: str,
        max_new_tokens: int = 64,
        device: Optional[torch.device] = None,
    ) -> str:
        device = device or next(self.base.parameters()).device
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        new_ids = self.generate_cached(ids, max_new_tokens)
        return tokenizer.decode(new_ids[0], skip_special_tokens=True)


class BaseKVGenerator:
    """Standart TinyLlama — KV-cache ile O(1) decode."""

    def __init__(self, base_model: nn.Module):
        self.base = base_model
        self._past_key_values = None

    @torch.inference_mode()
    def prefill(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._past_key_values = None
        out = self.base(input_ids=input_ids, use_cache=True)
        self._past_key_values = out.past_key_values
        return out.logits[:, -1, :]

    @torch.inference_mode()
    def decode_step(self, token_id: torch.Tensor) -> torch.Tensor:
        if token_id.dim() == 1:
            token_id = token_id.unsqueeze(1)
        out = self.base(
            input_ids=token_id,
            past_key_values=self._past_key_values,
            use_cache=True,
        )
        self._past_key_values = out.past_key_values
        return out.logits[:, -1, :]

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        logits = self.prefill(input_ids)
        next_tok = logits.argmax(-1, keepdim=True)
        tokens = [next_tok]
        for _ in range(max_new_tokens - 1):
            logits = self.decode_step(next_tok.squeeze(1))
            next_tok = logits.argmax(-1, keepdim=True)
            tokens.append(next_tok)
        return torch.cat(tokens, dim=1)


DEFAULT_MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_MODEL_DIR = Path(__file__).resolve().parent / "models" / "TinyLlama-1.1B-Chat-v1.0"


def resolve_model_path(model_id: str | None = None) -> str:
    local = LOCAL_MODEL_DIR
    if local.is_dir() and (local / "config.json").exists():
        has_weights = any(local.glob("*.safetensors")) or any(local.glob("*.bin"))
        if has_weights:
            return str(local)
    return model_id or DEFAULT_MODEL_ID


def load_tinyllama(device: torch.device, dtype: torch.dtype, model_path: str | None = None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    path = resolve_model_path(model_path)
    print(f"  model path: {path}")
    tokenizer = AutoTokenizer.from_pretrained(path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        path, dtype=dtype, low_cpu_mem_usage=True,
    )
    model = model.to(device)
    return model, tokenizer, path


def create_hybrid(
    device: torch.device,
    dtype: torch.dtype,
    adapter_path: Path | str | None = None,
    base_model: nn.Module | None = None,
    n_swap_layers: int = 0,
    **adapter_kw,
) -> tuple[TinyLlamaWithBulk, object, str]:
    if base_model is None:
        base, tokenizer, path = load_tinyllama(device, dtype)
    else:
        from transformers import AutoTokenizer
        path = resolve_model_path()
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = base_model

    hybrid = TinyLlamaWithBulk(
        base, freeze_base=True, n_swap_layers=n_swap_layers, **adapter_kw,
    ).to(device)

    if adapter_path is not None and Path(adapter_path).exists():
        ckpt = torch.load(adapter_path, map_location=device, weights_only=True)
        if isinstance(ckpt, dict) and "top_adapter" in ckpt:
            hybrid.adapter.load_state_dict(ckpt["top_adapter"])
            for bulk, state in zip(hybrid._swap_adapters, ckpt.get("swap_adapters", [])):
                bulk.load_state_dict(state)
        else:
            hybrid.adapter.load_state_dict(ckpt)
        print(f"  adapter yüklendi: {adapter_path}")
    return hybrid, tokenizer, path


def create_kvfree(
    device: torch.device,
    dtype: torch.dtype,
    base_model: nn.Module | None = None,
    n_swap_layers: int = 4,
    sliding_window: int = 512,
    swap_checkpoint: Path | str | None = None,
    **adapter_kw,
) -> tuple[Any, object, str]:
    """Faz 3: KV'siz kuyruk — TinyLlamaKVFreeTail."""
    from bulk_layer_swap import TinyLlamaKVFreeTail, load_bulk_swap_checkpoint

    if base_model is None:
        base, tokenizer, path = load_tinyllama(device, dtype)
    else:
        from transformers import AutoTokenizer
        path = resolve_model_path()
        tokenizer = AutoTokenizer.from_pretrained(path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = base_model

    model = TinyLlamaKVFreeTail(
        base,
        n_swap_layers=n_swap_layers,
        sliding_window=sliding_window,
        compact=adapter_kw.pop("compact", True),
        d_bulk=adapter_kw.pop("d_bulk", 256),
        **adapter_kw,
    ).to(device)

    if swap_checkpoint and Path(swap_checkpoint).exists():
        load_bulk_swap_checkpoint(model, str(swap_checkpoint), device)
        print(f"  swap checkpoint yüklendi: {swap_checkpoint}")
    return model, tokenizer, path


def save_adapter(hybrid: TinyLlamaWithBulk, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "top_adapter": hybrid.adapter.state_dict(),
        "swap_adapters": [b.state_dict() for b in hybrid._swap_adapters],
        "n_swap_layers": hybrid.n_swap_layers,
    }
    torch.save(payload, path)
