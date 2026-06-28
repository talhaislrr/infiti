"""
HFP LoRA Fine-tune — Küçük açık kaynak LLM
==========================================
Ablation modları:
  baseline    — standart LoRA, sabit epoch
  hfp_stiff   — aynı LoRA + StiffTransient erken durma (zenon KAPALI)

Kullanım:
  python3 hfp_lora_finetune.py --mode baseline --rank 64
  python3 hfp_lora_finetune.py --mode hfp_stiff --rank 64
  python3 lora_ablation.py
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from hfp_config import HFPConfig, HFPStiffTransientScheduler, HFPZenonQuantizationScheduler, _batch_grad_norm

try:
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
except ImportError as e:
    raise SystemExit("Kurulum: pip install transformers peft datasets accelerate") from e


DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
LORA_CONFIG_PATH = "hfp_lora_config.json"


def load_lora_settings(path: str = LORA_CONFIG_PATH) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {
        "max_epochs": 10,
        "stiffness_p": 1.0,
        "stiffness_threshold": 0.001,
        "stiffness_k": 3,
        "stiffness_min_epochs": 3,
        "initial_lr": 2e-4,
        "lora_dropout": 0.05,
        "use_zenon": False,
        "quick_epochs": 3,
    }


class InstructDataset(Dataset):
    def __init__(self, path: str, tokenizer, max_len: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.max_len = max_len
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                text = (
                    f"<|im_start|>user\n{row['instruction']}\n"
                    f"<|im_start|>assistant\n{row['output']}"
                )
                enc = tokenizer(text, truncation=True, max_length=max_len, padding=False)
                self.samples.append(enc)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        input_ids = torch.tensor(item["input_ids"], dtype=torch.long)
        labels = input_ids.clone()
        return {"input_ids": input_ids, "labels": labels, "attention_mask": torch.ones_like(input_ids)}


def collate(batch, pad_id: int):
    max_l = max(b["input_ids"].shape[0] for b in batch)
    input_ids, labels, mask = [], [], []
    for b in batch:
        l = b["input_ids"].shape[0]
        pad = max_l - l
        input_ids.append(torch.cat([b["input_ids"], torch.full((pad,), pad_id)]))
        labels.append(torch.cat([b["labels"], torch.full((pad,), -100)]))
        mask.append(torch.cat([b["attention_mask"], torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "attention_mask": torch.stack(mask),
    }


@torch.no_grad()
def eval_loss(model, loader, device) -> float:
    model.eval()
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        total += out.loss.item()
        n += 1
    return total / max(n, 1)


def build_hfp_cfg(mode: str, rank: int, settings: dict, quick: bool) -> HFPConfig:
    use_stiff = mode in ("hfp_stiff", "hfp")
    max_epochs = settings["quick_epochs"] if quick else settings["max_epochs"]
    return HFPConfig(
        bulk_rank=rank,
        stiffness_p=settings["stiffness_p"],
        stiffness_threshold=settings["stiffness_threshold"],
        stiffness_k=settings["stiffness_k"],
        stiffness_min_epochs=min(settings["stiffness_min_epochs"], max_epochs - 1),
        initial_lr=settings["initial_lr"],
        max_epochs=max_epochs,
        use_bulk=True,
        use_stiff=use_stiff,
        use_zenon=False,
        early_stop=use_stiff,
    )


def train_lora(
    mode: str,
    model_id: str,
    train_path: str,
    output_dir: str,
    rank: int,
    settings: dict,
    quick: bool,
    device: torch.device,
):
    if mode == "hfp":
        mode = "hfp_stiff"

    print(f"Model yükleniyor: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"trust_remote_code": True, "torch_dtype": torch.float32}
    if device.type == "cuda":
        load_kwargs["torch_dtype"] = torch.float16
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    if device.type == "cpu":
        model = model.to(device)

    lora_cfg = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=settings["lora_dropout"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    max_len = 128 if quick else 256
    ds = InstructDataset(train_path, tokenizer, max_len)
    n = len(ds)
    split = max(1, int(n * 0.85))
    train_ds = torch.utils.data.Subset(ds, range(split))
    val_ds = torch.utils.data.Subset(ds, range(split, n))

    batch_size = 2 if quick else 4
    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=lambda b: collate(b, pad_id),
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=lambda b: collate(b, pad_id),
    )

    cfg = build_hfp_cfg(mode, rank, settings, quick)
    lr = settings["initial_lr"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * cfg.max_epochs
    scheduler_hf = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    stiff = HFPStiffTransientScheduler(optimizer, cfg) if cfg.use_stiff else None
    zenon = None

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    best_val = float("inf")
    epochs_run = 0
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            out_fwd = model(**batch)
            loss = out_fwd.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler_hf.step()
            epoch_loss += loss.item()

        val_loss = eval_loss(model, val_loader, device)
        epochs_run = epoch
        print(f"  epoch {epoch} train_loss={epoch_loss/len(train_loader):.4f} val_loss={val_loss:.4f}")

        if stiff:
            _, stop = stiff.step(epoch, val_loss)
        else:
            stop = False

        if val_loss < best_val:
            best_val = val_loss
            model.save_pretrained(out / "best")

        if stop:
            print(f"  StiffTransient erken durdu (epoch {epoch})")
            break

    best_path = out / "best"
    if best_path.exists():
        import shutil
        for fname in ["adapter_model.safetensors", "adapter_config.json"]:
            src = best_path / fname
            if src.exists():
                shutil.copy2(src, out / fname)

    model.save_pretrained(out)
    tokenizer.save_pretrained(out)

    meta = {
        "mode": mode,
        "model_id": model_id,
        "lora_rank": rank,
        "trainable_params": trainable_params,
        "epochs_run": epochs_run,
        "max_epochs": cfg.max_epochs,
        "best_val_loss": best_val,
        "elapsed_sec": round(time.time() - t0, 2),
        "use_stiff": cfg.use_stiff,
        "use_zenon": False,
        "lr": lr,
    }
    with open(out / "train_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Adapter kaydedildi: {out} ({meta['elapsed_sec']}s, {epochs_run} epoch)")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["baseline", "hfp_stiff", "hfp"], default="hfp_stiff")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--train-data", default="train_data.jsonl")
    parser.add_argument("--output", default=None)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--config", default=LORA_CONFIG_PATH)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    settings = load_lora_settings(args.config)
    output = args.output or f"./adapters/{args.mode}_r{args.rank}"

    print("=" * 70)
    print(f"LoRA FINE-TUNE | mode={args.mode} | rank={args.rank} | device={device}")
    print("=" * 70)

    train_lora(
        mode=args.mode,
        model_id=args.model,
        train_path=args.train_data,
        output_dir=output,
        rank=args.rank,
        settings=settings,
        quick=args.quick,
        device=device,
    )


if __name__ == "__main__":
    main()
