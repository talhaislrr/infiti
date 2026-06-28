"""
Ollama'ya model/adapter import
==============================
Qwen2.5: Ollama safetensors ADAPTER henüz desteklemiyor (Llama/Mistral/Gemma only).
Çözümler:
  A) hf_lora_benchmark.py — doğrudan HF+PEFT karşılaştırması (önerilen)
  B) Merge + GGUF — llama.cpp ile (aşağıdaki merge_adapter)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Ollama ADAPTER safetensors: resmi destek listesi (modelfile.md)
OLLAMA_ADAPTER_ARCHS = {"llama", "mistral", "gemma"}


def write_modelfile(path: Path, base_model: str, adapter_dir: Path, temperature: float):
    # ADAPTER . → Modelfile ile aynı klasörde adapter dosyaları olmalı
    content = f"""FROM {base_model}
ADAPTER .
PARAMETER temperature {temperature}
PARAMETER num_predict 64
"""
    path.write_text(content)
    print(f"Modelfile yazıldı: {path}")


def verify_adapter(adapter: Path) -> list[str]:
    errors = []
    if not (adapter / "adapter_config.json").exists():
        errors.append("adapter_config.json eksik")
    if not (adapter / "adapter_model.safetensors").exists():
        errors.append("adapter_model.safetensors eksik")
    return errors


def ollama_create(name: str, work_dir: Path):
    cmd = ["ollama", "create", name, "-f", "Modelfile"]
    print(f"Çalıştırılıyor (cwd={work_dir}): {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=work_dir, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr, file=sys.stderr)
        return False, result.stderr.strip()
    print(result.stdout)
    return True, ""


def merge_adapter(model_id: str, adapter_dir: Path, out_dir: Path):
    """LoRA merge → tam model (GGUF dönüşümü için)."""
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise SystemExit("pip install transformers peft torch") from e

    print(f"Merge: {adapter_dir} → {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch.float16
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model = model.merge_and_unload()
    model.save_pretrained(out_dir, safe_serialization=True)
    tokenizer.save_pretrained(out_dir)
    print(f"Merged model: {out_dir}")
    print("\nGGUF için (llama.cpp gerekli):")
    print(f"  git clone https://github.com/ggerganov/llama.cpp")
    print(f"  python llama.cpp/convert_hf_to_gguf.py {out_dir.resolve()} --outfile {out_dir.name}.f16.gguf")
    print(f"  echo 'FROM ./{out_dir.name}.f16.gguf' > Modelfile && ollama create my-model -f Modelfile")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="hfp-qwen")
    parser.add_argument("--base", default="qwen2.5:0.5b")
    parser.add_argument("--adapter", default="./adapters/hfp")
    parser.add_argument("--hf-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--merge-only", action="store_true", help="LoRA merge et, Ollama deneme")
    parser.add_argument("--force-ollama", action="store_true", help="Qwen uyarısına rağmen dene")
    args = parser.parse_args()

    adapter = Path(args.adapter).resolve()
    if not adapter.exists():
        raise SystemExit(f"Adapter yok: {adapter}\nÖnce: python3 hfp_lora_finetune.py --mode hfp --quick")

    errs = verify_adapter(adapter)
    if errs:
        raise SystemExit("Adapter eksik:\n  " + "\n  ".join(errs))

    if args.merge_only:
        merge_adapter(args.hf_model, adapter, Path(f"./merged/{args.name}"))
        return

    # Qwen2.5 safetensors adapter → Ollama bilinen sınırlama
    if "qwen" in args.base.lower() and not args.force_ollama:
        print("=" * 70)
        print("UYARI: Ollama, Qwen2.5 safetensors ADAPTER'ı desteklemiyor.")
        print("  Desteklenen: Llama, Mistral, Gemma")
        print("  GitHub: ollama/ollama#8132")
        print()
        print("Önerilen karşılaştırma:")
        print("  python3 hf_lora_benchmark.py")
        print()
        print("Ollama'ya aktarmak için merge + GGUF:")
        print(f"  python3 export_ollama.py --merge-only --name {args.name} --adapter {adapter}")
        print("=" * 70)
        return

    write_modelfile(adapter / "Modelfile", args.base, adapter, args.temperature)

    try:
        subprocess.run(["ollama", "list"], check=True, capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise SystemExit("Ollama bulunamadı.")

    ok, err = ollama_create(args.name, adapter)
    if not ok:
        print("\nOllama import başarısız. HF benchmark kullanın:")
        print("  python3 hf_lora_benchmark.py")
        if err:
            print(f"  Hata: {err}")
        raise SystemExit(1)

    print(f"\nModel hazır: ollama run {args.name}")
    print(f"  python3 ollama_benchmark.py --models {args.base} {args.name}")


if __name__ == "__main__":
    main()
