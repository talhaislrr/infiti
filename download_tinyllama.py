#!/usr/bin/env python3
"""
TinyLlama'yı bir kez indir → models/ klasörüne kaydet
=====================================================
Kullanım:
  python3 download_tinyllama.py

Sonra eğitim:
  python3 bulk_trigger_hybrid_train.py --quick --device mps
"""

from __future__ import annotations

from pathlib import Path

MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LOCAL_DIR = Path(__file__).resolve().parent / "models" / "TinyLlama-1.1B-Chat-v1.0"


def main():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Önce kur: pip install huggingface_hub")
        return

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model:  {MODEL_ID}")
    print(f"Hedef:  {LOCAL_DIR}")
    print("İndiriliyor (~2.2 GB)...")

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=str(LOCAL_DIR),
        local_dir_use_symlinks=False,
    )

    files = [f for f in LOCAL_DIR.rglob("*") if f.is_file() and "cache" not in str(f)]
    total_mb = sum(f.stat().st_size for f in files) / 1e6
    weights = list(LOCAL_DIR.glob("*.safetensors")) + list(LOCAL_DIR.glob("*.bin"))

    print(f"\nTamam — {len(files)} dosya, {total_mb:.0f} MB")
    if weights:
        print(f"Ağırlık: {[w.name for w in weights]}")
    else:
        print("UYARI: safetensors/bin bulunamadı — indirme eksik olabilir, tekrar çalıştır.")

    print("\nTest:")
    print("  python3 -c \"from bulk_hybrid import load_tinyllama; from bulk_device import pick_device, pick_dtype; d=pick_device('mps'); load_tinyllama(d, pick_dtype(d))\"")


if __name__ == "__main__":
    main()
