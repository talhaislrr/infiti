#!/usr/bin/env python3
"""
Faz 3 — Güvenli tam test pipeline (Mac 24GB)
===========================================
Adımlar sırayla çalışır; her adım sonrası bellek temizlenir.

  python3 bulk_kvfree_safe_test.py --device mps
  python3 bulk_kvfree_safe_test.py --train-only
  python3 bulk_kvfree_safe_test.py --bench-only
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd: list[str], label: str) -> int:
    print("\n" + "=" * 70)
    print(f"▶ {label}")
    print("  " + " ".join(cmd))
    print("=" * 70)
    return subprocess.call(cmd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="mps")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--bench-only", action="store_true")
    parser.add_argument("--skip-demo", action="store_true")
    args = parser.parse_args()

    py = sys.executable
    root = Path(__file__).resolve().parent

    if not args.bench_only:
        rc = run(
            [
                py, str(root / "bulk_kvfree_train.py"),
                "--epochs", str(args.epochs),
                "--seq-len", "128",
                "--batch-size", "1",
                "--accum-steps", "4",
                "--device", args.device,
            ],
            "KVFree compact eğitim (3 epoch, batch=1)",
        )
        if rc != 0:
            print(f"\n✗ Eğitim başarısız (exit {rc})")
            sys.exit(rc)

    if not args.train_only:
        rc = run(
            [
                py, str(root / "bulk_kvfree_benchmark.py"),
                "--device", args.device,
                "--prompt-lens", "512,2048",
                "--skip-hybrid",
                "--new-tokens", "4",
            ],
            "KVFree benchmark (512, 2048 — hybrid atlandı)",
        )
        if rc != 0:
            sys.exit(rc)

        if not args.skip_demo:
            ckpt = root / "checkpoints/bulk_swap/bulk_v2.pt"
            if ckpt.exists():
                rc = run(
                    [
                        py, str(root / "bulk_longrange_demo.py"),
                        "--device", args.device,
                        "--kvfree",
                        "--doc-tokens", "1024",
                        "--swap-checkpoint", str(ckpt),
                    ],
                    "Uzun menzil QA demo (KVFree)",
                )
                if rc != 0:
                    sys.exit(rc)

    print("\n✓ Güvenli test pipeline tamamlandı.")
    print("  checkpoint: checkpoints/bulk_swap/bulk_v2.pt")
    print("  benchmark:  bulk_kvfree_benchmark_results.json")


if __name__ == "__main__":
    main()
