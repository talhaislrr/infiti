#!/usr/bin/env bash
# HFP LoRA Ablation — tam pipeline
set -e
cd "$(dirname "$0")"

echo "=== 1) Bağımlılıklar ==="
pip3 install -q -r requirements.txt

echo "=== 2) LoRA Ablation (baseline vs hfp_stiff, rank sweep) ==="
python3 lora_ablation.py "$@"

echo ""
echo "=== Bitti ==="
echo "  Eğitim meta : lora_ablation_train.json"
echo "  Benchmark   : lora_ablation_results.json"
