#!/usr/bin/env bash
# HFP Ollama test hattı
set -e
cd "$(dirname "$0")"

echo "=== 1) Bağımlılıklar ==="
pip3 install -q -r requirements.txt

echo "=== 2) Ollama base model ==="
ollama pull qwen2.5:0.5b || true

echo "=== 3) Baseline LoRA fine-tune ==="
python3 hfp_lora_finetune.py --mode baseline --rank 64 --quick

echo "=== 4) HFP Stiff LoRA fine-tune (zenon kapalı) ==="
python3 hfp_lora_finetune.py --mode hfp_stiff --rank 64 --quick

echo "=== 5) HF LoRA benchmark (asıl karşılaştırma) ==="
python3 hf_lora_benchmark.py --quick \
  --modes base baseline hfp_stiff \
  --adapter baseline=./adapters/baseline_r64 \
  --adapter hfp_stiff=./adapters/hfp_stiff_r64

echo "=== 6) Ollama base model benchmark ==="
python3 ollama_benchmark.py --models qwen2.5:0.5b

echo "=== 7) Ollama adapter import (Qwen: bilgi mesajı) ==="
python3 export_ollama.py --name baseline-qwen --adapter ./adapters/baseline || true
python3 export_ollama.py --name hfp-qwen --adapter ./adapters/hfp || true

echo ""
echo "=== Bitti ==="
echo "  HF karşılaştırma : hf_lora_results.json"
echo "  Ollama base      : ollama_results.json"
echo ""
echo "Tam HF benchmark (20 prompt):"
echo "  python3 hf_lora_benchmark.py"
