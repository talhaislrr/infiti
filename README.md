# Infiti

HFP BulkState tabanlı hibrit LLM — erken katmanlar KV-cache, kuyruk katmanları sabit bellekli BulkState.

**TinyLlama-1.1B** üzerinde teknik MVP: compact layer swap (~13M eğitilebilir param), RAM/decode benchmark, recall pipeline, multi-turn chat state persistence.

## Kurulum

```bash
pip install torch transformers datasets huggingface_hub accelerate

# Model (~2.2 GB, bir kez)
python3 download_tinyllama.py
```

Checkpoint (`checkpoints/bulk_swap/bulk_v2.pt`) repoda yok — eğitim sonrası lokal veya Google Drive.

## Hızlı komutlar

```bash
# Benchmark
python3 bulk_kvfree_benchmark.py --device mps --prompt-lens 512,2048 --skip-hybrid

# Recall eğitimi
python3 bulk_kvfree_train.py --recall --epochs 3 --device mps --resume checkpoints/bulk_swap/bulk_v2.pt

# Multi-turn chat (state persist)
python3 bulk_chat.py --kvfree --device mps

# Uzun menzil recall demo
python3 bulk_longrange_demo.py --kvfree --device mps --sliding-window 0
```

## Colab

```python
!git clone https://github.com/talhaislrr/infiti.git
%cd infiti
!pip install -q torch transformers datasets huggingface_hub accelerate
!python3 download_tinyllama.py
# checkpoint: Google Drive → checkpoints/bulk_swap/bulk_v2.pt
!python3 bulk_kvfree_benchmark.py --device cuda --skip-hybrid
```

## Mimari

```
[Layer 0–17]  Llama attention + KV-cache
[Layer 18–21] BulkTriggerDecoderLayerV2 (KV-free, BulkState)
```

## Lisans

Apache-2.0 (TinyLlama base model lisansına tabi).
