# HFP Makale Taslağı (Revize Çerçeve)

> **Durum:** İç taslak — mevcut `ai_test` benchmark verilerine dayalı  
> **Tarih:** Haziran 2026  
> **Ton:** Dürüst, ablation odaklı; “devrim” iddiası yok

---

## 1. Yeni Tez (Eski iddiadan kopuş)

### Eski (desteklenmiyor)
> “HFP yeni bir AI mimarisi; Bulk + Stiff + Zenon ile hem daha az parametre hem daha yüksek doğruluk.”

### Yeni (veriye uygun)
> **HFP-Stiff:** LoRA fine-tune sırasında val-loss platosuna dayalı **rank-aware erken durma** stratejisi. Amaç: sabit epoch baseline’a kıyasla **eğitim maliyetini düşürmek**, kabul edilebilir doğruluk kaybı pahasına.

**Zenon quantize** makaleden **çıkarıldı** — tüm deneylerde zarar veya nötr.

**BulkLinear** MNIST’te mimari sıkıştırma olarak kalır; LLM tarafında LoRA rank zaten low-rank olduğundan “bulk” = **rank seçimi** olarak yeniden adlandırılır.

---

## 2. Başlık Önerileri

1. **StiffTransient Early Stopping for Low-Rank LLM Adaptation: A Cost–Accuracy Trade-off Study** *(önerilen)*
2. Rank-Aware Training Schedulers for Parameter-Efficient Fine-Tuning
3. When Does Adaptive Early Stopping Match Full LoRA Training? An Ablation on Small Instruction Models

---

## 3. Özet (Abstract) — taslak

Küçük açık kaynak dil modellerinin (Qwen2.5-0.5B-Instruct) parametre-verimli fine-tune’unda, sabit epoch LoRA eğitimi ile val-loss tabanlı erken durma (StiffTransient) stratejisini karşılaştırıyoruz. Rank sweep (r ∈ {32, 64, 128}) altında 20 prompt’luk değerlendirme seti ve 30 örnek instruction verisi kullanıldı.

**Bulgular:**
- Ham model: %50–60 doğruluk; standart LoRA (r=64): **%100**
- HFP-Stiff (r=32): baseline ile **aynı %95 doğruluk**, **%31 daha kısa eğitim** (6 vs 10 epoch), **~%50 daha az trainable parametre**
- HFP-Stiff r=64’te baseline’ın altında (%90 vs %100)
- MNIST MLP’de BulkLinear + Stiff: ~%55 parametre tasarrufu, doğruluk farkı <0.2 pp
- Transformer sınıflandırmada HFP varyantları standart FFN’in gerisinde kaldı

**Sonuç:** StiffTransient, **düşük rank LoRA** rejiminde eğitim maliyetini düşüren pratik bir scheduler; yüksek doğruluk gerektiğinde standart yüksek-rank LoRA tercih edilmeli.

**Anahtar kelimeler:** LoRA, early stopping, PEFT, training efficiency, Qwen, ablation

---

## 4. Giriş (Introduction)

### 4.1 Problem
- Edge / yerel AI: küçük modeller + sınırlı GPU/CPU bütçesi
- LoRA rank ve epoch sayısı manuel seçiliyor; gereksiz epoch = maliyet
- “Daha akıllı durma” ihtiyacı

### 4.2 Katkılar (revize, dürüst)
1. **StiffTransient**’in LoRA ablation protokolü (rank × scheduler)
2. **Maliyet–doğruluk eğrisi:** trainable params, wall-clock, epoch vs accuracy
3. **Negative results:** Zenon quantize, transformer FFN’de HFP paketi
4. Açık kaynak benchmark hattı: HF + Ollama base karşılaştırması

### 4.3 Ne iddia etmiyoruz
- Yeni foundation architecture değil
- SOTA benchmark’ları geçmiyoruz
- Fiziksel sabitler (η̃ vb.) kullanılmıyor

---

## 5. İlgili Çalışmalar (Related Work)

| Alan | Referans yönü |
|------|----------------|
| LoRA / PEFT | Hu et al., Dettmers (QLoRA) |
| Early stopping | klasik val-loss, patience |
| Low-rank adaptation | rank seçimi, r etkisi |
| Küçük model fine-tune | instruction tuning, küçük veri |

**Konumlandırma:** “LoRA + akıllı durma” — mimari değil **training protocol** katkısı.

---

## 6. Yöntem (Method)

### 6.1 HFP-Stiff (makalede kullanılacak tek HFP bileşeni)

```
Her epoch sonu:
  val_loss ölç
  son k epoch'ta val_loss değişimi < ε ise DUR
  minimum epoch: m_min
```

Hiperparametreler (`hfp_lora_config.json`):
- `stiffness_threshold` = 0.001
- `stiffness_k` = 3
- `stiffness_min_epochs` = 3
- `max_epochs` = 10

### 6.2 Baseline
- Aynı LoRA config, aynı LR (2e-4), **sabit max_epochs**, erken durma yok

### 6.3 MNIST (tamamlayıcı)
- BulkLinear (rank-r FFN) + StiffTransient
- Zenon: ek ablation, ana metinde kısa negative result

### 6.4 Değerlendirme
- **numeric:** ilk çıkan sayı
- **contains:** alt string eşleşme
- 20 prompt (`prompts.jsonl`), 30 train örnek (`train_data.jsonl`)

---

## 7. Deneysel Kurulum (Experimental Setup)

| Bileşen | Değer |
|---------|-------|
| Base model | Qwen/Qwen2.5-0.5B-Instruct |
| LoRA targets | q,k,v,o,gate,up,down proj |
| Rank | 32, 64, 128 |
| LR | 2e-4, AdamW |
| Epoch | max 10 |
| Cihaz | CPU (mevcut); GPU tekrarı önerilir |
| Ollama karşılaştırma | qwen2.5:0.5b Q4, /api/chat |

---

## 8. Sonuçlar (Results) — mevcut verilerle Tablolar

### Tablo 1: LoRA Ablation (20 prompt, HF)

| Varyant | Rank | Epoch | Train (s) | Trainable (M) | Acc |
|---------|------|-------|-----------|---------------|-----|
| base | — | — | — | — | 50.0% |
| baseline_r32 | 32 | 10 | 27.5 | 17.6 | 95.0% |
| **hfp_stiff_r32** | 32 | **6** | **18.9** | 17.6 | **95.0%** |
| **baseline_r64** | 64 | 10 | 33.7 | 35.2 | **100.0%** |
| hfp_stiff_r64 | 64 | 10 | 34.4 | 35.2 | 90.0% |
| baseline_r128 | 128 | 10 | 36.0 | 70.4 | 95.0% |
| hfp_stiff_r128 | 128 | 10 | 37.0 | 70.4 | 95.0% |

**Şekil 1 önerisi:** Rank (x) vs Accuracy (y), iki çizgi: baseline / hfp_stiff  
**Şekil 2 önerisi:** Epoch (x) vs val_loss — r32 hfp’nin epoch 6’da durması

### Tablo 2: Unified karşılaştırma (normal vs fine-tune)

| Model | Backend | Acc | Not |
|-------|---------|-----|-----|
| ollama:qwen2.5:0.5b | Ollama Q4 | 60.0% | Fine-tune yok |
| hf:base | HF fp32 | 50.0% | Fine-tune yok |
| hf:baseline_r64 | HF+LoRA | **100.0%** | En iyi doğruluk |
| hf:hfp_stiff_r32 | HF+LoRA+HFP | 95.0% | En iyi maliyet/doğruluk dengesi |

### Tablo 3: MNIST MLP (tamamlayıcı)

| Model | Params | Acc | Epoch |
|-------|--------|-----|-------|
| Standart MLP | 670K | 97.8% | 10 |
| BulkLinear + Stiff | 303K | 97.7% | 5 |

### Tablo 4: Transformer (negative result, kısa)

| Model | FFN params | Acc |
|-------|------------|-----|
| Standart | 132K | **72.7%** |
| Bulk + Stiff | 197K | 70.8% |

---

## 9. Tartışma (Discussion)

### 9.1 Ana mesaj
HFP-Stiff **“ücretsiz öğle yemeği” değil** — doğruluk–maliyet trade-off:
- **Bütçe kısıtlı:** r=32 + stiff → %95, %31 daha hızlı eğitim
- **Doğruluk kritik:** r=64 baseline, stiff ekleme gereksiz/harmful

### 9.2 Neden r=64 HFP kötüleşti?
- Stiff epoch 10’da durdu ama val_loss zaten yükseliyordu
- Hipotez: düşük rank’ta stiff “doğru zamanda” duruyor; yüksek rank’ta overfit rejiminde geç kalıyor

### 9.3 Sınırlamalar (açıkça yaz)
- 30 train örneği — çok küçük
- Tek seed, CPU
- Tek model ailesi (Qwen 0.5B)
- Ollama vs HF farklı runtime — tok/s kıyası sınırlı
- Prompt seti 20 soru — genelleme iddiası zayıf

### 9.4 Zenon neden çıkarıldı
MNIST, transformer ve LoRA’da tutarlı zarar → metodolojiden elendi.

---

## 10. Sonuç (Conclusion)

StiffTransient, parametre-verimli LLM adaptasyonunda **eğitim süresini kısaltabilir**, özellikle düşük LoRA rank’ında ve küçük instruction setlerinde. Ancak **maksimum doğruluk** hedeflendiğinde standart yüksek-rank LoRA eğitimi üstün kalır. HFP, “yeni mimari” değil; **rank-aware training protocol** olarak konumlandırılmalıdır.

---

## 11. Eksik Deneyler (yayın öncesi yapılacaklar)

| Öncelik | Deney | Neden |
|---------|-------|-------|
| P0 | 3–5 seed, aynı ablation | İstatistiksel güven |
| P0 | GPU’da tekrar | CPU bias |
| P1 | Train set 200+ örnek | Küçük veri bias |
| P1 | Alpaca/LIMA subset | Genelleme |
| P2 | Mistral-7B veya Llama-3.2-1B | Model ailesi genelleme |
| P2 | Stiff hiperparam sweep (ε, k) | Hassasiyet analizi |

---

## 12. Makale Yapısı (bölüm uzunlukları)

```
Abstract           150–200 kelime
1. Introduction    1.5 sayfa
2. Related Work    1 sayfa
3. Method          1.5 sayfa
4. Experiments     1 sayfa
5. Results         2 sayfa (+ 4 tablo, 2 şekil)
6. Discussion      1 sayfa
7. Conclusion      0.5 sayfa
References         1–2 sayfa
```

**Hedef uzunluk:** 8–10 sayfa workshop / short paper (NeurIPS Tiny Papers, EMNLP Findings short, arXiv tech report)

---

## 13. Şekil ve Kod Referansları (repoda)

| Varlık | Dosya |
|--------|-------|
| LoRA ablation sonuçları | `lora_ablation_results.json` |
| Eğitim meta | `lora_ablation_train.json` |
| Unified karşılaştırma | `unified_compare_results.json` |
| MNIST | `results.json` |
| Transformer | `transformer_results.json` |
| Tekrar üretim | `./run_lora_ablation.sh`, `python3 unified_compare.py` |

---

## 14. Bir paragraf “elevator pitch” (Türkçe)

> Küçük dil modellerini yerelde fine-tune ederken hem GPU süresi hem adapter boyutu sınırlı. Biz, val-loss platosuna dayalı StiffTransient erken durmayı LoRA rank sweep ile birlikte test ettik. Düşük rank (r=32) ve stiff stopping, tam 10 epoch baseline ile aynı doğruluğu (%95) %31 daha kısa eğitimle verdi; en yüksek doğruluk (%100) ise stiff olmadan r=64 baseline’da kaldı. Zenon quantize ve transformer FFN paketi işe yaramadı. Sonuç: HFP bir mimari devrimi değil; **bütçe odaklı fine-tune protokolü**.

---

## 15. Sonraki adım

1. Bu taslağı İngilizceye çevir → `paper/draft.md`
2. Tablo 1–2’yi LaTeX’e aktar
3. P0 deneyleri (multi-seed) koştur
4. arXiv tech report veya workshop hedefle

İstersen bir sonraki adımda `paper/draft.md` İngilizce full draft veya multi-seed script yazılabilir.
