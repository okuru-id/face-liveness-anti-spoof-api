# Liveness Check Enhancement (FFT + MiniFASNet Hybrid)

## Overview
Penambahan modul FFT (Fast Fourier Transform) untuk meningkatkan deteksi spoofing, khususnya:

* Replay attack (layar iPad / HP)
* Screen-based spoofing
* High-quality video attack

Pendekatan menggunakan:

* Model AI (MiniFASNet ONNX)
* Signal processing (FFT)
* Multi-frame aggregation

---

## 1. Pipeline

```
Input Image / Video Frame
↓
Face Detection (crop wajah)
↓
Resize (80x80)
↓
Convert ke Y channel (YCrCb)
↓
FFT (2D)
↓
Log Magnitude Spectrum
↓
Frequency Band Analysis (low / mid / high)
↓
Compute FFT Score
↓
MiniFASNet Inference
↓
Score Fusion
↓
Decision (LIVE / SPOOF / UNCERTAIN)
```

---

## 2. FFT Feature Extraction

### Steps:

1. Convert face crop ke grayscale (Y channel)
2. Apply FFT 2D
3. Shift frequency ke center
4. Gunakan log scaling

### Formula:
```
magnitude = log(1 + |FFT(image)|)
```

---

### Frequency Band Extraction

```
Low Frequency   → pusat (struktur wajah)
Mid Frequency   → tekstur kulit
High Frequency  → noise / layar
```

### Implementation (pseudo):

```python
h, w = magnitude.shape
center = (h//2, w//2)

low = magnitude[center-5:center+5, center-5:center+5].mean()
mid = magnitude[center-15:center+15, center-15:center+15].mean()
high = magnitude.mean() - mid
```

---

## 3. FFT Scoring

Gunakan ratio (lebih stabil dibanding top-N features):

```python
fft_score = high / (low + 1e-6)
fft_score = clip(fft_score, 0, 1)
```

---

## 4. Multi-frame Aggregation (WAJIB)

```python
fft_score_final = mean(fft_score_frame_1 ... fft_score_frame_n)
```

Rekomendasi: 5–10 frame

---

## 5. MiniFASNet Score

```python
mini_fas_score = anti_spoof_result.confidence  # 0–1
```

---

## 6. Score Fusion

```python
final_score = (
mini_fas_score * 0.6 +
fft_score_final * 0.4
)
```

---

## 7. Decision Logic

```python
if final_score >= 0.6:
return "LIVE"
elif final_score <= 0.4:
return "SPOOF"
else:
return "UNCERTAIN"
```

---

## 8. Configuration

```python
fft_enabled = True
fft_weight = 0.4
fas_weight = 0.6

fft_log_scale = True
fft_multi_frame = 5

live_threshold = 0.6
spoof_threshold = 0.4
```

---

## 9. Summary

Pendekatan ini:

* ringan (CPU friendly)
* tidak butuh training tambahan
* efektif melawan replay attack
* scalable untuk production API
