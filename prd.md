# Product Requirements Document (PRD)
## API Liveness Detection & Anti-Spoofing — MVP

**Version:** 1.1.0  
**Status:** Draft  
**Tanggal:** April 2026  
**Owner:** Product Team  
**Changelog:** v1.1.0 — Arsitektur diperbarui ke CPU-only optimized stack (MiniFASNet INT8, ONNX Runtime, latency budget, concurrency strategy)

---

## 1. Overview

### 1.1 Latar Belakang

Verifikasi identitas digital semakin kritis di industri fintech, perbankan, dan layanan yang memerlukan KYC (Know Your Customer). Salah satu celah terbesar dalam proses onboarding digital adalah serangan spoofing — di mana pelaku menggunakan foto, video, atau mask untuk melewati sistem verifikasi wajah.

API Liveness Detection & Anti-Spoofing ini hadir sebagai layanan terpisah (headless) yang dapat diintegrasikan ke sistem manapun untuk memastikan bahwa subjek di depan kamera adalah orang nyata, bukan representasi palsu.

### 1.2 Tujuan Produk

- Menyediakan REST API yang dapat mendeteksi **liveness** (keberadaan orang nyata) dari input gambar/video
- Mendeteksi dan menolak berbagai jenis **spoofing attack**: print attack, replay attack, 3D mask attack
- Memberikan **confidence score** dan **reason** yang dapat dikonsumsi oleh sistem downstream
- Mendukung integrasi mudah ke pipeline KYC/onboarding yang sudah ada

### 1.3 Target Pengguna

| Segmen | Contoh Use Case |
|--------|----------------|
| Fintech / Neo-bank | Verifikasi pembukaan rekening digital |
| Platform pinjaman online | Selfie verification saat pengajuan kredit |
| E-wallet | Re-authentication transaksi besar |
| HR Tech | Verifikasi kehadiran karyawan remote |
| Developer / Integrator | Membangun fitur liveness di aplikasi sendiri |

---

## 2. Scope MVP

### 2.1 In Scope

- Passive liveness detection dari single image (selfie)
- Active liveness via challenge-response (blink, nod, atau senyum)
- Anti-spoof detection: print attack, digital screen replay, 3D mask
- REST API endpoint dengan autentikasi API Key
- Response JSON dengan score, verdict, dan reason
- Basic dashboard untuk monitoring penggunaan API
- Rate limiting dan quota management

### 2.2 Out of Scope (Post-MVP)

- Video stream liveness (real-time WebRTC)
- SDK mobile native (iOS/Android)
- Multi-face detection
- Emotion analysis
- Age estimation
- Face matching / recognition
- On-premise deployment

---

## 3. User Stories

### 3.1 Developer / Integrator

```
Sebagai developer,
Saya ingin mengirim gambar selfie ke API
Dan mendapatkan verdict apakah gambar tersebut liveness atau spoof,
Sehingga saya dapat memblokir atau melanjutkan proses onboarding pengguna.
```

```
Sebagai developer,
Saya ingin mendapatkan confidence score (0.0 - 1.0) dari hasil deteksi,
Sehingga saya dapat menentukan threshold sendiri sesuai kebutuhan bisnis.
```

```
Sebagai developer,
Saya ingin mendapatkan reason/label dari jenis serangan yang terdeteksi,
Sehingga saya dapat melakukan logging dan analisis fraud yang tepat.
```

### 3.2 Admin / Operator

```
Sebagai admin,
Saya ingin melihat total request, success rate, dan distribusi verdict di dashboard,
Sehingga saya dapat memantau performa dan anomali penggunaan API.
```

---

## 4. Functional Requirements

### 4.1 Endpoint API

#### `POST /v1/liveness/check`

Melakukan passive liveness detection dari satu gambar (base64 atau multipart).

**Request:**

```json
{
  "image": "base64_encoded_image_string",
  "mode": "passive",
  "options": {
    "return_face_crop": false,
    "min_face_size": 100
  }
}
```

**Response:**

```json
{
  "request_id": "req_abc123xyz",
  "verdict": "LIVE",
  "confidence": 0.97,
  "spoof_type": null,
  "face_detected": true,
  "quality_check": {
    "passed": true,
    "issues": []
  },
  "processing_time_ms": 320,
  "timestamp": "2026-04-20T08:30:00Z"
}
```

**Possible verdict values:**

| Verdict | Keterangan |
|---------|------------|
| `LIVE` | Wajah terdeteksi sebagai orang nyata |
| `SPOOF` | Terdeteksi sebagai serangan spoofing |
| `UNCERTAIN` | Confidence rendah, perlu verifikasi ulang |
| `NO_FACE` | Tidak ada wajah terdeteksi di gambar |
| `POOR_QUALITY` | Kualitas gambar tidak memenuhi syarat |

**Possible spoof_type values:**

| Spoof Type | Deskripsi |
|-----------|-----------|
| `PRINT_ATTACK` | Foto dicetak di kertas |
| `SCREEN_REPLAY` | Foto/video ditampilkan di layar |
| `3D_MASK` | Menggunakan mask 3D |
| `DEEPFAKE` | Terdeteksi manipulasi AI (best effort) |
| `null` | Tidak ada serangan terdeteksi |

---

#### `POST /v1/liveness/challenge/init`

Inisiasi sesi active liveness dengan challenge-response.

**Request:**

```json
{
  "session_config": {
    "challenges": ["BLINK", "SMILE"],
    "timeout_seconds": 30
  }
}
```

**Response:**

```json
{
  "session_id": "sess_def456uvw",
  "challenges": [
    { "order": 1, "type": "BLINK", "instruction": "Kedipkan mata Anda" },
    { "order": 2, "type": "SMILE", "instruction": "Tersenyumlah" }
  ],
  "expires_at": "2026-04-20T08:31:00Z"
}
```

---

#### `POST /v1/liveness/challenge/verify`

Submit frame untuk memverifikasi challenge aktif.

**Request:**

```json
{
  "session_id": "sess_def456uvw",
  "frames": [
    { "challenge_type": "BLINK", "image": "base64_image_1" },
    { "challenge_type": "SMILE", "image": "base64_image_2" }
  ]
}
```

**Response:**

```json
{
  "session_id": "sess_def456uvw",
  "verdict": "LIVE",
  "confidence": 0.99,
  "challenges_result": [
    { "type": "BLINK", "passed": true, "confidence": 0.98 },
    { "type": "SMILE", "passed": true, "confidence": 0.99 }
  ],
  "processing_time_ms": 510,
  "timestamp": "2026-04-20T08:30:45Z"
}
```

---

#### `GET /v1/health`

Health check endpoint.

```json
{
  "status": "ok",
  "version": "1.0.0",
  "model_version": "liveness-v1.2"
}
```

---

### 4.2 Autentikasi

- Semua endpoint (kecuali `/v1/health`) wajib menggunakan **API Key** via header:
  ```
  X-API-Key: your_api_key_here
  ```
- API Key di-generate dari dashboard dan dapat di-revoke kapan saja
- Support multiple API Key per akun (untuk environment staging/production)

### 4.3 Rate Limiting

| Plan | Request / menit | Request / bulan |
|------|----------------|----------------|
| Free | 10 | 500 |
| Starter | 60 | 10.000 |
| Pro | 300 | 100.000 |
| Enterprise | Custom | Custom |

Response header rate limit:
```
X-RateLimit-Limit: 60
X-RateLimit-Remaining: 45
X-RateLimit-Reset: 1713600060
```

### 4.4 Quality Check

Sebelum diproses ke model, gambar diperiksa kualitasnya:

- Resolusi minimum: 320×320 piksel
- Ukuran wajah minimum: 100×100 piksel (konfigurabel)
- Blur score: di atas threshold minimum
- Brightness: tidak terlalu gelap atau terlalu terang
- Jika gagal quality check → verdict `POOR_QUALITY` dengan detail `issues`

### 4.5 Error Handling

```json
{
  "error": {
    "code": "INVALID_IMAGE_FORMAT",
    "message": "Image must be JPEG or PNG, base64 encoded",
    "request_id": "req_abc123xyz"
  }
}
```

**Error codes:**

| HTTP Status | Error Code | Keterangan |
|------------|-----------|------------|
| 400 | `INVALID_IMAGE_FORMAT` | Format gambar tidak didukung |
| 400 | `IMAGE_TOO_LARGE` | Ukuran file melebihi 5MB |
| 400 | `INVALID_SESSION` | Session ID tidak valid atau expired |
| 401 | `UNAUTHORIZED` | API Key tidak valid |
| 429 | `RATE_LIMIT_EXCEEDED` | Kuota habis |
| 500 | `INTERNAL_ERROR` | Kesalahan server internal |
| 503 | `MODEL_UNAVAILABLE` | Model sedang tidak tersedia |

---

## 5. Non-Functional Requirements

### 5.1 Performa

| Metrik | Target MVP |
|--------|-----------|
| Latency P50 (passive) | < 500ms |
| Latency P95 (passive) | < 1500ms |
| Latency P50 (active) | < 800ms |
| Uptime SLA | 99.5% |
| False Acceptance Rate (FAR) | < 1% |
| False Rejection Rate (FRR) | < 5% |

### 5.2 Keamanan

- HTTPS wajib untuk semua komunikasi
- Gambar tidak disimpan secara permanen — hapus setelah proses (kecuali ada kebutuhan audit)
- API Key di-hash sebelum disimpan di database
- Request log hanya menyimpan metadata (request_id, verdict, timestamp) — bukan gambar
- Compliance dengan regulasi data pribadi (UU PDP Indonesia)

### 5.3 Skalabilitas

- Arsitektur stateless — horizontal scalable
- Model inferensi dijalankan di dedicated inference server (terpisah dari API server)
- Queue-based untuk request yang membutuhkan processing lebih lama

---

## 6. Arsitektur Teknis (High-Level)

```
Client
  │
  ▼
API Gateway (Auth, Rate Limit, Routing)
  │
  ▼
Liveness API Server (FastAPI)
  │
  ├──► Quality Check Module (OpenCV, ringan)
  │
  ├──► Inference Service — CPU-Optimized
  │         ├── Face Detector       (YOLOv5n-face / RetinaFace-MobileNet, ONNX INT8)
  │         ├── Passive Liveness    (MiniFASNet / Silent-Face, ONNX INT8)
  │         └── Active Challenge    (MediaPipe Face Mesh — native CPU)
  │
  └──► Response Formatter
           │
           ▼
        Client Response

Dashboard ──► Analytics DB (usage logs, metrics)
```

### 6.1 Model Stack — CPU-Optimized (MVP)

Seluruh inferensi dirancang berjalan di **CPU-only** tanpa dependensi GPU. Strategi optimasi diterapkan berlapis.

| Komponen | Model | Format | Teknik Optimasi |
|---------|-------|--------|----------------|
| Face Detection | YOLOv5n-face atau RetinaFace-MobileNetV1 | ONNX INT8 | Input resize ke 320px, early-exit jika no-face |
| Passive Liveness | MiniFASNet-v2 (Silent-Face-Anti-Spoofing) | ONNX INT8 | Model < 1MB, dual-scale input crop |
| Anti-Spoof Classifier | MiniFASNet-v2 (same model, output branch) | ONNX INT8 | Shared pipeline dengan passive liveness |
| Active Challenge | MediaPipe Face Mesh (Lite) | TFLite / native | CPU-native, landmark-based, no model inference tambahan |

#### Kenapa MiniFASNet-v2?

- Parameter: ~370K — sangat ringan
- Latency CPU: ~40–80ms per frame (Intel Xeon / AMD EPYC)
- Sudah terbukti di production environment tanpa GPU
- Open source, license MIT
- Input: 80×80 crop dari wajah (2 scale: 2.7x dan 4.0x)

### 6.2 CPU Inference Optimization Stack

#### A. Model Quantization (INT8)

Semua model di-convert ke **ONNX INT8** via post-training quantization sebelum deployment:

```
FP32 model  →  ONNX export  →  INT8 quantization  →  ONNX Runtime inference
```

- Ukuran model turun ~75% dari FP32
- Latency turun ~2–3x dibanding FP32 di CPU
- Akurasi turun minimal (< 0.5% pada benchmark internal)

Tools: `onnxruntime.quantization.quantize_dynamic()` atau `quantize_static()` dengan calibration dataset.

#### B. ONNX Runtime Session Config

```python
import onnxruntime as ort

opts = ort.SessionOptions()
opts.intra_op_num_threads = 2          # Sesuaikan dengan vCPU tersedia
opts.inter_op_num_threads = 1
opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
opts.enable_mem_pattern = True
opts.enable_cpu_mem_arena = True

session = ort.InferenceSession("model_int8.onnx", sess_options=opts,
                                providers=["CPUExecutionProvider"])
```

- `intra_op_num_threads = 2` → optimal untuk VPS 2–4 vCPU; hindari over-thread
- `ORT_ENABLE_ALL` → graph fusion, constant folding, kernel optimization otomatis
- Session di-load **sekali saat startup**, bukan per-request

#### C. Pre/Post Processing — OpenCV Optimized

```python
# Resize input sebelum ke model — jangan kirim full-res ke inferensi
face_crop = cv2.resize(face_region, (80, 80), interpolation=cv2.INTER_LINEAR)
img_array = face_crop.astype(np.float32) / 255.0
img_array = np.transpose(img_array, (2, 0, 1))[np.newaxis, :]  # NCHW
```

- Semua pre-processing menggunakan `numpy` + `OpenCV` (C-backed, tidak ada Python loop)
- Face crop dilakukan **sebelum** masuk model, bukan setelah
- Batch size = 1 (single inference per request — cocok untuk API use case)

#### D. Concurrency Strategy

| Skenario | Strategi |
|---------|---------|
| 1 vCPU | 1 worker process, `intra_op_num_threads=1` |
| 2–4 vCPU | 2 worker processes (Gunicorn), `intra_op_num_threads=2` |
| 4–8 vCPU | 4 worker processes, `intra_op_num_threads=2` |
| > 8 vCPU | 4–8 workers, pertimbangkan model caching per-worker |

- Gunakan **Gunicorn + Uvicorn workers** (bukan threading) untuk isolasi memori model
- Setiap worker process load model sendiri → hindari GIL bottleneck
- Session ONNX **tidak thread-safe** untuk concurrent inference — gunakan process isolation

#### E. Pipeline Urutan Eksekusi (Latency Budget)

```
Total target P95: < 1500ms (CPU)

[Quality Check]     ~20ms   — OpenCV blur/brightness check
[Face Detection]    ~50ms   — YOLOv5n-face ONNX INT8
[Face Crop]         ~5ms    — numpy crop + resize
[Liveness Model]    ~80ms   — MiniFASNet-v2 ONNX INT8 (2 scale)
[Response Format]   ~5ms
─────────────────────────
Total estimasi:     ~160ms  (P50 CPU single-core)
```

Latency aktual bergantung pada hardware — benchmark wajib dilakukan di target server sebelum go-live.

### 6.3 Spesifikasi Server Minimum (CPU-Only)

| Tier | CPU | RAM | Estimasi Throughput |
|------|-----|-----|-------------------|
| Dev / Staging | 2 vCPU | 2 GB | ~8–12 req/menit |
| Production MVP | 4 vCPU | 4 GB | ~20–30 req/menit |
| Production Scale | 8 vCPU | 8 GB | ~50–80 req/menit |

> Untuk throughput lebih tinggi di CPU: scale-out (tambah instance) lebih efektif daripada scale-up.

### 6.4 Tech Stack API

- **Bahasa:** Python 3.11+ (FastAPI + Uvicorn)
- **Inference Runtime:** ONNX Runtime 1.17+ (CPUExecutionProvider)
- **Model Format:** ONNX INT8 (quantized)
- **Face Landmark:** MediaPipe Face Mesh Lite (active liveness)
- **Image Processing:** OpenCV 4.x + NumPy
- **Process Manager:** Gunicorn (multi-worker)
- **Queue / Session:** Redis (active liveness session management)
- **Monitoring:** Prometheus + Grafana
- **Deployment:** Docker + Docker Compose (MVP); Kubernetes (scale)

---

## 7. Dashboard (Admin & Developer)

Fitur minimal dashboard MVP:

- **API Key Management:** Generate, lihat, revoke API key
- **Usage Stats:** Total request hari ini / bulan ini, breakdown per verdict
- **Quota Monitor:** Sisa quota, history penggunaan
- **Logs:** Request log (tanpa gambar) — request_id, verdict, confidence, timestamp
- **Documentation link:** Mengarah ke API docs

---

## 8. API Documentation

- Dokumentasi tersedia di `/docs` (Swagger UI auto-generated)
- Postman Collection tersedia untuk download
- Contoh integrasi tersedia untuk: Python, JavaScript, PHP, cURL

---

## 9. Acceptance Criteria

### Epic 1: Passive Liveness API

- [ ] Endpoint `POST /v1/liveness/check` menerima gambar base64 JPEG/PNG
- [ ] Mengembalikan verdict, confidence score, dan spoof_type
- [ ] Quality check berjalan sebelum inferensi
- [ ] Response time < 1500ms untuk P95
- [ ] FAR < 1% pada test set internal

### Epic 2: Active Liveness API

- [ ] Endpoint inisiasi challenge mengembalikan session_id dan instruksi
- [ ] Session expire otomatis setelah 30 detik
- [ ] Verifikasi challenge BLINK dan SMILE berjalan akurat > 90%
- [ ] Session tidak dapat digunakan setelah expire

### Epic 3: Autentikasi & Rate Limiting

- [ ] Request tanpa API Key mendapat 401
- [ ] Request melebihi quota mendapat 429 dengan header informasi reset
- [ ] API Key dapat di-revoke dan langsung tidak berfungsi

### Epic 4: Dashboard

- [ ] Developer dapat generate API Key baru
- [ ] Grafik penggunaan ditampilkan per hari (30 hari terakhir)
- [ ] Log request dapat difilter berdasarkan verdict dan tanggal

---

## 10. Milestones & Timeline

| Milestone | Deliverable | Estimasi |
|-----------|------------|----------|
| M1 | Model selection & benchmark internal | 2 minggu |
| M2 | Passive liveness API + auth + rate limit | 3 minggu |
| M3 | Active liveness API (challenge-response) | 2 minggu |
| M4 | Dashboard (basic) + API docs | 2 minggu |
| M5 | Internal QA + security review | 1 minggu |
| M6 | Soft launch (beta access) | 1 minggu |
| **Total** | | **~11 minggu** |

---

## 11. Risks & Mitigations

| Risiko | Dampak | Mitigasi |
|--------|--------|---------|
| Model accuracy tidak memenuhi target FAR/FRR | Tinggi | Benchmark 3+ model sebelum commit; siapkan ensemble |
| Latency melampaui target di CPU | Sedang | MiniFASNet INT8 target ~160ms P50; jika tidak tercapai: kurangi model scale dari 2 → 1, turunkan resolusi input, atau scale-out instance |
| Penyalahgunaan API | Sedang | Rate limiting ketat + anomaly detection di log |
| Regulasi data pribadi (UU PDP) | Tinggi | Tidak simpan gambar; privacy-by-design dari awal |
| Deepfake attack lolos deteksi | Sedang | Label sebagai "best effort" di MVP; roadmap post-MVP |

---

## 12. Open Questions

1. Apakah gambar perlu disimpan sementara untuk tujuan audit/dispute? Jika ya, berapa lama?
2. Apakah perlu integrasi langsung dengan provider KYC lain (Verihubs, Privy, dll)?
3. Target pasar awal: developer individu atau B2B enterprise?
4. Apakah perlu white-label / on-premise option di roadmap jangka pendek?

---

*Dokumen ini adalah living document. Semua perubahan major harus melalui sign-off Product Owner dan Tech Lead.*