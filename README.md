# Liveness Detection API (Passive Anti‑Spoofing MVP)

## Deskripsi
Backend ini menyediakan **API passive anti‑spoofing** berbasis gambar tunggal (selfie) untuk mendeteksi apakah subjek nyata atau spoof. Fokus MVP: endpoint `POST /v1/liveness/check` dengan otentikasi API‑Key, quality‑check, face‑detection, dan decision engine. Semua komponen dipisah menjadi service kecil agar dapat diganti (model, threshold, dll.) tanpa mengubah kontrak API.

## Tech Stack

| Layer | Teknologi |
|-------|-----------|
| **Runtime** | Python 3.13 |
| **Framework** | FastAPI + Uvicorn / Gunicorn |
| **Face Detection** | RetinaFace (OpenCV DNN) |
| **Anti-Spoof** | MiniFASNet V1SE + V2 (ONNXRuntime, model fusion) |
| **rPPG PhysNet** | PhysNet_padding_Encoder_Decoder_MAX (ONNXRuntime) |
| **Signal Analysis** | NumPy FFT |
| **Fusion Engine** | Custom weighted scoring (0.6 anti-spoof + 0.4 rPPG) |
| **Session Store** | Thread-safe in-memory (TTL configurable) |
| **Training** | PyTorch (hanya untuk fine-tune/export) |
| **Validation** | Pydantic v2 + Pydantic Settings |
| **Logging** | structlog |
| **Template** | Jinja2 |
| **Container** | Docker + Docker Compose |
| **API Docs** | Swagger UI / ReDoc (auto-generated) |

## Prasyarat
- **Python ≥ 3.12** (proyek ini diuji dengan 3.13)
- **pip** (atau `uv pip` bila memakai `uv` untuk manajemen paket)
- **Git** (opsional, untuk version control)
- **Virtual environment** (rekomendasi: `python -m venv .venv`)

## Instalasi (Development – Local)
```bash
# 1. Clone repo (jika belum ada)
# git clone <repo-url>
# cd liveness-detection

# 2. Buat dan aktifkan virtualenv runtime (CPU-only)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies runtime API (tanpa torch)
pip install -r requirements.runtime.txt

# 4. Siapkan environment variables (opsional)
# Salin .env.example -> .env dan sesuaikan (API_KEYS, path model, dll.)
cp .env.example .env
# edit .env bila perlu

# 5. Jalankan server lokal (hot‑reload)
uvicorn app.main:app --reload
```
Server akan tersedia di `http://127.0.0.1:8000`. Swagger UI dapat diakses di `http://127.0.0.1:8000/docs`.

### Mode Environment yang Direkomendasikan

Pisahkan environment runtime dan training:

- **Runtime API (CPU-only, ringan)**: gunakan `.venv` + `requirements.runtime.txt`
- **Training/Fine-tune (GPU)**: gunakan `.venv-training` + `requirements.training.txt` + `torch` CUDA wheel

Contoh setup environment training GPU (NVIDIA):

```bash
python3 -m venv .venv-training
source .venv-training/bin/activate
pip install -r requirements.runtime.txt
pip install --index-url https://download.pytorch.org/whl/cu126 torch
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Jika output `torch.cuda.is_available()` adalah `True`, training akan memakai GPU.

## Penggunaan API
### Health Check
```bash
curl http://127.0.0.1:8000/v1/health
```
### Liveness Check (Passive)
```bash
# Buat file JSON payload (image base64, mode=passive)
PAYLOAD=$(cat <<'EOF'
{
  "image": "<base64‑encoded‑jpeg>",
  "mode": "passive"
}
EOF
)

curl -X POST http://127.0.0.1:8000/v1/liveness/check \
  -H "Content-Type: application/json" \
  -H "X-API-Key: dev-test-key-001" \
  -d "$PAYLOAD"
```
Response JSON berisi `request_id`, `verdict`, `confidence`, `spoof_type`, `face_detected`, `quality_check`, `processing_time_ms`, dan `timestamp`.

## Deployment (Production Server)
### 1. Persiapan Docker (disarankan)
**Dockerfile** (disertakan di repo) membangun image runtime berbasis ONNX (tanpa `torch`) dan menyalakan `uvicorn` menggunakan workers Gunicorn untuk proses terisolasi.
```Dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.runtime.txt
EXPOSE 8000
CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", "app.main:app", "--bind", "0.0.0.0:8000"]
```
Build & push ke registry (Docker Hub, GitHub Packages, dll.)
```bash
docker build -t your-registry/liveness-api:latest .
# docker push your-registry/liveness-api:latest
```
### 2. Jalankan di server produksi
```bash
# contoh dengan Docker Compose
cat > docker-compose.yml <<'EOF'
version: "3"
services:
  api:
    image: your-registry/liveness-api:latest
    restart: unless-stopped
    environment:
      - APP_NAME=Liveness Detection API
      - APP_VERSION=1.0.0
      - API_KEYS=[\"prod-key-001\"]
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models   # mount folder berisi model ONNX
EOF

docker compose up -d
```
### 3. Konfigurasi Reverse Proxy (nginx contoh)
```nginx
server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```
Reload nginx setelah konfigurasi.

## Dokumentasi API

- **Swagger UI**: `http://localhost:8123/docs`
- **ReDoc**: `http://localhost:8123/redoc`
- **Demo UI**: `http://localhost:8123/demo`

Demo UI digunakan untuk upload selfie, mengisi API key, lalu mencoba endpoint passive anti-spoofing langsung dari browser.

## Model ONNX

### 1. Face-detector model
Saat ini face detector yang dipakai adalah **RetinaFace Caffe** lokal di dalam project ini. File runtime yang dipakai:

```text
models/detection_model/deploy.prototxt
models/detection_model/Widerface-RetinaFace.caffemodel
```

Detector ini dipakai agar bounding box wajah konsisten dengan pipeline referensi model anti-spoof.

### 2. Anti-spoof model
Model anti-spoof runtime yang dipakai saat ini adalah file `.onnx` lokal di dalam project:

```text
models/active/antispoof/best_2.7_80x80_MiniFASNetV2.onnx
models/active/antispoof/best_4_0_0_80x80_MiniFASNetV1SE.onnx
```

Keduanya dijalankan dengan model fusion agar sesuai dengan pipeline referensi repo sumber.

### 3. rPPG PhysNet model
Model rPPG PhysNet runtime untuk pipeline stream real-time menggunakan ONNX:

```text
models/active/rppg/UBFC-rPPG_PhysNet_DiffNormalized.onnx
```

Model ONNX ini berasal dari checkpoint `UBFC-rPPG_PhysNet_DiffNormalized.pth` yang diekspor via `scripts/export_pth_to_onnx.sh`.

### 4. Konfigurasi model
Pastikan `.env` menunjuk ke file lokal project:

```env
RETINAFACE_DEPLOY_PATH=models/detection_model/deploy.prototxt
RETINAFACE_CAFFEMODEL_PATH=models/detection_model/Widerface-RetinaFace.caffemodel
ANTI_SPOOF_MODEL_PATH=models/active/antispoof/best_2.7_80x80_MiniFASNetV2.onnx,models/active/antispoof/best_4_0_0_80x80_MiniFASNetV1SE.onnx
```

### 4. Verifikasi file model
Periksa apakah file model lokal tersedia:

```bash
ls -lh models/
ls -lh models/detection_model/
```

Semua dependency runtime sekarang harus berada di dalam project ini, tanpa path ke repo eksternal.

Jika menggunakan pipeline **stream real-time**, pastikan juga model rPPG tersedia:

```env
RPPG_MODEL_PATH=models/active/rppg/UBFC-rPPG_PhysNet_DiffNormalized.onnx
```

Catatan: model ONNX ini diekspor dari checkpoint repo `rPPG-Toolbox` (`final_model_release/UBFC-rPPG_PhysNet_DiffNormalized.pth`) agar runtime API tidak bergantung pada `torch`.


## Demo Stream

Demo UI sekarang mendukung mode stream berbasis snapshot berkala ke endpoint `POST /v1/liveness/check`.

Cara pakai:
- buka `/demo`
- isi `X-API-Key`
- klik `Start Stream` untuk mulai stream detection
- klik `Stop Stream` untuk menghentikan stream
- tombol `Upload File` tetap tersedia untuk single check manual

Catatan:
- stream demo ini bukan WebRTC backend stream
- browser hanya mengirim snapshot berkala dari video ke endpoint passive existing
- badge verdict utama memakai smoothing agar tidak flicker berlebihan
- saat terkena `429`, stream pause otomatis dan menampilkan countdown sampai retry
- panel kanan menampilkan mini history hasil terbaru selama sesi aktif
- iOS / browser tanpa `getUserMedia` akan fallback ke kamera/file picker native

## Real-Time Stream API (Multi-Frame)

Selain endpoint single-image, API ini menyediakan pipeline **multi-frame** yang mendukung deteksi liveness berbasis sinyal fisiologis (rPPG).

### Alur Kerja

```
Camera Stream → Face Detection → MiniFASNet (fast reject)
                                    ↓
                              rPPG PhysNet ONNX
                                    ↓
                            Signal Analysis
                                    ↓
                            Final Decision
```

### Endpoint Stream

| Method | Path | Deskripsi |
|--------|------|-----------|
| `POST` | `/v1/liveness/stream/init` | Buat session stream baru |
| `POST` | `/v1/liveness/stream/frame` | Upload satu frame |
| `GET`  | `/v1/liveness/stream/result` | Ambil hasil verdict |
| `POST` | `/v1/liveness/stream/end` | Hapus session |

### Cara Pakai

```bash
# 1. Inisialisasi session
INIT=$(curl -s -X POST http://127.0.0.1:8000/v1/liveness/stream/init \
  -H "X-API-Key: dev-test-key-001")
echo $INIT
# Output: {"session_id":"ee42f5e9dde7","window_ms":3000,"min_frames":6,...}

# 2. Upload frame (minimal 6 frame)
SESSION_ID="ee42f5e9dde7"
for i in $(seq 1 6); do
  curl -s -X POST http://127.0.0.1:8000/v1/liveness/stream/frame \
    -H "X-API-Key: dev-test-key-001" \
    -F "session_id=$SESSION_ID" \
    -F "frame=@foto.jpeg" > /dev/null
done

# 3. Ambil hasil
curl -s "http://127.0.0.1:8000/v1/liveness/stream/result?session_id=$SESSION_ID" \
  -H "X-API-Key: dev-test-key-001"

# 4. Hapus session (opsional)
curl -s -X POST http://127.0.0.1:8000/v1/liveness/stream/end \
  -H "X-API-Key: dev-test-key-001" \
  -F "session_id=$SESSION_ID"
```

### Komponen Pipeline

- **MiniFASNet Fast Reject**: setiap frame dicek segera; jika spoof terdeteksi dengan confidence tinggi, session langsung ditolak tanpa perlu menunggu frame cukup
- **rPPG PhysNet ONNX**: model ONNX untuk ekstraksi sinyal fisiologis dari sequence wajah (`models/active/rppg/UBFC-rPPG_PhysNet_DiffNormalized.onnx`)
- **Signal Analyzer**: hitung SNR, peak count, estimasi heart-rate, dan `signal_confidence` via FFT
- **Fusion Engine**: gabungkan skor MiniFASNet (bobot 0.6) dan rPPG (bobot 0.4) untuk keputusan akhir (`LIVE`, `SPOOF`, `UNCERTAIN`)

### Konfigurasi Stream (via `.env`)

```env
STREAM_WINDOW_MS=3000        # durasi window koleksi frame
STREAM_MIN_FRAMES=6          # frame minimum sebelum rPPG dijalankan
STREAM_FRAME_RATE=6.0        # target fps untuk estimation
FUSION_LIVE_THRESHOLD=0.7    # skor minimum untuk verdict LIVE
FUSION_SPOOF_THRESHOLD=0.3   # skor maksimum untuk verdict SPOOF
RPPG_MODEL_PATH=models/active/rppg/UBFC-rPPG_PhysNet_DiffNormalized.onnx
```

### Catatan

- Session disimpan di memory dengan TTL 30 detik
- Model `UBFC-rPPG_PhysNet_DiffNormalized.onnx` wajib ada agar `stream/result` bisa menghasilkan verdict berbasis rPPG
- Stream API belum menggunakan WebSocket; client perlu mengirim frame berkala via HTTP polling

## Fine-Tuning Model Anti-Spoof (Custom Dataset)

Repo ini sekarang menyediakan script untuk menyiapkan dataset dan fine-tune model MiniFASNet agar kompatibel dengan endpoint inference yang ada.

### Prasyarat tambahan

- gunakan environment terpisah (`.venv-training`) untuk training
- install dependency training (termasuk `torch`):

```bash
pip install -r requirements.training.txt
```

- untuk training GPU, install `torch` CUDA wheel (contoh CUDA 12.6):

```bash
pip install --index-url https://download.pytorch.org/whl/cu126 torch
```

- GPU sangat disarankan untuk proses training
- model file bawaan sudah tersedia di folder `models/`

### 1) Siapkan dataset training dari folder sumber

```bash
python3 scripts/train/prepare_antispoof_dataset.py \
  --dataset-live-root /home/kurob/Documents/KERJAAN/AI/dataset/anti-spoofing \
  --dataset-attack-root /home/kurob/Documents/KERJAAN/AI/dataset/anti-spoofing-1 \
  --output-root training_data/antispoof \
  --max-frames-per-video 8
```

Output:
- crop wajah 80x80 di `training_data/antispoof/images/`
- manifest di `training_data/antispoof/manifest.csv` (sudah berisi split train/val/test)

### 2) Fine-tune model

```bash
python3 scripts/train/train_mini_fasnet.py \
  --manifest training_data/antispoof/manifest.csv \
  --data-root training_data/antispoof \
  --model-name 2.7_80x80_MiniFASNetV2.pth \
  --epochs 12 \
  --batch-size 32 \
  --output-dir models/finetuned
```

Checkpoint terbaik akan disimpan ke:
- `models/finetuned/best_2.7_80x80_MiniFASNetV2.pth`

### 3) Pakai model hasil fine-tune di API

Set di `.env`:

```env
ANTI_SPOOF_MODEL_PATH=models/active/antispoof/best_2.7_80x80_MiniFASNetV2.onnx,models/active/antispoof/best_4_0_0_80x80_MiniFASNetV1SE.onnx
```

Lalu restart server.

### 4) Konversi model `.pth` ke `.onnx`

Gunakan script:

```bash
scripts/export_pth_to_onnx.sh --help
```

Contoh convert MiniFASNet:

```bash
scripts/export_pth_to_onnx.sh \
  --pth models/active/antispoof/best_2.7_80x80_MiniFASNetV2.pth \
  --onnx models/active/antispoof/best_2.7_80x80_MiniFASNetV2.onnx
```

Contoh convert PhysNet (rPPG):

```bash
scripts/export_pth_to_onnx.sh \
  --type physnet \
  --frames 6 \
  --pth models/active/rppg/UBFC-rPPG_PhysNet_DiffNormalized.pth \
  --onnx models/active/rppg/UBFC-rPPG_PhysNet_DiffNormalized.onnx
```

Catatan:
- Script akan auto-detect tipe model jika `--type` tidak diisi.
- Untuk PhysNet, output ONNX diekspor dari output sinyal rPPG (`rppg`).

## Catatan Penting
- **Threshold LIVE**: untuk deployment saat ini disarankan `LIVE_THRESHOLD=0.90` agar replay screen borderline tidak langsung lolos sebagai `LIVE`; skor di bawah itu akan turun menjadi `UNCERTAIN`.
- **Rate‑limit**: default 60 request per menit per API‑Key, header `X-RateLimit-Limit`, `X-RateLimit-Remaining`, `X-RateLimit-Reset` disertakan di setiap response.
- **Logging**: terstruktur, menuliskan `request_id`, verdict, confidence, dan durasi pemrosesan. Gambar mentah **tidak pernah ditulis ke log**.
- **Testing**: Tidak ada unit‑test otomatis di repo (policy). Tambahkan tes bila diminta.

## Pengembangan Selanjutnya
- Integrasi **model anti‑spoofing** yang lebih akurat (MiniFASNet‑v2, dll.).
- Menyediakan **challenge active** (`/v1/liveness/challenge/*`).
- Dashboard admin untuk API‑Key management & usage stats.
- Penambahan **CI/CD** pipeline (GitHub Actions) untuk lint, type‑check, dan image build otomatis.

---
*Dokumen ini di‑generate pada 2026‑04‑21 oleh Opencode.*
