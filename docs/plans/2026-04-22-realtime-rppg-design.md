# Desain Real‑Time rPPG untuk Liveness Detection

## Ringkasan
Desain ini menambahkan pipeline **multi‑frame** pada API yang sudah ada sehingga dapat memproses **rPPG PhysNet** secara realtime dengan latensi rendah. Kami menambahkan komponen **session stateful**, endpoint **stream**, serta servis **PhysNet** dan **Signal Analysis** yang terintegrasi dengan hasil **MiniFASNet** untuk keputusan akhir.

## Komponen Utama
| Nama | Deskripsi |
|------|-----------|
| **LivenessSession** | Menyimpan buffer frame (crop wajah, skor MiniFASNet, timestamp). State: `collecting`, `ready`, `decided`, `expired`. |
| **Stream Init Endpoint** (`POST /v1/liveness/stream/init`) | Membuat session UUID, mengembalikan konfigurasi window (mis 3000 ms) dan target frame‑rate. |
| **Frame Upload Endpoint** (`POST /v1/liveness/stream/frame`) | Menerima satu frame (base64 JPEG). Melakukan quality‑check, face‑detect, MiniFASNet. Jika spoof kuat → `fast_reject`. Jika lolos, frame ditambahkan ke buffer. |
| **Result Endpoint** (`GET /v1/liveness/stream/result`) | Bila buffer sudah mencapai `min_frames` dalam `window_ms`, memanggil **PhysNetService** untuk inferensi rPPG, lalu **SignalAnalyzer** menghitung metrik sinyal. Hasil **fusion** mengembalikan verdict (`LIVE`, `SPOOF`, `UNCERTAIN`, `TIMEOUT`). |
| **Stream End Endpoint** (`POST /v1/liveness/stream/end`) | Menghapus sesi dari memori (opsional). |
| **PhysNetService** (`app/services/rppg_physnet.py`) | Wrapper ONNXRuntime untuk model `models/rppg_physnet.onnx`. Input: tensor `[N, C, H, W]` (N = jumlah frame). Output: vektor sinyal rPPG. |
| **SignalAnalyzer** (`app/services/signal_analysis.py`) | Menghitung FFT, Peak‑to‑Noise Ratio, estimasi heart‑rate, dan menghasilkan `signal_confidence`. |
| **Fusion Engine** (`app/services/fusion.py`) | Menggabungkan skor MiniFASNet (bobot 0.6) dan confidence sinyal rPPG (bobot 0.4). Jika total ≥ 0.7 → `LIVE`, ≤ 0.3 → `SPOOF`, else `UNCERTAIN`. |
| **SessionStore** (`app/core/session_store.py`) | Dictionary thread‑safe dengan TTL 30 s. Menyediakan API `create_session`, `get_session`, `add_frame`, `expire_old`. |

## Alur Kerja (Flow)
1. **Init** – client memanggil `/stream/init`, server mengembalikan `session_id`. 
2. **Frame** – setiap frame dikirim ke `/stream/frame`. 
   - Jika MiniFASNet memberi label `SPOOF` dengan confidence ≥ `spoof_threshold` → server langsung mengirim `fast_reject`. 
   - Jika lolos, ROI wajah ditambahkan ke buffer. 
3. **Buffer Check** – setelah `window_ms` atau `min_frames` tercapai, endpoint `/result` memanggil:
   - `PhysNetService.infer(buffer)` → sinyal rPPG. 
   - `SignalAnalyzer.analyze(signal)` → `signal_confidence`. 
   - `FusionEngine.fuse(mini_fas_result, rppg_result, signal_metrics)` → verdict. 
4. **Response** – server mengembalikan `verdict`, `confidence`, `signal_confidence`, dan detail analisis. 
5. **End** – client optional memanggil `/stream/end` untuk membersihkan sesi.

## Model & Dependency
- Tambahkan model ONNX `rppg_physnet.onnx` ke folder `models/`.
- `requirements.txt` sudah mencakup `onnxruntime`; tidak perlu perubahan.
- Pastikan `opencv-python-headless` tetap untuk preprocessing.

## Perubahan Kode yang Diperlukan
1. **Buat service** `app/services/rppg_physnet.py` (load ONNX, infer). 
2. **Buat service** `app/services/signal_analysis.py`. 
3. **Buat modul** `app/services/fusion.py`. 
4. **Buat** `app/core/session_store.py`. 
5. **Tambah router** `app/api/routes/stream.py` dengan 4 endpoint di atas. 
6. **Update** `app/main.py` untuk `include_router(stream.router)`. 
7. **Modifikasi demo UI** (`app/templates/demo.html` + JS) untuk meng‑capture video, mengirim snapshot tiap 250 ms ke endpoint `/frame`. 

## Keamanan & Performansi
- Semua frame disimpan hanya dalam memori, tidak ditulis ke disk. 
- Buffer dibatasi ukuran `max_frames = window_ms / (1000/frame_rate)` (default 12). 
- MiniFASNet tetap dijalankan pada CPU; PhysNet dapat dipindahkan ke GPU bila tersedia (ONNX‑runtime `ExecutionProvider`). 
- TTL sesi 30 s untuk mencegah kebocoran memori.

## Dokumentasi Tambahan
- `docs/architecture/stream_state.md` – diagram kelas `LivenessSession`. 
- `docs/api/stream_endpoints.md` – spesifikasi request/response lengkap dengan contoh JSON.

## Timeline Implementasi (contoh)
| Minggu | Task |
|--------|------|
| 1 | Tambah `session_store`, service `rppg_physnet`, `signal_analysis`. |
| 2 | Implementasi router `stream` dan integrasi dengan `fusion`. |
| 3 | Update demo UI, testing manual, profiling latensi. |
| 4 | Dokumentasi, review code, persiapan merge. |

---
*Catatan*: Semua teks penjelasan ditulis dalam Bahasa Indonesia sesuai kebijakan global.
