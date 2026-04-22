# Passive Anti-Spoofing MVP Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Membangun MVP backend passive anti-spoofing berbasis single image dengan endpoint `POST /v1/liveness/check`, authentication, quality check, face detection, dan decision flow verdict yang konsisten.

**Architecture:** Sistem dibangun sebagai API `FastAPI` yang menerima image base64, menjalankan quality check dan face detection, lalu meneruskan crop wajah ke anti-spoofing inference service CPU-only. Business logic dipisah tipis per tahap pipeline agar threshold, error mapping, dan model backend bisa diganti tanpa mengubah kontrak API.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic, OpenCV, NumPy, ONNX Runtime, Gunicorn/Uvicorn, structured logging.

---

### Task 1: Bootstrap struktur project backend

**Files:**
- Create: `app/__init__.py`
- Create: `app/main.py`
- Create: `app/api/__init__.py`
- Create: `app/api/routes/__init__.py`
- Create: `app/core/__init__.py`
- Create: `app/services/__init__.py`
- Create: `app/schemas/__init__.py`
- Create: `requirements.txt`
- Create: `.env.example`

**Step 1: Buat package dan entrypoint aplikasi**

Tambahkan struktur package dasar `app/` dan `app/main.py` dengan factory atau inisialisasi `FastAPI` minimal.

**Step 2: Definisikan dependency Python awal**

Isi `requirements.txt` minimal dengan dependency yang dibutuhkan untuk API, image processing, dan ONNX runtime.

**Step 3: Tambahkan contoh environment**

Isi `.env.example` dengan placeholder untuk API key, path model, dan threshold dasar.

**Step 4: Jalankan server lokal untuk memastikan app bootstrap berhasil**

Run: `uvicorn app.main:app --reload`
Expected: aplikasi start tanpa error import.

### Task 2: Tambahkan konfigurasi aplikasi terpusat

**Files:**
- Create: `app/core/config.py`
- Modify: `app/main.py`

**Step 1: Buat settings terpusat**

Definisikan konfigurasi untuk:
- allowed API keys
- max request size
- minimum resolution
- minimum face size
- blur threshold
- brightness threshold
- live/spoof threshold
- model paths

**Step 2: Hubungkan config ke aplikasi**

Pastikan settings dapat diakses dari route dan service tanpa hardcode berulang.

**Step 3: Verifikasi config dapat dimuat dari environment**

Run: `uvicorn app.main:app --reload`
Expected: app start dan config invalid menghasilkan error startup yang jelas.

### Task 3: Tambahkan schema request dan response API

**Files:**
- Create: `app/schemas/liveness.py`
- Create: `app/schemas/common.py`
- Modify: `app/schemas/__init__.py`

**Step 1: Definisikan schema request**

Tambahkan model request untuk payload `image`, `mode`, dan `options`.

**Step 2: Definisikan schema response sukses**

Tambahkan model untuk:
- `quality_check`
- response liveness check

**Step 3: Definisikan schema error**

Tambahkan model error terstandar agar semua route konsisten.

**Step 4: Verifikasi schema muncul di OpenAPI**

Run: `uvicorn app.main:app --reload`
Expected: schema request/response tampil di `/docs`.

### Task 4: Implementasikan utility request metadata dan error handling

**Files:**
- Create: `app/core/errors.py`
- Create: `app/core/request_id.py`
- Modify: `app/main.py`

**Step 1: Buat generator `request_id`**

Pastikan setiap request dapat memiliki ID unik yang bisa dipakai pada log dan response.

**Step 2: Buat exception aplikasi**

Definisikan error terstruktur untuk `INVALID_IMAGE_FORMAT`, `IMAGE_TOO_LARGE`, `UNAUTHORIZED`, `RATE_LIMIT_EXCEEDED`, `MODEL_UNAVAILABLE`, dan `INTERNAL_ERROR`.

**Step 3: Tambahkan global exception handler**

Map exception internal ke HTTP status dan body error yang sesuai.

**Step 4: Verifikasi respons error konsisten**

Run: `uvicorn app.main:app --reload`
Expected: exception menghasilkan JSON error dengan `request_id`.

### Task 5: Implementasikan API key authentication

**Files:**
- Create: `app/api/dependencies/auth.py`
- Modify: `app/core/config.py`
- Modify: `app/api/routes/__init__.py`

**Step 1: Tambahkan dependency auth berbasis `X-API-Key`**

Validasi header request terhadap daftar API key yang diizinkan.

**Step 2: Pisahkan identitas key dari raw secret**

Jika memungkinkan, turunkan identifier aman untuk logging tanpa membocorkan raw key.

**Step 3: Verifikasi request tanpa key ditolak**

Run: `uvicorn app.main:app --reload`
Expected: request tanpa `X-API-Key` mendapat `401`.

### Task 6: Implementasikan rate limiter sederhana

**Files:**
- Create: `app/services/rate_limiter.py`
- Create: `app/api/dependencies/rate_limit.py`
- Modify: `app/core/config.py`

**Step 1: Buat rate limiter in-memory untuk MVP**

Track jumlah request per API key per window menit.

**Step 2: Tambahkan dependency rate limiting pada route**

Saat limit terlampaui, kembalikan `429 RATE_LIMIT_EXCEEDED`.

**Step 3: Tambahkan header rate limit ke response**

Set `X-RateLimit-Limit`, `X-RateLimit-Remaining`, dan `X-RateLimit-Reset`.

**Step 4: Verifikasi limit bekerja secara fungsional**

Run: `uvicorn app.main:app --reload`
Expected: request berulang pada key yang sama akhirnya menerima `429`.

### Task 7: Implementasikan image decoding dan validasi input

**Files:**
- Create: `app/services/image_decoder.py`
- Create: `app/services/image_validation.py`

**Step 1: Tambahkan decoder base64 ke image array**

Validasi bahwa payload benar-benar gambar `JPEG/PNG` yang dapat didecode.

**Step 2: Tambahkan validasi ukuran file dan dimensi minimum**

Reject input yang melebihi size maksimum atau dimensi tidak memenuhi syarat.

**Step 3: Verifikasi invalid input dipetakan ke error yang benar**

Run: `uvicorn app.main:app --reload`
Expected: base64 rusak atau format salah mendapat `400 INVALID_IMAGE_FORMAT`.

### Task 8: Implementasikan quality check dasar

**Files:**
- Create: `app/services/quality_check.py`

**Step 1: Tambahkan evaluasi blur dan brightness**

Gunakan OpenCV/NumPy untuk menghitung blur score dan brightness sederhana.

**Step 2: Kembalikan hasil quality check terstruktur**

Output harus berisi `passed` dan daftar `issues`.

**Step 3: Pastikan quality check tidak langsung bergantung pada route**

Jaga service ini reusable untuk pipeline utama.

### Task 9: Implementasikan face detection service

**Files:**
- Create: `app/services/face_detector.py`
- Modify: `app/core/config.py`

**Step 1: Buat wrapper service untuk face detector**

Service harus load model sekali saat startup atau lazy-init aman.

**Step 2: Tambahkan logika memilih wajah utama**

Jika ada beberapa wajah, pilih bounding box terbesar.

**Step 3: Definisikan kontrak hasil deteksi**

Output minimal: `face_detected`, `bbox`, dan ukuran wajah.

**Step 4: Verifikasi kasus no-face ditangani rapi**

Expected: pipeline bisa mengembalikan `NO_FACE` tanpa exception tak tertangani.

### Task 10: Implementasikan anti-spoofing inference service

**Files:**
- Create: `app/services/anti_spoof.py`
- Modify: `app/core/config.py`

**Step 1: Buat wrapper ONNX Runtime session**

Load model sekali, atur session options sesuai target CPU-only.

**Step 2: Tambahkan preprocessing crop wajah**

Resize dan normalisasi input sesuai format model.

**Step 3: Tambahkan output parser**

Ubah output model menjadi confidence live/spoof yang bisa dipakai business logic.

**Step 4: Tambahkan fallback error yang jelas jika model belum tersedia**

Expected: `MODEL_UNAVAILABLE` bila file model tidak ada atau gagal dimuat.

### Task 11: Implementasikan decision engine verdict

**Files:**
- Create: `app/services/verdict_engine.py`

**Step 1: Buat fungsi pemetaan score ke verdict**

Gunakan `live_threshold` dan `spoof_threshold` dari config.

**Step 2: Tambahkan aturan fallback `UNCERTAIN`**

Jika score berada di area abu-abu, jangan paksa `LIVE` atau `SPOOF`.

**Step 3: Tambahkan `spoof_type` default**

Untuk fase pertama, izinkan `spoof_type` bernilai `null` bila classifier detail belum tersedia.

### Task 12: Implementasikan route `POST /v1/liveness/check`

**Files:**
- Create: `app/api/routes/liveness.py`
- Modify: `app/main.py`
- Modify: `app/api/routes/__init__.py`

**Step 1: Rangkai seluruh pipeline di route handler**

Urutan harus mengikuti design: auth -> rate limit -> decode -> quality -> detect face -> inference -> verdict -> response.

**Step 2: Ukur `processing_time_ms`**

Gunakan timer sederhana untuk mengukur durasi proses.

**Step 3: Isi response sesuai kontrak API**

Pastikan field response lengkap dan konsisten pada semua jalur sukses.

**Step 4: Verifikasi endpoint muncul di Swagger**

Run: `uvicorn app.main:app --reload`
Expected: endpoint dapat diakses lewat `/docs`.

### Task 13: Tambahkan health endpoint

**Files:**
- Create: `app/api/routes/health.py`
- Modify: `app/main.py`

**Step 1: Tambahkan `GET /v1/health`**

Kembalikan `status`, `version`, dan `model_version` atau placeholder yang aman.

**Step 2: Verifikasi endpoint health tersedia tanpa auth**

Run: `uvicorn app.main:app --reload`
Expected: endpoint health merespons `200`.

### Task 14: Tambahkan structured logging request metadata

**Files:**
- Create: `app/core/logging.py`
- Modify: `app/main.py`
- Modify: `app/api/routes/liveness.py`

**Step 1: Konfigurasikan logger aplikasi**

Gunakan format log terstruktur atau key-value yang mudah diparsing.

**Step 2: Log metadata penting**

Log minimal `request_id`, verdict, confidence, spoof_type, processing time, dan key identifier aman.

**Step 3: Pastikan image raw tidak pernah ikut log**

Review jalur request agar payload besar tidak tercetak ke log.

### Task 15: Dokumentasikan cara menjalankan service

**Files:**
- Create: `README.md`

**Step 1: Tulis cara install dependency**

Jelaskan environment minimal dan perintah instalasi.

**Step 2: Tulis cara menjalankan server**

Sertakan contoh `uvicorn` command dan cara set `X-API-Key`.

**Step 3: Tulis contoh request `curl`**

Sertakan contoh payload base64 dan contoh response.

### Task 16: Verifikasi manual end-to-end

**Files:**
- Modify: `README.md`

**Step 1: Jalankan aplikasi lokal**

Run: `uvicorn app.main:app --reload`
Expected: startup sukses tanpa exception.

**Step 2: Uji health endpoint**

Run: `curl http://127.0.0.1:8000/v1/health`
Expected: response `200` dengan status `ok`.

**Step 3: Uji request tanpa API key**

Expected: response `401 UNAUTHORIZED`.

**Step 4: Uji request dengan payload invalid**

Expected: response `400 INVALID_IMAGE_FORMAT`.

**Step 5: Uji request valid dengan gambar yang tersedia**

Expected: response sukses dengan salah satu verdict yang didukung.

**Step 6: Catat batasan implementasi aktual**

Jika model detail atau `spoof_type` granular belum siap, dokumentasikan secara jujur di README.
