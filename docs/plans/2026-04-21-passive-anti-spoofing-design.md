# Passive Anti-Spoofing MVP Design

## Tujuan

MVP pertama dibatasi pada passive anti-spoofing berbasis single image untuk mendeteksi apakah selfie berasal dari subjek nyata atau spoof sederhana. Fokus implementasi adalah menyediakan satu endpoint yang stabil, ringan di CPU, dan cukup jelas output-nya untuk diintegrasikan ke pipeline onboarding.

## Scope MVP

Masuk scope:
- Endpoint `POST /v1/liveness/check`
- Input gambar `base64` dengan format `JPEG` atau `PNG`
- API Key authentication
- Quality check dasar sebelum inferensi
- Face detection untuk satu wajah utama
- Passive anti-spoofing inference berbasis model ringan CPU-only
- Response JSON dengan `verdict`, `confidence`, `spoof_type`, dan metadata proses
- Structured logging tanpa menyimpan gambar
- Rate limiting sederhana per menit

Di luar scope:
- Active liveness challenge
- Dashboard
- Multi-face support penuh
- Streaming / video
- Penyimpanan gambar untuk audit
- Klasifikasi `3D_MASK` dan `DEEPFAKE` pada fase pertama

## Pendekatan Teknis

Pipeline request yang dipilih:

1. Validasi header `X-API-Key`
2. Validasi payload JSON dan decode base64 image
3. Validasi ukuran file dan format gambar
4. Jalankan quality check dasar: resolusi, blur, brightness
5. Jalankan face detection
6. Pilih satu wajah utama dengan bounding box terbesar
7. Crop dan resize wajah sesuai kebutuhan model anti-spoofing
8. Jalankan anti-spoofing inference pada CPU
9. Petakan score model ke `verdict` dan `spoof_type`
10. Kembalikan response dan log metadata request

Keputusan utama anti-spoofing diambil dari model. Heuristik quality check hanya menentukan apakah input layak diproses dan membantu mengisi detail `issues` bila gambar ditolak sebelum inferensi.

## Komponen Sistem

### API Layer

`FastAPI` menangani routing, validasi request, authentication, error mapping, dan response formatting. Endpoint health check tetap tersedia, tetapi fokus MVP adalah endpoint liveness check.

### Authentication

Semua request ke endpoint liveness wajib mengirim `X-API-Key`. Untuk MVP, validasi API key dapat dimulai dari daftar key yang didefinisikan di environment atau config lokal. Hash-based storage tetap menjadi target jika persistence ditambahkan.

### Rate Limiting

Rate limiting sederhana per API key diterapkan per menit. Respons mengembalikan header standar:
- `X-RateLimit-Limit`
- `X-RateLimit-Remaining`
- `X-RateLimit-Reset`

Jika limit terlampaui, API mengembalikan `429 RATE_LIMIT_EXCEEDED`.

### Quality Check Module

Modul ini menolak input yang jelas buruk sebelum model berjalan. Cek minimum:
- resolusi minimum `320x320`
- face size minimum `100x100` setelah deteksi
- blur score di atas threshold minimum
- brightness dalam rentang aman

Jika gagal, response menggunakan verdict `POOR_QUALITY` dan mengisi `quality_check.issues`.

### Face Detection

Face detector ringan ONNX dipakai untuk menemukan wajah utama. Jika tidak ada wajah terdeteksi, sistem mengembalikan `NO_FACE`. Jika banyak wajah, MVP memilih wajah terbesar agar pipeline tetap sederhana.

### Passive Anti-Spoofing Inference

Model utama mengikuti arah PRD: `MiniFASNet-v2` atau model setara dalam format `ONNX INT8`. Inference session harus di-load sekali saat startup agar latency stabil. Hasil model dipetakan ke confidence dan verdict.

Pada fase pertama, `spoof_type` hanya menggunakan:
- `PRINT_ATTACK`
- `SCREEN_REPLAY`
- `null`

Jika model belum cukup spesifik untuk membedakan `PRINT_ATTACK` vs `SCREEN_REPLAY`, MVP boleh mengembalikan `SPOOF` dengan `spoof_type: null` terlebih dulu, selama kontrak API mengizinkan fallback tersebut.

## Kontrak API

### Request

Endpoint: `POST /v1/liveness/check`

Payload minimal:

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

Catatan:
- `mode` pada MVP hanya menerima `passive`
- `return_face_crop` dapat diabaikan dulu bila belum diperlukan
- `min_face_size` dapat diambil dari config dengan fallback default `100`

### Response Sukses

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

Verdict aktif pada MVP:
- `LIVE`
- `SPOOF`
- `UNCERTAIN`
- `NO_FACE`
- `POOR_QUALITY`

### Error Response

Format error distandarkan:

```json
{
  "error": {
    "code": "INVALID_IMAGE_FORMAT",
    "message": "Image must be JPEG or PNG, base64 encoded",
    "request_id": "req_abc123xyz"
  }
}
```

Error minimum yang perlu ada di MVP:
- `INVALID_IMAGE_FORMAT`
- `IMAGE_TOO_LARGE`
- `UNAUTHORIZED`
- `RATE_LIMIT_EXCEEDED`
- `INTERNAL_ERROR`
- `MODEL_UNAVAILABLE`

## Decision Flow

Urutan keputusan response:

1. Jika API key tidak valid -> `401 UNAUTHORIZED`
2. Jika payload / base64 / format invalid -> `400 INVALID_IMAGE_FORMAT`
3. Jika ukuran file terlalu besar -> `400 IMAGE_TOO_LARGE`
4. Jika quality check gagal -> `200` dengan `verdict=POOR_QUALITY`
5. Jika tidak ada wajah -> `200` dengan `verdict=NO_FACE`
6. Jika inferensi gagal karena model unavailable -> `503 MODEL_UNAVAILABLE`
7. Jika confidence spoof tinggi -> `200` dengan `verdict=SPOOF`
8. Jika confidence live tinggi -> `200` dengan `verdict=LIVE`
9. Selain itu -> `200` dengan `verdict=UNCERTAIN`

Threshold detail harus configurable, bukan hardcoded permanen di business logic.

## Konfigurasi Awal

Konfigurasi yang perlu disiapkan sejak awal:
- API keys yang diizinkan
- request size maksimum, default `5MB`
- minimum resolution
- minimum face size
- blur threshold
- brightness threshold bawah dan atas
- `live_threshold`
- `spoof_threshold`
- path model face detector
- path model anti-spoofing

## Observability dan Logging

Setiap request menghasilkan `request_id`. Log cukup menyimpan metadata:
- `request_id`
- timestamp
- verdict
- confidence
- spoof_type
- processing_time_ms
- api_key identifier non-raw jika tersedia

Gambar mentah tidak disimpan ke disk atau log.

## Risiko dan Mitigasi

### Model belum siap membedakan jenis spoof

Mitigasi: fase pertama mengutamakan keputusan `LIVE/SPOOF/UNCERTAIN`, sementara `spoof_type` boleh `null` bila classifier detail belum stabil.

### Latency CPU lebih tinggi dari target

Mitigasi: load session saat startup, pakai ONNX INT8, resize input lebih awal, dan batasi satu wajah utama.

### Input dunia nyata sangat bervariasi

Mitigasi: quality check eksplisit dan threshold configurable agar tuning tidak mengubah kontrak API.

## Deliverable Fase Implementasi Pertama

Fase implementasi pertama dianggap selesai bila tersedia:
- endpoint `POST /v1/liveness/check`
- API key auth
- rate limiting sederhana
- quality check dasar
- face detection
- passive anti-spoof inference stub atau model-backed
- response dan error contract yang konsisten
- structured logging tanpa penyimpanan gambar
