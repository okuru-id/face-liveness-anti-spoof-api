# Stream API Endpoints

Dokumen ini menjelaskan endpoint baru untuk pipeline liveness detection berbasis stream multi-frame.

## Ringkasan Endpoint

Base path mengikuti versi API aktif:

- `POST /v1/liveness/stream/init`
- `POST /v1/liveness/stream/frame`
- `GET /v1/liveness/stream/result`
- `POST /v1/liveness/stream/end`

Semua endpoint memerlukan header `X-API-Key`.

## 1. Init Session

`POST /v1/liveness/stream/init`

Membuat session stream baru dan mengembalikan parameter koleksi frame.

Contoh request:

```bash
curl -X POST http://127.0.0.1:8000/v1/liveness/stream/init \
  -H "X-API-Key: dev-test-key-001"
```

Contoh response:

```json
{
  "request_id": "req_123456789abc",
  "session_id": "1a2b3c4d5e6f",
  "window_ms": 3000,
  "min_frames": 6,
  "frame_interval_ms": 166,
  "processing_time_ms": 1
}
```

## 2. Upload Frame

`POST /v1/liveness/stream/frame`

Menerima satu frame gambar untuk session tertentu.

Field form-data:

- `session_id`: ID session dari endpoint init
- `frame`: file gambar JPEG/PNG

Contoh request:

```bash
curl -X POST http://127.0.0.1:8000/v1/liveness/stream/frame \
  -H "X-API-Key: dev-test-key-001" \
  -F "session_id=1a2b3c4d5e6f" \
  -F "frame=@foto.jpeg"
```

Contoh response sukses koleksi frame:

```json
{
  "request_id": "req_123456789abc",
  "session_id": "1a2b3c4d5e6f",
  "status": "collecting",
  "frame_count": 3,
  "processing_time_ms": 27
}
```

Contoh response fast reject:

```json
{
  "request_id": "req_123456789abc",
  "session_id": "1a2b3c4d5e6f",
  "status": "fast_reject",
  "reason": "spoof_confident",
  "confidence": 0.9342,
  "processing_time_ms": 18
}
```

Kemungkinan nilai `status`:

- `collecting`: frame diterima, buffer belum cukup
- `ready`: jumlah frame sudah memenuhi minimum
- `skip`: frame diabaikan, misalnya `no_face`
- `fast_reject`: ditolak cepat karena blur, brightness, atau spoof kuat

## 3. Get Result

`GET /v1/liveness/stream/result?session_id=<id>`

Mengembalikan hasil akhir jika frame sudah cukup. Jika belum, endpoint akan mengembalikan status `waiting`.

Contoh request:

```bash
curl "http://127.0.0.1:8000/v1/liveness/stream/result?session_id=1a2b3c4d5e6f" \
  -H "X-API-Key: dev-test-key-001"
```

Contoh response saat masih menunggu:

```json
{
  "request_id": "req_123456789abc",
  "session_id": "1a2b3c4d5e6f",
  "status": "waiting",
  "frame_count": 4,
  "min_frames": 6,
  "processing_time_ms": 0
}
```

Contoh response hasil akhir:

```json
{
  "request_id": "req_123456789abc",
  "verdict": "LIVE",
  "confidence": 0.7812,
  "spoof_type": null,
  "face_detected": true,
  "quality_check": {
    "passed": true,
    "issues": []
  },
  "processing_time_ms": 42,
  "timestamp": "2026-04-22T10:30:00+00:00",
  "face_bbox": {
    "x": 104,
    "y": 66,
    "w": 170,
    "h": 170
  },
  "anti_spoof_debug": {
    "mini_fas_label": "LIVE",
    "mini_fas_confidence": 0.8451,
    "rppg_signal_confidence": 0.6853,
    "snr_db": 7.4201,
    "estimated_hr": 84.0,
    "flags": [],
    "live_score": 0.7812,
    "frames_analyzed": 6
  }
}
```

## 4. End Session

`POST /v1/liveness/stream/end`

Menghapus session dari memory store.

Contoh request:

```bash
curl -X POST http://127.0.0.1:8000/v1/liveness/stream/end \
  -H "X-API-Key: dev-test-key-001" \
  -F "session_id=1a2b3c4d5e6f"
```

Contoh response:

```json
{
  "request_id": "req_123456789abc",
  "session_id": "1a2b3c4d5e6f",
  "ended": true
}
```

## Catatan Operasional

- Session disimpan di memory dan memiliki TTL.
- Verdict final memerlukan model `models/rppg_physnet.onnx`.
- Tanpa model rPPG, endpoint `stream/result` akan gagal saat tahap inferensi.
- Endpoint ini belum memakai WebSocket; client perlu mengirim frame berkala via HTTP.
