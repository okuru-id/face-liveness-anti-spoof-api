# Demo Stream Detection Design

## Tujuan

Menambahkan live detection berbasis stream di UI demo tanpa menambah endpoint backend baru. Fitur ini hanya berlaku untuk halaman `/demo` dan menggunakan endpoint `POST /v1/liveness/check` yang sudah ada.

## Scope

Masuk scope:
- Stream detection hanya di UI demo browser
- Kamera aktif via `getUserMedia` jika tersedia
- Fallback ke `input capture` untuk iOS / browser tanpa webcam API
- Frame dikirim berkala ke endpoint existing
- Polling cerdas: tidak kirim request paralel dan interval adaptif
- Badge status realtime dan panel debug response tetap ada

Di luar scope:
- WebRTC / backend streaming endpoint
- Session streaming di server
- Multi-frame temporal model
- Active challenge-response

## Pendekatan

Arsitektur dipilih: `debounced smart polling`.

Browser akan:
1. Menyalakan kamera
2. Mengambil snapshot dari video element ke canvas
3. Mengubah frame menjadi base64 JPEG
4. Mengirim ke `POST /v1/liveness/check`
5. Menampilkan hasil terakhir
6. Menunggu request selesai sebelum mengambil frame berikutnya

Kenapa pendekatan ini dipilih:
- perubahan backend minimal
- reuse endpoint passive yang sudah ada
- lebih hemat request dibanding polling buta
- cukup realistis untuk demo tanpa kompleksitas WebRTC

## Flow UI

State utama:
- `Idle`: kamera belum aktif
- `Streaming`: kamera aktif dan polling berjalan
- `InFlight`: satu request sedang diproses
- `Stopped`: stream dihentikan user
- `FallbackCamera`: browser tidak mendukung webcam stream, pakai camera/file picker native

Komponen baru di panel kanan:
- tombol `Start Stream`
- tombol `Stop Stream`
- badge status realtime
- indikator mode (`Idle`, `Streaming`, `Paused`, `Fallback`)
- informasi hasil terakhir (`verdict`, `confidence`, `processing_time_ms`)

Aturan UX:
- `Upload File` mematikan stream jika aktif
- `Start Stream` menonaktifkan mode upload sementara
- `Stop Stream` menghentikan polling dan kamera
- hasil JSON terakhir tetap tampil di panel debug

## Polling Rules

Aturan request:
- interval dasar: `1200ms`
- jika request masih berjalan, frame berikutnya tidak dikirim
- hanya kirim frame jika kamera aktif dan API key terisi
- snapshot diambil dari canvas, bukan dari upload image

Aturan adaptif:
- `NO_FACE` / `POOR_QUALITY` => interval naik ke `1600ms`
- `LIVE` / `SPOOF` / `UNCERTAIN` => interval kembali `1200ms`
- error network/server => interval `2000ms`

Tujuannya agar demo terasa realtime tetapi tidak membanjiri backend.

## Error Handling

Kasus yang ditangani:
- `getUserMedia` tidak tersedia => fallback ke `input capture`
- permission kamera ditolak => tampilkan pesan jelas, tidak retry otomatis
- endpoint mengembalikan non-JSON / non-200 => tampilkan status error dan body text
- user pindah mode saat stream aktif => stop track dan timer lebih dulu

## Implementasi Teknis

File utama yang akan disentuh:
- `app/templates/demo.html`
- opsional kecil pada `app/api/routes/demo.py` jika butuh flag context tambahan

Tidak perlu mengubah kontrak `POST /v1/liveness/check`.

JavaScript yang ditambahkan:
- state machine ringan untuk mode UI
- timer polling
- guard `isRequestInFlight`
- helper `captureCurrentFrame()`
- helper `scheduleNextPoll(delay)`
- helper `updateStatusBadge(result)`
- helper `startStream()` / `stopStream()`

## Risiko

1. Terlalu banyak request
Mitigasi: debounce, adaptive interval, no parallel request.

2. iOS Safari tidak mendukung webcam stream di context tertentu
Mitigasi: fallback ke `input capture`.

3. Latency backend membuat UI terasa lambat
Mitigasi: hanya tampilkan hasil terakhir dan tahan polling sampai request selesai.

## Acceptance Criteria

- User dapat klik `Start Stream` di `/demo`
- Saat kamera aktif, frame terkirim otomatis berkala ke endpoint existing
- Tidak ada request paralel ketika request sebelumnya belum selesai
- User dapat klik `Stop Stream` dan kamera benar-benar berhenti
- `Upload File` tetap berfungsi dan mematikan stream bila aktif
- iOS/browser tanpa `getUserMedia` tetap bisa memakai fallback kamera/file picker
