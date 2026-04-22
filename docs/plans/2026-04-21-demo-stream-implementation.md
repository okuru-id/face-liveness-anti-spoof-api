# Demo Stream Detection Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Menambahkan live detection berbasis stream ke UI demo dengan pendekatan debounced smart polling ke endpoint `POST /v1/liveness/check` yang sudah ada.

**Architecture:** Browser mengelola lifecycle kamera dan polling frame, lalu mengirim snapshot periodik ke endpoint passive existing. Tidak ada endpoint backend baru; backend tetap single-image per request dan frontend mengatur throttling, status UI, dan fallback iOS.

**Tech Stack:** FastAPI template HTML, vanilla JavaScript browser APIs (`getUserMedia`, `canvas`, `fetch`), CSS existing, existing passive liveness API.

---

### Task 1: Tambahkan state UI stream di demo page

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Tambahkan kontrol stream**

Tambahkan tombol `Start Stream` dan `Stop Stream` di panel kanan tanpa menghapus tombol upload yang sudah ada.

**Step 2: Tambahkan status display**

Tambahkan badge/status line untuk menampilkan state seperti `Idle`, `Streaming`, `Paused`, atau `Fallback`.

**Step 3: Tambahkan slot hasil terakhir**

Tampilkan verdict, confidence, dan latency terakhir secara ringkas di luar panel JSON debug.

### Task 2: Tambahkan state machine frontend

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Definisikan state runtime**

Tambahkan state JavaScript untuk:
- `isStreaming`
- `isRequestInFlight`
- `streamPollTimer`
- `lastResult`
- `currentMode`

**Step 2: Tambahkan helper start/stop stream**

Buat `startStream()` untuk membuka kamera dan `stopStream()` untuk menghentikan timer serta media tracks.

**Step 3: Pastikan mode upload dan stream saling eksklusif**

Saat stream aktif lalu user memilih upload, stream harus dihentikan terlebih dahulu.

### Task 3: Implementasikan capture frame periodik

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Tambahkan helper capture frame**

Gunakan `canvas` untuk mengambil frame terkini dari `video` dan ubah ke base64 JPEG.

**Step 2: Tambahkan helper schedule polling**

Buat `scheduleNextPoll(delay)` untuk mengatur interval adaptif.

**Step 3: Tambahkan guard request paralel**

Jika `isRequestInFlight` bernilai true, jangan kirim request baru.

### Task 4: Kirim frame stream ke endpoint existing

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Reuse request contract yang sama**

Gunakan body JSON yang sama dengan mode upload file:
```json
{
  "image": "<base64>",
  "mode": "passive"
}
```

**Step 2: Update hasil terakhir dari response**

Ambil `verdict`, `confidence`, `processing_time_ms`, dan JSON response penuh.

**Step 3: Atur adaptive delay**

Gunakan delay berbeda untuk `NO_FACE`, `POOR_QUALITY`, `LIVE`, `SPOOF`, `UNCERTAIN`, dan error.

### Task 5: Rapikan fallback iOS dan browser tanpa webcam stream

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Pertahankan fallback `input capture`**

Jika `getUserMedia` tidak tersedia atau halaman tidak secure, gunakan fallback yang sudah ada.

**Step 2: Bedakan mode fallback dengan stream sungguhan**

Tampilkan status berbeda agar user tahu browser sedang memakai fallback snapshot via picker, bukan stream live.

**Step 3: Pastikan pesan error jelas**

Jangan tampilkan error teknis mentah kalau bisa dijelaskan lebih ramah.

### Task 6: Rapikan layout demo untuk stream mode

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Pastikan preview kiri tetap stabil**

Saat stream aktif, ukuran video tidak boleh melompat-lompat.

**Step 2: Pastikan panel kanan cukup untuk status + response**

Atur spacing agar status, tombol, dan response panel tetap nyaman dibaca di desktop.

**Step 3: Pastikan layout mobile tetap satu kolom**

Jangan merusak layout mobile yang sudah ada.

### Task 7: Dokumentasikan cara memakai stream demo

**Files:**
- Modify: `README.md`

**Step 1: Tambahkan penjelasan fitur stream demo**

Jelaskan bahwa stream demo memakai snapshot berkala ke endpoint existing, bukan WebRTC backend stream.

**Step 2: Tambahkan instruksi penggunaan**

Jelaskan langkah `Start Stream`, `Stop Stream`, dan fallback iOS.

**Step 3: Tambahkan catatan beban request**

Jelaskan bahwa interval dikendalikan frontend untuk mengurangi spam request.

### Task 8: Verifikasi manual end-to-end

**Files:**
- Modify: `README.md`

**Step 1: Jalankan server lokal**

Run: `uvicorn app.main:app --reload`
Expected: halaman `/demo` terbuka normal.

**Step 2: Uji upload file tetap bekerja**

Expected: upload file tetap mengirim request manual seperti sebelumnya.

**Step 3: Uji start/stop stream di desktop**

Expected: kamera aktif, request periodik terkirim, dan tombol `Stop Stream` menghentikan proses.

**Step 4: Uji fallback iOS / browser terbatas**

Expected: tombol webcam beralih ke `input capture` tanpa error JavaScript fatal.

**Step 5: Uji non-200 response**

Expected: error tetap tampil rapi di panel kanan tanpa mematikan stream state machine secara permanen.
