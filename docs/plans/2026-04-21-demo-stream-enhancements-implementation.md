# Demo Stream Enhancements Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Menambahkan smoothing verdict, countdown rate limit, dan mini history pada UI stream demo tanpa mengubah arsitektur backend utama.

**Architecture:** Semua enhancement dilakukan di frontend template `/demo` menggunakan state JavaScript lokal. Backend tetap menyediakan response per-frame, sementara frontend menggabungkan hasil menjadi presentasi UI yang lebih stabil dan lebih mudah dibaca.

**Tech Stack:** FastAPI template HTML, vanilla JavaScript, existing passive liveness API.

---

### Task 1: Tambahkan state smoothing dan history

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Tambahkan buffer verdict**

Tambahkan array untuk menyimpan 5 verdict terakhir yang relevan.

**Step 2: Tambahkan history store**

Tambahkan array untuk menyimpan 8-10 hasil terakhir.

**Step 3: Tambahkan helper update state**

Buat helper untuk menambah item ke buffer dan menjaga panjang maksimum.

### Task 2: Implementasikan verdict smoothing

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Buat helper hitung verdict halus**

Aturan minimal:
- ambil 5 hasil terakhir
- jika mayoritas `LIVE`, tampilkan `LIVE`
- jika mayoritas `SPOOF`, tampilkan `SPOOF`
- jika campur, tampilkan `UNCERTAIN`

**Step 2: Pisahkan verdict mentah vs verdict halus**

Verdict mentah tetap masuk ke JSON debug; verdict halus dipakai untuk badge utama.

**Step 3: Update UI badge dengan verdict halus**

Pastikan warna badge mengikuti hasil smoothing, bukan hanya hasil frame terakhir.

### Task 3: Tambahkan mini history panel

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Tambahkan container history di panel kanan**

Sediakan area ringkas untuk hasil terbaru.

**Step 2: Render item history**

Tiap item menampilkan verdict, confidence, latency, dan waktu singkat.

**Step 3: Batasi panjang history**

Maksimal 8-10 item agar panel tetap ringkas.

### Task 4: Tambahkan countdown saat rate limit

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Tambahkan state countdown**

Simpan `rateLimitCountdownSeconds` dan timer interval untuk UI.

**Step 2: Render countdown di panel kanan**

Tampilkan teks seperti `Retry in 27s` saat stream pause.

**Step 3: Sinkronkan countdown dengan `X-RateLimit-Reset`**

Jika header ada, hitung waktu reset; jika tidak, fallback ke jeda tetap.

### Task 5: Integrasikan countdown dengan stream loop

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Saat `429`, set mode pause**

Jangan kirim request baru selama countdown aktif.

**Step 2: Restart polling setelah countdown selesai**

Pastikan stream lanjut otomatis tanpa harus klik `Start Stream` lagi.

**Step 3: Bersihkan countdown saat stream dihentikan manual**

Agar tidak ada timer yatim.

### Task 6: Rapikan visual history dan countdown

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Tambahkan card kecil untuk countdown**

Buat tampilannya konsisten dengan panel kanan existing.

**Step 2: Tambahkan list visual history**

Gunakan elemen sederhana, tidak berat, dan tetap readable.

**Step 3: Pastikan responsif di mobile**

History dan countdown tetap nyaman dibaca di satu kolom.

### Task 7: Dokumentasikan enhancement stream

**Files:**
- Modify: `README.md`

**Step 1: Tambahkan smoothing behavior**

Jelaskan bahwa badge utama memakai smoothing, sementara JSON debug tetap mentah.

**Step 2: Tambahkan rate limit pause behavior**

Jelaskan bahwa stream akan pause otomatis saat `429`.

**Step 3: Tambahkan mini history explanation**

Jelaskan bahwa history hanya tersimpan di browser selama sesi aktif.

### Task 8: Verifikasi manual

**Files:**
- Modify: `README.md`

**Step 1: Jalankan server lokal**

Run: `uvicorn app.main:app --reload`
Expected: `/demo` terbuka normal.

**Step 2: Verifikasi smoothing**

Expected: badge utama tidak berubah liar frame ke frame.

**Step 3: Verifikasi countdown rate limit**

Expected: saat `429`, UI menampilkan countdown dan polling pause.

**Step 4: Verifikasi history**

Expected: panel kanan menampilkan daftar hasil terbaru.
