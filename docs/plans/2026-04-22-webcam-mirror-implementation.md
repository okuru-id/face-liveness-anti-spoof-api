# Webcam Mirror Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Menambahkan mirror horizontal pada preview webcam demo dan menyamakan frame capture/stream yang dikirim ke API dengan tampilan yang dilihat user.

**Architecture:** Perubahan dibatasi ke frontend `app/templates/demo.html`. Mirror diterapkan pada layer preview webcam dan pada proses capture ke `canvas`, lalu koordinat overlay disesuaikan khusus saat sumber aktif adalah webcam supaya bbox tetap akurat.

**Tech Stack:** FastAPI template HTML, vanilla JavaScript browser APIs (`getUserMedia`, `canvas`), CSS existing.

---

### Task 1: Tambahkan state mirror untuk preview webcam

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Tambahkan styling mirror pada elemen webcam**

Tambahkan aturan CSS khusus pada elemen `#webcam` atau selector webcam yang sudah ada, misalnya:

```css
#webcam.is-mirrored {
  transform: scaleX(-1);
}
```

Tujuannya agar hanya preview webcam yang tampil mirror, tanpa mengubah preview upload file.

**Step 2: Aktifkan class mirror saat webcam atau stream dimulai**

Di alur `modeWebcam` dan `startStreamBtn`, tambahkan pengaturan class:

```javascript
webcam.classList.add('is-mirrored')
```

**Step 3: Lepas class mirror saat reset view**

Di helper seperti `resetView()` atau `clearPreview()`, pastikan class dibersihkan:

```javascript
webcam.classList.remove('is-mirrored')
```

**Step 4: Verifikasi manual preview**

Run: jalankan app dan buka `/demo`
Expected: saat webcam aktif, preview tampak mirror; saat mode upload, preview file tetap normal.

### Task 2: Mirror frame yang di-capture ke canvas

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Ubah `captureCurrentFrame()` agar menggambar dengan transform mirror**

Sesuaikan isi fungsi menjadi pola seperti berikut:

```javascript
function captureCurrentFrame() {
  if (!webcam.videoWidth || !webcam.videoHeight) return null
  const ctx = webcamCanvas.getContext('2d')
  webcamCanvas.width = webcam.videoWidth
  webcamCanvas.height = webcam.videoHeight
  ctx.save()
  ctx.scale(-1, 1)
  ctx.drawImage(webcam, -webcamCanvas.width, 0, webcamCanvas.width, webcamCanvas.height)
  ctx.restore()
  return webcamCanvas.toDataURL('image/jpeg', 0.9).split(',')[1]
}
```

Gunakan transform hanya untuk sumber webcam, karena mode upload tidak memakai fungsi ini.

**Step 2: Pastikan alur capture tunggal tidak berubah**

Jangan ubah call site `captureCurrentFrame()` di tombol `Capture` dan stream loop selain memanfaatkan hasil mirror dari fungsi yang sama.

**Step 3: Verifikasi manual hasil capture**

Run: aktifkan webcam, klik `Capture`
Expected: preview statis hasil capture tetap mirror dan cocok dengan preview live sebelum capture.

**Step 4: Verifikasi manual stream payload**

Run: aktifkan stream di `/demo`
Expected: frame yang diproses backend berasal dari hasil mirror yang sama dengan tampilan live.

### Task 3: Sesuaikan overlay bbox untuk sumber webcam mirror

**Files:**
- Modify: `app/templates/demo.html`

**Step 1: Tambahkan deteksi apakah media aktif adalah webcam**

Di `drawFaceBox(faceBbox, verdict)`, setelah menentukan `activeMedia`, tambahkan flag sederhana:

```javascript
const isMirroredWebcam = webcam.style.display === 'block'
```

**Step 2: Balik koordinat X bbox saat sumber webcam aktif**

Sebelum `strokeRect`, hitung posisi X gambar yang dipakai overlay. Pola minimalnya:

```javascript
const mirroredX = sourceWidth - (drawX + drawW)
const sourceDrawX = isMirroredWebcam ? mirroredX : drawX
```

Lalu pakai `sourceDrawX` untuk koordinat overlay.

**Step 3: Pertahankan jalur existing untuk upload file**

Jangan ubah alur `preview` statis dari upload. Jika media aktif bukan webcam live, overlay tetap memakai koordinat normal.

**Step 4: Verifikasi manual overlay**

Run: aktifkan webcam dan jalankan single check atau stream
Expected: bbox wajah tetap menempel pada wajah yang tampil mirror, bukan bergeser ke sisi berlawanan.

### Task 4: Final check dan dokumentasi perubahan

**Files:**
- Modify: `app/templates/demo.html`
- Review: `docs/plans/2026-04-22-webcam-mirror-design.md`

**Step 1: Lakukan smoke check seluruh flow**

Run: buka `/demo` dan cek alur berikut:
- webcam preview mirror
- capture menghasilkan preview mirror
- stream tetap jalan
- upload file tetap normal
- overlay tetap sejajar

Expected: tidak ada regression pada mode upload dan tidak ada error JavaScript fatal di browser.

**Step 2: Tinjau ulang perubahan agar tetap minimal**

Pastikan tidak ada helper tambahan yang tidak perlu, tidak ada perubahan backend, dan semua logika mirror tetap terpusat di file template demo.

**Step 3: Commit jika repository git tersedia**

```bash
git add app/templates/demo.html docs/plans/2026-04-22-webcam-mirror-design.md docs/plans/2026-04-22-webcam-mirror-implementation.md
git commit -m "feat: mirror webcam frames in demo"
```

Jika workspace bukan git repository, lewati langkah commit.
