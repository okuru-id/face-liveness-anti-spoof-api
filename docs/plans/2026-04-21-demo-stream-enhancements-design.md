# Demo Stream Enhancements Design

## Tujuan

Menambahkan tiga enhancement pada UI stream demo agar lebih stabil dan informatif:
- smoothing verdict 3-5 frame terakhir
- countdown saat stream pause karena rate limit
- mini score history di panel kanan

## Scope

Masuk scope:
- smoothing hanya di layer UI
- countdown hanya untuk status pause akibat `429`
- score history hanya untuk sesi aktif saat halaman dibuka
- tidak mengubah endpoint backend selain memanfaatkan response existing

Di luar scope:
- persistence history ke backend
- analytics chart kompleks
- state machine multi-session
- perubahan threshold backend

## Pendekatan

Fitur enhancement tetap berada di frontend `/demo` dan tidak menambah endpoint baru.

1. **Verdict smoothing**
Frontend menyimpan jendela hasil terakhir, misalnya 5 verdict. Verdict utama yang ditampilkan ke user bukan hasil frame tunggal, tetapi hasil mayoritas berbobot sederhana.

Aturan yang disarankan:
- gunakan buffer 5 hasil terakhir
- `LIVE` dan `SPOOF` punya bobot utama
- `NO_FACE` dan `POOR_QUALITY` tidak langsung menggantikan verdict stabil kecuali mendominasi berturut-turut
- jika distribusi terlalu campur, tampilkan `UNCERTAIN`

Tujuan smoothing adalah mengurangi flicker ketika hasil per frame sedikit naik turun.

2. **Rate limit countdown**
Saat backend mengembalikan `429`, frontend membaca `X-RateLimit-Reset` lalu menghitung waktu tunggu. UI menampilkan countdown detik yang terus menurun.

Behavior:
- mode badge berubah ke `Paused`
- tombol `Stop Stream` tetap aktif
- polling tidak mengirim frame sampai countdown selesai
- saat countdown selesai, stream lanjut otomatis

3. **Mini score history**
Panel kanan menampilkan riwayat pendek dari hasil stream terbaru.

Struktur yang disarankan:
- daftar 8-10 item terbaru
- setiap item menampilkan:
  - verdict
  - confidence
  - latency
  - timestamp lokal singkat
- urutan terbaru di atas

History ini bersifat ephemeral di browser, tidak disimpan ke backend.

## Interaksi UI

Panel kanan akan memiliki tiga lapisan informasi:
- status utama saat ini: mode, verdict, confidence, latency
- countdown jika sedang pause
- mini history untuk 8-10 hasil terbaru

Smoothing hanya memengaruhi badge/verdict utama. JSON debug tetap menampilkan response mentah terakhir dari backend agar tetap transparan.

## Data Flow

Untuk setiap frame sukses:
1. response mentah diterima
2. response dimasukkan ke history
3. verdict mentah dimasukkan ke smoothing buffer
4. UI menghitung verdict terhalus
5. badge utama di-update
6. history panel di-render ulang

Untuk `429`:
1. frontend baca reset header
2. hitung detik sisa
3. tampilkan countdown
4. schedule polling setelah waktu habis

## Acceptance Criteria

- badge utama tidak flicker berlebihan saat hasil frame berdekatan
- saat `429`, user melihat countdown yang jelas
- setelah countdown selesai, stream lanjut otomatis
- panel kanan menampilkan history ringkas hasil terbaru
- upload file tetap berfungsi normal
- JSON response mentah terakhir tetap tersedia
