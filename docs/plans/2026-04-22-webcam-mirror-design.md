# Webcam Mirror Design

## Tujuan

Membuat hasil webcam di halaman `/demo` tampil mirror secara visual dan memastikan frame yang dikirim ke API juga menggunakan orientasi mirror yang sama.

## Scope

Masuk scope:
- mirror hanya berlaku untuk mode webcam di `app/templates/demo.html`
- preview live webcam tampil horizontal flip seperti kamera selfie
- frame hasil capture dan stream polling ikut di-flip sebelum dikirim ke endpoint passive
- overlay face box tetap selaras dengan tampilan mirror saat webcam aktif

Di luar scope:
- perubahan backend atau kontrak endpoint `POST /v1/liveness/check`
- perubahan perilaku upload file
- penambahan setting UI untuk toggle mirror on/off

## Pendekatan

Pendekatan yang dipilih adalah mirror di dua layer yang berbeda tetapi konsisten:

1. Elemen `video` webcam di-mirror pada layer tampilan agar user melihat preview seperti kamera depan.
2. Fungsi capture webcam ke `canvas` juga melakukan horizontal flip sebelum `drawImage`, sehingga snapshot yang dihasilkan sama dengan yang dilihat user.
3. Perhitungan overlay face box menyesuaikan koordinat X saat sumber aktif adalah webcam, agar bounding box tetap berada di posisi wajah yang benar pada preview mirror.

Pendekatan ini dipilih karena paling kecil perubahan kodenya, menjaga ekspektasi UX selfie camera, dan menghindari mismatch antara preview dengan gambar yang diproses backend.

## Data Flow

Untuk mode webcam:
1. Browser membuka kamera dengan `getUserMedia`
2. `video` ditampilkan dalam mode mirror
3. Saat capture atau stream polling, frame ditulis ke `canvas` dengan transform horizontal flip
4. Hasil base64 dari `canvas` dikirim ke endpoint existing
5. Response face bbox dipetakan ke overlay mirror agar tetap sejajar dengan preview

Untuk mode upload:
1. Alur existing tetap dipakai tanpa perubahan
2. Preview dan overlay tetap menggunakan koordinat normal

## Error Handling

- Jika webcam tidak aktif atau dimensi video belum tersedia, capture tetap mengembalikan `null` seperti sekarang
- Jika user memakai mode upload, mirror tidak ikut aktif sehingga tidak ada dampak ke alur existing
- Jika response tidak memiliki `face_bbox`, overlay tetap dibersihkan seperti perilaku sekarang

## Acceptance Criteria

- preview webcam di `/demo` tampil mirror saat kamera aktif
- tombol `Capture` menghasilkan preview statis yang juga mirror
- stream polling mengirim frame mirror ke endpoint passive
- bounding box wajah tetap pas dengan wajah saat webcam aktif
- mode upload file tetap berfungsi tanpa perubahan perilaku
