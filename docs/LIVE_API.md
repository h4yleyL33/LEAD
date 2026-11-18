# Integrasi data realtime AIoT

Endpoint penerima sudah tersedia. Pengirim mengubah data dari API/sensor ke format JSON berikut, lalu mengirimnya ke backend AIoT. Pengambilan dari REST API pihak ketiga kini dapat diatur lewat menu **Koneksi API**; lihat [panduan koneksi](KONEKSI_API.md).

## Format kiriman

`POST http://127.0.0.1:8000/live/ingest`

Header: `Content-Type: application/json`

```json
{
  "records": [
    {
      "building_id": 101,
      "timestamp": "2026-11-18T00:00:00",
      "meter_reading": 125.4
    },
    {
      "building_id": 101,
      "timestamp": "2026-11-18T01:00:00",
      "meter_reading": null
    }
  ]
}
```

| Field | Tipe | Ketentuan integrasi |
|---|---|---|
| `records` | array | Minimal satu pembacaan; satu kiriman dapat berisi beberapa aset dan jam. |
| `building_id` | integer | ID aset yang stabil. Petakan ID teks dari sumber ke ID numerik yang konsisten. |
| `timestamp` | string | ISO 8601. Gunakan UTC tanpa akhiran zona, seperti contoh di atas; satu titik per jam pada menit `00`. Konversi WIB ke UTC sebelum mengirim. |
| `meter_reading` | number atau null | Energi interval satu jam dalam kWh. `null` menandai reading hilang/OUTAGE, bukan nol. |

Gunakan nama field persis seperti contoh. `meter_reading` dapat dihilangkan menurut kontrak API saat ini dan akan dianggap `null`; pengirim sebaiknya tetap menuliskannya. Angka desimal memakai titik. Jangan kirim `NaN` atau `Infinity`.

Model yang tersedia dilatih untuk energi gedung per jam. Bila API sumber memberi watt/kW setiap detik, hitung energi per jam terlebih dahulu. Bila sumber memberi meter kWh kumulatif, ambil selisih antarbatas jam dan tangani reset meter. Temperatur/getaran bukan input langsung yang setara untuk model ini.

File siap dipakai: [contoh payload](live-ingest.example.json) dan [JSON Schema dari model validasi backend](live-ingest.schema.json). Schema mencerminkan validasi tipe API saat ini; ketentuan UTC, interval jam, dan satuan di atas tetap menjadi tanggung jawab pengirim.

## Mengirim dari PowerShell

Jalankan dari root proyek setelah dashboard/server aktif:

```powershell
$payload = Get-Content -LiteralPath docs/live-ingest.example.json -Raw
Invoke-RestMethod -Method Post `
  -Uri 'http://127.0.0.1:8000/live/ingest' `
  -ContentType 'application/json' `
  -Body $payload
```

Contoh menggunakan curl:

```powershell
curl.exe -X POST http://127.0.0.1:8000/live/ingest -H "Content-Type: application/json" --data-binary "@docs/live-ingest.example.json"
```

Contoh tersebut memasukkan dua reading contoh ke buffer. Untuk data nyata, ganti ID, timestamp, dan nilai sebelum dikirim.

## Respons dan pembacaan dashboard

Respons ingest memuat `accepted`, `rejected`, `buildings`, `n_buildings_tracked`, dan `buffer`. `accepted` berarti reading masuk buffer, bukan sudah discor. Periksa `rejected` juga: timestamp yang tidak valid bisa ditolak meskipun HTTP-nya `200`. Payload yang gagal validasi tipe/struktur mendapat `422`.

Alur aktual:

```text
API eksternal / sensor
  -> pengirim/adapter memetakan field dan mengonversi satuan/waktu
  -> POST /live/ingest
  -> buffer memori backend
  -> GET /live/{building_id}?since=...&limit=600
  -> model menghitung skor
  -> dashboard memperbarui grafik dan alarm
```

Buka Live Tail, muat ulang status setelah reading pertama masuk, pilih aset, lalu tekan Mulai. Dashboard melakukan polling setiap 3 detik secara bawaan; interval UI dapat diubah antara 1 dan 30 detik. Ini frekuensi pemeriksaan pembaruan, sedangkan resolusi data model tetap per jam. Saat polling dihentikan atau halaman ditinggalkan, pengirim tetap bisa mengirim ke backend selama server aktif.

- Status buffer: `GET /live/status`.
- Snapshot berskor: `GET /live/101?limit=500`.
- Pembaruan berikutnya: `GET /live/101?since=2026-11-18T01%3A00%3A00&limit=500`. Batas `since` inklusif; abaikan titik batas yang sudah ditampilkan. Ini bukan jaminan pengiriman koreksi untuk data lama.
- Aset yang belum punya data mendapat `404` saat snapshot diminta.
- Dokumentasi interaktif dan kontrak lengkap: `http://127.0.0.1:8000/docs` dan `/openapi.json`.

## Ketentuan pengiriman

Kirim data berurutan menurut waktu. Kiriman awal sebaiknya memuat riwayat asli dari aset yang sama, lalu kirim reading jam berikutnya. Fitur model memakai jendela sampai 720 jam; hasil dengan history pendek tidak setara dengan hasil saat history lengkap. Tombol Seed dari held-out memuat data referensi historis untuk demo, bukan data sensor nyata.

Untuk pasangan aset+timestamp yang sama, backend menggunakan kiriman terakhir saat membuat snapshot. Karena ingest tidak memiliki ID transaksi, kiriman ulang masih menambah entri buffer; hindari retry tanpa batas dan koreksi data lama melalui polling biasa. Buffer saat ini berada di RAM dan hilang saat server restart; belum merupakan penyimpanan historis permanen atau antrean pengiriman andal.

Server launcher saat ini hanya menerima koneksi di `127.0.0.1`, sehingga pengirim harus berjalan pada komputer yang sama. API dari internet dapat dibaca oleh adapter lokal lalu diteruskan ke endpoint ini. Pengirim dari perangkat/komputer lain memerlukan konfigurasi jaringan serta autentikasi tersendiri; contoh ini tidak membuka server ke jaringan. Untuk mengambil data API eksternal dari backend lokal, gunakan menu Koneksi API.
