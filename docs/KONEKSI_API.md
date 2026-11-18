# Koneksi API IoT

Menu **Koneksi API** mengambil data dari satu sumber REST API memakai HTTP GET. Pengambilan dijalankan oleh server, sehingga tetap berjalan saat halaman/tab dashboard ditutup. WebSocket, MQTT, dan API dengan body POST/OAuth refresh belum termasuk.

## Menyambungkan sumber

1. Buka dashboard, lalu pilih **Koneksi API** di sidebar.
2. Masukkan nama, URL sumber JSON, dan interval pengambilan 5–3600 detik.
3. Untuk perangkat pada LAN/komputer yang sama, aktifkan **Izinkan perangkat di jaringan lokal**. API internet harus HTTPS. Token selalu memerlukan HTTPS; sertifikat TLS tetap diverifikasi.
4. Pilih tanpa autentikasi, Bearer token, atau API key di header; sesuaikan nama header bila diperlukan. Jangan taruh token di URL. Token hanya di RAM server dan tidak ditampilkan kembali. Setelah server restart, isi token lagi.
5. Sesuaikan pemetaan field. Gunakan titik untuk objek bertingkat (`data.readings`, `device.id`). Kosongkan path data bila array/objek berada di root. Kosongkan field ID untuk memakai ID aset tetap.
6. Klik **Uji koneksi**. Lima hasil pemetaan pertama ditampilkan; pengujian ini tidak memasukkan data ke buffer.
7. Klik **Simpan & mulai**. Status menampilkan waktu keberhasilan terakhir, jumlah data pada respons, data baru, dan error bila ada. Klik **Hentikan** sebelum mengedit konfigurasi yang sedang berjalan.
8. Buka **Live Tail**, pilih aset, lalu klik **Mulai** untuk menampilkan grafik. Pengambilan API tetap berjalan terpisah dari polling grafik.

Konfigurasi tanpa token tersimpan di `data/integrations/source.json` dan dikecualikan dari Git. Setelah server restart konfigurasi dimuat dalam keadaan berhenti; jalankan lagi secara eksplisit.

## Contoh pemetaan

Respons API sumber:

```json
{
  "data": {
    "readings": [
      {"device_id": 101, "time": "2026-11-18T07:00:00+07:00", "energy_kwh": 125.4}
    ]
  }
}
```

| Isian menu | Nilai |
|---|---|
| Path data JSON | `data.readings` |
| Field ID aset | `device_id` |
| Field waktu | `time` |
| Format waktu | ISO 8601 |
| Field energi per jam | `energy_kwh` |

Konektor mengubah waktu contoh menjadi `2026-11-18T00:00:00` UTC dan meneruskan reading ke live hub yang sama dengan `POST /live/ingest`. Format Unix detik/milidetik juga tersedia. Waktu ISO tanpa zona dianggap UTC.

Data harus berisi energi per jam dalam kWh pada menit 00, ID aset integer nonnegatif, dan nilai berhingga nonnegatif atau `null` (OUTAGE). Angka berbentuk string diterima. Nama ID berbentuk teks perlu dipetakan ke ID angka oleh penyedia/penghubung; atau gunakan ID tetap untuk satu aset. Daya instan, nilai kumulatif meter, suhu, dan getaran tidak otomatis dikonversi. Pembacaan sub-jam ditolak agar tidak salah dianggap sebagai energi per jam.

Respons sumber boleh berupa satu objek atau array objek di path yang dipilih. Batasnya 10000 baris dan 2 MB per respons. Respons bukan JSON, field yang hilang, atau baris tidak valid menggagalkan seluruh pengambilan itu; data lama tetap ada. Status HTTP sumber selain 200, redirect, atau kegagalan jaringan ditampilkan tanpa membocorkan isi respons/kredensial. Konektor mencoba kembali pada interval berikutnya.

Data identik pada aset+timestamp yang sama tidak dikirim ulang selama sesi koneksi. Pemetaan yang disimpan ulang atau restart server memulai sesi baru. Respons sumber harus menyertakan riwayat yang diperlukan model dan reading terbaru; konektor tidak mengikuti pagination atau mengambil riwayat tersembunyi secara otomatis. Koreksi data lama memerlukan pemuatan ulang grafik; `since` di Live Tail terutama ditujukan untuk data baru.

Fitur model menggunakan jendela sampai 720 jam. API sumber sebaiknya menyediakan riwayat awal dari aset yang sama. Buffer live menyimpan maksimal 2200 entri terakhir per aset, hanya dalam RAM, dan hilang saat server berhenti. Ini belum penyimpanan historis permanen. Interval mengambil API (mis. 30 detik) dan interval grafik (bawaan 3 detik) tidak mengubah resolusi model per jam.

## Uji tanpa API eksternal

Jalankan dari root proyek di terminal:

```powershell
.\.venv\Scripts\python.exe scripts/demo_iot_api.py
```

Script menjalankan server lokal sampai dihentikan dengan Ctrl+C. Semua pembacaannya **SIMULATED**; tidak mewakili sensor nyata. Keluaran berisi 96 jam data; tidak membuat jam baru setiap beberapa detik.

Isian koneksi:

- Nama: `SIMULATED - Uji lokal`
- URL: `http://127.0.0.1:8765/readings`
- Izinkan perangkat di jaringan lokal: aktif.
- Sumber data simulasi: aktif.
- Autentikasi: tanpa autentikasi.
- Path: `data.readings`.
- Field ID: `device_id`.
- Field waktu: `time`; format ISO 8601.
- Field energi: `energy_kwh`.

Uji koneksi lalu mulai. Data contoh hanya digunakan bila script dan koneksi dijalankan secara eksplisit; aplikasi tidak mengaktifkannya otomatis.

## Endpoint konektor

| Method dan path | Fungsi |
|---|---|
| `GET /connection` | Konfigurasi tanpa token serta status pengambilan |
| `PUT /connection` | Simpan konfigurasi saat berhenti |
| `POST /connection/test` | Uji konfigurasi dalam body tanpa ingest |
| `POST /connection/start` | Mulai worker pengambilan |
| `POST /connection/stop` | Hentikan worker |
| `POST /connection/poll` | Ambil sekali menggunakan konfigurasi tersimpan |

Kontrak lengkap tersedia di `/docs` dan `/openapi.json`. Autentikasi sumber adalah untuk permintaan keluar, bukan autentikasi dashboard. Server dashboard tetap terikat pada localhost, satu proses; jangan membukanya langsung ke internet. Pemeriksaan URL/DNS dan izin LAN bukan pengganti autentikasi dashboard atau isolasi jaringan untuk deployment publik.
