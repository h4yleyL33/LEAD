# AIoT LEAD

Proyek deteksi anomali pada telemetri listrik per jam. Alur penelitian mencakup audit data, rekayasa fitur kausal, pelatihan XGBoost, evaluasi pada gedung terpisah, interpretasi SHAP, dan inferensi lokal melalui FastAPI. Modul panas bumi memakai sinyal **SIMULASI** untuk menguji perpindahan model dan ambang antaraset.

## Hasil utama

Himpunan uji mencakup 30 gedung, 243.684 pembacaan, dan 5.454 pembacaan berlabel anomali. Model dan ambang dipilih pada validasi, sebelum evaluasi uji.

| Metrik | Nilai |
| --- | --- |
| ROC-AUC | 0,9524 dengan selang kepercayaan 95% dari 0,9249 sampai 0,9702 |
| PR-AUC | 0,8342 dengan proporsi anomali 2,24% |
| Skor Brier dan ECE | 0,00613 dan 0,0005 |
| Recall pada FPR 1% | 0,8089 |
| Presisi dan recall pada ambang 0,1283 | 0,7987 dan 0,7888 |
| Episode terdeteksi | 441 dari 536 |
| Median latensi dan alarm palsu | 1 jam dan 0,056 episode per gedung per hari |

Hasil negatif juga dilaporkan. Fitur kalender saja menghasilkan PR-AUC 0,064. Pembobotan kelas melipatgandakan skor Brier untuk kenaikan recall episode 1,6 poin. Recall per gedung berkisar 0,02–1,00. Pada simulasi transfer panas bumi, ambang yang dikalibrasi pada gedung tidak menandai satu pun dari 252 jam gangguan yang disuntikkan. Satu alarm tercatat pada jam normal di seri separator. Sinyal panas bumi bersifat **SIMULASI**, bukan pengukuran PLTP Kamojang.

## Menjalankan proyek

Dari root repo, gunakan Python 3.12 atau lebih baru. Siapkan lingkungan dan data sesuai ketentuan dataset. Jalur ini diverifikasi dengan CPython 3.14.9 di Windows 11. Dataset publik LEAD1.0-small berukuran sekitar 58 MB. Skrip memeriksa SHA-256 arsip dan CSV.

```powershell
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
python scripts/fetch_data.py
python scripts/run_all.py
python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Alur lengkap menjalankan audit, eksplorasi data, rekayasa fitur, pemisahan gedung, model pembanding, pelatihan, evaluasi holdout, SHAP, simulasi, dan tes. Opsi `--fast` melewati grid penalaan dan ablasi berbobot. Pada POSIX, aktifkan lingkungan dengan `source .venv/bin/activate`. Tahap individual dapat dijalankan lewat modul seperti `src.data.audit`, `src.features.build_features`, dan `src.models.train_xgboost`. Urutan dan dependensinya tercatat di `scripts/run_all.py`.

### Setup dan mulai di Windows

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_web.ps1
powershell -ExecutionPolicy Bypass -File scripts/start_web.ps1
powershell -ExecutionPolicy Bypass -File scripts/smoke_web.ps1
```

Setup menyiapkan data, model, dan artefak simulasi. Server memuat `models/xgb_final.json` saat mulai, lalu membuka dashboard di `http://127.0.0.1:8000/`. Dokumentasi API tersedia di `/docs`. Uji smoke memeriksa identitas model, simulasi, dan satu jalur penilaian. Jika artefak belum tersedia, skrip start menampilkan cara menyiapkannya.

Gunakan `setup_web.ps1 -Check` untuk memeriksa artefak, `-Deliverables model` untuk menyiapkan model saja, dan `-Force` untuk membangun ulang dari awal. Pemeriksaan menulis laporan ke `results/metrics/artifact_check.json`. Untuk menjalankan server di terminal, gunakan `start_web.ps1 -Foreground`. Gunakan `start_web.ps1 -Stop` untuk menghentikannya. Log server berada di `logs/web_server.log`. Label teknis `SIMULATED` tetap dipakai pada dashboard dan payload `/simulate`.

## Struktur repo

| Lokasi | Isi |
| --- | --- |
| `MASTER_PROMPT.md`, `PROJECT_PLAN.md`, `PROGRESS.md` | Spesifikasi, rencana, dan catatan progres |
| `DECISIONS.md`, `EXPERIMENT_LOG.md`, `BLOCKERS.md` | Keputusan, rekam eksperimen, dan hambatan |
| `data/`, `research/crisp_dm/`, `notebooks/` | Asal data, proses penelitian, dan notebook |
| `src/data/`, `src/features/`, `src/models/` | Audit, pembentukan fitur, pelatihan, dan pemilihan model |
| `src/evaluation/`, `src/explainability/` | Evaluasi, metrik, dan interpretasi |
| `src/inference/`, `src/simulation/` | Inferensi dan simulasi panas bumi |
| `backend/app/`, `backend/tests/`, `tests/` | API, dashboard, dan pengujian |
| `results/` | JSON metrik, tabel, gambar, dan prediksi |
| [`paper/indonesia/`](paper/indonesia/README.md) | Dua naskah bahasa Indonesia beserta PDF dan perintah kompilasi |
| `paper/main.tex` | Naskah gabungan historis dengan format IEEEtran |

## Keterulangan dan batas penggunaan

Versi dependensi dikunci di `requirements.txt`. Tes kebocoran memeriksa bahwa fitur hingga waktu tertentu tetap sama jika riwayat sesudahnya dihapus. Tes inferensi membandingkan hasil jalur API dengan matriks fitur eksperimen pada toleransi $10^{-6}$. Tes angka naskah gabungan memeriksa nilai headline terhadap artefak JSON. Untuk naskah terpisah, sumber angka tercantum pada [indeks naskah](paper/indonesia/README.md).

Hasil gedung berlaku untuk subset publik dan skema pemisahan yang digunakan. Angka kompetisi Kaggle memerlukan data dan protokol kompetisi. Uji transfer panas bumi memakai data sintetis dan tidak mengukur performa lapangan. Dashboard ditujukan untuk demonstrasi lokal dan belum mencakup autentikasi, deteksi pergeseran data, atau konektor SCADA.

Penggunaan ditujukan untuk riset dan pendidikan. Ketentuan dataset pihak ketiga tercantum di [data/README.md](data/README.md).
