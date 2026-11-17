# Naskah AIoT LEAD dalam bahasa Indonesia

| Naskah | Source | PDF | Bukti utama |
| --- | --- | --- | --- |
| Deteksi anomali konsumsi listrik gedung | [lead-gedung.tex](lead-gedung.tex) | [lead-gedung.pdf](lead-gedung.pdf) | `results/metrics/holdout_test_final.json`, `models/xgb_final.json`, dan metadata fitur |
| Uji transfer ambang pada simulasi panas bumi | [simulasi-geothermal.tex](simulasi-geothermal.tex) | [simulasi-geothermal.pdf](simulasi-geothermal.pdf) | `results/metrics/simulation_summary_SIMULATED.json` dan `results/metrics/simulation_detection_SIMULATED.json` |

Kedua naskah memakai IEEEtran mode conference dua kolom. Nama penulis dan afiliasi belum diisi; konfirmasikan sebelum naskah dikirim. Naskah gabungan lama tersimpan di `paper/main.tex`.

## Kompilasi

Dari direktori ini, jalankan dengan distribusi TeX yang menyediakan kelas IEEEtran.

```powershell
pdflatex -halt-on-error -interaction=nonstopmode lead-gedung.tex
pdflatex -halt-on-error -interaction=nonstopmode lead-gedung.tex
pdflatex -halt-on-error -interaction=nonstopmode simulasi-geothermal.tex
pdflatex -halt-on-error -interaction=nonstopmode simulasi-geothermal.tex
```

Jika TinyTeX belum ada pada PATH di Windows, gunakan `pdflatex.exe` dari direktori instalasinya. Jalankan dua kali agar sitasi dan rujukan silang terisi.

## Catatan angka

Artefak rekayasa fitur berisi 35 kandidat. Model final memakai 34 karena `f_is_missing` tidak tercantum dalam daftar masukan model. Tabel artikel menghitung keluarga fitur dari daftar masukan model.

Eksperimen panas bumi memakai data simulasi. Ambang tetap menghasilkan nol deteksi dari 252 jam gangguan. Separator menghasilkan satu alarm pada jam normal.
