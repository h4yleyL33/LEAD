# REDESIGN UI/UX & CSS — GALURA DATA JABAR DESIGN DIRECTION

Saya ingin melakukan redesign menyeluruh pada tampilan dashboard **Time-Series Monitoring & Anomaly Detection System** yang sedang dikembangkan.

Gunakan **Galura Data Jabar (https://galura.jabarprov.go.id/)** dan **Dashboard Jabar (https://dashboard.jabarprov.go.id/)** sebagai referensi utama visual design.

Tujuan utamanya adalah membuat dashboard terasa seperti platform data resmi yang modern, profesional, informatif, clean, dan nyaman digunakan — bukan dashboard template AI yang terlalu ramai dengan card, border, gradient, dan elemen dekoratif.

## 1. DESIGN DIRECTION

Terapkan karakter visual yang terinspirasi dari Galura Data Jabar:

- Clean, modern, minimal, dan data-centric.
- Tampilan profesional seperti portal data pemerintah modern.
- Hierarki informasi yang jelas.
- Whitespace yang cukup agar halaman tidak terasa padat.
- Komposisi layout yang rapi dan konsisten.
- Visualisasi data menjadi fokus utama.
- Typography yang mudah dibaca.
- Komponen UI sederhana tetapi tetap memiliki karakter.
- Gunakan aksen visual khas ekosistem digital Jawa Barat secara subtil.

Jangan membuat desain terlalu ramai atau terlalu futuristik.

Hindari:
- Terlalu banyak gradient.
- Glassmorphism berlebihan.
- Shadow yang terlalu tebal.
- Card kecil yang terlalu banyak.
- Border di setiap elemen.
- Jarak antarelemen yang terlalu rapat.
- Typography terlalu kecil.
- Warna yang terlalu mencolok.
- Tampilan seperti admin dashboard template generik.

## 2. COLOR SYSTEM

Buat sistem warna yang konsisten dan terinspirasi dari identitas visual portal data Jawa Barat.

Arah warna:
- Primary: biru sebagai warna identitas utama.
- Secondary: biru muda untuk aksen dan informasi pendukung.
- Background: putih atau very light cool gray.
- Surface: putih bersih.
- Text primary: dark navy atau charcoal.
- Text secondary: muted gray.
- Border: subtle light gray.

Gunakan warna status secara konsisten:
- Normal: hijau.
- Warning: amber/orange.
- Anomaly: merah.
- Informational: biru.

Pastikan warna status tidak mendominasi keseluruhan halaman.

Buat CSS variables/design tokens untuk seluruh warna agar mudah dikelola dan konsisten di semua halaman.

## 3. TYPOGRAPHY & SPACING

Gunakan font modern seperti Inter atau Plus Jakarta Sans.

Aturan typography:
- Page title harus jelas dan memiliki hierarki.
- Section heading konsisten.
- Label dan metadata tetap terbaca.
- Angka statistik harus mudah dipindai.
- Hindari terlalu banyak font weight dan ukuran berbeda.

Perbaiki spacing secara menyeluruh:
- Berikan ruang yang cukup antara page header dan konten.
- Berikan jarak konsisten antarseksi.
- Card tidak boleh terasa berdempetan.
- Isi card harus memiliki padding yang nyaman.
- Tabel harus memiliki row height yang cukup.
- Chart harus memiliki ruang yang memadai.

Gunakan sistem spacing yang konsisten, bukan margin dan padding acak di setiap komponen.

## 4. LAYOUT & NAVIGATION

Pertahankan dashboard sebagai aplikasi web desktop-first.

Buat layout yang lebih clean:
- Sidebar sederhana dan tidak terlalu dominan.
- Navigasi mudah dipahami.
- Active menu terlihat jelas.
- Header tidak terlalu tinggi.
- Konten utama menggunakan ruang secara efisien.
- Lebar konten konsisten.
- Responsive untuk tablet dan mobile.

Struktur navigasi yang harus dipertahankan:

1. Overview
2. Monitoring
3. Anomaly Events
4. Datasets
5. Reports
6. Settings

Jangan mengubah struktur navigasi tanpa alasan UX yang kuat.

## 5. OVERVIEW DASHBOARD

Halaman Overview harus terasa seperti portal visualisasi data profesional.

Prioritaskan:
- Ringkasan kondisi monitoring.
- Dataset atau equipment yang sedang dipantau.
- Total data records.
- Jumlah parameter.
- Total anomaly events.
- Status kualitas data.
- Grafik tren utama.
- Recent anomaly events.

Statistical cards harus:
- Memiliki ukuran proporsional.
- Tidak terlalu tinggi.
- Menggunakan typography yang kuat.
- Tidak memiliki dekorasi berlebihan.
- Memiliki alignment yang konsisten.

Grafik harus menjadi salah satu elemen visual utama, bukan sekadar elemen tambahan di antara banyak card.

## 6. MONITORING & DATA VISUALIZATION

Buat halaman Monitoring yang clean dan mudah digunakan untuk mengeksplorasi data time-series.

Perhatikan:
- Chart memiliki ukuran yang cukup besar.
- Filter parameter dan date range tertata rapi.
- Informasi min, max, average, dan anomaly ditampilkan secara jelas.
- Legend mudah dipahami.
- Tooltip informatif.
- Grid dan axis tidak terlalu ramai.
- Anomaly marker mudah dikenali.
- Multi-parameter comparison tetap mudah dibaca.

Jangan membuat semua informasi tampil sekaligus jika menyebabkan halaman terlalu padat.

## 7. ANOMALY EVENTS & TABLE

Gunakan gaya tabel portal data modern:
- Header tabel jelas.
- Row spacing nyaman.
- Alignment kolom konsisten.
- Status menggunakan badge sederhana.
- Informasi penting mudah dipindai.
- Hover state halus.
- Filter dan search tertata rapi.
- Pagination tidak mendominasi.

Detail anomaly harus memiliki hierarki informasi yang jelas antara:
- Waktu kejadian.
- Parameter terkait.
- Observed value.
- Anomaly score.
- Penjelasan hasil deteksi.
- Review status.

Jangan menyamakan anomaly score dengan probabilitas kerusakan alat.

## 8. DATASET UPLOAD

Halaman upload dataset harus terasa sederhana dan intuitif.

Gunakan:
- Upload area yang jelas.
- Drag-and-drop.
- File information.
- Column mapping.
- Data preview berbentuk tabel.
- Validation status.
- Error dan warning yang mudah dipahami.

Usahakan proses mapping terasa seperti mengelola spreadsheet, bukan mengisi banyak form satu per satu.

Pengguna harus bisa memahami data yang akan dianalisis sebelum menekan tombol Analyze.

## 9. RESPONSIVE & COMPONENT CONSISTENCY

Pastikan seluruh halaman menggunakan design system yang sama.

Buat konsistensi untuk:
- Button.
- Input.
- Select.
- Dropdown.
- Badge.
- Card.
- Table.
- Tabs.
- Modal.
- Tooltip.
- Chart container.
- Empty state.
- Loading state.
- Error state.

Jangan sampai setiap halaman memiliki style yang berbeda-beda.

## 10. IMPLEMENTATION RULES

Sebelum melakukan perubahan:

1. Audit struktur project dan seluruh komponen UI yang sudah ada.
2. Identifikasi stylesheet global, component styles, layout, dan design tokens.
3. Pahami framework dan library yang sedang digunakan.
4. Identifikasi komponen yang bisa digunakan kembali.

Kemudian implementasikan redesign langsung pada project.

PENTING:
- Pertahankan seluruh fungsi yang sudah berjalan.
- Jangan menghapus fitur yang sudah tersedia.
- Jangan mengganti framework atau struktur project tanpa alasan teknis yang kuat.
- Jangan mengubah logic backend, API, data processing, atau anomaly detection hanya untuk kepentingan styling.
- Jangan mengganti data asli dengan dummy data.
- Pertahankan integrasi chart dan tabel yang sudah ada.
- Hindari membuat duplikasi komponen atau CSS.
- Gunakan reusable components dan CSS variables.
- Pastikan tidak ada konflik CSS antarhalaman.

## 11. EXPECTED RESULT

Saya tidak ingin sekadar perubahan warna.

Saya ingin perubahan visual yang benar-benar terasa pada keseluruhan aplikasi:
- Lebih clean.
- Lebih spacious.
- Lebih modern.
- Lebih profesional.
- Lebih konsisten.
- Lebih mudah dibaca.
- Lebih mudah digunakan oleh engineer maupun supervisor.

Hasil akhirnya harus terasa seperti **platform data analytics modern dengan design language Galura Data Jabar**, tetapi tetap memiliki identitas sebagai sistem monitoring dan anomaly detection untuk data operasional.

Prioritaskan kualitas visual, readability, spacing, hierarchy, dan usability.

Jangan hanya memberikan rekomendasi atau contoh CSS. Implementasikan perubahan langsung pada seluruh halaman yang relevan, lalu lakukan pengecekan agar tidak ada layout yang rusak atau fitur yang hilang.