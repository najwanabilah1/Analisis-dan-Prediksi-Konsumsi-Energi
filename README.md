# Analisis dan Prediksi Konsumsi Energi

Proyek demonstrasi analisis dan machine learning menggunakan dataset **Power Laws: Forecasting Energy Consumption** yang tersedia di folder `dataset/`.

> Program membaca data training, cuaca, metadata, dan hari libur langsung dari folder `dataset/`; program tidak bergantung pada `dataset_energi.csv`. Nilai target, `Value`, dilaporkan dalam satuan asli dataset; satuan fisiknya tidak diasumsikan.

## 1. Tujuan

- Memeriksa dan membersihkan data konsumsi, cuaca, metadata, dan hari libur.
- Menggabungkan konsumsi dengan cuaca berdasarkan `SiteId` dan waktu terdekat.
- Menganalisis pola konsumsi berdasarkan jam.
- Melatih Random Forest untuk memprediksi `Value` dan mengevaluasi dengan MAE, RMSE, dan R².
- Mengilustrasikan perubahan prediksi jika suhu luar dinaikkan 2°C.

## 2. Struktur folder

```text
studi-kasus-energi-gedung/
├── analisis_energi.py
├── requirements.txt
├── README.md
├── dataset/
│   ├── power-laws-forecasting-energy-consumption-training-data.csv
│   ├── power-laws-forecasting-energy-consumption-weather.csv
│   ├── power-laws-forecasting-energy-consumption-metadata.csv
│   └── power-laws-forecasting-energy-consumption-holidays.csv
├── hasil_prediksi.csv
├── feature_importance.csv
├── hasil_simulasi_skenario.csv
└── gambar/
```

Dataset test dan submission yang juga ada di folder `dataset/` tidak diperlukan oleh alur evaluasi ini.

## 3. Cara menjalankan di VS Code (Windows)

Buka folder proyek di VS Code, lalu jalankan di Terminal:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
python analisis_energi.py
```

Jika `python` tidak dikenali, gunakan `py` sebagai penggantinya. Pastikan empat file yang tercantum di struktur folder berada di `dataset/`.

Data training berukuran besar, jadi program membacanya per bagian dan mengambil sampel acak maksimum **100.000 baris** agar pelatihan model lebih terjangkau. Sampel yang sama dapat direproduksi dengan seed tetap.

## 4. Data dan preprocessing

File Power Laws menggunakan pemisah titik koma (`;`).

| File | Kolom yang digunakan | Penggunaan |
|---|---|---|
| `...training-data.csv` | `SiteId`, `Timestamp`, `Value` | Data konsumsi dan target prediksi |
| `...weather.csv` | `SiteId`, `Timestamp`, `Temperature`, `Distance` | Data cuaca |
| `...metadata.csv` | `SiteId`, `Surface`, `Sampling`, `BaseTemperature` | Informasi lokasi |
| `...holidays.csv` | `SiteId`, `Date` | Penanda hari libur |

Setiap baris konsumsi dipasangkan dengan catatan cuaca pada `SiteId` yang sama dan waktu terdekat, maksimal berjarak satu jam. Cakupan cuaca tidak tersedia untuk semua lokasi; baris yang tidak mendapat pasangan tetap dipakai. Nilai `Temperature` dan `Distance` yang hilang diisi dengan median lokasi jika tersedia, atau median global sebagai fallback, dan fitur `weather_available` menandai apakah baris memiliki pasangan cuaca langsung. Baris yang masih memiliki target atau metadata wajib yang hilang dikeluarkan; jumlahnya dicetak saat program berjalan. Fitur model meliputi lokasi, waktu kalender, cuaca, penanda ketersediaan cuaca, dan metadata. `ForecastId` tidak digunakan sebagai fitur.

`Value` digunakan sebagai target dalam satuan asli file; program tidak mengubah atau menyebutnya sebagai kWh karena satuan tersebut tidak ditetapkan di sini. Data diurutkan berdasarkan waktu: 80% waktu yang lebih awal menjadi data training, sedangkan 20% waktu berikutnya menjadi data test. Ini lebih sesuai untuk data waktu daripada pembagian acak.

## 5. Hasil dan interpretasi

Program menampilkan jumlah data yang dipakai, rentang waktu, jumlah lokasi, dan MAE/RMSE/R². Hasil metrik dapat berubah bila dataset atau preprocessing berubah.

- **MAE**: rata-rata selisih absolut prediksi dan nilai aktual.
- **RMSE**: ukuran kesalahan yang memberi penalti lebih besar pada selisih besar.
- **R²**: ukuran kecocokan prediksi terhadap variasi data; dapat bernilai negatif pada data test.
- **Feature importance**: kontribusi relatif fitur pada model, bukan bukti sebab-akibat.

Grafik yang dibuat:

- `gambar/tren_konsumsi_per_jam.png`: rata-rata `Value` per jam.
- `gambar/aktual_vs_prediksi.png`: nilai aktual dan prediksi pada periode test.
- `gambar/feature_importance.png`: kepentingan relatif fitur.
- `gambar/simulasi_penghematan.png`: perbandingan prediksi dasar dan skenario suhu.

Skenario menaikkan `Temperature` sebesar 2°C untuk setiap baris. Ini hanya menunjukkan bagaimana prediksi model berubah pada masukan tersebut; ini **bukan** estimasi dampak kausal, penghematan energi, atau jaminan bahwa perubahan suhu akan menghasilkan perubahan konsumsi yang sama di dunia nyata.

File keluaran CSV:

- `hasil_prediksi.csv`: waktu, lokasi, nilai aktual, dan prediksi pada data test.
- `feature_importance.csv`: kepentingan relatif setiap fitur.
- `hasil_simulasi_skenario.csv`: prediksi awal dan skenario suhu untuk sampel yang dipakai.

## 6. Batasan

1. Sampel maksimum 100.000 baris digunakan agar pemodelan tetap praktis; hasil bukan evaluasi atas keseluruhan dataset.
2. Data cuaca dicocokkan pada waktu terdekat dengan toleransi satu jam; lokasi tanpa kecocokan menggunakan imputasi median yang dapat mengurangi ketepatan fitur cuaca.
3. Pemisahan waktu menguji performa pada periode yang lebih kemudian, tetapi tidak menjamin model akan berlaku pada semua lokasi/periode.
4. Simulasi suhu hanya analisis sensitivitas model, bukan bukti hubungan sebab-akibat.
5. Validasi operasional memerlukan pengetahuan satuan dan konteks pengukuran dari sumber dataset, serta data gedung yang representatif.

## 7. Hubungan dengan materi kuliah

| Konsep | Implementasi |
|---|---|
| Integrasi data | Menggabungkan konsumsi, cuaca, metadata, dan hari libur |
| Pengolahan data | Memeriksa data hilang, lokasi, waktu, dan target |
| Pemodelan komputasi | Prediksi dengan Random Forest |
| Evaluasi hasil | MAE, RMSE, dan R² pada periode waktu berikutnya |
| Analisis skenario | Mengubah suhu sebagai masukan model secara ilustratif |
