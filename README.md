# Analisis dan Prediksi Konsumsi Energi Menggunakan Data Analytics dan Machine Learning

## 1. Deskripsi Proyek

Proyek ini menganalisis data konsumsi energi dari dataset **Power Laws: Forecasting Energy Consumption**. Program menggabungkan data training konsumsi dengan data cuaca, metadata lokasi, dan hari libur untuk melihat pola konsumsi serta melatih model Random Forest Regressor.

Model diuji pada periode waktu yang lebih kemudian daripada data latih. Program juga membuat skenario ilustratif dengan menaikkan suhu sebesar 2°C untuk melihat perubahan prediksi model. Skenario ini bukan pengukuran penghematan energi nyata.

## 2. Latar Belakang

Data konsumsi energi dapat dianalisis bersama informasi waktu, cuaca, dan karakteristik lokasi untuk memahami pola yang muncul dalam pengamatan. Sebelum digunakan, beberapa sumber data perlu dibersihkan dan dicocokkan berdasarkan lokasi serta waktu.

Data Analytics digunakan untuk merangkum dan memvisualisasikan pola konsumsi. Machine Learning digunakan untuk mempelajari hubungan antara fitur yang tersedia dan nilai target `Value`. Hasilnya merupakan analisis pada dataset ini dan tidak otomatis berlaku untuk gedung tertentu.

## 3. Tujuan

1. Membaca dan memeriksa data konsumsi, cuaca, metadata, serta hari libur.
2. Menggabungkan data berdasarkan `SiteId` dan waktu pengamatan.
3. Menganalisis rata-rata nilai konsumsi berdasarkan jam.
4. Melatih Random Forest Regressor untuk memprediksi `Value`.
5. Mengevaluasi prediksi menggunakan MAE, RMSE, dan R² pada data test yang dipisahkan secara kronologis.
6. Melihat perubahan prediksi pada skenario suhu yang dinaikkan 2°C.

## 4. Dataset

**Nama dataset:** Power Laws: Forecasting Energy Consumption

Program menggunakan file berikut dari folder `dataset/`:

| File | Kolom yang digunakan | Fungsi |
|---|---|---|
| `power-laws-forecasting-energy-consumption-training-data.csv` | `SiteId`, `Timestamp`, `Value` | Data konsumsi dan target prediksi |
| `power-laws-forecasting-energy-consumption-weather.csv` | `SiteId`, `Timestamp`, `Temperature`, `Distance` | Data cuaca per lokasi dan waktu |
| `power-laws-forecasting-energy-consumption-metadata.csv` | `SiteId`, `Surface`, `Sampling`, `BaseTemperature` | Informasi lokasi |
| `power-laws-forecasting-energy-consumption-holidays.csv` | `SiteId`, `Date` | Informasi hari libur |

File menggunakan pemisah titik koma (`;`). Kolom `Value` dipakai sebagai target dalam satuan asli dataset. Program tidak mengasumsikan bahwa satuan `Value` adalah kWh.

Dataset training berisi **6.559.830 baris**. Agar penggunaan memori dan waktu pelatihan lebih terjangkau, program membacanya per bagian dan mengambil sampel acak maksimum **100.000 baris** dengan seed tetap. Setelah pembersihan, jumlah baris pada eksekusi yang dicatat adalah **98.667**.

File test dan submission yang juga tersedia di folder `dataset/` tidak digunakan oleh alur evaluasi program ini.

## 5. Metode dan Tahapan Analisis

### 5.1 Pengambilan dan Pembersihan Data

- Data training dibaca bertahap (*chunk*) dan dibatasi ke kolom yang diperlukan.
- Baris duplikat, data tanpa `SiteId`, waktu, atau target, serta nilai target negatif dikeluarkan.
- Data cuaca dicocokkan pada `SiteId` yang sama dan waktu terdekat, dengan toleransi maksimum satu jam.
- Pada eksekusi yang dicatat, **16.228 dari 98.667** sampel memiliki pasangan cuaca langsung.
- Baris tanpa pasangan cuaca tetap digunakan. `Temperature` dan `Distance` yang kosong diisi dengan median pada lokasi tersebut bila tersedia, atau median global. Fitur `weather_available` menandai apakah data cuaca berpasangan langsung.
- Data metadata lokasi dan hari libur ditambahkan jika tersedia.

### 5.2 Analisis Data

Program menghitung rata-rata `Value` untuk setiap jam dan menyimpan grafik tren. Program juga menyimpan grafik nilai aktual dibandingkan prediksi serta grafik *feature importance*.

### 5.3 Pemodelan dan Evaluasi

Fitur model terdiri atas `SiteId`, jam, hari dalam minggu, bulan, suhu, jarak cuaca, ketersediaan data cuaca, luas lokasi (`Surface`), interval sampling (`Sampling`), suhu dasar lokasi (`BaseTemperature`), dan penanda hari libur.

Data diurutkan berdasarkan waktu. Sekitar 80% data yang lebih awal digunakan untuk training dan periode berikutnya untuk test. Model menggunakan Random Forest Regressor dengan 200 pohon. Evaluasi dilakukan menggunakan:

- **MAE (Mean Absolute Error):** rata-rata selisih absolut antara nilai aktual dan prediksi.
- **RMSE (Root Mean Squared Error):** akar rata-rata kuadrat selisih; lebih sensitif terhadap kesalahan besar.
- **R²:** ukuran kecocokan model terhadap variasi nilai aktual; dapat bernilai negatif jika model lebih buruk daripada prediksi berdasarkan rata-rata pada data test.

### 5.4 Analisis Skenario

Untuk setiap baris pada sampel yang digunakan, program menaikkan `Temperature` sebesar 2°C lalu membandingkan prediksi model dengan prediksi awal. Ini hanya analisis sensitivitas model pada fitur suhu, bukan simulasi kausal atau estimasi penghematan nyata.

## 6. Hasil Analisis dan Evaluasi Model

Angka berikut berasal dari eksekusi program pada dataset dan konfigurasi yang tersedia saat ini. Hasil dapat berubah jika dataset atau preprocessing diubah.

### 6.1 Ringkasan Data

| Informasi | Hasil |
|---|---:|
| Baris training pada file | 6.559.830 |
| Maksimum sampel training | 100.000 |
| Sampel setelah pembersihan | 98.667 |
| Sampel dengan pasangan cuaca langsung | 16.228 |
| Baris data test | 19.734 |
| Jumlah lokasi pada sampel siap model | 267 |
| Periode data siap model | 26 Juni 2009–3 Desember 2017 |
| Periode data test | 25 Januari 2017–3 Desember 2017 |

### 6.2 Evaluasi Random Forest

| Metrik | Hasil |
|---|---:|
| MAE | 28.151,272 (satuan `Value`) |
| RMSE | 1.108.950,207 (satuan `Value`) |
| R² | -8,581 |

R² yang negatif menunjukkan bahwa model belum memprediksi variasi nilai pada periode test dengan baik. RMSE yang jauh lebih besar daripada MAE juga menunjukkan adanya kesalahan prediksi yang sangat besar pada sebagian data. Karena itu, metrik ini perlu dibaca bersama karakteristik distribusi data dan tidak dapat dianggap sebagai bukti bahwa model siap digunakan untuk prediksi operasional.

### 6.3 Analisis Skenario Suhu

| Informasi | Hasil |
|---|---:|
| Rata-rata prediksi awal | 127.140,294 |
| Rata-rata prediksi setelah suhu dinaikkan 2°C | 126.128,841 |
| Perubahan rata-rata prediksi | -1.011,453 (-0,80%) |

Perubahan tersebut adalah hasil respons model terhadap perubahan masukan suhu pada sampel ini. Nilai negatif itu **bukan** bukti bahwa kenaikan suhu menyebabkan konsumsi turun atau bahwa terjadi penghematan energi.

## 7. Struktur Proyek

```text
studi-kasus-energi-gedung/
├── dataset/
│   ├── power-laws-forecasting-energy-consumption-training-data.csv
│   ├── power-laws-forecasting-energy-consumption-weather.csv
│   ├── power-laws-forecasting-energy-consumption-metadata.csv
│   ├── power-laws-forecasting-energy-consumption-holidays.csv
│   ├── power-laws-forecasting-energy-consumption-test-data.csv
│   └── power-laws-forecasting-energy-consumption-submission-forecast-period.csv
├── gambar/
│   ├── aktual_vs_prediksi.png
│   ├── feature_importance.png
│   ├── simulasi_penghematan.png
│   └── tren_konsumsi_per_jam.png
├── analisis_energi.py
├── feature_importance.csv
├── hasil_prediksi.csv
├── hasil_simulasi_skenario.csv
├── requirements.txt
└── README.md
```

File test dan submission tercantum karena tersedia di folder dataset, tetapi tidak digunakan oleh program saat ini. File CSV hasil dan grafik dibuat atau diperbarui setiap kali program dijalankan.

## 8. Instalasi dan Cara Menjalankan

### Persyaratan

- Python 3.10 atau versi kompatibel dengan pustaka pada `requirements.txt`.
- Dataset yang diperlukan tersedia di folder `dataset/`.

### Instalasi pustaka

Buka Terminal pada folder proyek dan jalankan:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install -r requirements.txt
```

### Menjalankan program

```powershell
python analisis_energi.py
```

Program menampilkan ringkasan data dan metrik evaluasi, lalu menyimpan file hasil pada folder proyek dan grafik pada folder `gambar/`.

## 9. Output Program

| File | Keterangan |
|---|---|
| `hasil_prediksi.csv` | Waktu, `SiteId`, nilai aktual, dan prediksi untuk data test |
| `feature_importance.csv` | Kepentingan relatif fitur pada Random Forest |
| `hasil_simulasi_skenario.csv` | Prediksi awal, prediksi suhu +2°C, dan perubahannya |
| `gambar/tren_konsumsi_per_jam.png` | Grafik rata-rata `Value` berdasarkan jam |
| `gambar/aktual_vs_prediksi.png` | Grafik nilai aktual dibanding prediksi pada data test |
| `gambar/feature_importance.png` | Grafik kepentingan relatif fitur |
| `gambar/simulasi_penghematan.png` | Grafik skenario suhu; nama file dipertahankan, tetapi grafik bukan bukti penghematan |

## 10. Keterbatasan

1. Model dilatih pada sampel maksimum 100.000 baris, bukan keseluruhan data training.
2. Pencocokan cuaca langsung hanya mencakup sebagian sampel; imputasi median dapat mengurangi ketepatan fitur cuaca.
3. Nilai `Value` dilaporkan dalam satuan asli karena satuan fisiknya tidak diasumsikan oleh program.
4. R² negatif dan RMSE yang besar menunjukkan hasil prediksi pada periode test belum memadai.
5. *Feature importance* menunjukkan kepentingan relatif dalam model, bukan hubungan sebab-akibat.
6. Skenario suhu +2°C hanya mengukur perubahan keluaran model dan bukan estimasi dampak kausal atau penghematan.
7. Hasil dari dataset ini tidak otomatis mewakili konsumsi gedung tertentu atau dapat digunakan untuk keputusan operasional.

## 11. Kesimpulan

Proyek ini mengintegrasikan data konsumsi energi, cuaca, metadata lokasi, dan hari libur untuk membuat analisis pola serta model prediksi. Program berhasil menghasilkan evaluasi kronologis, file prediksi, *feature importance*, dan grafik.

Pada eksekusi yang dicatat, Random Forest memperoleh MAE 28.151,272, RMSE 1.108.950,207, dan R² -8,581. Hasil tersebut menunjukkan bahwa model belum memadai pada periode test. Pengembangan berikutnya perlu meninjau distribusi target dan data ekstrem, cakupan cuaca, serta strategi pemodelan dan evaluasi sebelum hasil digunakan untuk kebutuhan nyata.
