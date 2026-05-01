Berikut adalah format lengkap `README.md` yang telah diperbarui untuk mencakup seluruh fitur terbaru, termasuk modul **Deep Forensic Audit**, integrasi kamera, serta panduan teknis yang komprehensif:

---

# 🚀 Smart ML & Deep Forensic Auditor
###### Astra Digital Training - Advance Data Analytics & Computer Vision

Platform ini merupakan ekosistem kecerdasan buatan terintegrasi yang menggabungkan otomatisasi **Machine Learning (AutoML)** untuk data tabular dengan modul **Deep Forensic Audit** berbasis Computer Vision. Aplikasi ini dirancang untuk menangani alur kerja *data science* secara *end-to-end* sekaligus melakukan verifikasi integritas fisik dokumen melalui analisis citra tingkat lanjut.

Sistem ini dioptimalkan khusus untuk efisiensi memori (Smart Chunking) agar tetap berjalan lancar di lingkungan dengan sumber daya terbatas tanpa mengorbankan transparansi keputusan (Explainable AI).

---

## 🛠 Fitur Unggulan

### 1. 🔍 Deep Forensic Audit
Modul Computer Vision canggih untuk mendeteksi manipulasi fisik atau digital pada dokumen (seperti KTP atau NPWP):
*   **Live Capture Support**: Integrasi langsung dengan kamera perangkat menggunakan Media Devices API untuk pengambilan foto dokumen secara *real-time*.
*   **Pixel Integrity (ELA)**: Menggunakan *Error Level Analysis* untuk mendeteksi ketidakkonsistenan tingkat kompresi yang menandakan adanya manipulasi digital atau *tampering*.
*   **Media Integrity (FFT/Noise Analysis)**: Menganalisis karakteristik *grain* dan frekuensi gambar untuk mendeteksi apakah dokumen difoto langsung dari objek fisik atau hasil foto ulang dari layar (*screen reshoot*).
*   **Dynamic Verdict**: Memberikan penilaian otomatis (VALID, WARNING, atau DANGER) berdasarkan skor kepercayaan (*trust score*) yang dihitung dari berbagai metrik forensik.

### 2. 📊 Smart Data Chunking (Unlimited Data Handling)
Solusi untuk pengolahan dataset besar pada lingkungan dengan kapasitas RAM terbatas:
*   **Memory Efficient**: Menggunakan skema pemrosesan aliran (*streaming*) berbasis parameter `chunksize` pada Pandas untuk menghindari pemuatan data secara sekaligus.
*   **Non-Blocking Architecture**: Memungkinkan pemrosesan jutaan baris data secara bertahap tanpa memicu *Out of Memory* (OOM) pada server.

### 3. 🤖 Intelligent AutoML Leaderboard
Melakukan pelatihan dan komparasi otomatis terhadap berbagai algoritma untuk menemukan model terbaik:
*   **Multi-Model Training**: Melibatkan algoritma *Ensemble* (XGBoost, Random Forest) dan algoritma standar (Logistic Regression, Decision Tree).
*   **Dynamic Ranking**: Menampilkan perbandingan performa model (seperti akurasi atau MAPE) dalam bentuk tabel dan grafik interaktif.

### 4. 🧠 Explainable AI (SHAP Framework)
Menghilangkan konsep "Black Box" dengan memberikan visualisasi alasan di balik setiap prediksi AI:
*   **Global Feature Importance**: Memberikan gambaran fitur mana yang paling berpengaruh terhadap model secara keseluruhan.
*   **Local Waterfall Plot**: Menjelaskan kontribusi setiap variabel input terhadap hasil prediksi pada data individu tertentu.

---

## 🏗️ Arsitektur Teknologi

*   **Backend**: Python 3.9+ (Flask Framework)
*   **Frontend**: HTML5, Bootstrap 5, Vanilla JavaScript
*   **CV Engine**: OpenCV, Scikit-Image (Forensic Analysis)
*   **ML Engine**: Scikit-Learn, XGBoost, SHAP
*   **Data Engine**: Pandas (Chunk-based processing)

---

## 💻 Panduan Instalasi & Pengoperasian

### 1. Persiapan Repositori
```bash
git clone https://github.com/dansiapa/final-project-training.git
cd final-project-training
```

### 2. Instalasi Virtual Environment & Dependensi
Gunakan berkas `requirements.txt` yang sudah disediakan untuk memastikan semua pustaka (termasuk modul forensik) terpasang dengan benar:

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Menjalankan Aplikasi
```bash
python app.py
```
Akses aplikasi melalui peramban di: `[http://127.0.0.1:5000](http://127.0.0.1:5000)`

---

## 📊 Alur Kerja Pengguna

1.  **Analisis Data**: Unggah berkas CSV → Pilih target prediksi → Tekan "Start Learning" → Pantau ranking model dan grafik SHAP.
2.  **Audit Forensik**:
    *   Buka tab **Forensic Control**.
    *   Pilih kategori dokumen dan gunakan kamera (**Open Camera**) atau unggah foto manual.
    *   Tekan **Run Forensic Audit** untuk melihat analisis integritas citra (ELA, Noise, Sharpness).
3.  **Prediksi (Inference)**: Gunakan tab **Prediction** untuk memasukkan data baru dan melihat hasil estimasi instan beserta kontribusi variabelnya.

---

## ⚠️ Catatan Persyaratan Sistem
Untuk menjalankan modul Document Verification pada server Linux, pastikan pustaka sistem berikut telah terpasang:
```bash
sudo apt-get update && sudo apt-get install libgl1-mesa-glx libglib2.0-0
```