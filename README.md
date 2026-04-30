# 🚀 Final Project Training AMDI - Astra Digital - Advance Data Analytics

Aplikasi berbasis web ini merupakan platform otomatisasi Machine Learning (AutoML) yang dirancang untuk menangani alur kerja data science secara end-to-end. Fokus utama pengembangan adalah pada efisiensi penggunaan memori melalui **Smart Chunking** dan transparansi model menggunakan **Explainable AI (XAI)**.

Sistem ini dirancang khusus untuk berjalan optimal di lingkungan dengan sumber daya terbatas (seperti Google Cloud Shell) tanpa mengorbankan kemampuan pemrosesan dataset skala besar.

---

## 🛠 Fitur Utama

### 1. Smart Data Chunking (Unlimited Data Handling)
Dirancang untuk mengatasi keterbatasan hardware, sistem ini mengimplementasikan pembacaan data berbasis aliran (*streaming*):
* **Memory Efficient**: Menggunakan parameter `chunksize` pada Pandas untuk memproses dataset raksasa tanpa membebani RAM secara berlebihan.
* **Non-Blocking Architecture**: Mencegah terjadinya *Out of Memory* (OOM) pada lingkungan server dengan resource terbatas.
* **Real-time Monitoring**: Menampilkan log progres pemrosesan data (jumlah baris dan chunk) langsung di terminal server.

### 2. Intelligent Multi-Model Leaderboard
Sistem melakukan komparasi otomatis terhadap berbagai algoritma untuk menemukan model terbaik bagi data Anda:
* **Ensemble Models**: Menggunakan XGBoost dan Random Forest untuk akurasi tinggi pada pola data kompleks.
* **Standard Algorithms**: Logistic/Linear Regression dan Decision Tree sebagai baseline performa.
* **Dynamic Ranking**: Menampilkan perbandingan akurasi dan metrik performa (MAPE/RMSE) dalam bentuk tabel interaktif.

### 3. Auto-Inference & Task Detection
Sistem secara cerdas mendeteksi karakteristik kolom target untuk menentukan jenis tugas Machine Learning secara otomatis:
* **Classification**: Untuk prediksi kategori (misal: Ya/Tidak, Deteksi Fraud, Kategori Risiko).
* **Regression**: Untuk prediksi nilai kontinu (misal: Estimasi Harga, Prediksi Suhu, Skor Kredit).

### 4. Explainable AI (SHAP Framework)
Menghilangkan konsep "Black Box" dalam Machine Learning dengan memberikan alasan logis di balik setiap keputusan model:
* **Global Insights**: Visualisasi SHAP Plot untuk melihat pengaruh fitur secara keseluruhan terhadap model.
* **Local Decision (Waterfall)**: Visualisasi spesifik yang menjelaskan alasan model memberikan hasil prediksi tertentu pada data individu.

---

## 🏗️ Arsitektur Teknologi

*   **Backend**: Python 3.9+ (Flask Framework)
*   **Frontend**: HTML5, Bootstrap 5 (Modern UI), Vanilla JavaScript
*   **Data Engine**: Pandas (Chunk-based processing)
*   **ML Engine**: Scikit-Learn, XGBoost, SHAP
*   **Visualization**: Chart.js (Dashboard) & Matplotlib/Seaborn (Technical Plots)

---

## 💻 Panduan Instalasi & Pengoperasian

### 1. Persiapan Repositori
Buka terminal atau command prompt, lalu jalankan:
```bash
git clone [https://github.com/dansiapa/final-project-training.git](https://github.com/dansiapa/final-project-training.git)
cd final-project-training
```

### 2. Instalasi Berdasarkan Sistem Operasi

#### 🪟 Windows
1. Buka **Command Prompt** atau **PowerShell**.
2. Buat Virtual Environment:
   ```bash
   python -m venv venv
   venv\Scripts\activate
   ```
3. Install dependensi:
   ```bash
   pip install flask pandas scikit-learn xgboost shap matplotlib seaborn
   ```

#### 🐧 Linux (Ubuntu/Debian/CentOS)
1. Buka **Terminal**.
2. Install python-venv jika belum ada:
   ```bash
   sudo apt update && sudo apt install python3-venv python3-pip
   ```
3. Buat dan aktifkan Virtual Environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. Install dependensi:
   ```bash
   pip install flask pandas scikit-learn xgboost shap matplotlib seaborn
   ```

#### 🍎 macOS (Intel & Apple Silicon)
1. Buka **Terminal**.
2. Buat dan aktifkan Virtual Environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependensi:
   ```bash
   pip install flask pandas scikit-learn xgboost shap matplotlib seaborn
   ```

### 3. Menjalankan Aplikasi
Setelah semua library terinstal, jalankan perintah berikut:
```bash
python app.py
```
Akses aplikasi melalui browser di alamat: `http://127.0.0.1:5000`

---

## 📊 Alur Kerja Pengguna

1.  **Ingestion**: Unggah file CSV melalui modul di sidebar kiri.
2.  **Configuration**: Pilih kolom target yang ingin diprediksi. Sistem akan memvalidasi tipe data secara otomatis.
3.  **Optimization**: Tekan `Start Learning`. Sistem akan melakukan chunking, training, dan mencari model terbaik.
4.  **Interpretation**: 
    *   Buka tab **Models Ranking** untuk melihat pemenang akurasi.
    *   Buka tab **Analytics** untuk melihat korelasi fitur dan SHAP global.
    *   Buka tab **Summary** untuk mendapatkan narasi analisis otomatis.
5.  **Inference**: Masukkan data baru pada tab **Prediction** untuk mendapatkan estimasi instan beserta grafik kontribusi *SHAP Waterfall*.

---
2026 © **Precision Data Analytics Suite** | Astra Digital - Advance Data Analytics
```