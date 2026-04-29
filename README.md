# 🚀 Final Project Training AMDI - Astra Digital - Advance Data Analytics

Aplikasi berbasis web untuk otomasi alur kerja Machine Learning, mulai dari penanganan data skala besar (*Large Dataset Handling*) hingga interpretasi model menggunakan **Explainable AI (XAI)**. Sistem ini dirancang untuk berjalan efisien di lingkungan dengan sumber daya terbatas menggunakan teknik **Smart Chunking**.



## 🛠 Fitur Utama

### 1. Smart Data Chunking (Unlimited Mode)
Sistem menggunakan teknik pembacaan data secara bertahap menggunakan `chunksize`. Hal ini memungkinkan aplikasi untuk:
* **Memory Efficient**: Memproses dataset yang ukurannya lebih besar dari kapasitas RAM fisik server.
* **Non-Blocking**: Mencegah *Out of Memory* (OOM) pada lingkungan seperti Google Cloud Shell.
* **Real-time Monitoring**: Menampilkan log progres pemrosesan (jumlah baris dan chunk) langsung di terminal.

### 2. Multi-Model Leaderboard
Sistem melatih dan membandingkan 6 algoritma sekaligus untuk mencari akurasi terbaik:
* **Ensemble Learning**: XGBoost & Random Forest.
* **Standard Models**: Linear/Logistic Regression & Decision Tree.
* **Instance/Kernel Models**: K-Nearest Neighbors (KNN) & Support Vector Machine (SVM).

### 3. Auto-Inference Engine
Sistem secara cerdas mendeteksi tipe masalah berdasarkan distribusi data target:
* **Regression**: Untuk prediksi nilai kontinu (misal: harga, suhu, skor).
* **Classification**: Untuk prediksi kategori (misal: ya/tidak, kategori risiko).

### 4. Explainable AI (SHAP Waterfall)
Memberikan transparansi pada model "Black Box" sehingga pengguna dapat memahami alasan di balik setiap prediksi melalui visualisasi **SHAP Waterfall Plot**.



---

## 🏗️ Arsitektur Teknologi

* **Backend**: Python Flask
* **Frontend**: HTML5, Tailwind CSS, JavaScript (Vanilla)
* **ML Libraries**: Pandas, Scikit-Learn, XGBoost, SHAP
* **Server**: RAM-Optimized with Streaming Data Reader

---

## 💻 Instalasi & Cara Menjalankan

### 1. Clone Repository
```bash
git clone [https://github.com/dansiapa/final-project-training.git](https://github.com/dansiapa/final-project-training.git)
cd final-project-training