"""
compute_outlier_stats.py

TUJUAN:
Menghitung statistik jarak Mahalanobis dari data training (manifest_fitur.csv, 547 crop
yang sudah bersih) untuk dipakai sebagai basis deteksi "bukan jagung" (novelty/outlier
detection) di pipeline.py -- TANPA perlu dataset foto tambahan kelas ke-3, sesuai arahan
dosen.

CARA MENJALANKAN:
1. Pastikan file ini ditaruh di folder ROOT proyek (sejajar dengan folder data/ dan models/,
   BUKAN di dalam folder notebooks/) -- karena path di bawah pakai "data/..." dan "models/...".
   Kalau mau jalanin dari dalam notebooks/ (misal ditempel jadi cell baru di sebuah notebook),
   ganti "data/manifest_fitur.csv" jadi "../data/manifest_fitur.csv", dan
   "models/scaler.joblib" jadi "../models/scaler.joblib", dst.
2. Jalankan sekali lewat terminal: python compute_outlier_stats.py
   (atau copy-paste isinya jadi 1 cell notebook baru dan run di situ)
3. PENTING: pastikan manifest_fitur.csv yang dipakai di sini adalah versi yang SAMA PERSIS
   dengan data yang dipakai waktu melatih svm_model_v1.joblib dan scaler.joblib yang ada
   sekarang di folder models/. Kalau scaler/model dilatih dari data lama (103 data) sedangkan
   manifest_fitur.csv sekarang sudah 547 baris, hasilnya bisa tidak konsisten -- statistik
   outlier ini WAJIB dihitung dari data training yang sama dengan yang dipakai model aktif.
4. Hasilnya disimpan ke models/outlier_stats.joblib -- pipeline.py akan otomatis
   membacanya saat aplikasi FastAPI dijalankan ulang.
"""

import pandas as pd
import numpy as np
import joblib

# --- 1. Load manifest fitur (data training) dan scaler yang SUDAH dilatih ---
# Scaler yang dipakai di sini HARUS scaler yang sama dengan yang dipakai SVM saat prediksi,
# supaya skala fiturnya konsisten (mean=0, std=1 berdasarkan data training yang sama).
df = pd.read_csv("data/manifest_fitur.csv")
scaler = joblib.load("models/scaler.joblib")

# Urutan kolom fitur HARUS sama persis dengan urutan di pipeline.py (predict_single),
# kalau urutannya beda, hasil scaling & jarak akan salah tanpa error yang kelihatan.
feature_cols = ["contrast", "correlation", "energy", "homogeneity", "hue", "saturation", "value"]

X = df[feature_cols].values
y = df["label"].values

# --- 2. Scale fitur pakai scaler yang sama dengan yang dipakai SVM ---
# Ini WAJIB dilakukan supaya jarak Mahalanobis dihitung di ruang fitur yang sama dengan
# yang "dilihat" oleh SVM, bukan di skala mentah (mis. hue 0-179 vs energy 0-1).
X_scaled = scaler.transform(X)

# --- 3. Hitung mean vector & covariance matrix per kelas ---
class_stats = {}

for label in np.unique(y):
    # Ambil semua baris data training yang termasuk kelas ini saja
    X_class = X_scaled[y == label]

    # mean_vec: titik "pusat" kelas ini di ruang fitur (7 dimensi: contrast, correlation, dst)
    mean_vec = X_class.mean(axis=0)

    # cov_mat: menangkap bagaimana fitur-fitur ini saling berkorelasi & seberapa lebar
    # sebarannya. rowvar=False artinya tiap KOLOM adalah 1 variabel/fitur (bukan tiap baris).
    cov_mat = np.cov(X_class, rowvar=False)

    # cov_inv: invers dari covariance matrix. Dibalik di sini (bukan saat prediksi nanti)
    # supaya proses inferensi di web app lebih cepat -- tidak perlu invert matrix
    # setiap kali ada 1 foto baru masuk.
    cov_inv = np.linalg.inv(cov_mat)

    # --- 4. Hitung jarak Mahalanobis SETIAP data training ke mean kelasnya sendiri ---
    # Ini dipakai untuk menentukan threshold yang wajar: "seberapa jauh biji jagung ASLI
    # biasanya tersebar dari pusat kelasnya sendiri?"
    diffs = X_class - mean_vec
    # einsum di sini menghitung rumus jarak Mahalanobis: sqrt((x-mean) . cov_inv . (x-mean)^T)
    # untuk SEMUA baris sekaligus (lebih cepat daripada loop satu-satu).
    distances = np.sqrt(np.einsum('ij,jk,ik->i', diffs, cov_inv, diffs))

    class_stats[label] = {
        "mean": mean_vec,
        "cov_inv": cov_inv,
        # threshold = persentil ke-99 dari jarak data training sendiri.
        # Artinya: 99% biji jagung asli (kelas ini) di data training punya jarak
        # SEKECIL ATAU LEBIH KECIL dari angka ini ke pusat kelasnya. Jadi kalau ada foto
        # baru dengan jarak jauh melebihi ini, itu tanda kuat bukan biji jagung.
        "distance_threshold": float(np.percentile(distances, 99))
    }

    print(f"Kelas '{label}': jarak training min={distances.min():.2f}, "
          f"max={distances.max():.2f}, threshold(persentil 99%)={class_stats[label]['distance_threshold']:.2f}, "
          f"jumlah data={len(X_class)}")

# --- 5. Simpan hasilnya sebagai artifact baru ---
# File ini akan dibaca oleh pipeline.py setiap kali aplikasi FastAPI di-start.
joblib.dump(class_stats, "models/outlier_stats.joblib")
print("\nSelesai. Statistik outlier disimpan ke models/outlier_stats.joblib")
print("Restart aplikasi FastAPI (app.py) supaya perubahan ini terbaca oleh pipeline.")