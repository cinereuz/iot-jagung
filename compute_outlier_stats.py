"""
compute_outlier_stats.py

TUJUAN:
Menghitung statistik jarak Mahalanobis global dan sub-klaster pencahayaan dari data
training (`data/manifest_fitur.csv`) untuk dipakai sebagai basis deteksi "bukan_jagung"
(novelty/outlier detection) di `src/pipeline.py` -- TANPA perlu dataset foto tambahan
kelas ke-3, sesuai arahan dosen.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans

df = pd.read_csv("data/manifest_fitur.csv")
scaler = joblib.load("models/scaler.joblib")

feature_cols = ["contrast", "correlation", "energy", "homogeneity", "hue", "saturation", "value"]

X = df[feature_cols].values
y = df["label"].values
X_scaled = scaler.transform(X)

class_stats = {}

for label in ["sehat", "terkontaminasi"]:
    X_class = X_scaled[y == label]
    mean_vec = X_class.mean(axis=0)
    cov_mat = np.cov(X_class, rowvar=False) + np.eye(X_class.shape[1]) * 1e-6
    cov_inv = np.linalg.inv(cov_mat)

    diffs = X_class - mean_vec
    distances = np.sqrt(np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs))
    thr_global = max(float(np.percentile(distances, 99.5)), 11.5)

    km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_class)
    subclusters = []
    for cid in range(3):
        sub_pts = X_class[km.labels_ == cid]
        if len(sub_pts) < 10:
            continue
        s_mean = np.mean(sub_pts, axis=0)
        s_cov = np.cov(sub_pts, rowvar=False) + np.eye(sub_pts.shape[1]) * 1e-4
        s_inv = np.linalg.inv(s_cov)
        s_diffs = sub_pts - s_mean
        s_dists = np.sqrt(np.sum((s_diffs @ s_inv) * s_diffs, axis=1))
        s_thr = float(np.percentile(s_dists, 99.5))
        subclusters.append({
            "mean": s_mean,
            "cov_inv": s_inv,
            "distance_threshold": max(s_thr, 4.5)
        })

    class_stats[label] = {
        "mean": mean_vec,
        "cov_inv": cov_inv,
        "distance_threshold": thr_global,
        "percentile": 99.5,
        "n_samples": int(len(X_class)),
        "subclusters": subclusters
    }

    print(
        f"Kelas '{label}': jarak training min={distances.min():.2f}, "
        f"max={distances.max():.2f}, threshold={thr_global:.2f}, "
        f"subclusters={len(subclusters)}, jumlah data={len(X_class)}"
    )

joblib.dump(class_stats, "models/outlier_stats.joblib")
print("\nSelesai. Statistik outlier disimpan ke models/outlier_stats.joblib")
