import os
import glob
import shutil
import cv2
import time
import joblib
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nbformat
from nbclient import NotebookClient

# Import pipeline
import sys
sys.path.insert(0, os.path.abspath("."))
from src.pipeline import CornDiseasePipeline

p_instance = CornDiseasePipeline()

def process_single_raw(args):
    fpath, label = args
    fname = os.path.basename(fpath)
    img = cv2.imread(fpath)
    if img is None:
        return []

    # Auto resize & crop to standardized 224x224
    img_std = p_instance.auto_resize_and_crop(img, target_size=(224, 224))
    res = p_instance.segment_kernels_otsu(img_std)
    kernels = res["kernels"]
    if len(kernels) == 0:
        # If no separated component passed, verify if central region is valid corn
        return []

    # Extract features for all valid kernels found
    k_data = []
    for k in kernels:
        crop = k["crop"]
        area = k["area_piksel"]
        feats = p_instance.extract_features(crop)
        k_data.append({
            "kernel_id": k["kernel_id"],
            "area_piksel": area,
            "crop": crop,
            "features": feats
        })
    return (fname, label, k_data)

def main():
    t_start = time.time()
    print("=== Step 1: Discovering Raw Images ===")
    tasks = []
    for cls in ["sehat", "terkontaminasi"]:
        files = sorted(glob.glob(os.path.join("data", "raw", cls, "*")))
        for f in files:
            tasks.append((f, cls))
    print(f"Total raw images discovered: {len(tasks)}")

    print(f"=== Step 2: Parallel Segmentation with Artifact Filtering (using {os.cpu_count()} CPU cores) ===")
    results_raw = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for item in executor.map(process_single_raw, tasks, chunksize=16):
            if item:
                results_raw.append(item)

    print(f"Segmented images with valid kernels: {len(results_raw)} / {len(tasks)}")

    # Clean out old cropped and preprocessed directories
    for d in ["data/cropped/sehat", "data/cropped/terkontaminasi", "data/preprocessed/sehat", "data/preprocessed/terkontaminasi"]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)

    print("=== Step 3: Selecting Representative Kernels & Saving Crops ===")
    crop_rows = []
    fitur_rows = []

    for fname, label, k_data in results_raw:
        base_no_ext = os.path.splitext(fname)[0]
        # In multi-kernel photos, select up to 3 cleanest kernels
        if len(k_data) <= 3:
            selected_k = k_data
        else:
            # Sort by area & color saturation (representative of true batch)
            sorted_k = sorted(k_data, key=lambda x: x["features"]["saturation"], reverse=(label == "sehat"))
            selected_k = sorted_k[:3]

        for idx, k_item in enumerate(selected_k, start=1):
            crop_fname = f"{base_no_ext}_k{idx}.png"
            crop_path = os.path.join("data", "cropped", label, crop_fname)
            prep_path = os.path.join("data", "preprocessed", label, crop_fname)

            crop_img = k_item["crop"]
            cv2.imwrite(crop_path, crop_img)

            # Preprocessed 128x128 Gaussian smoothed version
            resized = cv2.resize(crop_img, (128, 128), interpolation=cv2.INTER_CUBIC)
            smoothed = cv2.GaussianBlur(resized, (3, 3), 0)
            cv2.imwrite(prep_path, smoothed)

            crop_rows.append({
                "file_asal": fname,
                "file_crop": crop_fname,
                "label": label,
                "area_piksel": int(k_item["area_piksel"])
            })

            f = k_item["features"]
            fitur_rows.append({
                "file_crop": crop_fname,
                "label": label,
                "contrast": f["contrast"],
                "correlation": f["correlation"],
                "energy": f["energy"],
                "homogeneity": f["homogeneity"],
                "hue": f["hue"],
                "saturation": f["saturation"],
                "value": f["value"]
            })

    df_crop = pd.DataFrame(crop_rows)
    df_fitur = pd.DataFrame(fitur_rows)
    df_crop.to_csv("data/manifest_crop.csv", index=False)
    df_fitur.to_csv("data/manifest_fitur.csv", index=False)

    print(f"Saved manifest_crop.csv: {len(df_crop)} rows, {df_crop['file_asal'].nunique()} unique source images")
    print(df_crop["label"].value_counts())
    print(f"Saved manifest_fitur.csv: {len(df_fitur)} rows")

    print("=== Step 4: Feature Scaling (StandardScaler) ===")
    feat_cols = ["contrast", "correlation", "energy", "homogeneity", "hue", "saturation", "value"]
    X = df_fitur[feat_cols].values
    y = df_fitur["label"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    print("Saved models/scaler.joblib")

    print("=== Step 5: Train/Test Split & Hyperparameter Tuning (GridSearchCV) ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.20, random_state=42, stratify=y
    )

    param_grid = {
        "C": [1, 10, 50, 100],
        "gamma": ["scale", 0.05, 0.1, 0.5, 1.0],
        "kernel": ["rbf"]
    }
    grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
    grid.fit(X_train, y_train)

    best_svm = grid.best_estimator_
    y_pred = best_svm.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    print(f"Best Params: {grid.best_params_}")
    print(f"Test Accuracy: {test_acc*100:.2f}%")
    print(classification_report(y_test, y_pred, digits=4))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    joblib.dump(best_svm, "models/svm_model_v1.joblib")
    print("Saved tuned model to models/svm_model_v1.joblib")

    print("=== Step 6: Multi-Subcluster Novelty Detection Stats (outlier_stats.joblib) ===")
    outlier_stats = {}
    for label in ["sehat", "terkontaminasi"]:
        X_cls = X_scaled[y == label]
        mean_vec = X_cls.mean(axis=0)
        cov_mat = np.cov(X_cls, rowvar=False) + np.eye(X_cls.shape[1]) * 1e-6
        cov_inv = np.linalg.inv(cov_mat)

        diffs = X_cls - mean_vec
        dists = np.sqrt(np.sum((diffs @ cov_inv) * diffs, axis=1))
        thr_global = max(float(np.percentile(dists, 99.5)), 11.5)

        km = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X_cls)
        subclusters = []
        for cid in range(3):
            sub_pts = X_cls[km.labels_ == cid]
            if len(sub_pts) < 10:
                continue
            s_mean = sub_pts.mean(axis=0)
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

        outlier_stats[label] = {
            "mean": mean_vec,
            "cov_inv": cov_inv,
            "distance_threshold": thr_global,
            "percentile": 99.5,
            "n_samples": int(len(X_cls)),
            "subclusters": subclusters
        }
        print(f"Class '{label}': n={len(X_cls)}, threshold={thr_global:.2f}, subclusters={len(subclusters)}")

    joblib.dump(outlier_stats, "models/outlier_stats.joblib")
    print("Saved models/outlier_stats.joblib")

    print(f"Dataset processing and model retraining completed in {time.time()-t_start:.1f}s!")

if __name__ == "__main__":
    main()
