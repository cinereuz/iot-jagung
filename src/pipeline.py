"""
Corn Kernel Mold & Disease Detection Pipeline
Based on: 'Detection of Corn Leaf Blight Disease Based on GLCM and HSV Feature Extraction Using Support Vector Machine'
Adapted for corn kernel mold/fungus detection.

UPDATE: ditambah deteksi "bukan_jagung" pakai pendekatan negasi (novelty/outlier detection
berbasis jarak Mahalanobis terhadap data training yang sudah ada) -- tanpa perlu dataset
foto tambahan untuk kelas ketiga, sesuai arahan dosen.
"""

import os
import cv2
import numpy as np
import pandas as pd
import joblib
from skimage.feature import graycomatrix, graycoprops

class CornDiseasePipeline:
    def __init__(self, model_path="models/svm_model_v1.joblib", scaler_path="models/scaler.joblib",
                 outlier_stats_path="models/outlier_stats.joblib", outlier_margin=1.15):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.outlier_stats_path = outlier_stats_path
        # outlier_margin: kelonggaran tambahan di atas threshold hasil training (15% lebih
        # longgar). Ini penting supaya biji jagung asli yang kebetulan agak "ekstrem" (mis.
        # dari batch foto overexposed) tidak gampang salah ditolak jadi "bukan jagung".
        # Kalau nanti false negative (biji jagung asli malah dibilang bukan jagung) masih
        # sering terjadi, naikkan angka ini (mis. 1.3). Kalau sebaliknya (benda lain malah
        # lolos dianggap jagung), turunkan (mis. 1.05).
        self.outlier_margin = outlier_margin
        self.model = None
        self.scaler = None
        self.outlier_stats = None  # dict: {"sehat": {...}, "terkontaminasi": {...}}
        self.feature_names = ["contrast", "correlation", "energy", "homogeneity", "hue", "saturation", "value"]
        self.load_models()

    def load_models(self):
        """Loads trained SVM model, StandardScaler, dan statistik outlier (untuk deteksi 'bukan jagung')."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            return False

        # Statistik outlier bersifat opsional: kalau belum dihitung (belum jalanin
        # compute_outlier_stats.py), pipeline tetap jalan seperti sebelumnya (tanpa deteksi
        # "bukan jagung"), supaya tidak merusak fungsi yang sudah ada.
        if os.path.exists(self.outlier_stats_path):
            self.outlier_stats = joblib.load(self.outlier_stats_path)
        else:
            self.outlier_stats = None
            print(f"[Peringatan] '{self.outlier_stats_path}' tidak ditemukan -- deteksi "
                  f"'bukan jagung' dinonaktifkan, model hanya memprediksi sehat/terkontaminasi. "
                  f"Jalankan compute_outlier_stats.py untuk mengaktifkannya.")

        return True

    def preprocess_image(self, img, target_size=(128, 128)):
        """
        Step 1: Preprocessing
        - Grayscale conversion
        - Noise reduction using Gaussian Blur
        - Resizing to standardized target size
        """
        if img is None:
            raise ValueError("Citra input tidak valid (None)")
        
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Standardized resized version for feature extraction
        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        resized_blur = cv2.GaussianBlur(resized_gray, (3, 3), 0)

        return {
            "original": img,
            "gray": gray,
            "blur": blur,
            "resized_img": resized_img,
            "resized_gray": resized_gray,
            "resized_blur": resized_blur
        }

    def segment_kernels_otsu(self, img, min_area=100, max_area=50000, margin=4):
        """
        Step 2A: Object Isolation using Otsu Thresholding and Morphological Filtering.
        Detects individual corn kernels in the macro image.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Otsu thresholding
        _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological opening with elliptical structuring element to detach touching kernels
        kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_morph, iterations=2)

        # Contours detection
        contours, hierarchy = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_kernels = []
        h_img, w_img = img.shape[:2]

        for idx, c in enumerate(contours):
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, w_img)
            y2 = min(y + h + margin, h_img)

            crop = img[y1:y2, x1:x2]
            mask_crop = opened[y1:y2, x1:x2]

            valid_kernels.append({
                "kernel_id": idx + 1,
                "bbox": (x1, y1, x2 - x1, y2 - y1),
                "area_piksel": area,
                "crop": crop,
                "mask": mask_crop,
                "contour": c
            })

        return {
            "otsu_mask": opened,
            "kernels": valid_kernels
        }

    def segment_fungus_kmeans(self, kernel_crop, k=3):
        """
        Step 2B: Lesion / Mold Disease Area Isolation using K-Means Clustering.
        Applies K-Means clustering on the color space of the kernel to separate
        healthy golden/yellow kernel tissue from discolored fungal spots / mycelium.
        """
        if kernel_crop is None or kernel_crop.size == 0:
            return None

        # Convert to HSV color space for color-texture differentiation
        hsv_crop = cv2.cvtColor(kernel_crop, cv2.COLOR_BGR2HSV)
        
        # Exclude pure black background pixels
        gray_crop = cv2.cvtColor(kernel_crop, cv2.COLOR_BGR2GRAY)
        _, fg_mask = cv2.threshold(gray_crop, 15, 255, cv2.THRESH_BINARY)
        
        # Flatten foreground pixels for K-Means
        fg_indices = np.where(fg_mask > 0)
        if len(fg_indices[0]) < 10:
            return {
                "clustered_bgr": kernel_crop,
                "mold_mask": np.zeros(kernel_crop.shape[:2], dtype=np.uint8),
                "mold_ratio": 0.0,
                "cluster_centers": []
            }

        pixels = hsv_crop[fg_indices]
        pixels = np.float32(pixels)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        compactness, labels, centers = cv2.kmeans(pixels, k, None, criteria, 10, flags)

        centers = np.uint8(centers)
        
        # Reconstruct clustered image
        clustered_hsv = np.zeros_like(hsv_crop)
        clustered_labels = np.zeros(kernel_crop.shape[:2], dtype=np.int32) - 1
        
        for idx in range(len(labels)):
            py = fg_indices[0][idx]
            px = fg_indices[1][idx]
            cluster_id = labels[idx][0]
            clustered_hsv[py, px] = centers[cluster_id]
            clustered_labels[py, px] = cluster_id

        clustered_bgr = cv2.cvtColor(clustered_hsv, cv2.COLOR_HSV2BGR)

        # Identify mold cluster: lowest health score (low saturation, whitish/grayish, or dark necrosis)
        mold_cluster_id = None
        min_health_score = 999999.0
        
        for cid in range(k):
            h_c, s_c, v_c = centers[cid]
            health_score = float(s_c) * 1.5 + float(v_c) * 0.8
            if health_score < min_health_score:
                min_health_score = health_score
                mold_cluster_id = cid

        mold_mask = np.zeros(kernel_crop.shape[:2], dtype=np.uint8)
        if mold_cluster_id is not None:
            mold_mask[clustered_labels == mold_cluster_id] = 255

        total_fg_pixels = len(fg_indices[0])
        mold_pixels = np.sum(mold_mask > 0)
        mold_ratio = float(mold_pixels) / float(total_fg_pixels) if total_fg_pixels > 0 else 0.0

        return {
            "clustered_bgr": clustered_bgr,
            "mold_mask": mold_mask,
            "mold_ratio": mold_ratio,
            "cluster_centers": centers.tolist()
        }

    def extract_features(self, kernel_img):
        """
        Step 3: Feature Extraction (GLCM + HSV)
        - 4 GLCM Texture Features: Contrast, Correlation, Energy, Homogeneity
          (calculated using graycomatrix across angles [0, 45, 90, 135 deg], distance 1)
        - 3 HSV Color Features: Mean Hue, Mean Saturation, Mean Value
        """
        resized = cv2.resize(kernel_img, (128, 128), interpolation=cv2.INTER_CUBIC)
        smoothed = cv2.GaussianBlur(resized, (3, 3), 0)
        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

        # GLCM Texture Features
        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True
        )

        contrast = float(graycoprops(glcm, 'contrast').mean())
        correlation = float(graycoprops(glcm, 'correlation').mean())
        energy = float(graycoprops(glcm, 'energy').mean())
        homogeneity = float(graycoprops(glcm, 'homogeneity').mean())

        # HSV Color Features
        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        fg_mask = gray > 10
        if np.sum(fg_mask) > 0:
            h_mean = float(hsv[:, :, 0][fg_mask].mean())
            s_mean = float(hsv[:, :, 1][fg_mask].mean())
            v_mean = float(hsv[:, :, 2][fg_mask].mean())
        else:
            h_mean = float(hsv[:, :, 0].mean())
            s_mean = float(hsv[:, :, 1].mean())
            v_mean = float(hsv[:, :, 2].mean())

        return {
            "contrast": round(contrast, 4),
            "correlation": round(correlation, 4),
            "energy": round(energy, 4),
            "homogeneity": round(homogeneity, 4),
            "hue": round(h_mean, 2),
            "saturation": round(s_mean, 2),
            "value": round(v_mean, 2)
        }

    def _mahalanobis_distance(self, x_scaled, mean_vec, cov_inv):
        """
        Menghitung jarak Mahalanobis dari 1 titik data (x_scaled) ke sebuah titik pusat
        (mean_vec), dengan mempertimbangkan bentuk sebaran data training (cov_inv) --
        BUKAN jarak lurus biasa (Euclidean).

        Kenapa bukan jarak lurus biasa? Karena beberapa fitur (mis. contrast vs hue) bisa
        punya sebaran/korelasi yang berbeda-beda. Mahalanobis distance otomatis "menormalkan"
        itu, jadi 1 satuan jarak di fitur yang sebarannya sempit dihitung lebih signifikan
        daripada 1 satuan jarak di fitur yang sebarannya lebar.
        """
        diff = x_scaled - mean_vec
        # Rumus: sqrt( (x - mean) . cov_inv . (x - mean)^T )
        distance = np.sqrt(diff @ cov_inv @ diff.T)
        return float(distance)

    def check_is_corn(self, X_scaled_row):
        """
        Step 3.5 (BARU): Novelty/Outlier Detection -- deteksi "bukan jagung" TANPA perlu
        dataset foto kelas ketiga (pendekatan negasi, sesuai arahan dosen).

        Logikanya: dari 547 data training (sehat + terkontaminasi) yang sudah ada, kita tahu
        persis "wilayah" ruang fitur (GLCM+HSV) yang biasanya ditempati biji jagung asli.
        Kalau fitur objek baru terlalu jauh dari KEDUA kelas itu, objek itu dianggap BUKAN
        biji jagung -- tanpa pernah dilatih dengan contoh foto "bukan jagung" sama sekali.

        Return:
            is_corn (bool): True kalau dianggap biji jagung (lolos novelty check)
            nearest_class (str atau None): kelas training terdekat (buat info/debug)
            distance (float atau None): jarak Mahalanobis ke kelas terdekat itu
            threshold (float atau None): batas jarak yang dipakai (sudah termasuk margin)
        """
        if self.outlier_stats is None:
            # Statistik belum dihitung (compute_outlier_stats.py belum dijalankan) --
            # fallback ke perilaku lama: semua objek dianggap jagung, tidak ada penolakan.
            return True, None, None, None

        best_class = None
        best_distance = float("inf")
        best_threshold = None

        # Cek jarak ke SETIAP kelas (sehat & terkontaminasi), ambil yang PALING DEKAT.
        # Alasan pakai yang paling dekat: objek dianggap "jagung" kalau dia mirip dengan
        # SALAH SATU dari kedua kelas itu (tidak perlu mirip keduanya).
        for label, stats in self.outlier_stats.items():
            dist = self._mahalanobis_distance(X_scaled_row, stats["mean"], stats["cov_inv"])
            if dist < best_distance:
                best_distance = dist
                best_class = label
                best_threshold = stats["distance_threshold"]

        # Terapkan margin toleransi (lihat penjelasan outlier_margin di __init__)
        effective_threshold = best_threshold * self.outlier_margin
        is_corn = best_distance <= effective_threshold

        return is_corn, best_class, best_distance, effective_threshold

    def predict_single(self, features_dict):
        """
        Step 4: Klasifikasi.
        Urutan BARU (setelah ditambah negasi):
          4a. Scaling fitur (seperti sebelumnya).
          4b. BARU -- Novelty check: apakah fitur ini masuk "wilayah" biji jagung?
              Kalau TIDAK -> langsung dilabeli "bukan_jagung", SVM tidak usah dipanggil.
          4c. Kalau LOLOS (dianggap jagung) -> baru diklasifikasi sehat/terkontaminasi
              pakai SVM, persis seperti kode sebelumnya.
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model SVM atau Scaler belum dimuat.")

        feat_values = np.array([[
            features_dict["contrast"],
            features_dict["correlation"],
            features_dict["energy"],
            features_dict["homogeneity"],
            features_dict["hue"],
            features_dict["saturation"],
            features_dict["value"]
        ]])

        X_scaled = self.scaler.transform(feat_values)

        # --- Langkah baru: novelty/outlier check (pendekatan negasi) ---
        is_corn, nearest_class, distance, threshold = self.check_is_corn(X_scaled[0])

        if not is_corn:
            # Objek dianggap BUKAN biji jagung -- berhenti di sini, tidak perlu tanya SVM
            # sama sekali (SVM memang tidak pernah dilatih untuk kasus ini).
            return {
                "prediction": "bukan_jagung",
                "confidence": None,
                "probabilities": {"sehat": 0.0, "terkontaminasi": 0.0},
                "decision_score": None,
                "outlier_distance": round(distance, 3) if distance is not None else None,
                "outlier_threshold": round(threshold, 3) if threshold is not None else None,
                "outlier_nearest_class": nearest_class
            }

        # --- Kalau lolos novelty check, lanjut klasifikasi SVM seperti kode semula ---
        prediction = self.model.predict(X_scaled)[0]

        decision = self.model.decision_function(X_scaled)[0]
        prob_contam = 1.0 / (1.0 + np.exp(-decision))
        prob_sehat = 1.0 - prob_contam

        classes = list(self.model.classes_)
        if classes[0] == "sehat":
            prob_dict = {"sehat": round(prob_sehat * 100, 1), "terkontaminasi": round(prob_contam * 100, 1)}
        else:
            prob_dict = {"terkontaminasi": round(prob_sehat * 100, 1), "sehat": round(prob_contam * 100, 1)}

        return {
            "prediction": prediction,
            "confidence": prob_dict.get(prediction, 90.0),
            "probabilities": prob_dict,
            "decision_score": round(float(decision), 3),
            "outlier_distance": round(distance, 3) if distance is not None else None,
            "outlier_threshold": round(threshold, 3) if threshold is not None else None,
            "outlier_nearest_class": nearest_class
        }

    def process_image(self, img):
        """
        Full End-to-End Pipeline:
        1. Preprocessing (Grayscale + Gaussian Noise Reduction)
        2. Segmentation:
           - Otsu Thresholding to isolate individual corn kernels
           - K-Means Clustering on each kernel to isolate fungal mold infection
        3. Feature Extraction (GLCM: Contrast, Correlation, Energy, Homogeneity; HSV: H, S, V)
        3.5. BARU -- Novelty check (negasi): tolak objek yang bukan biji jagung
        4. Classification (SVM + Scaler) -- hanya untuk objek yang lolos novelty check
        5. Annotated Visuals & Batch Report Generation
        """
        prep = self.preprocess_image(img)
        seg_otsu = self.segment_kernels_otsu(img)
        kernels = seg_otsu["kernels"]

        # If no distinct kernels detected, treat image as single kernel
        if len(kernels) == 0:
            kernels = [{
                "kernel_id": 1,
                "bbox": (0, 0, img.shape[1], img.shape[0]),
                "area_piksel": img.shape[0] * img.shape[1],
                "crop": img,
                "mask": np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255,
                "contour": None
            }]

        annotated_img = img.copy()
        kernel_results = []
        sehat_count = 0
        kontam_count = 0
        bukan_jagung_count = 0  # BARU

        for k in kernels:
            crop = k["crop"]
            # K-Means segmentation for mold/fungus
            kmeans_res = self.segment_fungus_kmeans(crop, k=3)
            
            # Feature extraction (GLCM + HSV)
            features = self.extract_features(crop)

            # SVM prediction (sekarang termasuk novelty check di dalamnya)
            pred_res = self.predict_single(features)
            label = pred_res["prediction"]

            if label == "sehat":
                sehat_count += 1
                box_color = (46, 204, 113)  # Green BGR
                label_text = f"Sehat ({pred_res['confidence']}%)"
            elif label == "terkontaminasi":
                kontam_count += 1
                box_color = (60, 76, 231)  # Red BGR
                label_text = f"Jamur ({pred_res['confidence']}%)"
            else:
                # label == "bukan_jagung"
                bukan_jagung_count += 1
                box_color = (150, 150, 150)  # Abu-abu BGR -- menandakan objek diabaikan
                label_text = "Bukan Jagung"

            x, y, w, h = k["bbox"]
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                annotated_img,
                f"#{k['kernel_id']}: {label_text}",
                (x, max(y - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                box_color,
                1,
                cv2.LINE_AA
            )

            kernel_results.append({
                "kernel_id": k["kernel_id"],
                "bbox": [int(v) for v in k["bbox"]],
                "prediction": label,
                # is_moldy sekarang HANYA true untuk "terkontaminasi" -- sebelumnya kode lama
                # pakai (label != "sehat"), yang kalau tidak diperbaiki akan salah menghitung
                # "bukan_jagung" sebagai kontaminasi juga.
                "is_moldy": (label == "terkontaminasi"),
                "is_corn": (label != "bukan_jagung"),  # field baru, berguna untuk frontend
                "confidence": pred_res["confidence"],
                "features": features,
                "mold_ratio": round(kmeans_res["mold_ratio"] * 100, 1) if kmeans_res else 0.0,
                "probabilities": pred_res["probabilities"],
                "outlier_distance": pred_res.get("outlier_distance"),
                "outlier_threshold": pred_res.get("outlier_threshold"),
                "crop_img": crop,
                "kmeans_img": kmeans_res["clustered_bgr"] if kmeans_res else None,
                "mold_mask": kmeans_res["mold_mask"] if kmeans_res else None
            })

        total_kernels = len(kernel_results)
        # Persentase kontaminasi sekarang dihitung dari objek yang MEMANG biji jagung saja
        # (sehat + terkontaminasi), supaya objek "bukan_jagung" tidak menggeser persentase.
        total_jagung = sehat_count + kontam_count
        contam_pct = round((kontam_count / total_jagung) * 100, 1) if total_jagung > 0 else 0.0

        if total_jagung == 0:
            quality_status = "Tidak ada biji jagung terdeteksi pada citra ini"
            status_badge = "secondary"
        elif contam_pct == 0.0:
            quality_status = "Kualitas Sangat Baik (Aman Konsumsi & Benih)"
            status_badge = "success"
        elif contam_pct <= 15.0:
            quality_status = "Kualitas Standar (Perlu Sortir Ringan)"
            status_badge = "warning"
        else:
            quality_status = "Kualitas Buruk (Terkontaminasi Jamur / Tidak Layak)"
            status_badge = "danger"

        return {
            "summary": {
                "total_biji": total_kernels,
                "sehat": sehat_count,
                "terkontaminasi": kontam_count,
                "bukan_jagung": bukan_jagung_count,  # field baru
                "persen_kontaminasi": contam_pct,
                "status_mutu": quality_status,
                "badge": status_badge
            },
            "kernels": kernel_results,
            "images": {
                "original": img,
                "gray": prep["gray"],
                "blur": prep["blur"],
                "otsu_mask": seg_otsu["otsu_mask"],
                "annotated": annotated_img
            }
        }

    def process_image_file(self, file_path):
        """Helper to process from file path."""
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Tidak dapat membaca file: {file_path}")
        return self.process_image(img)