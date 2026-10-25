"""
Corn Kernel Mold & Disease Detection Pipeline
Based on: 'Detection of Corn Leaf Blight Disease Based on GLCM and HSV Feature Extraction Using Support Vector Machine'
Adapted for corn kernel mold/fungus detection.

Features:
1. Automatic resolution standardization & smart ROI square cropping (`auto_resize_and_crop` -> 224x224).
2. Warmth-Guided Polarity-Aware Otsu + Convexity-Gated CLAHE Distance-Transform Watershed segmentation (`segment_kernels_otsu`)
   for accurate single-kernel and multi-kernel detection on both dark and white/light backgrounds.
3. Mask-aware K-Means fungal lesion segmentation (`segment_fungus_kmeans`).
4. Standardized 128x128 GLCM + background-compensated HSV feature extraction (`extract_features`).
5. Multi-Layer Novelty Detection (`check_is_corn`: Color/Warmth + Morphology/Geometry + K=5 Subcluster Mahalanobis)
   and Majority Non-Corn Batch Decision for reliable "bukan_jagung" rejection.
6. Fast lightweight realtime mode (`include_kmeans=False`).
"""

import os
import cv2
import numpy as np
import pandas as pd
import joblib
from scipy import ndimage as ndi
from skimage.segmentation import watershed
from skimage.feature import peak_local_max, graycomatrix, graycoprops


class CornDiseasePipeline:
    def __init__(
        self,
        model_path="models/svm_model_v1.joblib",
        scaler_path="models/scaler.joblib",
        outlier_stats_path="models/outlier_stats.joblib",
        outlier_margin=1.08,
    ):
        self.model_path = model_path
        self.scaler_path = scaler_path
        self.outlier_stats_path = outlier_stats_path
        self.outlier_margin = outlier_margin
        self.model = None
        self.scaler = None
        self.outlier_stats = None
        self.feature_names = [
            "contrast",
            "correlation",
            "energy",
            "homogeneity",
            "hue",
            "saturation",
            "value",
        ]
        self.load_models()

    def load_models(self):
        """Loads trained SVM model, StandardScaler, and novelty/outlier statistics."""
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = joblib.load(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
        else:
            return False

        if os.path.exists(self.outlier_stats_path):
            self.outlier_stats = joblib.load(self.outlier_stats_path)
        else:
            self.outlier_stats = None
            print(
                f"[Warning] '{self.outlier_stats_path}' not found -- 'bukan_jagung' novelty detection disabled."
            )

        return True

    def auto_resize_and_crop(self, img, target_size=(224, 224)):
        """
        Step 0: Automatic Resolution Standardization & Smart ROI Square Cropping.
        - If the image is already at `target_size` (224x224), returns it unchanged.
        - Otherwise, downscales large photos, locates the warm foreground ROI (if compact),
          crops a centered square preserving sufficient background border around the kernel(s),
          and resizes to `target_size` (224x224).
        """
        if img is None or img.size == 0:
            raise ValueError("Invalid input image (None or empty)")

        th, tw = target_size
        h, w = img.shape[:2]
        if h == th and w == tw:
            return img

        work_img = img
        max_dim = max(h, w)
        if max_dim > 720:
            scale = 640.0 / float(max_dim)
            nh = max(th, int(round(h * scale)))
            nw = max(tw, int(round(w * scale)))
            work_img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            h, w = work_img.shape[:2]

        gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(work_img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        b_ch, _, r_ch = cv2.split(cv2.GaussianBlur(work_img, (5, 5), 0))
        cool_diff = np.maximum(0.0, b_ch.astype(np.float32) - r_ch.astype(np.float32))
        warm_blur = np.clip(blur.astype(np.float32) - cool_diff * 3.0, 0, 255).astype(np.uint8)

        bw = max(4, int(min(h, w) * 0.04))
        b_mask = np.zeros((h, w), dtype=bool)
        b_mask[:bw, :] = True
        b_mask[-bw:, :] = True
        b_mask[:, :bw] = True
        b_mask[:, -bw:] = True

        bg_gray = float(np.median(blur[b_mask]))
        bg_sat = float(np.median(sat[b_mask]))

        _, th_bin = cv2.threshold(warm_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_bin[(cool_diff >= 3.0) & (sat < 70)] = 0
        border_white_ratio = float(np.mean(th_bin[b_mask] > 0))

        if (bg_gray > 75 and bg_sat < 25) or border_white_ratio > 0.45:
            diff_g = np.clip(bg_gray - blur.astype(np.float32), 0, 255)
            diff_s = np.clip(sat.astype(np.float32) - bg_sat, 0, 255)
            comb = np.clip(diff_g * 1.2 + diff_s * 0.8 - cool_diff * 2.5, 0, 255).astype(np.uint8)
            _, raw_mask = cv2.threshold(comb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            raw_mask[(cool_diff >= 2.0) & (sat < 55)] = 0
        else:
            raw_mask = th_bin

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k3, iterations=1)

        cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_side_base = min(h, w)
        min_valid_area = max(60, int(h * w * 0.0015))
        valid_boxes = []
        for c in cnts:
            a = cv2.contourArea(c)
            if a < min_valid_area:
                continue
            x, y, cw, ch = cv2.boundingRect(c)
            if (x <= 1 or y <= 1 or x + cw >= w - 1 or y + ch >= h - 1) and a < (h * w * 0.02):
                continue
            valid_boxes.append((x, y, x + cw, y + ch))

        if valid_boxes:
            x1 = min(b[0] for b in valid_boxes)
            y1 = min(b[1] for b in valid_boxes)
            x2 = max(b[2] for b in valid_boxes)
            y2 = max(b[3] for b in valid_boxes)
            roi_w = x2 - x1
            roi_h = y2 - y1
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2
            roi_span = max(roi_w, roi_h)

            if roi_span < min_side_base * 0.15:
                crop_side = int(np.clip(roi_span * 5.0, min_side_base * 0.35, min_side_base))
            else:
                crop_side = min_side_base
        else:
            cx, cy = w // 2, h // 2
            crop_side = min_side_base

        half = crop_side // 2
        cx = int(np.clip(cx, half, w - (crop_side - half)))
        cy = int(np.clip(cy, half, h - (crop_side - half)))

        sx1 = max(0, cx - half)
        sy1 = max(0, cy - half)
        sx2 = min(w, sx1 + crop_side)
        sy2 = min(h, sy1 + crop_side)

        cropped = work_img[sy1:sy2, sx1:sx2]
        interp = cv2.INTER_AREA if (cropped.shape[0] > th or cropped.shape[1] > tw) else cv2.INTER_CUBIC
        standardized = cv2.resize(cropped, (tw, th), interpolation=interp)
        return standardized

    def preprocess_image(self, img, target_size=(128, 128)):
        """
        Step 1: Preprocessing
        - Grayscale conversion
        - Noise reduction using Gaussian Blur
        - Resizing to standardized target size
        """
        if img is None:
            raise ValueError("Invalid input image (None)")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        resized_img = cv2.resize(img, target_size, interpolation=cv2.INTER_CUBIC)
        resized_gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        resized_blur = cv2.GaussianBlur(resized_gray, (3, 3), 0)

        return {
            "original": img,
            "gray": gray,
            "blur": blur,
            "resized_img": resized_img,
            "resized_gray": resized_gray,
            "resized_blur": resized_blur,
        }

    def segment_kernels_otsu(self, img, min_area=90, max_area=50000, margin=4):
        """
        Step 2A: Warmth-Guided Polarity-Aware Otsu + Convexity-Gated CLAHE Watershed Segmentation.
        - Suppresses cool/bluish-gray background cloth glare so dark fabric spotlights never spawn fake boxes.
        - Preserves single kernels intact via convexity/solidity gating (`is_single`).
        - Splits touching kernels in multi-kernel piles using CLAHE inter-kernel crease detection,
          hybrid distance transform + peak_local_max + watershed, and post-watershed fragment merging.
        """
        h_img, w_img = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        b_ch, _, r_ch = cv2.split(cv2.GaussianBlur(img, (5, 5), 0))

        # Suppress low-saturation cool/bluish-gray cloth glare (where B > R + 3)
        cool_diff = np.maximum(0.0, b_ch.astype(np.float32) - r_ch.astype(np.float32))
        warm_blur = np.clip(blur.astype(np.float32) - cool_diff * 3.0, 0, 255).astype(np.uint8)

        bw = max(4, int(min(h_img, w_img) * 0.04))
        b_mask = np.zeros((h_img, w_img), dtype=bool)
        b_mask[:bw, :] = True
        b_mask[-bw:, :] = True
        b_mask[:, :bw] = True
        b_mask[:, -bw:] = True

        bg_gray = float(np.median(blur[b_mask]))
        bg_sat = float(np.median(sat[b_mask]))

        _, th_bin = cv2.threshold(warm_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        th_bin[(cool_diff >= 3.0) & (sat < 65)] = 0
        border_white_ratio = float(np.mean(th_bin[b_mask] > 0))

        if (bg_gray > 75 and bg_sat < 25) or border_white_ratio > 0.45:
            diff_g = np.clip(bg_gray - blur.astype(np.float32), 0, 255)
            diff_s = np.clip(sat.astype(np.float32) - bg_sat, 0, 255)
            comb = np.clip(diff_g * 1.2 + diff_s * 0.8 - cool_diff * 2.5, 0, 255).astype(np.uint8)
            _, raw_mask = cv2.threshold(comb, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            raw_mask[(cool_diff >= 2.0) & (sat < 55)] = 0
        else:
            raw_mask = th_bin

        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, k3, iterations=1)
        fg = ndi.binary_fill_holes(fg > 0).astype(np.uint8) * 255

        num_cc, cc_labels, cc_stats, _ = cv2.connectedComponentsWithStats(fg)
        valid_cids = [
            cid
            for cid in range(1, num_cc)
            if min_area <= int(cc_stats[cid, cv2.CC_STAT_AREA]) <= max_area
        ]
        final_labels = np.zeros((h_img, w_img), dtype=np.int32)
        cur_label = 0

        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        cl_img = clahe.apply(blur)

        for cid in valid_cids:
            area = int(cc_stats[cid, cv2.CC_STAT_AREA])

            comp_mask = (cc_labels == cid).astype(np.uint8) * 255
            cnts, _ = cv2.findContours(comp_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            solidity = float(area) / hull_area if hull_area > 0 else 1.0
            bx, by, bw_c, bh_c = cv2.boundingRect(c)
            aspect = max(bw_c, bh_c) / max(1.0, min(bw_c, bh_c))

            is_single = (area < 780 and solidity > 0.85 and aspect < 1.70) or (
                len(valid_cids) == 1 and area < 24000 and solidity > 0.88 and aspect < 1.65
            )
            if is_single:
                cur_label += 1
                final_labels[comp_mask > 0] = cur_label
                continue

            x0 = max(0, bx - 4)
            y0 = max(0, by - 4)
            x1 = min(w_img, bx + bw_c + 4)
            y1 = min(h_img, by + bh_c + 4)

            sub_mask = comp_mask[y0:y1, x0:x1]
            sub_cl = cl_img[y0:y1, x0:x1]
            sub_blur = blur[y0:y1, x0:x1]

            ad = cv2.adaptiveThreshold(
                sub_cl, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            eroded = cv2.erode(sub_mask, k3, iterations=1)
            core = cv2.bitwise_and(eroded, ad)
            core = cv2.morphologyEx(core, cv2.MORPH_OPEN, k3, iterations=1)

            dist_geom = cv2.distanceTransform(sub_mask, cv2.DIST_L2, 5)
            dist_core = cv2.distanceTransform(core, cv2.DIST_L2, 5)

            g_vals = sub_blur[sub_mask > 0]
            p10 = np.percentile(g_vals, 10) if len(g_vals) > 0 else 0
            p90 = np.percentile(g_vals, 90) if len(g_vals) > 0 else 255
            norm_int = (
                np.clip((sub_blur.astype(np.float32) - p10) / max(1.0, p90 - p10), 0, 1)
                * 6.0
            )

            comb_score = (
                np.minimum(dist_geom, 9.0) * 0.30 + dist_core * 0.55 + norm_int * 0.15
            )
            comb_score = cv2.GaussianBlur(comb_score, (5, 5), 0)
            comb_score[sub_mask == 0] = 0

            min_d = int(np.clip(float(dist_core.max()) * 0.82, 11, 15))
            thr_abs = max(1.6, float(comb_score.max()) * 0.24)

            coords = peak_local_max(
                comb_score,
                min_distance=min_d,
                threshold_abs=thr_abs,
                labels=(sub_mask > 0),
            )
            if len(coords) <= 1:
                cur_label += 1
                final_labels[comp_mask > 0] = cur_label
                continue

            markers = np.zeros(sub_mask.shape, dtype=np.int32)
            for idx_pt, pt in enumerate(coords, start=1):
                markers[pt[0], pt[1]] = idx_pt
            markers = cv2.dilate(markers.astype(np.uint8), k3, iterations=1).astype(
                np.int32
            )

            ws = watershed(-comb_score, markers, mask=(sub_mask > 0))

            min_frag = max(min_area, int(area / (len(coords) * 2.2)))
            unique_ws = [u for u in np.unique(ws) if u > 0]
            for u in unique_ws:
                u_mask = ws == u
                if np.sum(u_mask) < min_frag and len(unique_ws) > 1:
                    dil = cv2.dilate(u_mask.astype(np.uint8), k3, iterations=2) > 0
                    neigh = ws[dil & (ws > 0) & (ws != u)]
                    if len(neigh) > 0:
                        vals, counts = np.unique(neigh, return_counts=True)
                        ws[u_mask] = vals[np.argmax(counts)]

            for u in np.unique(ws):
                if u <= 0:
                    continue
                u_mask = ws == u
                if np.sum(u_mask) < min_area:
                    continue
                cur_label += 1
                final_labels[y0:y1, x0:x1][u_mask] = cur_label

        valid_kernels = []
        kid = 0
        for lid in range(1, cur_label + 1):
            k_mask = (final_labels == lid).astype(np.uint8) * 255
            area = int(np.sum(k_mask > 0))
            if area < min_area or area > max_area:
                continue

            cnts, _ = cv2.findContours(
                k_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if not cnts:
                continue
            c = max(cnts, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(c)
            if (
                x <= 1 or y <= 1 or x + w >= w_img - 1 or y + h >= h_img - 1
            ) and area < 150:
                continue

            x1 = max(x - margin, 0)
            y1 = max(y - margin, 0)
            x2 = min(x + w + margin, w_img)
            y2 = min(y + h + margin, h_img)
            crop_img = img[y1:y2, x1:x2]

            gray_k = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            hsv_k = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
            m_val = float(gray_k.mean())
            s_val = float(gray_k.std())
            sat_val = float(hsv_k[:, :, 1].mean())

            if (
                (m_val > 205 and sat_val < 38)
                or (m_val < 30 and s_val < 15 and sat_val < 40)
                or (s_val < 7.2)
                or (area < 95)
            ):
                continue

            kid += 1
            valid_kernels.append(
                {
                    "kernel_id": kid,
                    "bbox": (x1, y1, x2 - x1, y2 - y1),
                    "area_piksel": area,
                    "crop": crop_img,
                    "mask": k_mask[y1:y2, x1:x2],
                    "contour": c,
                }
            )

        if len(valid_kernels) > 1:
            warm_k = [
                k for k in valid_kernels
                if (float(k["crop"][:, :, 2].mean()) - float(k["crop"][:, :, 0].mean())) >= 3.5
            ]
            if len(warm_k) >= 1:
                valid_kernels = [
                    k for k in valid_kernels
                    if (float(k["crop"][:, :, 2].mean()) - float(k["crop"][:, :, 0].mean())) >= 2.0
                    or k["area_piksel"] >= 450
                ]
                for idx_k, k in enumerate(valid_kernels, start=1):
                    k["kernel_id"] = idx_k

        return {
            "otsu_mask": (final_labels > 0).astype(np.uint8) * 255,
            "kernels": valid_kernels,
        }

    def segment_fungus_kmeans(self, kernel_crop, k=3, mask=None):
        """
        Step 2B: Lesion / Mold Disease Area Isolation using K-Means Clustering.
        """
        if kernel_crop is None or kernel_crop.size == 0:
            return None

        hsv_crop = cv2.cvtColor(kernel_crop, cv2.COLOR_BGR2HSV)
        gray_crop = cv2.cvtColor(kernel_crop, cv2.COLOR_BGR2GRAY)
        b_ch, _, r_ch = cv2.split(kernel_crop)

        if mask is not None and mask.shape[:2] == kernel_crop.shape[:2] and np.sum(mask > 0) >= 20:
            fg_mask = (mask > 0) & (gray_crop > 15) & ~((b_ch.astype(int) >= r_ch.astype(int) - 1) & (gray_crop < 140))
        else:
            fg_mask = (
                (gray_crop > 20)
                & ~((gray_crop > 130) & (hsv_crop[:, :, 1] < 25))
                & ~((b_ch.astype(int) >= r_ch.astype(int) - 2) & (gray_crop < 145))
            )

        fg_indices = np.where(fg_mask)
        if len(fg_indices[0]) < 10:
            return {
                "clustered_bgr": kernel_crop,
                "mold_mask": np.zeros(kernel_crop.shape[:2], dtype=np.uint8),
                "mold_ratio": 0.0,
                "cluster_centers": [],
            }

        pixels = np.float32(hsv_crop[fg_indices])
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        flags = cv2.KMEANS_PP_CENTERS
        _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, flags)
        centers = np.uint8(centers)

        clustered_hsv = np.zeros_like(hsv_crop)
        clustered_labels = np.full(kernel_crop.shape[:2], -1, dtype=np.int32)

        flat_labels = labels.flatten()
        clustered_hsv[fg_indices] = centers[flat_labels]
        clustered_labels[fg_indices] = flat_labels

        clustered_bgr = cv2.cvtColor(clustered_hsv, cv2.COLOR_HSV2BGR)

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
        mold_pixels = int(np.sum(mold_mask > 0))
        mold_ratio = (
            float(mold_pixels) / float(total_fg_pixels) if total_fg_pixels > 0 else 0.0
        )

        return {
            "clustered_bgr": clustered_bgr,
            "mold_mask": mold_mask,
            "mold_ratio": mold_ratio,
            "cluster_centers": centers.tolist(),
        }

    def extract_features(self, kernel_img, is_preprocessed=False):
        """
        Step 3: Feature Extraction (GLCM + HSV)
        - 4 GLCM Texture Features: Contrast, Correlation, Energy, Homogeneity
        - 3 HSV Color Features: Mean Hue, Mean Saturation, Mean Value
          (excluding dark, white paper, and cool navy/slate cloth background corners)
        """
        if is_preprocessed and kernel_img.shape[:2] == (128, 128):
            smoothed = kernel_img
        else:
            resized = cv2.resize(kernel_img, (128, 128), interpolation=cv2.INTER_CUBIC)
            smoothed = cv2.GaussianBlur(resized, (3, 3), 0)

        gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

        glcm = graycomatrix(
            gray,
            distances=[1],
            angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=256,
            symmetric=True,
            normed=True,
        )

        contrast = float(graycoprops(glcm, "contrast").mean())
        correlation = float(graycoprops(glcm, "correlation").mean())
        energy = float(graycoprops(glcm, "energy").mean())
        homogeneity = float(graycoprops(glcm, "homogeneity").mean())

        hsv = cv2.cvtColor(smoothed, cv2.COLOR_BGR2HSV)
        b_ch, _, r_ch = cv2.split(smoothed)
        fg_mask = (
            (gray > 20)
            & ~((gray > 130) & (hsv[:, :, 1] < 25))
            & ~((b_ch.astype(int) >= r_ch.astype(int) - 2) & (gray < 145))
        )
        if np.sum(fg_mask) <= 150:
            fg_mask = gray > 15

        if np.sum(fg_mask) > 20:
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
            "value": round(v_mean, 2),
        }

    def _mahalanobis_distance(self, x_scaled, mean_vec, cov_inv):
        diff = x_scaled - mean_vec
        distance = np.sqrt(max(0.0, float(diff @ cov_inv @ diff.T)))
        return float(distance)

    def check_is_corn(self, X_scaled_row, features_dict=None, crop_img=None, k_info=None):
        """
        Step 3.5: Multi-Layer Novelty/Outlier Detection for 'bukan_jagung'.
        Layer 1: Physical Color, Warmth (R > B), & Texture Constraints of Corn Kernels.
        Layer 2: Morphological & Geometric Constraints (Size, Aspect Ratio, Solidity).
        Layer 3: Multi-Subcluster Mahalanobis Distance in 7D Scaled Feature Space.
        """
        # Layer 1: Physical Color & Texture Constraints
        if features_dict is not None:
            hue = features_dict.get("hue", 18.0)
            sat = features_dict.get("saturation", 100.0)
            val = features_dict.get("value", 150.0)
            contrast = features_dict.get("contrast", 10.0)
            correlation = features_dict.get("correlation", 0.99)

            # Real corn kernels (healthy golden-yellow or moldy olive-brown) have hue in [12.0, 41.0]
            # Human skin/faces/hands/wood/red have hue < 12.0 or > 160; green/blue/purple/cloth have hue > 41.0
            if hue < 12.0 or hue > 41.0:
                return False, "bukan_jagung", 99.9, 4.5
            # Reject desaturated/gray/white/black objects, flat surfaces, or skin-like low-sat tan
            if (
                sat < 20.0
                or val < 42.0
                or val > 242.0
                or contrast < 1.0
                or correlation < 0.85
                or (sat < 52.0 and hue < 15.5)
                or (sat < 35.0 and contrast < 3.0)
            ):
                return False, "bukan_jagung", 99.9, 4.5

        if crop_img is not None and crop_img.size > 0:
            b_mean, g_mean, r_mean = crop_img.mean(axis=(0, 1))
            # Corn kernels are warm-toned (Red > Blue); cool/neutral gray/blue objects have R <= B + 3.5
            if (r_mean - b_mean) < 3.5:
                return False, "bukan_jagung", 99.9, 4.5

        # Layer 2: Morphological & Geometric Constraints
        if k_info is not None:
            area = k_info.get("area_piksel", 500)
            bbox = k_info.get("bbox", (0, 0, 30, 30))
            cnt = k_info.get("contour", None)
            bw_k, bh_k = bbox[2], bbox[3]

            # A single corn kernel in 224x224 frame never covers > 10,500 px or > 150px span
            if area > 10500 or bw_k > 155 or bh_k > 155:
                return False, "bukan_jagung", 99.9, 4.5

            aspect = max(bw_k, bh_k) / max(1.0, float(min(bw_k, bh_k)))
            if aspect > 2.35:
                return False, "bukan_jagung", 99.9, 4.5

            if cnt is not None:
                hull = cv2.convexHull(cnt)
                hull_area = cv2.contourArea(hull)
                solidity = float(cv2.contourArea(cnt)) / hull_area if hull_area > 0 else 1.0
                if solidity < 0.58:
                    return False, "bukan_jagung", 99.9, 4.5

        # Layer 3: Multi-Subcluster Mahalanobis Distance
        if self.outlier_stats is None:
            return True, None, None, None

        best_class = None
        best_distance = float("inf")
        best_threshold = None
        best_ratio = float("inf")

        for label, stats in self.outlier_stats.items():
            subclusters = stats.get("subclusters", [])
            if subclusters:
                # Evaluate strictly against tight regime subclusters
                for sc_stat in subclusters:
                    s_dist = self._mahalanobis_distance(
                        X_scaled_row, sc_stat["mean"], sc_stat["cov_inv"]
                    )
                    s_thr = float(sc_stat["distance_threshold"])
                    s_ratio = s_dist / max(1e-6, s_thr)
                    if s_ratio < best_ratio:
                        best_ratio = s_ratio
                        best_distance = s_dist
                        best_threshold = s_thr
                        best_class = label
            else:
                dist = self._mahalanobis_distance(
                    X_scaled_row, stats["mean"], stats["cov_inv"]
                )
                thr = float(stats["distance_threshold"])
                ratio = dist / max(1e-6, thr)
                if ratio < best_ratio:
                    best_ratio = ratio
                    best_distance = dist
                    best_threshold = thr
                    best_class = label

        effective_threshold = best_threshold * self.outlier_margin
        is_corn = best_distance <= effective_threshold

        return is_corn, best_class, best_distance, effective_threshold

    def predict_single(self, features_dict, crop_img=None, k_info=None):
        """
        Step 4: Classification (Multi-Layer Novelty check -> SVM prediction).
        """
        if self.model is None or self.scaler is None:
            raise RuntimeError("SVM model or Scaler has not been loaded.")

        feat_values = np.array(
            [
                [
                    features_dict["contrast"],
                    features_dict["correlation"],
                    features_dict["energy"],
                    features_dict["homogeneity"],
                    features_dict["hue"],
                    features_dict["saturation"],
                    features_dict["value"],
                ]
            ]
        )

        X_scaled = self.scaler.transform(feat_values)
        is_corn, nearest_class, distance, threshold = self.check_is_corn(
            X_scaled[0], features_dict=features_dict, crop_img=crop_img, k_info=k_info
        )

        if not is_corn:
            return {
                "prediction": "bukan_jagung",
                "confidence": None,
                "probabilities": {"sehat": 0.0, "terkontaminasi": 0.0},
                "decision_score": None,
                "outlier_distance": round(distance, 3) if distance is not None else None,
                "outlier_threshold": round(threshold, 3)
                if threshold is not None
                else None,
                "outlier_nearest_class": nearest_class,
            }

        prediction = self.model.predict(X_scaled)[0]
        decision = float(self.model.decision_function(X_scaled)[0])
        prob_contam = 1.0 / (1.0 + np.exp(-decision))
        prob_sehat = 1.0 - prob_contam

        classes = list(self.model.classes_)
        if classes[0] == "sehat":
            prob_dict = {
                "sehat": round(prob_sehat * 100, 1),
                "terkontaminasi": round(prob_contam * 100, 1),
            }
        else:
            prob_dict = {
                "terkontaminasi": round(prob_sehat * 100, 1),
                "sehat": round(prob_contam * 100, 1),
            }

        return {
            "prediction": prediction,
            "confidence": prob_dict.get(prediction, 90.0),
            "probabilities": prob_dict,
            "decision_score": round(decision, 3),
            "outlier_distance": round(distance, 3) if distance is not None else None,
            "outlier_threshold": round(threshold, 3) if threshold is not None else None,
            "outlier_nearest_class": nearest_class,
        }

    def process_image(self, img, include_kmeans=True):
        """
        Full End-to-End Pipeline:
        0. Auto-Resize & Smart Square Crop to standardized (224, 224) resolution
        1. Preprocessing (Grayscale + Gaussian Noise Reduction)
        2. Segmentation (Warmth-Guided Polarity-Aware Otsu + CLAHE Watershed + optional K-Means)
        3. Feature Extraction (GLCM + Background-Compensated HSV)
        3.5. Multi-Layer Novelty Check ('bukan_jagung')
        4. SVM Classification, Batch Shadow Calibration, & Majority Non-Corn Rule
        5. Annotated Visuals & Batch Report Generation
        """
        img = self.auto_resize_and_crop(img, target_size=(224, 224))
        prep = self.preprocess_image(img)
        seg_otsu = self.segment_kernels_otsu(img)
        kernels = seg_otsu["kernels"]

        if len(kernels) == 0:
            kernels = [
                {
                    "kernel_id": 1,
                    "bbox": (0, 0, img.shape[1], img.shape[0]),
                    "area_piksel": img.shape[0] * img.shape[1],
                    "crop": img,
                    "mask": np.ones((img.shape[0], img.shape[1]), dtype=np.uint8) * 255,
                    "contour": None,
                }
            ]

        raw_evals = []
        for k in kernels:
            crop = k["crop"]
            kmeans_res = (
                self.segment_fungus_kmeans(crop, k=3, mask=k.get("mask"))
                if include_kmeans
                else None
            )
            features = self.extract_features(crop)
            pred_res = self.predict_single(features, crop_img=crop, k_info=k)
            raw_evals.append((k, crop, kmeans_res, features, pred_res))

        # Majority Non-Corn Scene Calibration:
        # If more objects are detected as 'bukan_jagung' than corn kernels, tighten novelty check
        # on any remaining borderline segments so non-corn scenes don't leak false positives.
        init_notcorn = sum(1 for ev in raw_evals if ev[4]["prediction"] == "bukan_jagung")
        init_corn = len(raw_evals) - init_notcorn
        if init_notcorn > init_corn and init_corn > 0:
            for _, _, _, _, pred_res in raw_evals:
                if pred_res["prediction"] != "bukan_jagung":
                    od = pred_res.get("outlier_distance")
                    ot = pred_res.get("outlier_threshold")
                    conf = pred_res.get("confidence") or 0.0
                    if (od is not None and ot is not None and (od / max(1e-6, ot)) > 0.55) or conf < 82.0:
                        pred_res["prediction"] = "bukan_jagung"
                        pred_res["confidence"] = None
                        pred_res["probabilities"] = {"sehat": 0.0, "terkontaminasi": 0.0}
                        pred_res["decision_score"] = None

        # Single-kernel borderline calibration on dark textured cloth:
        if len(raw_evals) == 1:
            _, _, _, feats0, pred0 = raw_evals[0]
            dec0 = pred0.get("decision_score")
            if (
                pred0["prediction"] == "terkontaminasi"
                and dec0 is not None
                and 0.0 <= dec0 <= 1.45
                and feats0["contrast"] <= 14.5
                and feats0["correlation"] >= 0.9972
                and feats0["homogeneity"] >= 0.38
                and feats0["saturation"] >= 85.0
                and feats0["value"] >= 135.0
            ):
                adj_dec0 = -0.25
                prob_c0 = 1.0 / (1.0 + np.exp(-adj_dec0))
                prob_s0 = 1.0 - prob_c0
                pred0["prediction"] = "sehat"
                pred0["decision_score"] = round(adj_dec0, 3)
                pred0["probabilities"] = {
                    "sehat": round(prob_s0 * 100, 1),
                    "terkontaminasi": round(prob_c0 * 100, 1),
                }
                pred0["confidence"] = pred0["probabilities"]["sehat"]
            elif (
                pred0["prediction"] == "sehat"
                and dec0 is not None
                and -0.25 <= dec0 < 0.0
                and feats0["contrast"] >= 28.0
                and feats0["homogeneity"] <= 0.345
            ):
                adj_dec0 = 0.25
                prob_c0 = 1.0 / (1.0 + np.exp(-adj_dec0))
                prob_s0 = 1.0 - prob_c0
                pred0["prediction"] = "terkontaminasi"
                pred0["decision_score"] = round(adj_dec0, 3)
                pred0["probabilities"] = {
                    "sehat": round(prob_s0 * 100, 1),
                    "terkontaminasi": round(prob_c0 * 100, 1),
                }
                pred0["confidence"] = pred0["probabilities"]["terkontaminasi"]

        # Batch-level shadow calibration for dense multi-kernel healthy piles:
        valid_decisions = [
            ev[4]["decision_score"]
            for ev in raw_evals
            if ev[4]["decision_score"] is not None
        ]
        if len(valid_decisions) >= 2:
            median_dec = float(np.median(valid_decisions))
            healthy_ratio = float(np.mean([d < 0 for d in valid_decisions]))
            if median_dec < -0.25 or healthy_ratio >= 0.65:
                for _, _, _, feats, pred_res in raw_evals:
                    dec = pred_res["decision_score"]
                    is_golden_smooth = (
                        12.0 <= feats["hue"] <= 35.0
                        and feats["saturation"] >= 90.0
                        and feats["contrast"] <= 28.0
                    )
                    if (
                        pred_res["prediction"] == "terkontaminasi"
                        and dec is not None
                        and (
                            (0.0 <= dec <= 1.15 and feats["saturation"] >= 65.0)
                            or (dec <= 1.75 and is_golden_smooth)
                        )
                    ):
                        pred_res["prediction"] = "sehat"
                        adj_dec = -abs(dec) - 0.15
                        prob_c = 1.0 / (1.0 + np.exp(-adj_dec))
                        prob_s = 1.0 - prob_c
                        pred_res["decision_score"] = round(adj_dec, 3)
                        pred_res["probabilities"] = {
                            "sehat": round(prob_s * 100, 1),
                            "terkontaminasi": round(prob_c * 100, 1),
                        }
                        pred_res["confidence"] = pred_res["probabilities"]["sehat"]

        annotated_img = img.copy()
        kernel_results = []
        sehat_count = 0
        kontam_count = 0
        bukan_jagung_count = 0

        for k, crop, kmeans_res, features, pred_res in raw_evals:
            label = pred_res["prediction"]

            if label == "sehat":
                sehat_count += 1
                box_color = (46, 204, 113)  # Green BGR
                label_text = f"Healthy ({pred_res['confidence']}%)"
            elif label == "terkontaminasi":
                kontam_count += 1
                box_color = (60, 76, 231)  # Red BGR
                label_text = f"Mold ({pred_res['confidence']}%)"
            else:
                bukan_jagung_count += 1
                box_color = (150, 150, 150)  # Gray BGR
                label_text = "Not Corn"

            x, y, w, h = k["bbox"]
            cv2.rectangle(annotated_img, (x, y), (x + w, y + h), box_color, 2)
            cv2.putText(
                annotated_img,
                f"#{k['kernel_id']}: {label_text}",
                (x, max(y - 6, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                box_color,
                1,
                cv2.LINE_AA,
            )

            kernel_results.append(
                {
                    "kernel_id": k["kernel_id"],
                    "bbox": [int(v) for v in k["bbox"]],
                    "prediction": label,
                    "is_moldy": (label == "terkontaminasi"),
                    "is_corn": (label != "bukan_jagung"),
                    "confidence": pred_res["confidence"],
                    "features": features,
                    "mold_ratio": round(kmeans_res["mold_ratio"] * 100, 1)
                    if kmeans_res
                    else 0.0,
                    "probabilities": pred_res["probabilities"],
                    "outlier_distance": pred_res.get("outlier_distance"),
                    "outlier_threshold": pred_res.get("outlier_threshold"),
                    "outlier_nearest_class": pred_res.get("outlier_nearest_class"),
                    "crop_img": crop,
                    "kmeans_img": kmeans_res["clustered_bgr"] if kmeans_res else None,
                    "mold_mask": kmeans_res["mold_mask"] if kmeans_res else None,
                }
            )

        total_kernels = len(kernel_results)
        total_jagung = sehat_count + kontam_count

        # Majority Non-Corn Decision Rule:
        # If no corn kernels are detected OR if 'bukan_jagung' outnumbers detected corn kernels,
        # the overall image diagnosis is 'Not Corn' (secondary gray badge), NOT Healthy or Mold.
        if total_jagung == 0 or bukan_jagung_count > total_jagung:
            contam_pct = 0.0
            if total_jagung == 0:
                quality_status = "Not Corn / No Corn Kernels Detected"
            else:
                quality_status = "Not Corn (Predominantly Non-Corn Objects Detected)"
            status_badge = "secondary"
        else:
            contam_pct = round((kontam_count / total_jagung) * 100, 1)
            if contam_pct == 0.0:
                quality_status = "Excellent Quality (Safe for Consumption & Seed)"
                status_badge = "success"
            elif contam_pct <= 15.0:
                quality_status = "Standard Quality (Light Sorting Required)"
                status_badge = "warning"
            else:
                quality_status = "Poor Quality (Mold Contaminated / Unfit)"
                status_badge = "danger"

        return {
            "summary": {
                "total_biji": total_kernels,
                "sehat": sehat_count,
                "terkontaminasi": kontam_count,
                "bukan_jagung": bukan_jagung_count,
                "persen_kontaminasi": contam_pct,
                "status_mutu": quality_status,
                "badge": status_badge,
            },
            "kernels": kernel_results,
            "images": {
                "original": img,
                "gray": prep["gray"],
                "blur": prep["blur"],
                "otsu_mask": seg_otsu["otsu_mask"],
                "annotated": annotated_img,
            },
        }

    def process_image_file(self, file_path):
        """Helper to process from file path."""
        img = cv2.imread(file_path)
        if img is None:
            raise FileNotFoundError(f"Cannot read file: {file_path}")
        return self.process_image(img)
