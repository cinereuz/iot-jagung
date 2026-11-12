"""
Crowdsourcing Dataset & Continuous Training (Human-in-the-Loop)
================================================================

Alur:
1. Setiap deteksi user (upload / stop kamera) yang berisi biji jagung valid
   otomatis disimpan sebagai FOTO UTUH ke area staging:
       data/staging/sehat/            -> semua biji valid terdeteksi sehat
       data/staging/terkontaminasi/   -> semua biji valid terdeteksi berjamur
       data/staging/campuran/         -> ada biji sehat + berjamur sekaligus
   Foto yang "bukan jagung" (tidak ada biji valid / mayoritas bukan jagung)
   TIDAK disimpan sama sekali.
2. Admin mengkurasi staging: terima (dengan label yang bisa diubah) ke
   data/raw/<label>/ atau hapus.
3. Admin menekan "Re-train": fitur foto baru diekstrak secara INKREMENTAL
   (foto lama memakai fitur yang sudah ada di manifest_fitur.csv), lalu
   StandardScaler, SVM, dan outlier_stats dilatih ulang. Model lama
   di-backup ke models/history/ supaya bisa di-rollback.
"""

import os
import re
import json
import time
import glob
import shutil
import hashlib
import secrets
import threading
from datetime import datetime

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# ------------------------------------------------------------------
# Konstanta path
# ------------------------------------------------------------------
DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")
STAGING_DIR = os.path.join(DATA_DIR, "staging")
HISTORY_LOG = os.path.join(DATA_DIR, "crowdsource_history.jsonl")
MANIFEST_CROP = os.path.join(DATA_DIR, "manifest_crop.csv")
MANIFEST_FITUR = os.path.join(DATA_DIR, "manifest_fitur.csv")

MODELS_DIR = "models"
MODEL_FILES = ["svm_model_v1.joblib", "scaler.joblib", "outlier_stats.joblib"]
MODEL_META = os.path.join(MODELS_DIR, "model_meta.json")
MODEL_HISTORY_DIR = os.path.join(MODELS_DIR, "history")

DATASET_CLASSES = ["sehat", "terkontaminasi"]
STAGING_CATEGORIES = ["sehat", "terkontaminasi", "campuran"]
FEATURE_COLS = ["contrast", "correlation", "energy", "homogeneity", "hue", "saturation", "value"]
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

# Penamaan file dataset resmi, diseragamkan dengan dataset asli tim:
#   data/raw/sehat/normal_001.JPG, normal_002.JPG, ...
#   data/raw/terkontaminasi/virus_001.JPG, virus_002.JPG, ...
# Foto hasil crowdsourcing yang di-approve melanjutkan nomor terakhir.
CLASS_PREFIX = {"sehat": "normal_", "terkontaminasi": "virus_"}
DATASET_EXT = ".JPG"

# Registry foto crowdsourcing yang sudah masuk dataset resmi:
#   { "<hash>": {"label", "filename", "staging_id", "source", "approved_at"} }
# Dipakai untuk (1) mencegah duplikat walaupun nama file sudah diseragamkan,
# dan (2) menghitung berapa foto dataset yang berasal dari user.
CROWD_REGISTRY = os.path.join(DATA_DIR, "crowd_registry.json")

# Cache hash seluruh foto di data/raw/ (termasuk dataset awal normal_001.. & virus_001..):
#   { "sehat/normal_001.JPG": "<12-char-sha1>", ... }
# Supaya kalau foto dari dataset awal di-upload ke halaman deteksi, sistem langsung
# tahu foto itu SUDAH ADA di dataset training dan tidak memasukkannya ke staging
# (serta tidak memunculkan tombol feedback crowdsourcing).
RAW_HASH_CACHE = os.path.join(DATA_DIR, "raw_hashes.json")
_raw_hashes_mem = None  # set of hashes di memori untuk pengecekan O(1)

# ID item staging hanya boleh berisi karakter aman (mencegah path traversal)
_SAFE_ID = re.compile(r"^[A-Za-z0-9_\-]+$")

_fs_lock = threading.Lock()  # melindungi operasi file staging/raw


def _ensure_dirs():
    for c in STAGING_CATEGORIES:
        os.makedirs(os.path.join(STAGING_DIR, c), exist_ok=True)
    for c in DATASET_CLASSES:
        os.makedirs(os.path.join(RAW_DIR, c), exist_ok=True)


def _log_history(entry):
    entry = {"timestamp": datetime.now().isoformat(timespec="seconds"), **entry}
    with open(HISTORY_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _read_history(limit=None):
    if not os.path.exists(HISTORY_LOG):
        return []
    rows = []
    with open(HISTORY_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows[-limit:] if limit else rows


def validate_id(item_id):
    return bool(item_id) and bool(_SAFE_ID.match(item_id))


def validate_category(category):
    return category in STAGING_CATEGORIES


# ==================================================================
# 1. AUTENTIKASI ADMIN SEDERHANA (password + token di memori)
# ==================================================================
class AdminAuth:
    """
    Login admin sederhana tanpa database:
    - Password HANYA diambil dari environment variable ADMIN_PASSWORD, yang
      dibaca otomatis dari file `.env` di root proyek (file itu di-.gitignore,
      jadi password tidak pernah masuk ke repo). Tidak ada password default
      di kode -- kalau ADMIN_PASSWORD kosong, login admin dinonaktifkan.
    - Login berhasil -> server membuat token acak yang berlaku TOKEN_TTL detik.
    - Endpoint admin wajib mengirim token lewat header 'X-Admin-Token'
      (atau query ?token=... khusus untuk <img src>).
    """

    TOKEN_TTL = 8 * 3600  # 8 jam

    def __init__(self):
        self.password = os.environ.get("ADMIN_PASSWORD", "").strip()
        self._tokens = {}
        self._lock = threading.Lock()
        if not self.password:
            print(
                "[Warning] ADMIN_PASSWORD is not set -- admin login is DISABLED. "
                "Create a '.env' file (see '.env.example') containing ADMIN_PASSWORD=..."
            )

    @property
    def enabled(self):
        return bool(self.password)

    def login(self, password):
        if not self.enabled:
            return None
        if not secrets.compare_digest(str(password or ""), self.password):
            return None
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._purge()
            self._tokens[token] = time.time() + self.TOKEN_TTL
        return token

    def logout(self, token):
        with self._lock:
            self._tokens.pop(token, None)

    def is_valid(self, token):
        if not token:
            return False
        with self._lock:
            exp = self._tokens.get(token)
            if exp is None:
                return False
            if exp < time.time():
                self._tokens.pop(token, None)
                return False
            return True

    def _purge(self):
        now = time.time()
        for t in [t for t, e in self._tokens.items() if e < now]:
            self._tokens.pop(t, None)


# ==================================================================
# 2. AUTO-SAVE KE STAGING
# ==================================================================
def decide_staging_category(summary):
    """
    Menentukan ke folder staging mana sebuah hasil deteksi disimpan.
    Return None jika foto TIDAK boleh disimpan (bukan jagung).
    """
    sehat = int(summary.get("sehat", 0))
    kontam = int(summary.get("terkontaminasi", 0))
    total_jagung = sehat + kontam

    # Mengikuti 'Majority Non-Corn Decision Rule' di pipeline:
    # badge 'secondary' = tidak ada jagung / mayoritas bukan jagung -> buang.
    if total_jagung == 0 or summary.get("badge") == "secondary":
        return None
    if sehat > 0 and kontam > 0:
        return "campuran"
    return "sehat" if sehat > 0 else "terkontaminasi"


def _image_hash(img):
    return hashlib.sha1(img.tobytes()).hexdigest()[:12]


def _check_duplicate_hash(img_hash):
    """
    Cek apakah foto dengan `img_hash` sudah ada di dataset resmi (`data/raw/`)
    atau di antrean `data/staging/`.
    Return:
      - "already_in_dataset" jika sudah ada di data/raw/ (baik dataset awal maupun hasil approve user)
      - "duplicate" jika sudah ada di antrean staging
      - None jika belum ada (foto benar-benar baru)
    """
    if img_hash in get_raw_hash_set():
        return "already_in_dataset"
    entry = _load_registry().get(img_hash)
    if entry and os.path.exists(os.path.join(RAW_DIR, entry["label"], entry["filename"])):
        return "already_in_dataset"
    for c in STAGING_CATEGORIES:
        if glob.glob(os.path.join(STAGING_DIR, c, f"*_{img_hash}.json")):
            return "duplicate"
    return None


def _hash_exists(img_hash):
    return _check_duplicate_hash(img_hash) is not None


def sync_raw_hashes():
    """
    Membangun/menyinkronkan cache hash untuk SEMUA foto di `data/raw/sehat` dan
    `data/raw/terkontaminasi` (termasuk ribuan foto dataset awal).
    Hanya foto yang belum ada di `data/raw_hashes.json` yang dihitung hash-nya,
    sehingga startup berikutnya instan (< 5 ms).
    Juga otomatis membersihkan item di `data/staging/` yang ternyata duplikat
    dari foto di `data/raw/`.
    """
    global _raw_hashes_mem
    from concurrent.futures import ThreadPoolExecutor

    _ensure_dirs()
    cache = {}
    if os.path.exists(RAW_HASH_CACHE):
        try:
            with open(RAW_HASH_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        except Exception:
            cache = {}

    current_files = {}
    for c in DATASET_CLASSES:
        folder = os.path.join(RAW_DIR, c)
        if os.path.isdir(folder):
            for fname in os.listdir(folder):
                if fname.lower().endswith(IMG_EXTS):
                    rel_key = f"{c}/{fname}"
                    current_files[rel_key] = os.path.join(folder, fname)

    # Hapus entri cache yang filenya sudah dihapus dari data/raw/
    dirty = False
    for k in list(cache.keys()):
        if k not in current_files:
            cache.pop(k, None)
            dirty = True

    missing = [(k, path) for k, path in current_files.items() if k not in cache]
    if missing:
        print(f"[Crowdsource] Indexing pixel hashes for {len(missing)} dataset image(s) in data/raw/...")

        def _hash_one(item):
            k, path = item
            img = cv2.imread(path)
            if img is None:
                return k, None
            return k, _image_hash(img)

        with ThreadPoolExecutor(max_workers=min(8, (os.cpu_count() or 4))) as ex:
            for k, h in ex.map(_hash_one, missing):
                if h:
                    cache[k] = h
        dirty = True

    if dirty:
        tmp = RAW_HASH_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, RAW_HASH_CACHE)

    _raw_hashes_mem = set(cache.values())

    # Bersihkan item staging yang ternyata sudah ada di data/raw/
    for c in STAGING_CATEGORIES:
        for meta_path in glob.glob(os.path.join(STAGING_DIR, c, "*.json")):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                h = meta.get("hash")
                item_id = meta.get("id")
                if h and h in _raw_hashes_mem and item_id:
                    folder = os.path.join(STAGING_DIR, c)
                    for p in (
                        os.path.join(folder, f"{item_id}.jpg"),
                        os.path.join(folder, f"{item_id}_annotated.jpg"),
                        meta_path,
                    ):
                        if os.path.exists(p):
                            os.remove(p)
            except Exception:
                continue

    return _raw_hashes_mem


def get_raw_hash_set():
    global _raw_hashes_mem
    if _raw_hashes_mem is None:
        return sync_raw_hashes()
    return _raw_hashes_mem


def _register_raw_hash(label, filename, img_hash):
    """Tambahkan 1 foto baru ke cache raw_hashes saat admin klik Approve."""
    global _raw_hashes_mem
    if _raw_hashes_mem is None:
        sync_raw_hashes()
    _raw_hashes_mem.add(img_hash)
    try:
        cache = {}
        if os.path.exists(RAW_HASH_CACHE):
            with open(RAW_HASH_CACHE, "r", encoding="utf-8") as f:
                cache = json.load(f)
        cache[f"{label}/{filename}"] = img_hash
        tmp = RAW_HASH_CACHE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cache, f, ensure_ascii=False)
        os.replace(tmp, RAW_HASH_CACHE)
    except Exception:
        pass


# ------------------------------------------------------------------
# Registry & penamaan berurutan
# ------------------------------------------------------------------
def _load_registry():
    if os.path.exists(CROWD_REGISTRY):
        try:
            with open(CROWD_REGISTRY, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_registry(reg):
    tmp = CROWD_REGISTRY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(reg, f, ensure_ascii=False, indent=2)
    os.replace(tmp, CROWD_REGISTRY)  # atomic: registry tidak pernah setengah tertulis


def _next_dataset_filename(label, registry=None):
    """
    Nama file berikutnya untuk kelas `label`, melanjutkan nomor tertinggi.
    Nomor diambil dari GABUNGAN: isi folder data/raw/<label>, manifest training,
    dan registry -- supaya nomor yang pernah dipakai (lalu filenya dihapus)
    tidak dipakai ulang. Ini penting karena re-training inkremental mengenali
    foto lama dari namanya; kalau nama dipakai ulang, fitur lama bisa terbawa.
    """
    prefix = CLASS_PREFIX[label]
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+)\.[A-Za-z]+$", re.IGNORECASE)
    names = set()

    folder = os.path.join(RAW_DIR, label)
    if os.path.isdir(folder):
        names.update(os.listdir(folder))
    if os.path.exists(MANIFEST_CROP):
        try:
            df = pd.read_csv(MANIFEST_CROP, usecols=["file_asal", "label"])
            names.update(df.loc[df["label"] == label, "file_asal"].astype(str))
        except Exception:
            pass
    for e in (registry if registry is not None else _load_registry()).values():
        if e.get("label") == label:
            names.add(e.get("filename", ""))

    max_num = 0
    for n in names:
        m = pattern.match(n)
        if m:
            max_num = max(max_num, int(m.group(1)))
    return f"{prefix}{max_num + 1:03d}{DATASET_EXT}"


def migrate_legacy_crowd_files():
    """
    Satu kali (idempotent): ganti nama file lama 'crowd_<id>_<hash>.jpg' di data/raw
    menjadi format seragam normal_XXX.JPG / virus_XXX.JPG dan catat di registry.
    Manifest training ikut di-update supaya fitur yang sudah diekstrak tidak hilang.
    """
    legacy = []
    for label in DATASET_CLASSES:
        for p in glob.glob(os.path.join(RAW_DIR, label, "crowd_*")):
            legacy.append((label, p))
    if not legacy:
        return []

    renamed = []
    with _fs_lock:
        reg = _load_registry()
        df_crop = pd.read_csv(MANIFEST_CROP) if os.path.exists(MANIFEST_CROP) else None
        for label, old_path in sorted(legacy):
            old_name = os.path.basename(old_path)
            stem = os.path.splitext(old_name)[0]          # crowd_<YYYYmmdd>_<HHMMSS>_<rand>_<hash>
            parts = stem.split("_")
            img_hash = parts[-1]
            staging_id = "_".join(parts[1:])
            new_name = _next_dataset_filename(label, reg)
            os.rename(old_path, os.path.join(RAW_DIR, label, new_name))
            reg[img_hash] = {
                "label": label,
                "filename": new_name,
                "staging_id": staging_id,
                "source": None,
                "approved_at": datetime.now().isoformat(timespec="seconds"),
                "migrated_from": old_name,
            }
            if df_crop is not None:
                mask = (df_crop["label"] == label) & (df_crop["file_asal"] == old_name)
                df_crop.loc[mask, "file_asal"] = new_name
            renamed.append((old_name, new_name))
        _save_registry(reg)
        if df_crop is not None:
            df_crop.to_csv(MANIFEST_CROP, index=False)
    for old, new in renamed:
        print(f"[Crowdsource] Renamed legacy dataset file {old} -> {new}")
    return renamed


def save_to_staging(original_img, results, source):
    """
    Menyimpan foto utuh (asli, sebelum di-resize) + gambar anotasi + metadata
    ke data/staging/<kategori>/.

    Return dict info staging untuk dikirim ke frontend, contoh:
        {"saved": True, "category": "sehat", "id": "..."}
        {"saved": False, "reason": "not_corn"}
    """
    try:
        summary = results["summary"]
        category = decide_staging_category(summary)
        if category is None:
            return {"saved": False, "reason": "not_corn"}

        img_hash = _image_hash(original_img)
        with _fs_lock:
            _ensure_dirs()
            dup_reason = _check_duplicate_hash(img_hash)
            if dup_reason:
                return {"saved": False, "reason": dup_reason}

            item_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{secrets.token_hex(2)}_{img_hash}"
            folder = os.path.join(STAGING_DIR, category)

            cv2.imwrite(os.path.join(folder, f"{item_id}.jpg"), original_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            annotated = results.get("images", {}).get("annotated")
            if annotated is not None:
                cv2.imwrite(os.path.join(folder, f"{item_id}_annotated.jpg"), annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

            valid_kernels = [k for k in results.get("kernels", []) if k.get("prediction") != "bukan_jagung"]
            confs = [float(k.get("confidence", 0)) for k in valid_kernels]
            meta = {
                "id": item_id,
                "category": category,
                "source": source,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "hash": img_hash,
                "summary": {
                    "total_biji": int(summary.get("total_biji", 0)),
                    "sehat": int(summary.get("sehat", 0)),
                    "terkontaminasi": int(summary.get("terkontaminasi", 0)),
                    "bukan_jagung": int(summary.get("bukan_jagung", 0)),
                    "persen_kontaminasi": float(summary.get("persen_kontaminasi", 0)),
                },
                "avg_confidence": round(float(np.mean(confs)), 2) if confs else None,
                "min_confidence": round(float(np.min(confs)), 2) if confs else None,
                "user_feedback": None,
            }
            with open(os.path.join(folder, f"{item_id}.json"), "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)

        return {"saved": True, "category": category, "id": item_id}
    except Exception as e:  # auto-save tidak boleh menggagalkan deteksi user
        print(f"[Crowdsource] Gagal menyimpan ke staging: {e}")
        return {"saved": False, "reason": "error"}


def _find_item(item_id):
    """Cari item staging berdasarkan id. Return (category, meta_path) atau (None, None)."""
    for c in STAGING_CATEGORIES:
        p = os.path.join(STAGING_DIR, c, f"{item_id}.json")
        if os.path.exists(p):
            return c, p
    return None, None


def submit_user_feedback(item_id, is_correct, suggested_label=None):
    """Feedback opsional dari user biasa: 'hasil benar' atau 'seharusnya <label>'."""
    if not validate_id(item_id):
        return False
    if suggested_label is not None and suggested_label not in DATASET_CLASSES:
        return False
    with _fs_lock:
        category, meta_path = _find_item(item_id)
        if not meta_path:
            return False
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        meta["user_feedback"] = {
            "is_correct": bool(is_correct),
            "suggested_label": suggested_label,
            "at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    return True


# ==================================================================
# 3. KURASI ADMIN
# ==================================================================
def count_images(folder):
    if not os.path.isdir(folder):
        return 0
    return sum(1 for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))


def list_staging():
    _ensure_dirs()
    items = []
    for c in STAGING_CATEGORIES:
        for meta_path in glob.glob(os.path.join(STAGING_DIR, c, "*.json")):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                meta["category"] = c
                items.append(meta)
            except Exception:
                continue
    items.sort(key=lambda m: m.get("created_at", ""), reverse=True)
    return items


def get_dataset_stats():
    _ensure_dirs()
    raw = {c: count_images(os.path.join(RAW_DIR, c)) for c in DATASET_CLASSES}
    # Foto dari user dihitung dari registry (nama file sudah seragam dengan dataset asli)
    crowd = {c: 0 for c in DATASET_CLASSES}
    for e in _load_registry().values():
        if e.get("label") in crowd and os.path.exists(os.path.join(RAW_DIR, e["label"], e["filename"])):
            crowd[e["label"]] += 1
    staging = {
        c: len(glob.glob(os.path.join(STAGING_DIR, c, "*.json")))
        for c in STAGING_CATEGORIES
    }
    return {
        "raw": raw,
        "raw_total": sum(raw.values()),
        "crowd_approved": crowd,
        "staging": staging,
        "staging_total": sum(staging.values()),
        "pending_retrain": count_pending_retrain(),
        "model": load_model_meta(),
        "history": list(reversed(_read_history(limit=15))),
    }


def staging_image_path(category, item_id, kind="original"):
    if not validate_category(category) or not validate_id(item_id):
        return None
    suffix = "_annotated.jpg" if kind == "annotated" else ".jpg"
    path = os.path.join(STAGING_DIR, category, f"{item_id}{suffix}")
    return path if os.path.exists(path) else None


def approve_item(item_id, label):
    """Pindahkan foto dari staging ke data/raw/<label>/ (label boleh diubah admin)."""
    if not validate_id(item_id) or label not in DATASET_CLASSES:
        raise ValueError("Invalid id or label")
    with _fs_lock:
        category, meta_path = _find_item(item_id)
        if not meta_path:
            raise FileNotFoundError("Staging item not found")
        folder = os.path.join(STAGING_DIR, category)
        src_img = os.path.join(folder, f"{item_id}.jpg")
        if not os.path.exists(src_img):
            raise FileNotFoundError("Staging image file missing")

        os.makedirs(os.path.join(RAW_DIR, label), exist_ok=True)

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        # Nama seragam: normal_XXX.JPG / virus_XXX.JPG (lanjut dari nomor terakhir).
        # Hash disimpan di registry (bukan di nama file) untuk pencegahan duplikat.
        reg = _load_registry()
        new_name = _next_dataset_filename(label, reg)
        dst_img = os.path.join(RAW_DIR, label, new_name)
        shutil.move(src_img, dst_img)

        img_hash = meta.get("hash") or item_id.rsplit("_", 1)[-1]
        reg[img_hash] = {
            "label": label,
            "filename": new_name,
            "staging_id": item_id,
            "source": meta.get("source"),
            "approved_at": datetime.now().isoformat(timespec="seconds"),
        }
        _save_registry(reg)
        _register_raw_hash(label, new_name, img_hash)

        for extra in (meta_path, os.path.join(folder, f"{item_id}_annotated.jpg")):
            if os.path.exists(extra):
                os.remove(extra)

        _log_history({
            "action": "approve",
            "id": item_id,
            "from_category": category,
            "label": label,
            "relabeled": category != label,
            "source": meta.get("source"),
            "filename": new_name,
            "hash": img_hash,
        })
    return {"id": item_id, "label": label, "filename": new_name, "path": dst_img}


def reject_item(item_id):
    if not validate_id(item_id):
        raise ValueError("Invalid id")
    with _fs_lock:
        category, meta_path = _find_item(item_id)
        if not meta_path:
            raise FileNotFoundError("Staging item not found")
        folder = os.path.join(STAGING_DIR, category)
        for p in (
            os.path.join(folder, f"{item_id}.jpg"),
            os.path.join(folder, f"{item_id}_annotated.jpg"),
            meta_path,
        ):
            if os.path.exists(p):
                os.remove(p)
        _log_history({"action": "reject", "id": item_id, "from_category": category})
    return {"id": item_id}


# ==================================================================
# 4. RE-TRAINING INKREMENTAL + VERSIONING
# ==================================================================
def load_model_meta():
    if os.path.exists(MODEL_META):
        try:
            with open(MODEL_META, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _list_raw_files():
    """Return set of (label, filename) untuk semua foto di data/raw/."""
    files = set()
    for c in DATASET_CLASSES:
        folder = os.path.join(RAW_DIR, c)
        if os.path.isdir(folder):
            for f in os.listdir(folder):
                if f.lower().endswith(IMG_EXTS):
                    files.add((c, f))
    return files


def _load_manifests():
    if os.path.exists(MANIFEST_CROP) and os.path.exists(MANIFEST_FITUR):
        return pd.read_csv(MANIFEST_CROP), pd.read_csv(MANIFEST_FITUR)
    return (
        pd.DataFrame(columns=["file_asal", "file_crop", "label", "area_piksel"]),
        pd.DataFrame(columns=["file_crop", "label"] + FEATURE_COLS),
    )


def count_pending_retrain():
    """Jumlah foto di data/raw yang belum pernah diekstrak fiturnya (belum ikut training)."""
    try:
        df_crop, _ = _load_manifests()
        known = set(zip(df_crop["label"], df_crop["file_asal"]))
        # Foto yang sudah pernah diproses tapi tidak menghasilkan biji valid
        # juga dicatat supaya tidak dihitung terus sebagai 'pending'.
        known |= _load_skipped()
        return len(_list_raw_files() - known)
    except Exception:
        return 0


SKIPPED_LOG = os.path.join(DATA_DIR, "retrain_skipped.json")


def _load_skipped():
    if os.path.exists(SKIPPED_LOG):
        try:
            with open(SKIPPED_LOG, "r", encoding="utf-8") as f:
                return {tuple(x) for x in json.load(f)}
        except Exception:
            return set()
    return set()


def _save_skipped(skipped):
    with open(SKIPPED_LOG, "w", encoding="utf-8") as f:
        json.dump(sorted(list(skipped)), f)


def _extract_new_image(pipeline, fpath, label):
    """
    Sama persis dengan logika scripts/process_all_and_train.py:
    auto_resize_and_crop -> segment_kernels_otsu -> pilih maks. 3 biji -> extract_features.
    """
    img = cv2.imread(fpath)
    if img is None:
        return []
    img_std = pipeline.auto_resize_and_crop(img, target_size=(224, 224))
    kernels = pipeline.segment_kernels_otsu(img_std)["kernels"]
    if not kernels:
        return []
    k_data = [
        {"area_piksel": k["area_piksel"], "crop": k["crop"], "features": pipeline.extract_features(k["crop"])}
        for k in kernels
    ]
    if len(k_data) > 3:
        k_data = sorted(k_data, key=lambda x: x["features"]["saturation"], reverse=(label == "sehat"))[:3]
    return k_data


def _compute_outlier_stats(X_scaled, y):
    """Sama dengan compute_outlier_stats.py (global + 5 subcluster Mahalanobis)."""
    class_stats = {}
    for label in DATASET_CLASSES:
        X_class = X_scaled[y == label]
        mean_vec = X_class.mean(axis=0)
        cov_mat = np.cov(X_class, rowvar=False) + np.eye(X_class.shape[1]) * 1e-5
        cov_inv = np.linalg.inv(cov_mat)
        diffs = X_class - mean_vec
        distances = np.sqrt(np.einsum("ij,jk,ik->i", diffs, cov_inv, diffs))
        thr_global = float(np.percentile(distances, 98.5))

        km = KMeans(n_clusters=5, random_state=42, n_init=10).fit(X_class)
        subclusters = []
        for cid in range(5):
            sub_pts = X_class[km.labels_ == cid]
            if len(sub_pts) < 15:
                continue
            s_mean = np.mean(sub_pts, axis=0)
            s_cov = np.cov(sub_pts, rowvar=False) + np.eye(sub_pts.shape[1]) * 1e-4
            s_inv = np.linalg.inv(s_cov)
            s_diffs = sub_pts - s_mean
            s_dists = np.sqrt(np.sum((s_diffs @ s_inv) * s_diffs, axis=1))
            s_thr = float(np.percentile(s_dists, 98.8))
            subclusters.append({"mean": s_mean, "cov_inv": s_inv, "distance_threshold": max(s_thr, 3.8)})

        class_stats[label] = {
            "mean": mean_vec,
            "cov_inv": cov_inv,
            "distance_threshold": thr_global,
            "percentile": 98.8,
            "n_samples": int(len(X_class)),
            "subclusters": subclusters,
        }
    return class_stats


def _backup_current_models():
    """Salin model aktif ke models/history/<timestamp>/ sebelum ditimpa."""
    if not all(os.path.exists(os.path.join(MODELS_DIR, f)) for f in MODEL_FILES):
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dst = os.path.join(MODEL_HISTORY_DIR, stamp)
    os.makedirs(dst, exist_ok=True)
    for f in MODEL_FILES + ["model_meta.json"]:
        src = os.path.join(MODELS_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(dst, f))
    # manifest juga di-backup supaya rollback konsisten
    for m in (MANIFEST_CROP, MANIFEST_FITUR):
        if os.path.exists(m):
            shutil.copy2(m, os.path.join(dst, os.path.basename(m)))
    return stamp


def list_model_backups():
    if not os.path.isdir(MODEL_HISTORY_DIR):
        return []
    return sorted(os.listdir(MODEL_HISTORY_DIR), reverse=True)


class RetrainManager:
    """Menjalankan re-training di background thread + menyimpan status progresnya."""

    def __init__(self, pipeline):
        self.pipeline = pipeline
        self._lock = threading.Lock()
        self.state = {"running": False, "stage": "idle", "progress": 0, "message": "", "result": None, "error": None}

    def status(self):
        return dict(self.state)

    def _set(self, **kw):
        self.state.update(kw)

    def start(self):
        if not self._lock.acquire(blocking=False):
            return False
        self.state = {"running": True, "stage": "starting", "progress": 0, "message": "Preparing...", "result": None, "error": None}
        threading.Thread(target=self._run_wrapper, daemon=True).start()
        return True

    def _run_wrapper(self):
        try:
            result = self._run()
            self._set(running=False, stage="done", progress=100, message="Re-training complete", result=result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self._set(running=False, stage="error", message=str(e), error=str(e))
        finally:
            self._lock.release()

    def _run(self):
        t0 = time.time()
        with _fs_lock:
            raw_files = _list_raw_files()
        df_crop, df_fitur = _load_manifests()
        skipped = _load_skipped()

        # --- Sinkronisasi manifest dengan isi data/raw ---
        # (a) buang baris manifest yang fotonya sudah tidak ada di data/raw
        crop_keys = list(zip(df_crop["label"], df_crop["file_asal"]))
        keep_mask = [k in raw_files for k in crop_keys]
        removed_crops = set(df_crop.loc[[not m for m in keep_mask], "file_crop"])
        df_crop = df_crop[keep_mask].reset_index(drop=True)
        if removed_crops:
            df_fitur = df_fitur[~df_fitur["file_crop"].isin(removed_crops)].reset_index(drop=True)

        # (b) foto baru = ada di data/raw tapi belum ada di manifest
        known = set(zip(df_crop["label"], df_crop["file_asal"])) | skipped
        new_files = sorted(raw_files - known)
        self._set(stage="extract", message=f"Extracting features from {len(new_files)} new image(s)...")

        new_crop_rows, new_fitur_rows, n_new_ok = [], [], 0
        for i, (label, fname) in enumerate(new_files, start=1):
            fpath = os.path.join(RAW_DIR, label, fname)
            try:
                k_data = _extract_new_image(self.pipeline, fpath, label)
            except Exception as e:
                print(f"[Retrain] Skip {fpath}: {e}")
                k_data = []
            if not k_data:
                skipped.add((label, fname))
            else:
                n_new_ok += 1
                base = os.path.splitext(fname)[0]
                for idx, k in enumerate(k_data, start=1):
                    crop_fname = f"{base}_k{idx}.png"
                    # simpan crop & preprocessed seperti script training asli
                    for sub in ("cropped", "preprocessed"):
                        os.makedirs(os.path.join(DATA_DIR, sub, label), exist_ok=True)
                    cv2.imwrite(os.path.join(DATA_DIR, "cropped", label, crop_fname), k["crop"])
                    prep = cv2.GaussianBlur(cv2.resize(k["crop"], (128, 128), interpolation=cv2.INTER_CUBIC), (3, 3), 0)
                    cv2.imwrite(os.path.join(DATA_DIR, "preprocessed", label, crop_fname), prep)
                    new_crop_rows.append({"file_asal": fname, "file_crop": crop_fname, "label": label, "area_piksel": int(k["area_piksel"])})
                    f = k["features"]
                    new_fitur_rows.append({"file_crop": crop_fname, "label": label, **{c: f[c] for c in FEATURE_COLS}})
            self._set(progress=int(5 + 35 * i / max(1, len(new_files))))

        if new_crop_rows:
            df_crop = pd.concat([df_crop, pd.DataFrame(new_crop_rows)], ignore_index=True)
            df_fitur = pd.concat([df_fitur, pd.DataFrame(new_fitur_rows)], ignore_index=True)

        counts = df_fitur["label"].value_counts().to_dict()
        if any(counts.get(c, 0) < 20 for c in DATASET_CLASSES):
            raise RuntimeError(f"Not enough training samples per class: {counts}")

        # --- Training ---
        self._set(stage="train", progress=45, message=f"Training SVM on {len(df_fitur)} kernel samples (GridSearchCV)...")
        X = df_fitur[FEATURE_COLS].values.astype(float)
        y = df_fitur["label"].values

        X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

        # Akurasi model LAMA pada test set yang sama (untuk perbandingan adil)
        old_acc = None
        if self.pipeline.model is not None and self.pipeline.scaler is not None:
            try:
                old_acc = float(accuracy_score(y_test, self.pipeline.model.predict(self.pipeline.scaler.transform(X_test_raw))))
            except Exception:
                old_acc = None

        scaler = StandardScaler().fit(X)  # sama seperti script asli: scaler di-fit pada seluruh data
        X_scaled = scaler.transform(X)
        X_train, X_test = scaler.transform(X_train_raw), scaler.transform(X_test_raw)

        param_grid = {"C": [1, 10, 50, 100], "gamma": ["scale", 0.05, 0.1, 0.5, 1.0], "kernel": ["rbf"]}
        grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)
        best = grid.best_estimator_
        y_pred = best.predict(X_test)
        new_acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(y_test, y_pred, digits=4, output_dict=True)

        self._set(stage="outlier", progress=85, message="Updating novelty detection statistics (Mahalanobis)...")
        outlier_stats = _compute_outlier_stats(X_scaled, y)

        # --- Simpan (dengan backup model lama) ---
        self._set(stage="save", progress=92, message="Saving new model version...")
        prev_meta = load_model_meta() or {}
        backup = _backup_current_models()
        version = int(prev_meta.get("version", 1)) + 1

        joblib.dump(best, os.path.join(MODELS_DIR, "svm_model_v1.joblib"))
        joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
        joblib.dump(outlier_stats, os.path.join(MODELS_DIR, "outlier_stats.joblib"))
        df_crop.to_csv(MANIFEST_CROP, index=False)
        df_fitur.to_csv(MANIFEST_FITUR, index=False)
        _save_skipped(skipped)

        meta = {
            "version": version,
            "trained_at": datetime.now().isoformat(timespec="seconds"),
            "accuracy": round(new_acc * 100, 2),
            "previous_accuracy_same_test": round(old_acc * 100, 2) if old_acc is not None else None,
            "best_params": {k: (v if isinstance(v, (int, float, str)) else str(v)) for k, v in grid.best_params_.items()},
            "n_samples": int(len(df_fitur)),
            "n_samples_per_class": {k: int(v) for k, v in counts.items()},
            "n_images": int(df_crop["file_asal"].nunique()),
            "n_new_images": int(n_new_ok),
            "n_skipped_new_images": int(len(new_files) - n_new_ok),
            "f1_per_class": {c: round(report[c]["f1-score"] * 100, 2) for c in DATASET_CLASSES if c in report},
            "backup_of_previous": backup,
            "duration_sec": round(time.time() - t0, 1),
        }
        with open(MODEL_META, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # Reload model di server secara live (tanpa restart app.py)
        self.pipeline.load_models()
        _log_history({"action": "retrain", "version": version, "accuracy": meta["accuracy"],
                      "previous_accuracy": meta["previous_accuracy_same_test"], "n_new_images": n_new_ok})
        return meta

    def rollback(self):
        """Kembalikan model + manifest ke backup terakhir."""
        if self.state.get("running"):
            raise RuntimeError("Re-training is in progress")
        backups = list_model_backups()
        if not backups:
            raise FileNotFoundError("No model backup available")
        latest = backups[0]
        src = os.path.join(MODEL_HISTORY_DIR, latest)
        for f in MODEL_FILES:
            shutil.copy2(os.path.join(src, f), os.path.join(MODELS_DIR, f))
        meta_src = os.path.join(src, "model_meta.json")
        if os.path.exists(meta_src):
            shutil.copy2(meta_src, MODEL_META)
        elif os.path.exists(MODEL_META):
            os.remove(MODEL_META)
        for m in (MANIFEST_CROP, MANIFEST_FITUR):
            b = os.path.join(src, os.path.basename(m))
            if os.path.exists(b):
                shutil.copy2(b, m)
        # Foto yang dulu dilewati mungkin perlu dievaluasi lagi setelah rollback
        if os.path.exists(SKIPPED_LOG):
            os.remove(SKIPPED_LOG)
        shutil.rmtree(src, ignore_errors=True)
        self.pipeline.load_models()
        _log_history({"action": "rollback", "restored_backup": latest})
        return {"restored": latest, "model": load_model_meta()}
