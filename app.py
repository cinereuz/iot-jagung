import os
import io
import base64
import cv2
import numpy as np
from dotenv import load_dotenv

# BARU -- baca konfigurasi rahasia (ADMIN_PASSWORD) dari file .env di root proyek.
# Harus dipanggil SEBELUM import modul yang membaca os.environ.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from typing import Optional
from fastapi import FastAPI, File, UploadFile, Request, HTTPException, Header, Query
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.pipeline import CornDiseasePipeline
from src import crowdsource

app = FastAPI(title="Corn Kernel Mold Detector")

# Mount static and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize pipeline
pipeline = CornDiseasePipeline()

# BARU -- Crowdsourcing dataset & continuous training
admin_auth = crowdsource.AdminAuth()
retrain_manager = crowdsource.RetrainManager(pipeline)
# Seragamkan nama file crowdsourcing versi lama (crowd_*.jpg) -> normal_XXX / virus_XXX
crowdsource.migrate_legacy_crowd_files()
# Index hash seluruh foto di data/raw/ (termasuk dataset awal) supaya foto dataset
# yang di-upload ulang untuk tes tidak masuk ke staging & tidak memunculkan feedback
crowdsource.sync_raw_hashes()

def mat_to_base64(mat, is_rgb=False):
    """Encodes OpenCV image (BGR or Gray) into base64 JPEG string."""
    if mat is None or mat.size == 0:
        return None
    if is_rgb:
        bgr = cv2.cvtColor(mat, cv2.COLOR_RGB2BGR)
    else:
        bgr = mat
    _, buffer = cv2.imencode('.jpg', bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request=request, name="index.html")

@app.get("/api/samples")
async def get_samples():
    """Returns available sample images from data/raw/ for quick testing."""
    samples = []
    
    # Healthy samples
    sehat_dir = "data/raw/sehat"
    if os.path.exists(sehat_dir):
        files = sorted(os.listdir(sehat_dir))[:8]
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append({
                    "id": f"sehat_{f}",
                    "name": f"Healthy Kernel ({f})",
                    "category": "sehat",
                    "path": os.path.join(sehat_dir, f)
                })

    # Moldy samples
    kontam_dir = "data/raw/terkontaminasi"
    if os.path.exists(kontam_dir):
        files = sorted(os.listdir(kontam_dir))[:8]
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append({
                    "id": f"kontam_{f}",
                    "name": f"Moldy Kernel ({f})",
                    "category": "terkontaminasi",
                    "path": os.path.join(kontam_dir, f)
                })

    return {"samples": samples}

class SampleRequest(BaseModel):
    sample_path: str

@app.post("/api/detect/sample")
async def detect_sample(req: SampleRequest):
    if not os.path.exists(req.sample_path):
        raise HTTPException(status_code=404, detail="Sample file not found")

    img = cv2.imread(req.sample_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Failed to read the sample image")

    return run_detection(img)

@app.post("/api/detect/upload")
async def detect_upload(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Invalid or corrupted image file format")

    # BARU -- hasil upload user otomatis masuk staging crowdsourcing
    return run_detection(img, staging_source="upload")

def run_detection(img, staging_source=None):
    """
    Runs pipeline and formats JSON response with base64 images.

    staging_source: kalau diisi ("upload" / "webcam"), foto utuh otomatis disimpan
    ke data/staging/ (kecuali terdeteksi bukan jagung). Untuk sampel bawaan
    dibiarkan None supaya tidak menduplikasi dataset.
    """
    try:
        results = pipeline.process_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    staging_info = None
    if staging_source:
        staging_info = crowdsource.save_to_staging(img, results, staging_source)

    # Format images to base64
    images_b64 = {
        "raw_input": mat_to_base64(results["images"].get("raw_input", results["images"]["original"])),
        "original": mat_to_base64(results["images"]["original"]),
        "gray": mat_to_base64(results["images"]["gray"]),
        "blur": mat_to_base64(results["images"]["blur"]),
        "resized_blur": mat_to_base64(results["images"].get("resized_blur", results["images"]["blur"])),
        "otsu_mask": mat_to_base64(results["images"]["otsu_mask"]),
        "seg_overlay": mat_to_base64(results["images"].get("seg_overlay", results["images"]["annotated"])),
        "annotated": mat_to_base64(results["images"]["annotated"])
    }

    # Format individual kernel images
    kernels_formatted = []
    for k in results["kernels"]:
        k_dict = {
            "kernel_id": k["kernel_id"],
            "bbox": k["bbox"],
            "area_piksel": k.get("area_piksel"),
            "solidity": k.get("solidity"),
            "aspect_ratio": k.get("aspect_ratio"),
            "prediction": k["prediction"],
            "is_moldy": k["is_moldy"],
            "is_corn": k.get("is_corn"),  # BARU -- untuk debugging & tampilan frontend nanti
            "confidence": k["confidence"],
            "decision_score": k.get("decision_score"),
            "mold_ratio": k["mold_ratio"],
            "cluster_centers": k.get("cluster_centers", []),
            "probabilities": k["probabilities"],
            "features": k["features"],
            "scaled_features": k.get("scaled_features"),
            # BARU -- 3 field ini dipakai untuk mengecek & men-tuning deteksi 'bukan jagung'
            "outlier_distance": k.get("outlier_distance"),
            "outlier_threshold": k.get("outlier_threshold"),
            "outlier_nearest_class": k.get("outlier_nearest_class"),
            "crop_b64": mat_to_base64(k["crop_img"]),
            "kmeans_b64": mat_to_base64(k["kmeans_img"]) if k["kmeans_img"] is not None else None,
            "mold_mask_b64": mat_to_base64(k["mold_mask"]) if k["mold_mask"] is not None else None
        }
        kernels_formatted.append(k_dict)

    return {
        "status": "success",
        "summary": results["summary"],
        "stage_meta": results.get("stage_meta", {}),
        "images": images_b64,
        "kernels": kernels_formatted,
        "staging": staging_info
    }


# ================================================================
# BARU -- Mode Realtime Webcam
# ================================================================

class FrameRequest(BaseModel):
    image: str  # base64 string dari canvas.toDataURL(), formatnya "data:image/jpeg;base64,xxxxx..."


def strip_base64_prefix(b64_string):
    """
    Browser mengirim base64 dengan awalan "data:image/jpeg;base64," yang menempel
    di depan teksnya. Awalan itu HARUS dibuang dulu sebelum di-decode, kalau tidak
    base64.b64decode() akan error karena bagian "data:image/jpeg;base64," itu
    bukan karakter base64 yang valid.
    """
    if "," in b64_string:
        return b64_string.split(",", 1)[1]
    return b64_string


def run_detection_lightweight(img):
    """
    Versi ringan dari run_detection(), khusus dipakai mode realtime webcam.

    Bedanya dengan run_detection() biasa:
    - TIDAK mengirim balik gambar tahap preprocessing (gray, blur, otsu_mask) --
      hanya gambar 'annotated' (yang sudah ada kotak deteksi) yang dikirim,
      karena cuma itu yang perlu ditampilkan realtime.
    - TIDAK menjalankan K-Means clustering (include_kmeans=False) dan TIDAK
      mengirim balik crop_img / kmeans_img / mold_mask per biji -- membuat
      deteksi realtime ~4-5x lebih cepat dan bebas lag.
    """
    try:
        results = pipeline.process_image(img, include_kmeans=False)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

    # Ambil hanya info penting tiap biji, TANPA gambar-gambar beratnya
    kernels_light = []
    for k in results["kernels"]:
        kernels_light.append({
            "kernel_id": k["kernel_id"],
            "bbox": k["bbox"],
            "prediction": k["prediction"],
            "confidence": k["confidence"],
        })

    return {
        "status": "success",
        "summary": results["summary"],
        "annotated": mat_to_base64(results["images"]["annotated"]),
        "kernels": kernels_light
    }


@app.post("/api/detect/frame")
def detect_frame(req: FrameRequest):
    """
    Endpoint khusus untuk 1 frame webcam.

    PENTING -- perhatikan fungsi ini pakai 'def' BIASA, BUKAN 'async def' seperti
    endpoint lain di atas (/api/detect/upload, /api/detect/sample). Ini disengaja:

    - Isi fungsi ini kerja berat di CPU (OpenCV, K-Means, GLCM) dan tidak pernah
      pakai 'await' di dalamnya sama sekali.
    - Kalau ditulis 'async def' tapi isinya kerja berat tanpa 'await', FastAPI
      akan menjalankannya di 'jalur utama' server -- artinya SELURUH server ikut
      macet total selama 1 frame ini diproses, termasuk endpoint lain yang lagi
      dipakai orang lain di saat bersamaan.
    - Dengan 'def' biasa, FastAPI otomatis memindahkan fungsi ini ke 'thread pool'
      (semacam antrean pekerja terpisah), jadi server tetap bisa melayani request
      lain sambil frame ini diproses.
    - Ini jauh lebih krusial di endpoint ini dibanding endpoint upload/sample
      biasa, karena endpoint ini dipanggil BERULANG KALI tiap detik selama
      kamera menyala -- kalau blocking, aplikasi bisa terasa macet total,
      bukan cuma "agak lambat".
    """
    b64_clean = strip_base64_prefix(req.image)
    try:
        img_bytes = base64.b64decode(b64_clean)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode the image frame")

    return run_detection_lightweight(img)


@app.post("/api/detect/frame/full")
def detect_frame_full(req: FrameRequest):
    """
    Endpoint untuk analisis PENUH dari frame webcam terakhir.

    Dipakai saat user menghentikan kamera setelah deteksi biji.
    Bedanya dengan /api/detect/frame biasa:
    - Mengembalikan SEMUA data lengkap: gambar tahapan preprocessing (gray, blur,
      otsu), crop tiap biji, K-Means cluster, mold mask, fitur GLCM & HSV, dll.
    - Response-nya lebih besar, tapi hanya dipanggil SEKALI saat kamera berhenti
      (bukan tiap detik seperti /api/detect/frame).

    Pakai 'def' biasa (bukan async def) supaya FastAPI menjalankannya di thread pool,
    sama seperti /api/detect/frame -- alasannya sama: kerja berat CPU.
    """
    b64_clean = strip_base64_prefix(req.image)
    try:
        img_bytes = base64.b64decode(b64_clean)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image data")

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Failed to decode the image frame")

    # BARU -- frame terakhir saat kamera dihentikan ikut masuk staging
    return run_detection(img, staging_source="webcam")


# ================================================================
# BARU -- Crowdsourcing Dataset & Continuous Training
# ================================================================

class FeedbackRequest(BaseModel):
    id: str
    is_correct: bool
    suggested_label: Optional[str] = None


@app.post("/api/staging/feedback")
def staging_feedback(req: FeedbackRequest):
    """Feedback opsional dari user biasa (tanpa login) tentang hasil deteksinya."""
    if not crowdsource.submit_user_feedback(req.id, req.is_correct, req.suggested_label):
        raise HTTPException(status_code=404, detail="Staging item not found or invalid label")
    return {"status": "success"}


# ---------- Autentikasi admin ----------
class LoginRequest(BaseModel):
    password: str


def require_admin(token: Optional[str]):
    if not admin_auth.is_valid(token):
        raise HTTPException(status_code=401, detail="Admin login required")


@app.post("/api/admin/login")
def admin_login(req: LoginRequest):
    if not admin_auth.enabled:
        raise HTTPException(
            status_code=503,
            detail="Admin login is disabled: ADMIN_PASSWORD is not configured in the server's .env file",
        )
    token = admin_auth.login(req.password)
    if not token:
        raise HTTPException(status_code=401, detail="Wrong admin password")
    return {"status": "success", "token": token}


@app.post("/api/admin/logout")
def admin_logout(x_admin_token: Optional[str] = Header(None)):
    admin_auth.logout(x_admin_token)
    return {"status": "success"}


@app.get("/api/admin/check")
def admin_check(x_admin_token: Optional[str] = Header(None)):
    return {"valid": admin_auth.is_valid(x_admin_token)}


# ---------- Kurasi staging ----------
@app.get("/api/admin/stats")
def admin_stats(x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    return crowdsource.get_dataset_stats()


@app.get("/api/admin/staging")
def admin_staging(x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    return {"items": crowdsource.list_staging()}


@app.get("/api/admin/staging/image/{category}/{item_id}")
def admin_staging_image(
    category: str,
    item_id: str,
    kind: str = "original",
    token: Optional[str] = Query(None),
    x_admin_token: Optional[str] = Header(None),
):
    # <img src> tidak bisa mengirim header, jadi token juga diterima via query ?token=
    require_admin(x_admin_token or token)
    path = crowdsource.staging_image_path(category, item_id, kind)
    if not path:
        raise HTTPException(status_code=404, detail="Image not found")
    return FileResponse(path, media_type="image/jpeg")


class ApproveRequest(BaseModel):
    id: str
    label: str  # label final pilihan admin: "sehat" / "terkontaminasi"


@app.post("/api/admin/staging/approve")
def admin_approve(req: ApproveRequest, x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    try:
        return {"status": "success", **crowdsource.approve_item(req.id, req.label)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


class RejectRequest(BaseModel):
    id: str


@app.post("/api/admin/staging/reject")
def admin_reject(req: RejectRequest, x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    try:
        return {"status": "success", **crowdsource.reject_item(req.id)}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


# ---------- Re-training ----------
@app.post("/api/admin/retrain")
def admin_retrain(x_admin_token: Optional[str] = Header(None)):
    """Memulai re-training di background thread (status dipantau via /api/admin/retrain/status)."""
    require_admin(x_admin_token)
    if not retrain_manager.start():
        raise HTTPException(status_code=409, detail="Re-training is already running")
    return {"status": "started"}


@app.get("/api/admin/retrain/status")
def admin_retrain_status(x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    return retrain_manager.status()


@app.post("/api/admin/rollback")
def admin_rollback(x_admin_token: Optional[str] = Header(None)):
    require_admin(x_admin_token)
    try:
        return {"status": "success", **retrain_manager.rollback()}
    except (RuntimeError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)