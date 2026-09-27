import os
import io
import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from src.pipeline import CornDiseasePipeline

app = FastAPI(title="Corn Kernel Mold Detector")

# Mount static and templates
os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Initialize pipeline
pipeline = CornDiseasePipeline()

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
        files = sorted(os.listdir(sehat_dir))[:6]
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append({
                    "id": f"sehat_{f}",
                    "name": f"Biji Sehat ({f})",
                    "category": "sehat",
                    "path": os.path.join(sehat_dir, f)
                })

    # Moldy samples
    kontam_dir = "data/raw/terkontaminasi"
    if os.path.exists(kontam_dir):
        files = sorted(os.listdir(kontam_dir))[:6]
        for f in files:
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                samples.append({
                    "id": f"kontam_{f}",
                    "name": f"Biji Berjamur ({f})",
                    "category": "terkontaminasi",
                    "path": os.path.join(kontam_dir, f)
                })

    return {"samples": samples}

class SampleRequest(BaseModel):
    sample_path: str

@app.post("/api/detect/sample")
async def detect_sample(req: SampleRequest):
    if not os.path.exists(req.sample_path):
        raise HTTPException(status_code=404, detail="File sampel tidak ditemukan")

    img = cv2.imread(req.sample_path)
    if img is None:
        raise HTTPException(status_code=400, detail="Gagal membaca citra sampel")

    return run_detection(img)

@app.post("/api/detect/upload")
async def detect_upload(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Format file citra tidak valid atau rusak")

    return run_detection(img)

def run_detection(img):
    """Runs pipeline and formats JSON response with base64 images."""
    try:
        results = pipeline.process_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pemrosesan: {str(e)}")

    # Format images to base64
    images_b64 = {
        "original": mat_to_base64(results["images"]["original"]),
        "gray": mat_to_base64(results["images"]["gray"]),
        "blur": mat_to_base64(results["images"]["blur"]),
        "otsu_mask": mat_to_base64(results["images"]["otsu_mask"]),
        "annotated": mat_to_base64(results["images"]["annotated"])
    }

    # Format individual kernel images
    kernels_formatted = []
    for k in results["kernels"]:
        k_dict = {
            "kernel_id": k["kernel_id"],
            "bbox": k["bbox"],
            "prediction": k["prediction"],
            "is_moldy": k["is_moldy"],
            "is_corn": k.get("is_corn"),  # BARU -- untuk debugging & tampilan frontend nanti
            "confidence": k["confidence"],
            "mold_ratio": k["mold_ratio"],
            "probabilities": k["probabilities"],
            "features": k["features"],
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
        "images": images_b64,
        "kernels": kernels_formatted
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
    - TIDAK mengirim balik crop_img / kmeans_img / mold_mask per biji -- kalau
      difoto ada 10 biji, itu berarti 30 gambar base64 tambahan tiap detik.
      Itu akan membuat response jadi besar sekali dan bikin lag.
    - Hasilnya: response JSON jauh lebih kecil & lebih cepat, cocok dipanggil
      berulang-ulang tiap detik.
    """
    try:
        results = pipeline.process_image(img)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error pemrosesan: {str(e)}")

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
        raise HTTPException(status_code=400, detail="Data gambar base64 tidak valid")

    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        raise HTTPException(status_code=400, detail="Gagal decode frame gambar")

    return run_detection_lightweight(img)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)