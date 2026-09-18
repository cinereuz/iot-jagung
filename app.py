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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8080, reload=True)