from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np
import io
import time

app = FastAPI(title="ProtocolAura AI Service")

# -----------------------------
# CORS
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Image analysis (REAL math)
# -----------------------------
def analyze_image(image: Image.Image):
    img = image.resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0

    gray = np.mean(arr, axis=2)

    texture_variance = float(np.var(gray))
    edge_energy = float(np.mean(np.abs(np.diff(gray, axis=0))))
    entropy = float(-np.mean(gray * np.log(gray + 1e-6)))

    humanity_score = min(100, texture_variance * 5000)
    confidence = min(1.0, max(0.3, entropy))

    return {
        "humanityScore": round(humanity_score, 2),
        "confidence": round(confidence, 2),
        "textureVariance": round(texture_variance, 6),
        "edgeEnergy": round(edge_energy, 6),
    }

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "AI Service running"}

@app.post("/liveness/level2")
async def liveness_level_2(file: UploadFile = File(...)):
    start_time = time.time()

    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    signals = analyze_image(image)

    verdict = (
        "likely_real"
        if signals["humanityScore"] > 40 and signals["confidence"] > 0.5
        else "needs_review"
    )

    return {
        "level": 2,
        "verdict": verdict,
        "signals": signals,
        "processingTimeMs": round((time.time() - start_time) * 1000, 2),
        "timestamp": int(time.time() * 1000),
    }
