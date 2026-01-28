from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import time
from PIL import Image
import io

app = FastAPI(title="ProtocolAura AI Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Utility: real image analysis
# -----------------------------
def analyze_image(image: np.ndarray):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Texture variance (real signal)
    variance = float(np.var(gray))

    # Edge density (real signal)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = float(np.sum(edges > 0) / edges.size)

    humanity_score = min(100.0, (variance / 5000.0) * 100)
    confidence = min(1.0, max(0.3, edge_density * 5))

    return {
        "textureVariance": round(variance, 2),
        "edgeDensity": round(edge_density, 4),
        "humanityScore": round(humanity_score, 2),
        "confidence": round(confidence, 2),
    }

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
def health():
    return {"status": "AI Service running"}

@app.post("/liveness/level2")
async def liveness_level_2(file: UploadFile = File(...)):
    start = time.time()

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    signals = analyze_image(image_np)

    verdict = (
        "likely_real"
        if signals["humanityScore"] > 35 and signals["confidence"] > 0.5
        else "needs_review"
    )

    return {
        "level": 2,
        "verdict": verdict,
        "signals": signals,
        "processingTimeMs": round((time.time() - start) * 1000, 2),
        "timestamp": int(time.time() * 1000),
    }
