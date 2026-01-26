from fastapi import FastAPI, File, UploadFile
import numpy as np

app = FastAPI()

@app.post("/analyze")
async def analyze_media(file: UploadFile = File(...)):
    content = await file.read()
    
    # Dummy AI detection (replace with real model later)
    fake_score = np.random.uniform(0, 1)  # 0 = real, 1 = fake
    verdict = "synthetic" if fake_score > 0.5 else "authentic"

    return {
        "verdict": verdict,
        "confidence": round(fake_score, 2)
    }
