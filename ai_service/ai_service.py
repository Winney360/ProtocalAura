# File: ai_service/ai_service.py (UPDATED)
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
from typing import List

# Import temporal analyzer
from analysis.temporal import TemporalAnalyzer

app = FastAPI(title="Protocol Aura AI Service")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize models
device = torch.device("cpu")
resnet = models.resnet18(pretrained=True)
resnet.eval()
resnet.to(device)
feature_extractor = torch.nn.Sequential(*list(resnet.children())[:-1])

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image: Image.Image) -> np.ndarray:
    """Extract ResNet-18 features."""
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feature_extractor(image_tensor)
    return features.squeeze().cpu().numpy()

def compute_visual_metrics(features: np.ndarray) -> Dict:
    """Compute visual metrics (existing function)."""
    feature_variance = np.var(features)
    hist, _ = np.histogram(features, bins=50, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist))
    
    return {
        'feature_variance': float(feature_variance),
        'feature_entropy': float(entropy),
        'feature_mean': float(np.mean(features)),
        'feature_std': float(np.std(features))
    }

def calculate_humanity_score(metrics: Dict) -> Dict:
    """Calculate humanity score (existing function)."""
    variance_score = min(metrics['feature_variance'] / 10.0, 1.0)
    entropy = metrics['feature_entropy']
    entropy_score = 1.0 - abs(entropy - 4.0) / 4.0
    entropy_score = max(0.0, min(1.0, entropy_score))
    
    humanity_score = (0.5 * variance_score) + (0.5 * entropy_score)
    confidence = min(1.0, humanity_score * 1.2)
    
    return {
        'humanity_score': float(humanity_score),
        'confidence': float(confidence)
    }

@app.post("/liveness/level2")
async def analyze_image_frames(files: List[UploadFile] = File(...)):
    """Analyze multiple frames with temporal consistency."""
    try:
        frame_results = []
        temporal_analyzer = TemporalAnalyzer(window_size=10)
        
        for idx, file in enumerate(files):
            # Read image
            contents = await file.read()
            image = Image.open(io.BytesIO(contents)).convert('RGB')
            
            # Extract features
            features = extract_features(image)
            
            # Add to temporal analyzer
            temporal_analyzer.add_frame(features)
            
            # Get temporal analysis
            temporal_metrics = temporal_analyzer.analyze_current()
            
            # Compute visual metrics
            visual_metrics = compute_visual_metrics(features)
            
            # Calculate humanity score
            scores = calculate_humanity_score(visual_metrics)
            
            # Adjust score based on temporal anomalies
            if temporal_metrics['has_anomaly']:
                scores['humanity_score'] *= 0.8
                scores['confidence'] *= 0.9
            
            # Determine verdict
            if scores['humanity_score'] > 0.7 and not temporal_metrics['has_anomaly']:
                verdict = "likely_real"
            elif scores['humanity_score'] > 0.4:
                verdict = "needs_review"
            else:
                verdict = "suspicious"
            
            frame_results.append({
                'frame_index': idx,
                'humanity_score': scores['humanity_score'],
                'confidence': scores['confidence'],
                'verdict': verdict,
                'visual_metrics': visual_metrics,
                'temporal_metrics': temporal_metrics
            })
        
        # Get temporal summary
        temporal_summary = temporal_analyzer.get_summary()
        
        # Calculate final scores
        if frame_results:
            avg_humanity = np.mean([r['humanity_score'] for r in frame_results])
            avg_confidence = np.mean([r['confidence'] for r in frame_results])
            
            # Adjust with temporal consistency
            temporal_factor = temporal_summary['temporal_consistency_score']
            final_humanity = avg_humanity * (0.7 + 0.3 * temporal_factor)
            
            # Determine final verdict
            if final_humanity > 0.75 and temporal_summary['anomaly_count'] == 0:
                final_verdict = "likely_real"
            elif final_humanity > 0.5 and temporal_summary['anomaly_count'] < 2:
                final_verdict = "needs_review"
            else:
                final_verdict = "suspicious"
        else:
            final_humanity = 0.0
            avg_confidence = 0.0
            final_verdict = "error"
        
        # Response
        response = {
            "success": True,
            "frame_count": len(frame_results),
            "aggregated_metrics": {
                "avg_humanity_score": float(avg_humanity) if frame_results else 0.0,
                "avg_confidence": float(avg_confidence) if frame_results else 0.0,
                "final_humanity_score": float(final_humanity),
                "final_verdict": final_verdict,
                "final_confidence": float(avg_confidence * 0.8 + temporal_summary['temporal_consistency_score'] * 0.2)
            },
            "temporal_analysis": temporal_summary,
            "frame_details": frame_results[:5],  # First 5 frames for brevity
            "explainability": {
                "key_factors": [
                    f"Visual authenticity: {'High' if final_humanity > 0.7 else 'Medium' if final_humanity > 0.5 else 'Low'}",
                    f"Temporal consistency: {temporal_summary['stability'].upper()}",
                    f"Anomalies detected: {temporal_summary['anomaly_count']}"
                ]
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)