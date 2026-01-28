# File: ai_service/ai_service.py (COMPLETE with Audio Analysis)
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io
import logging
from typing import List, Dict, Any, Optional
import time
from scipy import stats

# Import analyzers
from analysis.audio import AudioAnalyzer
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

# Initialize analyzers
audio_analyzer = AudioAnalyzer(sample_rate=22050)

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image: Image.Image) -> np.ndarray:
    """Extract ResNet-18 features."""
    try:
        image_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            features = feature_extractor(image_tensor)
        return features.squeeze().cpu().numpy()
    except Exception as e:
        logger.error(f"Feature extraction error: {e}")
        raise

def compute_visual_metrics(features: np.ndarray) -> Dict[str, float]:
    """Compute visual metrics from features."""
    try:
        # Feature variance (texture complexity)
        feature_variance = np.var(features)
        
        # Entropy of feature distribution
        hist, _ = np.histogram(features, bins=50, density=True)
        hist = hist[hist > 0]
        entropy = -np.sum(hist * np.log(hist))
        
        # Feature statistics
        feature_mean = np.mean(features)
        feature_std = np.std(features)
        
        # Kurtosis and skewness (distribution shape)
        kurtosis_value = float(stats.kurtosis(features.flatten()))
        skewness_value = float(stats.skew(features.flatten()))
        
        return {
            'feature_variance': float(feature_variance),
            'feature_entropy': float(entropy),
            'feature_mean': float(feature_mean),
            'feature_std': float(feature_std),
            'kurtosis': kurtosis_value,
            'skewness': skewness_value
        }
    except Exception as e:
        logger.error(f"Metrics computation error: {e}")
        return {
            'feature_variance': 0.0,
            'feature_entropy': 0.0,
            'feature_mean': 0.0,
            'feature_std': 0.0,
            'kurtosis': 0.0,
            'skewness': 0.0
        }

def calculate_humanity_score(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate humanity score based on visual metrics."""
    try:
        # Real metrics-based scoring (no randomness)
        
        # Higher variance typically indicates more natural textures
        variance_score = min(metrics['feature_variance'] / 10.0, 1.0)
        
        # Moderate entropy indicates natural complexity
        entropy = metrics['feature_entropy']
        entropy_score = 1.0 - abs(entropy - 4.0) / 4.0  # Target entropy around 4
        entropy_score = max(0.0, min(1.0, entropy_score))
        
        # Natural images have moderate kurtosis
        kurtosis = abs(metrics['kurtosis'])
        kurtosis_score = 1.0 - min(kurtosis / 10.0, 1.0)
        
        # Combined score (weighted average)
        humanity_score = (
            0.4 * variance_score +
            0.4 * entropy_score +
            0.2 * kurtosis_score
        )
        
        # Calculate confidence based on metric consistency
        confidence = min(1.0, humanity_score * 1.2)
        
        return {
            'humanity_score': float(humanity_score),
            'confidence': float(confidence)
        }
    except Exception as e:
        logger.error(f"Score calculation error: {e}")
        return {'humanity_score': 0.5, 'confidence': 0.5}

def generate_explanations(frame_results: List[Dict], temporal_summary: Dict) -> List[str]:
    """Generate human-readable explanations."""
    factors = []
    
    # Visual metrics explanations
    if frame_results:
        visual_scores = [r['humanity_score'] for r in frame_results if 'humanity_score' in r]
        avg_visual = np.mean(visual_scores) if visual_scores else 0
        
        if avg_visual > 0.7:
            factors.append("High visual authenticity")
        elif avg_visual > 0.5:
            factors.append("Moderate visual authenticity")
        else:
            factors.append("Low visual authenticity")
    
    # Temporal explanations
    if temporal_summary:
        stability = temporal_summary.get('stability', 'unknown')
        if stability == "high":
            factors.append("Excellent temporal consistency")
        elif stability == "medium":
            factors.append("Good temporal consistency")
        elif stability == "low":
            factors.append("Poor temporal consistency")
        else:
            factors.append("Unstable temporal patterns")
        
        # Anomaly explanations
        anomaly_count = temporal_summary.get('anomaly_count', 0)
        if anomaly_count == 0:
            factors.append("No temporal anomalies detected")
        else:
            factors.append(f"{anomaly_count} temporal anomalies detected")
        
        abrupt_changes = any(
            frame.get('temporal_metrics', {}).get('is_abrupt_change', False)
            for frame in frame_results
        )
        if abrupt_changes:
            factors.append("Contains abrupt visual changes")
    
    return factors if factors else ["Insufficient data for detailed analysis"]

def generate_recommendation(final_verdict: str, temporal_summary: Dict) -> str:
    """Generate recommendation based on analysis."""
    if final_verdict == "likely_real":
        return "Content appears authentic with high confidence"
    elif final_verdict == "needs_review":
        anomaly_count = temporal_summary.get('anomaly_count', 0)
        if anomaly_count > 0:
            return "Review recommended due to detected anomalies"
        else:
            return "Review recommended for additional verification"
    else:
        anomaly_count = temporal_summary.get('anomaly_count', 0)
        if anomaly_count > 3:
            return "High suspicion: Multiple temporal anomalies detected"
        else:
            return "Content shows signs of manipulation"

def combine_verdicts(video_response: Dict, audio_response: Optional[Dict]) -> Dict:
    """Combine video and audio analysis for final verdict."""
    video_verdict = video_response.get('aggregated_metrics', {}).get('final_verdict', 'unknown')
    video_confidence = video_response.get('aggregated_metrics', {}).get('final_confidence', 0.5)
    
    if not audio_response or not audio_response.get('success', False):
        return {
            'verdict': video_verdict,
            'confidence': video_confidence,
            'sources': ['video_only'],
            'notes': 'Audio analysis not available'
        }
    
    audio_verdict = audio_response.get('final_audio_verdict', 'unknown')
    audio_confidence = audio_response.get('confidence', 0.5)
    
    # Combine weights: 70% video, 30% audio
    combined_confidence = (video_confidence * 0.7) + (audio_confidence * 0.3)
    
    # Determine combined verdict
    if video_verdict == 'likely_real' and audio_verdict == 'likely_authentic':
        verdict = 'highly_authentic'
    elif video_verdict == 'suspicious' or audio_verdict == 'suspicious':
        verdict = 'suspicious'
    elif video_verdict == 'needs_review' or audio_verdict == 'needs_review':
        verdict = 'needs_review'
    else:
        verdict = 'likely_authentic'
    
    return {
        'verdict': verdict,
        'confidence': combined_confidence,
        'sources': ['video', 'audio'],
        'video_contribution': video_confidence,
        'audio_contribution': audio_confidence
    }

# ========== VIDEO ANALYSIS ENDPOINT ==========
@app.post("/liveness/level2")
async def analyze_image_frames(files: List[UploadFile] = File(...)):
    """Analyze multiple frames with temporal consistency."""
    try:
        start_time = time.time()
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided")
        
        frame_results = []
        temporal_analyzer = TemporalAnalyzer(window_size=10)
        
        logger.info(f"Processing {len(files)} frames with temporal analysis")
        
        for idx, file in enumerate(files):
            try:
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
                if temporal_metrics.get('has_anomaly', False):
                    scores['humanity_score'] *= 0.8
                    scores['confidence'] *= 0.9
                
                # Determine verdict
                humanity_score = scores['humanity_score']
                if humanity_score > 0.7 and not temporal_metrics.get('has_anomaly', False):
                    verdict = "likely_real"
                elif humanity_score > 0.4:
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
                
            except Exception as e:
                logger.error(f"Error processing frame {idx}: {e}")
                continue
        
        # Get temporal summary
        temporal_summary = temporal_analyzer.get_summary()
        
        # Calculate final scores
        if frame_results:
            humanity_scores = [r['humanity_score'] for r in frame_results]
            confidence_scores = [r['confidence'] for r in frame_results]
            
            avg_humanity = np.mean(humanity_scores) if humanity_scores else 0
            avg_confidence = np.mean(confidence_scores) if confidence_scores else 0
            
            # Adjust with temporal consistency
            temporal_score = temporal_summary.get('temporal_consistency_score', 0.5)
            anomaly_count = temporal_summary.get('anomaly_count', 0)
            
            # Penalize for anomalies
            anomaly_penalty = min(anomaly_count * 0.1, 0.3)
            temporal_factor = temporal_score * (1 - anomaly_penalty)
            
            # Combined score: 70% visual, 30% temporal
            final_humanity = (avg_humanity * 0.7) + (temporal_factor * 0.3)
            final_humanity = max(0.0, min(1.0, final_humanity))
            
            # Determine final verdict
            if final_humanity > 0.75 and anomaly_count == 0 and temporal_score > 0.7:
                final_verdict = "likely_real"
            elif final_humanity > 0.6 and anomaly_count <= 1 and temporal_score > 0.5:
                final_verdict = "likely_real"
            elif final_humanity > 0.4:
                final_verdict = "needs_review"
            else:
                final_verdict = "suspicious"
                
            final_confidence = (avg_confidence * 0.7) + (temporal_score * 0.3)
            final_confidence = max(0.0, min(1.0, final_confidence))
            
        else:
            avg_humanity = 0.0
            avg_confidence = 0.0
            final_humanity = 0.0
            final_confidence = 0.0
            final_verdict = "error"
        
        # Generate explanations
        key_factors = generate_explanations(frame_results, temporal_summary)
        recommendation = generate_recommendation(final_verdict, temporal_summary)
        
        processing_time = time.time() - start_time
        
        # Build response
        response = {
            "success": True,
            "processing_time": processing_time,
            "frame_count": len(frame_results),
            "aggregated_metrics": {
                "avg_humanity_score": float(avg_humanity),
                "avg_confidence": float(avg_confidence),
                "final_humanity_score": float(final_humanity),
                "final_verdict": final_verdict,
                "final_confidence": float(final_confidence)
            },
            "temporal_analysis": temporal_summary,
            "frame_details": frame_results[:10],  # Limit to first 10 frames
            "explainability": {
                "key_factors": key_factors,
                "recommendation": recommendation
            }
        }
        
        logger.info(f"Video analysis completed in {processing_time:.2f}s: {final_verdict}")
        return response
        
    except Exception as e:
        logger.error(f"Video analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== AUDIO ANALYSIS ENDPOINT ==========
@app.post("/audio/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """Analyze audio file for authenticity."""
    try:
        start_time = time.time()
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Analyze audio
        result = audio_analyzer.analyze_audio(tmp_path)
        result['processing_time'] = time.time() - start_time
        
        # Cleanup
        os.unlink(tmp_path)
        
        logger.info(f"Audio analysis completed in {result['processing_time']:.2f}s")
        return result
        
    except Exception as e:
        logger.error(f"Audio analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== COMBINED ANALYSIS ENDPOINT ==========
@app.post("/combined/analyze")
async def analyze_combined(
    video_frames: List[UploadFile] = File(...),
    audio_file: Optional[UploadFile] = File(None)
):
    """Combined analysis of video frames and audio."""
    try:
        start_time = time.time()
        
        # Analyze video frames
        video_response = await analyze_image_frames(video_frames)
        
        # Analyze audio if provided
        audio_response = None
        if audio_file:
            audio_response = await analyze_audio(audio_file)
        
        # Combine results
        combined_verdict = combine_verdicts(video_response, audio_response)
        
        response = {
            'success': True,
            'processing_time': time.time() - start_time,
            'video_analysis': video_response,
            'audio_analysis': audio_response if audio_response else {'available': False},
            'combined_verdict': combined_verdict,
            'multimodal_analysis': True,
            'modes_analyzed': ['visual', 'temporal'] + (['audio'] if audio_response else [])
        }
        
        logger.info(f"Combined analysis completed in {response['processing_time']:.2f}s")
        return response
        
    except Exception as e:
        logger.error(f"Combined analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ========== HEALTH CHECK ==========
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "Protocol Aura AI Service",
        "version": "2.0.0",
        "features": ["visual_analysis", "temporal_analysis", "audio_analysis"],
        "timestamp": time.time()
    }

# ========== ROOT ENDPOINT ==========
@app.get("/")
async def root():
    """Root endpoint with service information."""
    return {
        "message": "Protocol Aura AI Service",
        "version": "2.0.0",
        "endpoints": {
            "/liveness/level2": "POST - Analyze video frames with temporal consistency",
            "/audio/analyze": "POST - Analyze audio for authenticity",
            "/combined/analyze": "POST - Combined video + audio analysis",
            "/health": "GET - Health check",
            "/": "GET - This information"
        },
        "features": [
            "Visual authenticity detection",
            "Temporal consistency analysis",
            "Audio authenticity verification",
            "Multimodal deepfake detection"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)