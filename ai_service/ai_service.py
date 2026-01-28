# File: ai_service/ai_service.py (COMPLETE with Enhanced Explainability)
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

# ========== ENHANCED EXPLAINABILITY FUNCTIONS ==========
def generate_detailed_explanations(frame_results: List[Dict], 
                                  temporal_summary: Dict,
                                  visual_score: float,
                                  temporal_score: float,
                                  anomaly_count: int,
                                  final_verdict: str,
                                  final_confidence: float) -> Dict[str, Any]:
    """Generate detailed explanations with confidence breakdown."""
    
    explanations = []
    confidence_breakdown = []
    anomaly_details = []
    frame_markers = []
    key_findings = []
    
    # 1. Visual analysis breakdown
    visual_explanations = []
    if frame_results:
        # Get metrics from first frame
        first_frame = frame_results[0] if frame_results else {}
        visual_metrics = first_frame.get('visual_metrics', {})
        
        variance = visual_metrics.get('feature_variance', 0)
        entropy = visual_metrics.get('feature_entropy', 0)
        
        if variance > 8:
            visual_explanations.append("Rich texture complexity (natural)")
            key_findings.append("High texture complexity")
        elif variance < 3:
            visual_explanations.append("Low texture complexity (potentially synthetic)")
            key_findings.append("Low texture complexity")
        
        if 3.5 < entropy < 4.5:
            visual_explanations.append("Natural visual entropy")
            key_findings.append("Natural visual patterns")
        elif entropy < 3:
            visual_explanations.append("Simplified visual patterns")
            key_findings.append("Simplified patterns")
    
    # 2. Visual score explanation
    if visual_score > 0.7:
        explanations.append("✅ **High Visual Authenticity**")
        visual_confidence = 0.8
        visual_impact = "positive"
    elif visual_score > 0.5:
        explanations.append("⚠️ **Moderate Visual Authenticity**")
        visual_confidence = 0.6
        visual_impact = "neutral"
    else:
        explanations.append("❌ **Low Visual Authenticity**")
        visual_confidence = 0.3
        visual_impact = "negative"
    
    confidence_breakdown.append({
        "factor": "Visual Analysis",
        "score": float(visual_score),
        "confidence": float(visual_confidence),
        "weight": 40,
        "impact": visual_impact,
        "details": visual_explanations
    })
    
    # 3. Temporal analysis breakdown
    temporal_explanations = []
    stability = temporal_summary.get('stability', 'unknown')
    avg_drift = temporal_summary.get('average_drift', 0)
    
    if stability == "high":
        explanations.append("✅ **Excellent Temporal Consistency**")
        temporal_explanations.append("Smooth frame transitions")
        temporal_confidence = 0.9
        temporal_impact = "positive"
    elif stability == "medium":
        explanations.append("⚠️ **Good Temporal Consistency**")
        temporal_explanations.append("Minor inconsistencies")
        temporal_confidence = 0.7
        temporal_impact = "neutral"
    else:
        explanations.append("❌ **Poor Temporal Consistency**")
        temporal_explanations.append("Unnatural frame transitions")
        temporal_confidence = 0.4
        temporal_impact = "negative"
    
    if avg_drift < 0.05:
        temporal_explanations.append("Minimal frame-to-frame changes")
    elif avg_drift > 0.15:
        temporal_explanations.append("Large frame-to-frame changes detected")
        key_findings.append("High frame drift")
    
    confidence_breakdown.append({
        "factor": "Temporal Analysis",
        "score": float(temporal_score),
        "confidence": float(temporal_confidence),
        "weight": 30,
        "impact": temporal_impact,
        "details": temporal_explanations
    })
    
    # 4. Anomaly detection breakdown
    anomaly_explanations = []
    if anomaly_count == 0:
        explanations.append("✅ **No Anomalies Detected**")
        anomaly_confidence = 1.0
        anomaly_impact = "positive"
        anomaly_explanations.append("Consistent throughout video")
    elif anomaly_count == 1:
        explanations.append("⚠️ **1 Anomaly Detected**")
        anomaly_confidence = 0.7
        anomaly_impact = "neutral"
        anomaly_explanations.append("Minor irregularity found")
    else:
        explanations.append(f"❌ **{anomaly_count} Anomalies Detected**")
        anomaly_confidence = max(0.3, 1.0 - (anomaly_count * 0.1))
        anomaly_impact = "negative"
        anomaly_explanations.append(f"Multiple irregularities ({anomaly_count} total)")
        key_findings.append(f"{anomaly_count} anomalies found")
    
    # Get anomaly timeline
    anomaly_timeline = temporal_summary.get('anomaly_timeline', [])
    for anomaly in anomaly_timeline[:3]:  # Show top 3 anomalies
        anomaly_details.append({
            "frame": anomaly.get('frame_index', 0),
            "reason": anomaly.get('reason', 'unknown').replace('_', ' ').title(),
            "score": float(anomaly.get('drift', 0))
        })
    
    confidence_breakdown.append({
        "factor": "Anomaly Detection",
        "anomaly_count": anomaly_count,
        "confidence": float(anomaly_confidence),
        "weight": 30,
        "impact": anomaly_impact,
        "details": anomaly_explanations
    })
    
    # 5. Frame markers for suspicious frames
    suspicious_threshold = 0.4
    for i, frame in enumerate(frame_results[:20]):  # Check first 20 frames
        score = frame.get('humanity_score', 0)
        if score < suspicious_threshold:
            severity = 'high' if score < 0.3 else 'medium'
            frame_markers.append({
                'frame': i,
                'score': float(score),
                'severity': severity,
                'reason': 'Low visual authenticity score'
            })
    
    # 6. Calculate overall confidence
    overall_confidence = (
        (visual_confidence * 0.4) +
        (temporal_confidence * 0.3) +
        (anomaly_confidence * 0.3)
    )
    
    # 7. Generate recommendation
    recommendation = ""
    if final_verdict == "likely_real" and overall_confidence > 0.8:
        recommendation = "✅ **HIGH CONFIDENCE**: Content appears authentic"
    elif final_verdict == "likely_real":
        recommendation = "✅ **LIKELY AUTHENTIC**: Minor concerns detected"
    elif final_verdict == "needs_review":
        recommendation = "⚠️ **REVIEW RECOMMENDED**: Further verification needed"
    elif final_verdict == "suspicious":
        recommendation = "❌ **SUSPICIOUS**: Strong indicators of manipulation"
    else:
        recommendation = "🔍 **ANALYSIS INCONCLUSIVE**: Insufficient data"
    
    # 8. Generate summary
    summary_parts = []
    if visual_score > 0.7:
        summary_parts.append("high visual authenticity")
    elif visual_score < 0.4:
        summary_parts.append("low visual authenticity")
    
    if stability == "high":
        summary_parts.append("excellent temporal consistency")
    elif stability == "low" or stability == "unstable":
        summary_parts.append("poor temporal consistency")
    
    if anomaly_count > 0:
        summary_parts.append(f"{anomaly_count} anomaly{'s' if anomaly_count != 1 else ''} detected")
    
    summary = "Content shows " + ", ".join(summary_parts) if summary_parts else "Analysis complete"
    
    return {
        "explanations": explanations,
        "confidence_breakdown": confidence_breakdown,
        "anomaly_details": anomaly_details,
        "frame_markers": frame_markers[:5],  # Top 5 suspicious frames
        "key_findings": key_findings,
        "recommendation": recommendation,
        "summary": summary,
        "overall_confidence": float(overall_confidence),
        "calculated_confidence": float(final_confidence)
    }

def generate_audio_explanations(audio_analysis: Dict) -> Dict[str, Any]:
    """Generate detailed audio explanations."""
    if not audio_analysis.get('success', False):
        return {
            "available": False,
            "note": "Audio analysis not available"
        }
    
    explanations = []
    audio_score = audio_analysis.get('authenticity_scores', {}).get('overall_authenticity', 0)
    insights = audio_analysis.get('insights', {})
    
    if audio_score > 0.7:
        explanations.append("✅ **Authentic Audio** - Natural voice characteristics")
    elif audio_score > 0.5:
        explanations.append("⚠️ **Questionable Audio** - Some synthetic characteristics")
    else:
        explanations.append("❌ **Synthetic Audio** - Strong AI indicators")
    
    return {
        "available": True,
        "explanations": explanations,
        "score": float(audio_score),
        "strengths": insights.get('strengths', []),
        "anomalies": insights.get('anomalies', []),
        "recommendations": insights.get('recommendations', [])
    }

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
        
        # Generate ENHANCED explanations with explainability
        explainability = generate_detailed_explanations(
            frame_results=frame_results,
            temporal_summary=temporal_summary,
            visual_score=float(avg_humanity),
            temporal_score=float(temporal_score),
            anomaly_count=anomaly_count,
            final_verdict=final_verdict,
            final_confidence=final_confidence
        )
        
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
            "explainability": explainability  # Enhanced explainability
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
        
        # Add audio explanations
        audio_explanations = generate_audio_explanations(result)
        result['explainability'] = audio_explanations
        
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
        
        # Combine explainability
        combined_explainability = {
            "video": video_response.get('explainability', {}),
            "audio": audio_response.get('explainability', {}) if audio_response else {"available": False},
            "combined_verdict": combined_verdict
        }
        
        response = {
            'success': True,
            'processing_time': time.time() - start_time,
            'video_analysis': video_response,
            'audio_analysis': audio_response if audio_response else {'available': False},
            'combined_verdict': combined_verdict,
            'explainability': combined_explainability,
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
        "features": ["visual_analysis", "temporal_analysis", "audio_analysis", "explainability"],
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
            "Explainability and human-readable insights",
            "Multimodal deepfake detection"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)