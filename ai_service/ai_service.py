# File: ai_service/ai_service.py (COMPLETE - NO TORCH VERSION)
import tempfile
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from PIL import Image, ImageFilter
import io
import logging
from typing import List, Dict, Any, Optional
import time
from scipy import stats
import cv2
from skimage.feature import local_binary_pattern

# Import analyzers
from analysis.audio import AudioAnalyzer
from analysis.temporal import TemporalAnalyzer
from analysis.deepfake_detector import get_deepfake_detector

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

# Initialize analyzers
audio_analyzer = AudioAnalyzer(sample_rate=22050)
deepfake_detector = get_deepfake_detector()

# ========== HANDCRAFTED FEATURE EXTRACTION FUNCTIONS ==========
def extract_handcrafted_features(image: Image.Image) -> np.ndarray:
    """Extract handcrafted visual features without PyTorch."""
    try:
        # Convert PIL Image to grayscale numpy array for most analyses
        img_gray_uint8 = np.array(image.convert('L')).astype(np.uint8)
        img_gray = img_gray_uint8.astype(np.float32) / 255.0
        
        # Calculate multiple feature types
        features = []
        
        # 1. BASIC STATISTICS
        features.extend([
            np.mean(img_gray),           # Brightness
            np.std(img_gray),            # Contrast
            np.var(img_gray),            # Variance
            stats.skew(img_gray.flatten()),   # Distribution symmetry
            stats.kurtosis(img_gray.flatten()) # Distribution tails
        ])
        
        # 2. EDGE FEATURES (Canny edge detection)
        edges = cv2.Canny(img_gray_uint8, 50, 150)
        edges_norm = edges.astype(np.float32) / 255.0
        features.extend([
            np.mean(edges_norm),                     # Edge strength
            np.var(edges_norm),                      # Edge consistency
            np.sum(edges > 0) / edges.size           # Edge density (0-1)
        ])
        
        # 3. TEXTURE FEATURES (Local Binary Patterns)
        radius = 2
        n_points = 8 * radius
        lbp = local_binary_pattern(img_gray_uint8, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), density=True)
        features.extend(hist[:5])  # Use first 5 histogram bins as texture features
        
        # 4. FREQUENCY DOMAIN FEATURES (FFT analysis)
        fft = np.fft.fft2(img_gray)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.log(np.abs(fft_shift) + 1)
        features.extend([
            np.mean(magnitude_spectrum),
            np.std(magnitude_spectrum),
            np.var(magnitude_spectrum)
        ])
        
        # 5. COLOR FEATURES (if color image)
        if image.mode == 'RGB':
            img_color = np.array(image).astype(np.float32) / 255.0
            # Color variance in each channel
            for channel in range(3):
                features.append(np.var(img_color[:, :, channel]))
        
        return np.array(features, dtype=np.float32)
        
    except Exception as e:
        logger.error(f"Handcrafted feature extraction error: {e}")
        # Return consistent zero features
        return np.zeros(19, dtype=np.float32)

def compute_handcrafted_metrics(features: np.ndarray) -> Dict[str, float]:
    """Compute visual metrics from handcrafted features."""
    try:
        # Extract specific metrics from our feature vector
        # Features structure: [0:4] basic stats, [5:7] edge features, 
        # [8:12] texture, [13:15] frequency, [16:19] color (if available)
        
        return {
            'feature_variance': float(np.var(features)),
            'feature_entropy': float(stats.entropy(np.histogram(features, bins=20)[0] + 1e-10)),
            'feature_mean': float(np.mean(features)),
            'feature_std': float(np.std(features)),
            'edge_density': float(features[7] if len(features) > 7 else 0),
            'texture_complexity': float(np.mean(features[8:12]) if len(features) > 12 else 0),
            'frequency_energy': float(features[13] if len(features) > 13 else 0),
            'color_variance': float(np.mean(features[16:]) if len(features) > 16 else 0)
        }
    except Exception as e:
        logger.error(f"Handcrafted metrics computation error: {e}")
        return {
            'feature_variance': 0.0,
            'feature_entropy': 0.0,
            'feature_mean': 0.0,
            'feature_std': 0.0,
            'edge_density': 0.0,
            'texture_complexity': 0.0,
            'frequency_energy': 0.0,
            'color_variance': 0.0
        }

def calculate_handcrafted_humanity_score(metrics: Dict[str, float]) -> Dict[str, float]:
    """Calculate humanity score based on handcrafted visual metrics."""
    try:
        # Score components from handcrafted metrics - RELAXED thresholds for real videos
        
        # 1. Texture variance score - natural images have moderate variance
        variance = metrics['feature_variance']
        # Much wider acceptable range
        if variance > 0.01:
            variance_score = min(variance * 5.0, 1.0)  # Reward any reasonable variance
        else:
            variance_score = 0.5
        
        # 2. Edge density score - natural images have organic edge patterns
        edge_density = metrics['edge_density']
        # Accept wide range, just reward presence of edges
        if edge_density > 0.02:
            edge_score = min(0.5 + (edge_density * 0.5), 1.0)
        else:
            edge_score = 0.4
        
        # 3. Texture complexity score
        texture = metrics['texture_complexity']
        texture_score = min(0.5 + (texture * 2.0), 1.0)  # Base score + bonus
        
        # 4. Frequency distribution score - accept wide range
        freq = metrics['frequency_energy']
        if freq > 0.5:
            freq_score = min(freq / 5.0 + 0.5, 1.0)
        else:
            freq_score = 0.6
        
        # 5. Entropy score - natural images have higher entropy
        entropy = metrics['feature_entropy']
        entropy_score = min(0.4 + (entropy / 4.0), 1.0)  # Base + scaled bonus
        
        # Combined weighted score - reduced weight on edge features
        humanity_score = (
            0.30 * variance_score +
            0.10 * edge_score +      # Reduced from 0.20
            0.25 * texture_score +
            0.15 * freq_score +
            0.20 * entropy_score
        )
        
        # Clamp to 0-1 range
        humanity_score = max(0.0, min(1.0, humanity_score))
        
        # Calculate confidence based on metric consistency
        scores = [variance_score, edge_score, texture_score, freq_score, entropy_score]
        confidence = 0.5 + (0.5 * (1.0 - np.std(scores)))  # Higher confidence if scores agree
        
        return {
            'humanity_score': float(humanity_score),
            'confidence': float(confidence)
        }
    except Exception as e:
        logger.error(f"Handcrafted score calculation error: {e}")
        return {'humanity_score': 0.5, 'confidence': 0.5}

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
        edge_density = visual_metrics.get('edge_density', 0)
        
        if variance > 0.3:
            visual_explanations.append("Rich texture complexity (natural)")
            key_findings.append("High texture complexity")
        elif variance < 0.1:
            visual_explanations.append("Low texture complexity (potentially synthetic)")
            key_findings.append("Low texture complexity")
        
        if entropy > 2.0:
            visual_explanations.append("Natural visual entropy")
            key_findings.append("Natural visual patterns")
        elif entropy < 1.0:
            visual_explanations.append("Simplified visual patterns")
            key_findings.append("Simplified patterns")
            
        if edge_density > 0.2:
            visual_explanations.append("High edge detail (natural)")
            key_findings.append("Detailed edges")
        elif edge_density < 0.05:
            visual_explanations.append("Low edge detail (possibly smoothed)")
            key_findings.append("Low edge detail")
    
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
    
    # 7. Generate recommendation with actionable guidance
    recommendation = ""
    next_steps = []
    
    if final_verdict == "likely_real" and overall_confidence > 0.8:
        recommendation = "✅ **HIGH CONFIDENCE AUTHENTIC**: Content appears genuine"
        next_steps = ["Content likely authentic", "Can be used with confidence"]
    elif final_verdict == "likely_real":
        recommendation = "✅ **LIKELY AUTHENTIC**: Content appears genuine with minor inconsistencies"
        next_steps = ["Check context and source", "Look for corroborating evidence"]
    elif final_verdict == "needs_review":
        recommendation = "⚠️ **INCONCLUSIVE**: AI cannot determine authenticity with confidence"
        # Add specific guidance based on what triggered the need for review
        if visual_score < 0.5 and temporal_score < 0.5:
            next_steps = ["Submit for expert forensic analysis", "Both visual and temporal metrics inconclusive"]
        elif visual_score < 0.5:
            next_steps = ["Expert review recommended", "Visual analysis shows inconsistencies"]
        elif temporal_score < 0.5:
            next_steps = ["Expert review recommended", "Frame transitions show irregularities"]
        else:
            next_steps = ["Obtain additional context", "Verify through independent sources"]
    elif final_verdict == "suspicious":
        recommendation = "❌ **LIKELY SYNTHETIC**: Strong indicators of manipulation detected"
        next_steps = ["Treat content as potentially manipulated", "Verify through original source"]
    elif final_verdict == "synthetic" or final_verdict == "likely_synthetic":
        recommendation = "❌ **SYNTHETIC DETECTED**: Content appears AI-generated or heavily manipulated"
        next_steps = ["Content flagged as likely synthetic", "Do not use for verification purposes"]
    elif final_verdict == "authentic":
        recommendation = "✅ **AUTHENTIC**: High confidence - content is genuine"
        next_steps = ["Content verified as authentic", "Safe to use and share"]
    else:
        recommendation = "🔍 **ANALYSIS INCONCLUSIVE**: Insufficient data for determination"
        next_steps = ["Insufficient data", "Please provide clearer video"]
    
    
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
        "next_steps": next_steps,
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

def combine_verdicts(video_response: Dict, audio_response: Optional[Dict]) -> Dict:
    """Combine video and audio analysis for final verdict - SIMPLIFIED & PRACTICAL."""
    video_verdict = video_response.get('aggregated_metrics', {}).get('final_verdict', 'unknown')
    video_confidence = video_response.get('aggregated_metrics', {}).get('final_confidence', 0.5)
    temporal_score = video_response.get('temporal_analysis', {}).get('temporal_consistency_score', 0.5)
    
    if not audio_response or not audio_response.get('success', False):
        # Video-only: use video verdict directly
        return {
            'verdict': video_verdict,
            'confidence': video_confidence * 0.85,
            'sources': ['video_only'],
            'notes': 'Audio analysis not available'
        }
    
    audio_verdict = audio_response.get('final_audio_verdict', 'unknown')
    audio_confidence = audio_response.get('confidence', 0.5)
    
    combined_confidence = (video_confidence * 0.7) + (audio_confidence * 0.3)
    
    # SIMPLIFIED LOGIC:
    # Authentic: Good video + audio not strongly suspicious
    # Synthetic: Bad video AND audio suspicious OR (perfect temporal + audio suspicious)
    # Needs review: Everything else (be honest about limits)
    
    # ===== AUTHENTIC PATH =====
    if video_verdict == 'likely_real' and audio_confidence < 0.45:
        # Good video + audio not strongly against it
        verdict = 'likely_authentic'
    
    # ===== SYNTHETIC PATH =====
    # Both bad: video says suspicious AND audio says suspicious
    elif video_verdict == 'suspicious' and audio_verdict == 'suspicious':
        verdict = 'likely_synthetic'
    
    # Perfect temporal (AI indicator) + suspicious audio
    elif temporal_score > 0.95 and audio_verdict == 'suspicious':
        verdict = 'suspicious'  # Likely AI-generated
    
    # Audio very confident about synthesis (> 0.55)
    elif audio_verdict == 'suspicious' and audio_confidence > 0.55:
        verdict = 'likely_synthetic'
    
    # ===== NEEDS REVIEW PATH (EVERYTHING ELSE) =====
    else:
        verdict = 'needs_review'
    
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
                
                # Extract handcrafted features (REPLACED PyTorch)
                features = extract_handcrafted_features(image)
                
                # Add to temporal analyzer
                temporal_analyzer.add_frame(features)
                
                # Get temporal analysis
                temporal_metrics = temporal_analyzer.analyze_current()
                
                # Compute visual metrics from handcrafted features
                visual_metrics = compute_handcrafted_metrics(features)
                
                # Calculate humanity score from handcrafted metrics
                scores = calculate_handcrafted_humanity_score(visual_metrics)
                
                # GET DEEP LEARNING PREDICTION (NEW)
                frame_array = np.array(image)
                dl_analysis = deepfake_detector.analyze_frame_ensemble(frame_array)
                dl_deepfake_score = dl_analysis.get('ensemble_score', 0.5)
                
                # Adjust handcrafted score based on deep learning signal
                # If DL thinks it's deepfake (score > 0.6), lower the humanity score
                if dl_deepfake_score > 0.65:
                    scores['humanity_score'] *= 0.85  # Reduce by 15%
                elif dl_deepfake_score > 0.55:
                    scores['humanity_score'] *= 0.95  # Reduce by 5%
                
                # Adjust confidence if DL is very confident
                if dl_deepfake_score < 0.35:
                    scores['confidence'] = max(scores['confidence'], 0.85)  # High confidence in authenticity
                
                # Adjust score based on temporal anomalies
                if temporal_metrics.get('has_anomaly', False):
                    scores['humanity_score'] *= 0.9
                    scores['confidence'] *= 0.95
                
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
                    'temporal_metrics': temporal_metrics,
                    'deepfake_analysis': dl_analysis  # Include DL results
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
            anomaly_penalty = min(anomaly_count * 0.05, 0.2)
            temporal_factor = temporal_score * (1 - anomaly_penalty)
            
            # Combined score: 70% visual, 30% temporal
            final_humanity = (avg_humanity * 0.7) + (temporal_factor * 0.3)
            final_humanity = max(0.0, min(1.0, final_humanity))
            
            # Determine final verdict - SIMPLE & PRACTICAL
            # Good video (>0.65) = likely_real
            # Bad video (<0.40) = suspicious
            # Middle = needs_review (be honest)
            
            if final_humanity > 0.65 and temporal_score > 0.75:
                final_verdict = "likely_real"  # Good video + reasonable temporal
            elif final_humanity < 0.40:
                final_verdict = "suspicious"  # Clearly low score
            else:
                final_verdict = "needs_review"  # Middle ground - inconclusive
                
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