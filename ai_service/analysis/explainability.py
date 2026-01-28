# File: ai_service/analysis/explainability.py
"""
Explainability Module for Protocol Aura
Step 6: Human-readable explanations for analysis results
"""

from typing import Dict, List, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ExplainabilityEngine:
    """
    Generates human-readable explanations for analysis results.
    Translates technical metrics into understandable insights.
    """
    
    def __init__(self):
        self.explanation_templates = {
            'visual': {
                'high_variance': "Rich texture details detected (natural images typically have complex textures)",
                'low_variance': "Limited texture variation (AI-generated images often have uniform textures)",
                'high_entropy': "Complex visual patterns detected",
                'low_entropy': "Simplistic visual patterns (common in synthetic images)",
                'natural_kurtosis': "Natural distribution of visual features",
                'abnormal_kurtosis': "Unusual distribution of visual features"
            },
            'temporal': {
                'stable': "Smooth transitions between frames",
                'unstable': "Inconsistent frame-to-frame transitions",
                'abrupt_change': "Sudden visual jump detected at frame {frame}",
                'consistent_embeddings': "Visual features remain consistent over time",
                'drifting_embeddings': "Visual features change unpredictably"
            },
            'audio': {
                'natural_pitch': "Natural pitch variations in speech",
                'robotic_pitch': "Unnaturally consistent pitch (robotic sounding)",
                'good_hnr': "Clear voice harmonics detected",
                'poor_hnr': "Voice lacks natural harmonics",
                'normal_zcr': "Natural speech-silence patterns",
                'abnormal_zcr': "Unusual speech rhythm detected"
            },
            'anomalies': {
                'single_anomaly': "1 unusual frame detected",
                'multiple_anomalies': "{count} unusual frames detected",
                'consecutive_anomalies': "Sustained period of irregularities",
                'temporal_inconsistency': "Inconsistent visual flow throughout video"
            }
        }
    
    def generate_video_explanation(self, 
                                  visual_metrics: Dict,
                                  temporal_metrics: Dict,
                                  frame_results: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive explanation for video analysis.
        
        Args:
            visual_metrics: Aggregated visual metrics
            temporal_metrics: Temporal analysis results
            frame_results: Per-frame analysis results
            
        Returns:
            Dictionary with explanations, scores, and recommendations
        """
        explanations = []
        confidence_factors = []
        anomaly_details = []
        
        # 1. Visual metrics explanations
        visual_score = visual_metrics.get('avg_humanity_score', 0)
        
        if visual_score > 0.7:
            explanations.append("✅ **High visual authenticity** - Image textures appear natural")
            confidence_factors.append(("Visual authenticity", 0.8))
        elif visual_score > 0.5:
            explanations.append("⚠️ **Moderate visual authenticity** - Some textures appear synthetic")
            confidence_factors.append(("Visual authenticity", 0.6))
        else:
            explanations.append("❌ **Low visual authenticity** - Textures suggest AI generation")
            confidence_factors.append(("Visual authenticity", 0.3))
        
        # 2. Temporal consistency explanations
        temporal_score = temporal_metrics.get('temporal_consistency_score', 0)
        anomaly_count = temporal_metrics.get('anomaly_count', 0)
        
        if temporal_score > 0.8:
            explanations.append("✅ **Excellent temporal consistency** - Smooth video flow")
            confidence_factors.append(("Temporal consistency", 0.9))
        elif temporal_score > 0.6:
            explanations.append("⚠️ **Good temporal consistency** - Minor inconsistencies")
            confidence_factors.append(("Temporal consistency", 0.7))
        else:
            explanations.append("❌ **Poor temporal consistency** - Unnatural frame transitions")
            confidence_factors.append(("Temporal consistency", 0.4))
        
        # 3. Anomaly detection explanations
        if anomaly_count == 0:
            explanations.append("✅ **No anomalies detected** - Consistent throughout")
        elif anomaly_count == 1:
            explanations.append("⚠️ **1 anomaly detected** - Minor irregularity")
        else:
            explanations.append(f"❌ **{anomaly_count} anomalies detected** - Multiple irregularities")
        
        # 4. Identify specific anomaly frames
        for i, frame in enumerate(frame_results[:10]):  # Check first 10 frames
            if frame.get('temporal_metrics', {}).get('has_anomaly', False):
                reason = frame['temporal_metrics'].get('anomaly_reason', 'unknown')
                anomaly_details.append({
                    'frame': i,
                    'timestamp': f"{i/len(frame_results)*100:.1f}%",
                    'reason': self._translate_reason(reason),
                    'visual_score': frame.get('humanity_score', 0)
                })
        
        # 5. Generate frame markers for suspicious frames
        frame_markers = []
        suspicious_threshold = 0.4
        
        for i, frame in enumerate(frame_results):
            score = frame.get('humanity_score', 0)
            if score < suspicious_threshold:
                frame_markers.append({
                    'frame': i,
                    'score': score,
                    'reason': 'Low visual authenticity',
                    'severity': 'high' if score < 0.3 else 'medium'
                })
        
        # 6. Calculate overall confidence breakdown
        confidence_breakdown = self._calculate_confidence_breakdown(
            visual_score, temporal_score, anomaly_count
        )
        
        # 7. Generate recommendation
        recommendation = self._generate_recommendation(
            visual_score, temporal_score, anomaly_count, len(anomaly_details)
        )
        
        return {
            'explanations': explanations,
            'confidence_breakdown': confidence_breakdown,
            'anomaly_details': anomaly_details[:5],  # Top 5 anomalies
            'frame_markers': frame_markers[:10],     # Top 10 suspicious frames
            'key_findings': self._extract_key_findings(visual_metrics, temporal_metrics),
            'recommendation': recommendation,
            'summary': self._generate_summary(visual_score, temporal_score, anomaly_count)
        }
    
    def generate_audio_explanation(self, audio_analysis: Dict) -> Dict[str, Any]:
        """Generate explanations for audio analysis."""
        if not audio_analysis.get('success', False):
            return {'available': False, 'note': 'Audio analysis not available'}
        
        explanations = []
        audio_score = audio_analysis.get('authenticity_scores', {}).get('overall_authenticity', 0)
        insights = audio_analysis.get('insights', {})
        
        if audio_score > 0.7:
            explanations.append("✅ **Authentic audio** - Natural voice characteristics detected")
        elif audio_score > 0.5:
            explanations.append("⚠️ **Questionable audio** - Some synthetic characteristics")
        else:
            explanations.append("❌ **Synthetic audio** - Strong indicators of AI generation")
        
        # Add specific audio insights
        for strength in insights.get('strengths', []):
            explanations.append(f"✓ {strength}")
        
        for anomaly in insights.get('anomalies', []):
            explanations.append(f"⚠️ {anomaly}")
        
        return {
            'available': True,
            'explanations': explanations,
            'score': audio_score,
            'strengths': insights.get('strengths', []),
            'anomalies': insights.get('anomalies', []),
            'recommendations': insights.get('recommendations', [])
        }
    
    def _translate_reason(self, reason: str) -> str:
        """Translate technical reason to human-readable."""
        translations = {
            'unusual_frame_change': 'Unusual visual change',
            'abrupt_jump': 'Abrupt visual jump',
            'extreme_frame_change': 'Extreme frame difference',
            'insufficient_frames': 'Not enough data',
            'normal': 'Normal'
        }
        return translations.get(reason, reason.replace('_', ' ').title())
    
    def _calculate_confidence_breakdown(self, 
                                       visual_score: float, 
                                       temporal_score: float, 
                                       anomaly_count: int) -> List[Dict]:
        """Calculate confidence contribution from each factor."""
        factors = []
        
        # Visual factor (40% weight)
        visual_contribution = visual_score * 0.4
        factors.append({
            'factor': 'Visual Authenticity',
            'score': visual_score,
            'weight': 40,
            'contribution': visual_contribution,
            'explanation': 'Based on texture complexity and natural patterns'
        })
        
        # Temporal factor (30% weight)
        temporal_penalty = min(anomaly_count * 0.05, 0.2)
        temporal_contribution = temporal_score * 0.3 * (1 - temporal_penalty)
        factors.append({
            'factor': 'Temporal Consistency',
            'score': temporal_score,
            'weight': 30,
            'contribution': temporal_contribution,
            'explanation': 'Based on frame-to-frame consistency'
        })
        
        # Anomaly factor (30% weight)
        anomaly_penalty = min(anomaly_count * 0.1, 0.3)
        anomaly_contribution = 0.3 - (anomaly_penalty * 0.3)
        factors.append({
            'factor': 'Anomaly Detection',
            'anomaly_count': anomaly_count,
            'weight': 30,
            'contribution': anomaly_contribution,
            'explanation': 'Based on detected irregularities'
        })
        
        return factors
    
    def _extract_key_findings(self, visual_metrics: Dict, temporal_metrics: Dict) -> List[str]:
        """Extract the most important findings."""
        findings = []
        
        # Visual findings
        variance = visual_metrics.get('feature_variance', 0)
        if variance > 8:
            findings.append("High texture complexity (natural)")
        elif variance < 3:
            findings.append("Low texture complexity (potentially synthetic)")
        
        entropy = visual_metrics.get('feature_entropy', 0)
        if 3.5 < entropy < 4.5:
            findings.append("Natural visual entropy")
        
        # Temporal findings
        if temporal_metrics.get('stability') == 'high':
            findings.append("Stable temporal patterns")
        elif temporal_metrics.get('stability') == 'unstable':
            findings.append("Unstable temporal patterns")
        
        return findings
    
    def _generate_recommendation(self, 
                                visual_score: float, 
                                temporal_score: float, 
                                anomaly_count: int,
                                detailed_anomalies: int) -> str:
        """Generate final recommendation."""
        if visual_score > 0.7 and temporal_score > 0.7 and anomaly_count == 0:
            return "✅ HIGH CONFIDENCE: Content appears authentic"
        elif visual_score > 0.5 and temporal_score > 0.5 and anomaly_count <= 1:
            return "⚠️ MODERATE CONFIDENCE: Likely authentic, minor concerns"
        elif anomaly_count > 3 or visual_score < 0.3:
            return "❌ HIGH RISK: Strong indicators of manipulation"
        else:
            return "⚠️ REVIEW RECOMMENDED: Further verification needed"
    
    def _generate_summary(self, 
                         visual_score: float, 
                         temporal_score: float, 
                         anomaly_count: int) -> str:
        """Generate one-line summary."""
        if visual_score > 0.7 and temporal_score > 0.7:
            return "Authentic media with high confidence"
        elif anomaly_count > 0:
            return f"Contains {anomaly_count} anomaly{'s' if anomaly_count != 1 else ''}"
        elif visual_score < 0.5:
            return "Low visual authenticity score"
        else:
            return "Mixed signals detected"

# Singleton instance
explainability_engine = ExplainabilityEngine()