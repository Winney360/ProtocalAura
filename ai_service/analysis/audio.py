# File: ai_service/analysis/audio.py
"""
Audio Analysis Module for Protocol Aura
Step 4: Audio authenticity verification
"""

import numpy as np
import librosa
import librosa.display
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """
    Analyzes audio for signs of AI-generation or manipulation.
    
    Detects:
    1. Unnatural pitch patterns (too perfect/too erratic)
    2. Spectral anomalies (common in synthetic audio)
    3. Voice consistency (natural voice vs AI-generated)
    4. Background noise patterns
    
    All calculations based on real audio signals, no randomness.
    """
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        logger.info(f"AudioAnalyzer initialized (sample_rate={sample_rate})")
    
    def analyze_audio(self, audio_path: str) -> Dict:
        """
        Analyze an audio file for authenticity.
        
        Args:
            audio_path: Path to audio file
            
        Returns:
            Dictionary with audio analysis results
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=self.sample_rate)
            
            # Extract audio features
            features = self._extract_audio_features(y, sr)
            
            # Calculate authenticity scores
            scores = self._calculate_authenticity_scores(features)
            
            # Generate human-readable insights
            insights = self._generate_insights(features, scores)
            
            return {
                'success': True,
                'audio_metrics': features,
                'authenticity_scores': scores,
                'insights': insights,
                'final_audio_verdict': self._determine_verdict(scores),
                'confidence': scores.get('overall_confidence', 0.5)
            }
            
        except Exception as e:
            logger.error(f"Audio analysis error: {e}")
            return self._error_response(str(e))
    
    def _extract_audio_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract all audio features for analysis."""
        features = {}
        
        # 1. Basic statistics
        features['duration'] = len(y) / sr
        features['amplitude_mean'] = float(np.mean(np.abs(y)))
        features['amplitude_std'] = float(np.std(y))
        
        # 2. Pitch analysis (fundamental frequency)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = pitches[pitches > 0]
        
        if len(pitch_values) > 0:
            features['pitch_mean'] = float(np.mean(pitch_values))
            features['pitch_std'] = float(np.std(pitch_values))
            features['pitch_range'] = float(np.ptp(pitch_values))
        else:
            features['pitch_mean'] = 0.0
            features['pitch_std'] = 0.0
            features['pitch_range'] = 0.0
        
        # 3. Spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features['spectral_centroid_mean'] = float(np.mean(spectral_centroid))
        features['spectral_centroid_std'] = float(np.std(spectral_centroid))
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
        
        # 4. MFCCs (Mel-frequency cepstral coefficients) for voice characteristics
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = float(np.mean(mfccs))
        features['mfcc_std'] = float(np.std(mfccs))
        
        # 5. Zero-crossing rate (speech vs silence)
        zcr = librosa.feature.zero_crossing_rate(y)
        features['zcr_mean'] = float(np.mean(zcr))
        
        # 6. Harmonics-to-noise ratio (voice quality)
        harmonic, percussive = librosa.effects.hpss(y)
        if np.sum(np.abs(harmonic)) > 0:
            hnr = 10 * np.log10(np.sum(harmonic**2) / np.sum(percussive**2))
            features['hnr'] = float(hnr)
        else:
            features['hnr'] = 0.0
        
        # 7. Spectral rolloff (frequency range)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
        
        return features
    
    def _calculate_authenticity_scores(self, features: Dict) -> Dict:
        """Calculate authenticity scores based on audio features."""
        scores = {}
        
        # 1. Pitch naturalness score
        pitch_std = features.get('pitch_std', 0)
        if pitch_std > 0:
            # Natural speech has moderate pitch variation
            # Too low = robotic, too high = erratic
            pitch_score = 1.0 - abs(pitch_std - 20) / 40  # Target ~20 Hz std
            pitch_score = max(0.0, min(1.0, pitch_score))
        else:
            pitch_score = 0.3  # No pitch detected
        
        scores['pitch_naturalness'] = pitch_score
        
        # 2. Spectral consistency score
        spectral_centroid_std = features.get('spectral_centroid_std', 0)
        spectral_score = 1.0 - min(spectral_centroid_std / 500, 1.0)
        scores['spectral_consistency'] = spectral_score
        
        # 3. Voice quality score (based on HNR)
        hnr = features.get('hnr', 0)
        if hnr > 0:
            voice_quality = min(hnr / 30, 1.0)  # Higher HNR = better voice quality
        else:
            voice_quality = 0.3
        scores['voice_quality'] = voice_quality
        
        # 4. Speech naturalness (ZCR + amplitude)
        zcr_mean = features.get('zcr_mean', 0)
        amp_std = features.get('amplitude_std', 0)
        
        # Natural speech has moderate ZCR
        zcr_score = 1.0 - abs(zcr_mean - 0.1) / 0.2
        zcr_score = max(0.0, min(1.0, zcr_score))
        
        # Natural speech has moderate amplitude variation
        amp_score = min(amp_std * 10, 1.0)
        
        speech_score = (zcr_score * 0.6) + (amp_score * 0.4)
        scores['speech_naturalness'] = speech_score
        
        # 5. Overall audio authenticity
        overall = (
            scores['pitch_naturalness'] * 0.3 +
            scores['spectral_consistency'] * 0.25 +
            scores['voice_quality'] * 0.25 +
            scores['speech_naturalness'] * 0.2
        )
        scores['overall_authenticity'] = overall
        
        # 6. Confidence based on feature consistency
        confidence = overall * 0.8 + 0.2  # Base confidence
        scores['overall_confidence'] = min(confidence, 1.0)
        
        return scores
    
    def _generate_insights(self, features: Dict, scores: Dict) -> Dict:
        """Generate human-readable insights from analysis."""
        insights = {
            'strengths': [],
            'anomalies': [],
            'recommendations': []
        }
        
        overall_score = scores.get('overall_authenticity', 0)
        
        # Strengths
        if scores.get('pitch_naturalness', 0) > 0.7:
            insights['strengths'].append('Natural pitch variation')
        if scores.get('voice_quality', 0) > 0.7:
            insights['strengths'].append('Good voice quality')
        if scores.get('spectral_consistency', 0) > 0.7:
            insights['strengths'].append('Consistent spectral patterns')
        
        # Anomalies
        if features.get('pitch_std', 1000) < 5:  # Too monotone
            insights['anomalies'].append('Unnaturally consistent pitch (robotic)')
        elif features.get('pitch_std', 0) > 50:  # Too erratic
            insights['anomalies'].append('Excessively erratic pitch')
        
        if features.get('hnr', 0) < 5:  # Low harmonics-to-noise
            insights['anomalies'].append('Poor voice harmonics (possible synthesis)')
        
        if features.get('zcr_mean', 0) < 0.05:  # Very low zero-crossing
            insights['anomalies'].append('Unnatural speech-silence pattern')
        
        # Recommendations
        if overall_score > 0.8:
            insights['recommendations'].append('Audio appears authentic')
        elif overall_score > 0.6:
            insights['recommendations'].append('Audio likely authentic - minor anomalies detected')
        elif overall_score > 0.4:
            insights['recommendations'].append('Audio questionable - review recommended')
        else:
            insights['recommendations'].append('Audio shows signs of manipulation')
        
        return insights
    
    def _determine_verdict(self, scores: Dict) -> str:
        """Determine final audio authenticity verdict."""
        overall = scores.get('overall_authenticity', 0)
        confidence = scores.get('overall_confidence', 0)
        
        if overall > 0.75 and confidence > 0.7:
            return "likely_authentic"
        elif overall > 0.6:
            return "probably_authentic"
        elif overall > 0.4:
            return "needs_review"
        else:
            return "suspicious"
    
    def _error_response(self, error_msg: str) -> Dict:
        """Return error response."""
        return {
            'success': False,
            'error': error_msg,
            'audio_metrics': {},
            'authenticity_scores': {},
            'insights': {
                'strengths': [],
                'anomalies': ['Audio analysis failed'],
                'recommendations': ['Manual review required']
            },
            'final_audio_verdict': 'analysis_error',
            'confidence': 0.0
        }