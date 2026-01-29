# File: ai_service/analysis/temporal.py (UPDATED with imports)
"""
Temporal Analysis Module for Protocol Aura
Step 3: Frame-to-frame consistency analysis
"""

import numpy as np
from collections import deque
from typing import Dict, List, Deque
import logging

logger = logging.getLogger(__name__)

class TemporalAnalyzer:
    """Analyzes temporal consistency between video frames."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.frame_embeddings: Deque[np.ndarray] = deque(maxlen=window_size)
        self.drift_history: List[float] = []
        self.anomaly_timeline: List[Dict] = []
    
    def add_frame(self, embedding: np.ndarray) -> None:
        """Add a frame for temporal analysis."""
        self.frame_embeddings.append(embedding)
        
        if len(self.frame_embeddings) >= 2:
            drift = self._calculate_drift()
            self.drift_history.append(drift)
    
    def analyze_current(self) -> Dict:
        """Analyze the most recent frame."""
        if len(self.frame_embeddings) < 2:
            return self._empty_response()
        
        drift = self._calculate_drift()
        is_abrupt = drift > 0.2
        
        # Simple anomaly detection
        is_anomaly = False
        reason = "normal"
        
        if len(self.drift_history) >= 8:
            recent_drifts = self.drift_history[-8:]
            avg_drift = float(np.mean(recent_drifts)) if recent_drifts else 0.0
            drift_std = float(np.std(recent_drifts)) if len(recent_drifts) > 1 else 0.0
            threshold = max(avg_drift + 2.5 * drift_std, 0.2)
            
            # Check for unusual changes
            if drift > threshold:
                is_anomaly = True
                reason = "unusual_frame_change"
                if is_abrupt:
                    reason += "|abrupt_jump"
            
            # Also check for very low similarity (sudden complete change)
            if drift > 0.35:
                is_anomaly = True
                reason = "extreme_frame_change"
        
        if is_anomaly:
            self.anomaly_timeline.append({
                'frame_index': len(self.drift_history),
                'drift': drift,
                'reason': reason
            })
        
        return {
            'embedding_drift': float(drift),
            'cosine_similarity': float(1.0 - drift),
            'has_anomaly': is_anomaly,
            'anomaly_reason': reason,
            'is_abrupt_change': is_abrupt
        }
    
    def get_summary(self) -> Dict:
        """Get overall temporal summary."""
        if not self.drift_history:
            return {
                'temporal_consistency_score': 0.0,
                'anomaly_count': 0,
                'stability': 'unknown',
                'average_drift': 0.0,
                'total_frames': 0
            }
        
        drifts = np.array(self.drift_history)
        avg_drift = float(np.mean(drifts))
        drift_std = float(np.std(drifts)) if len(drifts) > 1 else 0.0
        
        # Consistency score: lower drift and lower std = higher consistency
        consistency_score = max(0.0, 1.0 - (avg_drift * 1.5 + drift_std * 2.0))
        
        # Determine stability
        if consistency_score > 0.8:
            stability = "high"
        elif consistency_score > 0.6:
            stability = "medium"
        elif consistency_score > 0.4:
            stability = "low"
        else:
            stability = "unstable"
        
        return {
            'temporal_consistency_score': consistency_score,
            'stability': stability,
            'average_drift': avg_drift,
            'drift_standard_deviation': drift_std,
            'anomaly_count': len(self.anomaly_timeline),
            'total_frames': len(self.drift_history),
            'anomaly_timeline': self.anomaly_timeline[-10:] if self.anomaly_timeline else []
        }
    
    def _calculate_drift(self) -> float:
        """Calculate cosine distance between last two frames."""
        if len(self.frame_embeddings) < 2:
            return 0.0
        
        emb1 = self.frame_embeddings[-2].flatten()
        emb2 = self.frame_embeddings[-1].flatten()
        
        # Normalize vectors
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Cosine similarity
        cosine_sim = np.dot(emb1/norm1, emb2/norm2)
        # Ensure valid range and calculate distance
        drift = max(0.0, 1.0 - cosine_sim)
        return float(drift)
    
    def _empty_response(self) -> Dict:
        return {
            'embedding_drift': 0.0,
            'cosine_similarity': 1.0,
            'has_anomaly': False,
            'anomaly_reason': 'insufficient_frames',
            'is_abrupt_change': False
        }
    
    def reset(self):
        """Reset for new video."""
        self.frame_embeddings.clear()
        self.drift_history.clear()
        self.anomaly_timeline.clear()