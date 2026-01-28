# File: ai_service/analysis/temporal.py
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
        is_abrupt = drift > 0.15
        
        # Simple anomaly detection
        is_anomaly = False
        reason = "normal"
        
        if len(self.drift_history) >= 5:
            avg_drift = np.mean(self.drift_history[-5:])
            if drift > avg_drift * 2.5:  # Simple threshold
                is_anomaly = True
                reason = "unusual_frame_change"
                if is_abrupt:
                    reason += "|abrupt_jump"
        
        if is_anomaly:
            self.anomaly_timeline.append({
                'frame': len(self.drift_history),
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
            return {'temporal_consistency_score': 0.0, 'anomaly_count': 0}
        
        drifts = np.array(self.drift_history)
        avg_drift = float(np.mean(drifts))
        
        # Consistency score: lower drift = higher consistency
        consistency_score = max(0.0, 1.0 - (avg_drift * 3.0))
        
        return {
            'temporal_consistency_score': consistency_score,
            'average_drift': avg_drift,
            'anomaly_count': len(self.anomaly_timeline),
            'stability': 'high' if consistency_score > 0.7 else 
                        'medium' if consistency_score > 0.5 else 'low',
            'total_frames': len(self.drift_history)
        }
    
    def _calculate_drift(self) -> float:
        """Calculate cosine distance between last two frames."""
        if len(self.frame_embeddings) < 2:
            return 0.0
        
        emb1 = self.frame_embeddings[-2].flatten()
        emb2 = self.frame_embeddings[-1].flatten()
        
        # Cosine distance
        norm1 = np.linalg.norm(emb1)
        norm2 = np.linalg.norm(emb2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        cosine_sim = np.dot(emb1/norm1, emb2/norm2)
        return float(max(0.0, 1.0 - cosine_sim))
    
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