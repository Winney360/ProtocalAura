# File: temporal_analyzer.py
import numpy as np
from scipy.spatial.distance import cosine
from collections import deque
import warnings
warnings.filterwarnings('ignore')

class TemporalAnalyzer:
    """
    Analyzes temporal consistency between video frames.
    Detects abrupt changes, embedding drift, and statistical anomalies.
    """
    
    def __init__(self, window_size=15, anomaly_threshold=2.5):
        """
        Initialize the temporal analyzer.
        
        Args:
            window_size: How many previous frames to consider for analysis
            anomaly_threshold: Z-score threshold for flagging anomalies
        """
        self.window_size = window_size
        self.anomaly_threshold = anomaly_threshold
        
        # Store recent frames for comparison
        self.frame_embeddings = deque(maxlen=window_size)  # ResNet feature vectors
        self.visual_metrics = deque(maxlen=window_size)    # Feature variance, entropy, etc.
        
        # History for statistical baselines
        self.drift_history = []
        self.consistency_history = []
        
        print(f"[TemporalAnalyzer] Initialized with window_size={window_size}")
    
    def add_frame(self, embedding, metrics):
        """
        Add a new frame for temporal analysis.
        
        Args:
            embedding: ResNet-18 feature vector (2048-dim for ResNet)
            metrics: Dictionary of visual metrics for this frame
        """
        self.frame_embeddings.append(embedding)
        self.visual_metrics.append(metrics)
        
        # Store for statistical baselines
        if len(self.frame_embeddings) >= 2:
            drift = self._calculate_embedding_drift()
            self.drift_history.append(drift)
    
    def analyze_current(self):
        """
        Analyze temporal consistency for the most recent frame.
        
        Returns:
            dict: Temporal metrics for current frame
        """
        if len(self.frame_embeddings) < 2:
            return self._get_insufficient_data_response()
        
        # 1. Calculate embedding drift (how much features changed)
        embedding_drift = self._calculate_embedding_drift()
        
        # 2. Calculate cosine similarity (1 - drift)
        cosine_similarity = max(0.0, 1.0 - embedding_drift)
        
        # 3. Calculate temporal variance (stability over window)
        temporal_variance = self._calculate_temporal_variance()
        
        # 4. Detect anomalies
        anomaly_result = self._detect_anomalies(embedding_drift, temporal_variance)
        
        # 5. Check for abrupt changes (common in deepfakes)
        is_abrupt = embedding_drift > 0.15  # Threshold for abrupt visual jumps
        
        return {
            'embedding_drift': float(embedding_drift),
            'cosine_similarity': float(cosine_similarity),
            'temporal_variance': float(temporal_variance),
            'anomaly_score': float(anomaly_result['score']),
            'has_anomaly': bool(anomaly_result['is_anomaly']),
            'anomaly_reason': anomaly_result['reason'],
            'is_abrupt_change': bool(is_abrupt),
            'frame_count': len(self.frame_embeddings)
        }
    
    def _calculate_embedding_drift(self):
        """Calculate how much features changed between last two frames."""
        if len(self.frame_embeddings) < 2:
            return 0.0
        
        # Get the last two embeddings
        emb1 = self.frame_embeddings[-2].flatten()
        emb2 = self.frame_embeddings[-1].flatten()
        
        # Normalize vectors to unit length for cosine distance
        emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
        emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
        
        # Cosine distance = 1 - cosine similarity
        drift = 1.0 - np.dot(emb1_norm, emb2_norm)
        
        # Ensure valid range
        return float(np.clip(drift, 0.0, 1.0))
    
    def _calculate_temporal_variance(self):
        """Calculate variance of embeddings over the window."""
        if len(self.frame_embeddings) < 2:
            return 0.0
        
        # Convert embeddings to array
        embeddings_array = np.array([emb.flatten() for emb in self.frame_embeddings])
        
        # Calculate variance across frames for each feature dimension
        per_dimension_variance = np.var(embeddings_array, axis=0)
        
        # Return average variance (normalized)
        avg_variance = np.mean(per_dimension_variance)
        return float(avg_variance / 100.0)  # Normalize to reasonable range
    
    def _detect_anomalies(self, current_drift, current_variance):
        """
        Detect statistical anomalies using historical data.
        
        Returns:
            dict: anomaly_score, is_anomaly, reason
        """
        # Need enough history for statistical detection
        if len(self.drift_history) < 10:
            return {
                'score': 0.0,
                'is_anomaly': False,
                'reason': 'insufficient_history'
            }
        
        # Calculate Z-score for current drift
        drift_mean = np.mean(self.drift_history)
        drift_std = np.std(self.drift_history) + 1e-10  # Avoid division by zero
        
        drift_zscore = abs((current_drift - drift_mean) / drift_std)
        
        # Check if this is an anomaly
        is_anomaly = drift_zscore > self.anomaly_threshold
        
        # Determine reason
        reason = "normal"
        if is_anomaly:
            if current_drift > drift_mean:
                reason = "unusually_large_change"
            else:
                reason = "unusually_small_change"
            
            if current_drift > 0.15:
                reason += "|abrupt_visual_jump"
        
        # Anomaly score (0-1, higher = more anomalous)
        anomaly_score = min(drift_zscore / 5.0, 1.0)
        
        return {
            'score': float(anomaly_score),
            'is_anomaly': bool(is_anomaly),
            'reason': reason
        }
    
    def _get_insufficient_data_response(self):
        """Return response when there's not enough data."""
        return {
            'embedding_drift': 0.0,
            'cosine_similarity': 1.0,
            'temporal_variance': 0.0,
            'anomaly_score': 0.0,
            'has_anomaly': False,
            'anomaly_reason': 'insufficient_frames',
            'is_abrupt_change': False,
            'frame_count': len(self.frame_embeddings)
        }
    
    def get_summary(self):
        """
        Get overall temporal consistency summary.
        
        Returns:
            dict: Summary metrics across all analyzed frames
        """
        if len(self.drift_history) == 0:
            return self._get_empty_summary()
        
        # Calculate overall statistics
        avg_drift = np.mean(self.drift_history)
        drift_std = np.std(self.drift_history)
        
        # Count anomalies (drift > 2.5 std deviations)
        anomaly_threshold = avg_drift + (self.anomaly_threshold * drift_std)
        anomaly_count = sum(1 for d in self.drift_history if d > anomaly_threshold)
        
        # Determine stability level
        if drift_std < 0.02:
            stability = "high"
        elif drift_std < 0.05:
            stability = "medium"
        else:
            stability = "low"
        
        # Consistency score (higher = more consistent)
        # Based on low drift and low variance
        consistency_score = 1.0 - min(avg_drift * 5.0 + drift_std * 10.0, 1.0)
        
        return {
            'temporal_consistency_score': float(consistency_score),
            'average_drift': float(avg_drift),
            'drift_std': float(drift_std),
            'stability': stability,
            'anomaly_count': int(anomaly_count),
            'total_frames': len(self.drift_history),
            'anomaly_percentage': float(anomaly_count / len(self.drift_history))
        }
    
    def _get_empty_summary(self):
        """Return empty summary when no data."""
        return {
            'temporal_consistency_score': 0.0,
            'average_drift': 0.0,
            'drift_std': 0.0,
            'stability': 'unknown',
            'anomaly_count': 0,
            'total_frames': 0,
            'anomaly_percentage': 0.0
        }
    
    def reset(self):
        """Reset the analyzer for a new video."""
        self.frame_embeddings.clear()
        self.visual_metrics.clear()
        self.drift_history.clear()
        self.consistency_history.clear()
        print("[TemporalAnalyzer] Reset for new analysis")

# Simple test to verify it works
if __name__ == "__main__":
    print("Testing TemporalAnalyzer...")
    
    # Create analyzer
    analyzer = TemporalAnalyzer(window_size=10)
    
    # Test with dummy data
    dummy_embedding = np.random.randn(512)  # Simulated ResNet features
    dummy_metrics = {'variance': 0.5, 'entropy': 4.2}
    
    # Add some frames
    for i in range(5):
        analyzer.add_frame(dummy_embedding + np.random.randn(512)*0.1, dummy_metrics)
        analysis = analyzer.analyze_current()
        print(f"Frame {i+1}: Drift={analysis['embedding_drift']:.3f}, Anomaly={analysis['has_anomaly']}")
    
    # Get summary
    summary = analyzer.get_summary()
    print(f"\nSummary: Consistency={summary['temporal_consistency_score']:.2f}, Stability={summary['stability']}")
    
    print("✅ TemporalAnalyzer test complete!")