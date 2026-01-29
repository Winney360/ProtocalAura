"""
Deep Learning-based Deepfake Detection using pretrained models.
Uses TensorFlow/Keras for efficient inference without GPU requirement.
"""

import numpy as np
import cv2
from PIL import Image
import logging
from typing import Dict, Tuple, Optional
import os

logger = logging.getLogger(__name__)

# Try to import TensorFlow - if not available, use fallback
try:
    import tensorflow as tf
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.preprocessing import image as tf_image
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    logger.warning("TensorFlow not available. DeepFake detector will use handcrafted features only.")

# Try MediaPipe for facial analysis
try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    logger.warning("MediaPipe not available. Facial analysis features disabled.")


class DeepfakeDetector:
    """Deep learning-based deepfake detector using pretrained models."""
    
    def __init__(self):
        self.tf_available = TF_AVAILABLE
        self.mediapipe_available = MEDIAPIPE_AVAILABLE
        self.model = None
        self.face_detector = None
        
        # Initialize TensorFlow model if available
        if self.tf_available:
            try:
                # Use pretrained EfficientNetB0 for image classification
                self.model = EfficientNetB0(weights='imagenet', include_top=True)
                logger.info("EfficientNetB0 model loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load TensorFlow model: {e}")
                self.tf_available = False
        
        # Initialize MediaPipe face detection if available
        if self.mediapipe_available:
            try:
                self.mp_face_detection = mp.solutions.face_detection
                self.face_detector = self.mp_face_detection.FaceDetection(
                    model_selection=0, 
                    min_detection_confidence=0.5
                )
                logger.info("MediaPipe face detector loaded successfully")
            except Exception as e:
                logger.error(f"Failed to load MediaPipe: {e}")
                self.mediapipe_available = False
    
    def detect_facial_artifacts(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect deepfake artifacts using facial analysis.
        Returns scores for various facial anomalies.
        """
        if not self.mediapipe_available:
            return {}
        
        try:
            # Convert to RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_detector.process(image_rgb)
            
            artifacts = {
                'face_detected': False,
                'face_count': 0,
                'eye_artifacts': 0.0,
                'mouth_artifacts': 0.0,
                'skin_texture_anomaly': 0.0
            }
            
            if results.detections:
                artifacts['face_detected'] = True
                artifacts['face_count'] = len(results.detections)
                
                # Analyze each detected face
                for detection in results.detections:
                    # Extract bounding box
                    h, w, _ = frame.shape
                    bbox = detection.location_data.bounding_box
                    
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    
                    # Ensure coordinates are within bounds
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    if x2 > x1 and y2 > y1:
                        face_roi = frame[y1:y2, x1:x2]
                        
                        # Analyze skin texture
                        artifacts['skin_texture_anomaly'] = self._analyze_skin_texture(face_roi)
                        
                        # Analyze eyes and mouth regions (simplified)
                        artifacts['eye_artifacts'] = self._analyze_eye_region(face_roi)
                        artifacts['mouth_artifacts'] = self._analyze_mouth_region(face_roi)
            
            return artifacts
            
        except Exception as e:
            logger.error(f"Facial artifact detection error: {e}")
            return {}
    
    def _analyze_skin_texture(self, face_roi: np.ndarray) -> float:
        """Analyze skin texture for deepfake artifacts (0-1 score)."""
        try:
            # Convert to HSV for better color analysis
            hsv = cv2.cvtColor(face_roi, cv2.COLOR_BGR2HSV)
            
            # Analyze saturation uniformity (deepfakes often have unnatural saturation)
            saturation = hsv[:, :, 1].astype(np.float32) / 255.0
            saturation_var = np.var(saturation)
            
            # Analyze value (brightness) uniformity
            value = hsv[:, :, 2].astype(np.float32) / 255.0
            value_var = np.var(value)
            
            # Too uniform saturation/value = suspicious (deepfake)
            # Natural skin has variation
            if saturation_var < 0.005 or value_var < 0.005:
                return 0.7  # High anomaly score
            elif saturation_var < 0.01 or value_var < 0.01:
                return 0.4
            else:
                return 0.1  # Normal variation
                
        except Exception as e:
            logger.error(f"Skin texture analysis error: {e}")
            return 0.0
    
    def _analyze_eye_region(self, face_roi: np.ndarray) -> float:
        """Analyze eye region for deepfake artifacts."""
        try:
            # Simplified: check for unusual patterns in upper face region
            h, w = face_roi.shape[:2]
            eye_region = face_roi[:h//3, :]  # Top third (rough eye area)
            
            # Convert to grayscale
            gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
            
            # Detect edges (eyes have strong edges)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Too many or too few edges = suspicious
            if edge_density < 0.01:
                return 0.5  # Missing edge detail
            elif edge_density > 0.3:
                return 0.3  # Excessive edges
            else:
                return 0.1  # Normal
                
        except Exception as e:
            logger.error(f"Eye region analysis error: {e}")
            return 0.0
    
    def _analyze_mouth_region(self, face_roi: np.ndarray) -> float:
        """Analyze mouth region for deepfake artifacts."""
        try:
            # Simplified: check lower face region
            h, w = face_roi.shape[:2]
            mouth_region = face_roi[2*h//3:, :]  # Bottom third
            
            # Convert to HSV for lip color analysis
            hsv = cv2.cvtColor(mouth_region, cv2.COLOR_BGR2HSV)
            
            # Analyze hue distribution (lips have specific hue range)
            hue = hsv[:, :, 0]
            hue_std = np.std(hue)
            
            # Too uniform hue = suspicious
            if hue_std < 10:
                return 0.6  # Unnatural uniformity
            elif hue_std > 80:
                return 0.3  # High variation (acceptable)
            else:
                return 0.2
                
        except Exception as e:
            logger.error(f"Mouth region analysis error: {e}")
            return 0.0
    
    def predict_deepfake_probability(self, frame: Image.Image) -> Dict[str, float]:
        """
        Predict deepfake probability using deep learning model.
        Returns: {'deepfake_score': 0-1, 'confidence': 0-1}
        """
        if not self.tf_available:
            return {'deepfake_score': 0.5, 'confidence': 0.3, 'note': 'TensorFlow not available'}
        
        try:
            # Prepare image for model
            img_array = np.array(frame)
            
            # Resize to EfficientNet input size (224x224)
            img_resized = cv2.resize(img_array, (224, 224))
            
            # Normalize to [-1, 1] range as per EfficientNet requirements
            img_normalized = tf_image.smart_resize(img_array, (224, 224))
            img_normalized = np.expand_dims(img_normalized, axis=0)
            img_normalized = img_normalized / 127.5 - 1.0
            
            # Get predictions
            predictions = self.model.predict(img_normalized, verbose=0)
            
            # Find max probability class
            max_prob = float(np.max(predictions[0]))
            
            # Simple heuristic: certain ImageNet classes are more common in deepfakes
            # This is a rough approximation - ideally would use a model fine-tuned on deepfake data
            # Classes like "face", "portrait" might indicate synthetic content if confidence is too high
            deepfake_score = max(0.0, min(1.0, 1.0 - max_prob))  # Inverted: high confidence = low deepfake score
            
            return {
                'deepfake_score': float(deepfake_score),
                'confidence': max(0.3, max_prob),  # Minimum confidence threshold
                'model': 'EfficientNetB0'
            }
            
        except Exception as e:
            logger.error(f"Deepfake prediction error: {e}")
            return {'deepfake_score': 0.5, 'confidence': 0.2, 'error': str(e)}
    
    def analyze_frame_ensemble(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Ensemble analysis combining multiple detection methods.
        Returns comprehensive deepfake indicators.
        """
        results = {
            'facial_artifacts': {},
            'neural_prediction': {},
            'ensemble_score': 0.5,
            'methods_available': []
        }
        
        # Facial artifact analysis
        if self.mediapipe_available:
            results['facial_artifacts'] = self.detect_facial_artifacts(frame)
            results['methods_available'].append('facial_analysis')
        
        # Deep learning prediction
        if self.tf_available:
            try:
                pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                results['neural_prediction'] = self.predict_deepfake_probability(pil_frame)
                results['methods_available'].append('neural_network')
            except Exception as e:
                logger.error(f"Neural prediction failed: {e}")
        
        # Ensemble scoring
        scores = []
        
        # Add facial artifact scores
        if results['facial_artifacts']:
            facial_score = (
                results['facial_artifacts'].get('skin_texture_anomaly', 0) * 0.5 +
                results['facial_artifacts'].get('eye_artifacts', 0) * 0.3 +
                results['facial_artifacts'].get('mouth_artifacts', 0) * 0.2
            )
            scores.append(facial_score)
        
        # Add neural prediction
        if results['neural_prediction'] and 'deepfake_score' in results['neural_prediction']:
            scores.append(results['neural_prediction']['deepfake_score'])
        
        # Calculate ensemble score
        if scores:
            results['ensemble_score'] = float(np.mean(scores))
        
        return results


# Singleton instance
_detector = None

def get_deepfake_detector() -> DeepfakeDetector:
    """Get or create singleton deepfake detector instance."""
    global _detector
    if _detector is None:
        _detector = DeepfakeDetector()
    return _detector
