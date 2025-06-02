#
import cv2
import numpy as np
import logging
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from .face_detection import FaceDetection
from .image_loader import ImageLoader

logger = logging.getLogger(__name__)

class EmotionClassifier:
    def __init__(self):
        """Initialize emotion classifier"""
        try:
            self.face_detector = FaceDetection()
            self.image_loader = ImageLoader()
            
            # Define emotion classes
            self.emotion_classes = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            
            logger.info("Emotion classifier initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing emotion classifier: {str(e)}")
            raise

    def predict(self, image: np.ndarray) -> Dict[str, Any]:
        """Predict emotions in facial image"""
        try:
            if image is None:
                return self._empty_prediction()
            
            # Detect faces
            faces = self.face_detector.detect_faces(image)
            
            if len(faces) == 0:
                logger.warning("No faces detected in image")
                return self._no_face_prediction()
            
            # Get the largest face (most prominent)
            largest_face = self.face_detector.get_largest_face(faces)
            
            # Extract face region
            face_region = self.face_detector.extract_face_region(image, largest_face)
            
            if face_region is None:
                return self._no_face_prediction()
            
            # Extract features from face
            features = self._extract_facial_features(face_region, largest_face)
            
            # Detect facial features
            facial_features = self.face_detector.detect_facial_features(image, largest_face)
            
            # Classify emotion using rule-based approach
            emotions = self._classify_emotions(features, facial_features, face_region)
            
            # Determine dominant emotion
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            confidence = emotions[dominant_emotion]
            
            return {
                'emotions': emotions,
                'dominant_emotion': dominant_emotion,
                'confidence': confidence,
                'face_detected': True,
                'num_faces': len(faces),
                'face_area': largest_face[2] * largest_face[3],
                'facial_features': facial_features,
                'features': features
            }
            
        except Exception as e:
            logger.error(f"Error predicting emotions: {str(e)}")
            return self._empty_prediction()

    def _extract_facial_features(self, face_region: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Dict[str, float]:
        """Extract features from face region for emotion classification"""
        try:
            features = {}
            
            # Convert to grayscale for feature extraction
            if len(face_region.shape) == 3:
                if face_region.max() <= 1.0:
                    gray_face = cv2.cvtColor((face_region * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray_face = cv2.cvtColor(face_region.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray_face = face_region.astype(np.uint8) if face_region.max() <= 1.0 else face_region
            
            # Basic intensity features
            features['mean_intensity'] = np.mean(gray_face)
            features['std_intensity'] = np.std(gray_face)
            features['intensity_range'] = np.max(gray_face) - np.min(gray_face)
            
            # Edge features (emotional expressions create distinctive edge patterns)
            edges = cv2.Canny(gray_face, 50, 150)
            features['edge_density'] = np.sum(edges > 0) / edges.size
            
            # Texture features using Local Binary Pattern approximation
            features['texture_variance'] = self._compute_texture_variance(gray_face)
            
            # Geometric features
            h, w = gray_face.shape
            features['aspect_ratio'] = w / h if h > 0 else 1.0
            features['face_area'] = h * w
            
            # Symmetry features
            features['horizontal_symmetry'] = self._compute_horizontal_symmetry(gray_face)
            
            # Regional analysis (divide face into regions)
            eye_region = gray_face[:h//3, :]  # Upper third - eye region
            mouth_region = gray_face[2*h//3:, :]  # Lower third - mouth region
            
            features['eye_region_intensity'] = np.mean(eye_region)
            features['mouth_region_intensity'] = np.mean(mouth_region)
            features['eye_mouth_intensity_diff'] = features['eye_region_intensity'] - features['mouth_region_intensity']
            
            # Gradient features
            grad_x = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
            
            features['gradient_magnitude_mean'] = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            features['gradient_direction_variance'] = np.var(np.arctan2(grad_y, grad_x + 1e-10))
            
            # Face geometry analysis
            geometry = self.face_detector.analyze_face_geometry(face_region)
            features.update(geometry)
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting facial features: {str(e)}")
            return {}

    def _compute_texture_variance(self, gray_image: np.ndarray) -> float:
        """Compute texture variance as a simple texture measure"""
        try:
            # Apply Gaussian blur and compute difference
            blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)
            texture = gray_image.astype(np.float32) - blurred.astype(np.float32)
            return np.var(texture)
        except:
            return 0.0

    def _compute_horizontal_symmetry(self, gray_image: np.ndarray) -> float:
        """Compute horizontal symmetry score"""
        try:
            h, w = gray_image.shape
            left_half = gray_image[:, :w//2]
            right_half = cv2.flip(gray_image[:, w//2:], 1)
            
            # Resize to same size if needed
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Compute correlation
            correlation = cv2.matchTemplate(left_half.astype(np.float32), 
                                          right_half.astype(np.float32), 
                                          cv2.TM_CCOEFF_NORMED)
            return float(np.max(correlation))
        except:
            return 0.5

    def _classify_emotions(self, features: Dict[str, float], facial_features: Dict[str, Any], face_region: np.ndarray) -> Dict[str, float]:
        """Classify emotions using rule-based approach"""
        try:
            emotions = {emotion: 0.0 for emotion in self.emotion_classes}
            
            if not features:
                emotions['neutral'] = 0.7
                return emotions
            
            # Extract key features
            mean_intensity = features.get('mean_intensity', 128)
            std_intensity = features.get('std_intensity', 10)
            edge_density = features.get('edge_density', 0.1)
            eye_mouth_diff = features.get('eye_mouth_intensity_diff', 0)
            symmetry = features.get('horizontal_symmetry', 0.5)
            texture_var = features.get('texture_variance', 100)
            gradient_mag = features.get('gradient_magnitude_mean', 10)
            has_smile = facial_features.get('has_smile', False)
            eye_symmetry = facial_features.get('eye_symmetry', 0)
            
            # Normalize features
            intensity_norm = mean_intensity / 255.0
            edge_norm = min(edge_density * 10, 1.0)
            
            # Rule-based emotion classification
            
            # Joy/Happiness - typically brighter, more symmetric, smiles detected
            if intensity_norm > 0.6 and symmetry > 0.7 and edge_density > 0.15:
                emotions['joy'] += 0.6
                emotions['joy'] += min(0.3, (intensity_norm - 0.6) * 2)
                
            if has_smile:
                emotions['joy'] += 0.4
                
            # Sadness - often darker, less symmetric, drooping features
            if intensity_norm < 0.4 and symmetry < 0.6:
                emotions['sadness'] += 0.5
                emotions['sadness'] += min(0.4, (0.4 - intensity_norm) * 2)
                
            # Anger - high contrast, sharp features, asymmetric
            if std_intensity > 30 and edge_density > 0.2 and symmetry < 0.5:
                emotions['anger'] += 0.5
                emotions['anger'] += min(0.3, (std_intensity - 30) / 50)
                
            # Fear - high variance, tension in features
            if texture_var > 200 and gradient_mag > 15 and eye_mouth_diff > 10:
                emotions['fear'] += 0.4
                emotions['fear'] += min(0.3, (texture_var - 200) / 300)
                
            # Surprise - high gradient, distinctive patterns, wide eyes
            if gradient_mag > 20 and edge_density > 0.25:
                emotions['surprise'] += 0.4
                emotions['surprise'] += min(0.3, (gradient_mag - 20) / 30)
                
            if facial_features.get('eyes', 0) >= 2 and eye_symmetry > 0.7:
                emotions['surprise'] += 0.2
                
            # Disgust - specific asymmetric patterns
            if symmetry < 0.4 and edge_density > 0.18 and eye_mouth_diff < -5:
                emotions['disgust'] += 0.5
                emotions['disgust'] += min(0.3, (0.4 - symmetry) * 2)
                
            # Neutral - balanced features
            if (0.4 <= intensity_norm <= 0.6 and 
                0.5 <= symmetry <= 0.8 and 
                10 <= std_intensity <= 25):
                emotions['neutral'] += 0.6
            
            # Ensure we have at least some emotion
            if all(score < 0.1 for score in emotions.values()):
                emotions['neutral'] = 0.5
            
            # Additional contextual rules
            if edge_density < 0.05:  # Very smooth face
                emotions['neutral'] += 0.2
                emotions['sadness'] += 0.1
            
            if std_intensity > 50:  # Very high contrast
                emotions['anger'] += 0.2
                emotions['fear'] += 0.1
            
            # Normalize scores
            total_score = sum(emotions.values())
            if total_score > 0:
                for emotion in emotions:
                    emotions[emotion] = min(1.0, emotions[emotion] / total_score * 2)
            
            return emotions
            
        except Exception as e:
            logger.error(f"Error classifying emotions: {str(e)}")
            return {emotion: 0.0 for emotion in self.emotion_classes}

    def _empty_prediction(self) -> Dict[str, Any]:
        """Return empty prediction"""
        return {
            'emotions': {emotion: 0.0 for emotion in self.emotion_classes},
            'dominant_emotion': 'neutral',
            'confidence': 0.0,
            'face_detected': False,
            'num_faces': 0,
            'face_area': 0,
            'facial_features': {},
            'features': {}
        }

    def _no_face_prediction(self) -> Dict[str, Any]:
        """Return prediction when no face is detected"""
        emotions = {emotion: 0.0 for emotion in self.emotion_classes}
        emotions['neutral'] = 0.3  # Low confidence neutral
        
        return {
            'emotions': emotions,
            'dominant_emotion': 'neutral',
            'confidence': 0.3,
            'face_detected': False,
            'num_faces': 0,
            'face_area': 0,
            'facial_features': {},
            'features': {}
        }

    def predict_batch(self, images: List[np.ndarray]) -> List[Dict[str, Any]]:
        """Predict emotions for a batch of images"""
        try:
            results = []
            for i, image in enumerate(images):
                result = self.predict(image)
                results.append(result)
                logger.info(f"Processed image {i+1}/{len(images)}")
            
            logger.info(f"Completed batch emotion classification for {len(images)} images")
            return results
            
        except Exception as e:
            logger.error(f"Error predicting emotions for batch: {str(e)}")
            raise