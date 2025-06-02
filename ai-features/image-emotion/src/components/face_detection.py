import cv2
import numpy as np
import logging
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)

class FaceDetection:
    def __init__(self):
        """Initialize face detection"""
        try:
            # Load Haar Cascade classifiers
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            
            logger.info("Face detection initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing face detection: {str(e)}")
            raise

    def detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image"""
        try:
            if image is None:
                return []
            
            # Convert to grayscale
            if len(image.shape) == 3:
                if image.max() <= 1.0:
                    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8) if image.max() <= 1.0 else image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            return [tuple(face) for face in faces]
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []

    def detect_facial_features(self, image: np.ndarray, face_region: Tuple[int, int, int, int]) -> dict:
        """Detect facial features within a face region"""
        try:
            x, y, w, h = face_region
            
            # Extract face region
            if len(image.shape) == 3:
                if image.max() <= 1.0:
                    gray = cv2.cvtColor((image * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = image.astype(np.uint8) if image.max() <= 1.0 else image
                
            face_gray = gray[y:y+h, x:x+w]
            
            # Detect eyes
            eyes = self.eye_cascade.detectMultiScale(face_gray, scaleFactor=1.1, minNeighbors=5)
            
            # Detect smile
            smiles = self.smile_cascade.detectMultiScale(face_gray, scaleFactor=1.8, minNeighbors=20)
            
            features = {
                'eyes': len(eyes),
                'eye_positions': eyes.tolist() if len(eyes) > 0 else [],
                'smiles': len(smiles),
                'smile_positions': smiles.tolist() if len(smiles) > 0 else [],
                'has_smile': len(smiles) > 0,
                'eye_symmetry': self._calculate_eye_symmetry(eyes) if len(eyes) >= 2 else 0
            }
            
            return features
            
        except Exception as e:
            logger.error(f"Error detecting facial features: {str(e)}")
            return {
                'eyes': 0,
                'eye_positions': [],
                'smiles': 0,
                'smile_positions': [],
                'has_smile': False,
                'eye_symmetry': 0
            }

    def get_largest_face(self, faces: List[Tuple[int, int, int, int]]) -> Optional[Tuple[int, int, int, int]]:
        """Get the largest face from detected faces"""
        if not faces:
            return None
        
        return max(faces, key=lambda f: f[2] * f[3])

    def extract_face_region(self, image: np.ndarray, face: Tuple[int, int, int, int], padding: int = 20) -> Optional[np.ndarray]:
        """Extract face region with padding"""
        try:
            x, y, w, h = face
            
            # Calculate bounds with padding
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            # Extract face region
            face_region = image[y_start:y_end, x_start:x_end]
            
            return face_region
            
        except Exception as e:
            logger.error(f"Error extracting face region: {str(e)}")
            return None

    def _calculate_eye_symmetry(self, eyes: np.ndarray) -> float:
        """Calculate symmetry between detected eyes"""
        try:
            if len(eyes) < 2:
                return 0.0
            
            # Take first two eyes
            eye1, eye2 = eyes[:2]
            
            # Calculate centers
            center1 = (eye1[0] + eye1[2] // 2, eye1[1] + eye1[3] // 2)
            center2 = (eye2[0] + eye2[2] // 2, eye2[1] + eye2[3] // 2)
            
            # Calculate distance and vertical alignment
            horizontal_distance = abs(center1[0] - center2[0])
            vertical_distance = abs(center1[1] - center2[1])
            
            # Symmetry score (higher when eyes are horizontally aligned)
            if horizontal_distance > 0:
                symmetry = 1.0 - (vertical_distance / horizontal_distance)
                return max(0.0, min(1.0, symmetry))
            
            return 0.0
            
        except Exception as e:
            logger.error(f"Error calculating eye symmetry: {str(e)}")
            return 0.0

    def analyze_face_geometry(self, face_region: np.ndarray) -> dict:
        """Analyze geometric properties of face region"""
        try:
            if face_region is None:
                return {}
            
            h, w = face_region.shape[:2]
            
            # Basic geometric features
            aspect_ratio = w / h if h > 0 else 1.0
            area = h * w
            
            # Convert to grayscale for analysis
            if len(face_region.shape) == 3:
                if face_region.max() <= 1.0:
                    gray = cv2.cvtColor((face_region * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
                else:
                    gray = cv2.cvtColor(face_region.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                gray = face_region.astype(np.uint8) if face_region.max() <= 1.0 else face_region
            
            # Calculate symmetry
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            
            # Resize to same size if needed
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            # Calculate correlation
            try:
                correlation = cv2.matchTemplate(left_half.astype(np.float32), 
                                              right_half.astype(np.float32), 
                                              cv2.TM_CCOEFF_NORMED)
                symmetry_score = float(np.max(correlation))
            except:
                symmetry_score = 0.5
            
            return {
                'aspect_ratio': aspect_ratio,
                'area': area,
                'width': w,
                'height': h,
                'symmetry_score': symmetry_score
            }
            
        except Exception as e:
            logger.error(f"Error analyzing face geometry: {str(e)}")
            return {
                'aspect_ratio': 1.0,
                'area': 0,
                'width': 0,
                'height': 0,
                'symmetry_score': 0.5
            }