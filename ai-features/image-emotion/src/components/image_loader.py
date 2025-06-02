import cv2
import numpy as np
import requests
from PIL import Image
import io
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class ImageLoader:
    def __init__(self):
        """Initialize image loader"""
        try:
            # Load Haar Cascade for face detection
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            logger.info("Image loader initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing image loader: {str(e)}")
            raise

    def load_from_url(self, url: str, timeout: int = 10) -> Optional[np.ndarray]:
        """Load image from URL"""
        try:
            # Make request with timeout
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Convert to PIL Image
            image = Image.open(io.BytesIO(response.content))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            
            logger.info(f"Successfully loaded image from URL: {url}")
            return image_array
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error downloading image from {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error processing image from {url}: {str(e)}")
            return None

    def preprocess_image(self, image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """Preprocess image for emotion detection"""
        try:
            if image is None:
                raise ValueError("Input image is None")
            
            # Convert BGR to RGB if needed (OpenCV loads as BGR)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume it's already RGB from PIL
                rgb_image = image
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize image
            resized_image = cv2.resize(rgb_image, target_size)
            
            # Normalize pixel values to [0, 1]
            normalized_image = resized_image.astype(np.float32) / 255.0
            
            return normalized_image
            
        except Exception as e:
            logger.error(f"Error preprocessing image: {str(e)}")
            # Return a black image as fallback
            return np.zeros((target_size[0], target_size[1], 3), dtype=np.float32)

    def detect_faces(self, image: np.ndarray) -> list:
        """Detect faces in image using Haar Cascades"""
        try:
            if image is None:
                return []
            
            # Convert to grayscale for face detection
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            logger.info(f"Detected {len(faces)} faces in image")
            return faces.tolist() if len(faces) > 0 else []
            
        except Exception as e:
            logger.error(f"Error detecting faces: {str(e)}")
            return []

    def extract_face_region(self, image: np.ndarray, face_coords: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
        """Extract face region from image"""
        try:
            x, y, w, h = face_coords
            
            # Add some padding around the face
            padding = 20
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(image.shape[1] - x, w + 2 * padding)
            h = min(image.shape[0] - y, h + 2 * padding)
            
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            
            return face_region
            
        except Exception as e:
            logger.error(f"Error extracting face region: {str(e)}")
            return None

    def validate_image(self, image: np.ndarray) -> bool:
        """Validate if image is suitable for emotion detection"""
        try:
            if image is None:
                return False
            
            # Check image dimensions
            if len(image.shape) not in [2, 3]:
                logger.warning("Invalid image dimensions")
                return False
            
            # Check image size
            if image.shape[0] < 32 or image.shape[1] < 32:
                logger.warning("Image too small for emotion detection")
                return False
            
            # Check if image has content (not all zeros)
            if np.all(image == 0):
                logger.warning("Image appears to be empty")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating image: {str(e)}")
            return False

    def enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance image quality for better emotion detection"""
        try:
            if image is None:
                return image
            
            # Convert to uint8 if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    image = (image * 255).astype(np.uint8)
                else:
                    image = image.astype(np.uint8)
            
            # Apply histogram equalization to improve contrast
            if len(image.shape) == 3:
                # Convert to LAB color space
                lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
                # Apply CLAHE to L channel
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                lab[:,:,0] = clahe.apply(lab[:,:,0])
                # Convert back to RGB
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
            else:
                # Grayscale image
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
                enhanced = clahe.apply(image)
            
            return enhanced
            
        except Exception as e:
            logger.error(f"Error enhancing image: {str(e)}")
            return image