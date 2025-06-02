import logging
from typing import Dict, Any, List
from datetime import datetime
import numpy as np
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

from shared.util.supabase_client import SupabaseClient
from ..components.image_loader import ImageLoader
from ..components.emotion_classifier import EmotionClassifier
from ..entity.artifact_entity import EmotionResult, BatchEmotionResult
from ..exceptions import PredictionException

logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        """Initialize prediction pipeline"""
        try:
            self.supabase_client = SupabaseClient()
            self.image_loader = ImageLoader()
            self.emotion_classifier = EmotionClassifier()
            self.emotion_classes = ['anger', 'disgust', 'fear', 'joy', 'neutral', 'sadness', 'surprise']
            logger.info("Image prediction pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing prediction pipeline: {str(e)}")
            raise PredictionException(f"Error initializing prediction pipeline: {str(e)}")

    def predict_single(self, image: np.ndarray, image_id: str = None, image_url: str = None) -> EmotionResult:
        """Predict emotion for single image"""
        try:
            if image_id is None:
                image_id = f"image_{datetime.now().timestamp()}"
            
            # Validate image
            if not self.image_loader.validate_image(image):
                logger.warning(f"Invalid image for prediction: {image_id}")
                return EmotionResult(
                    image_id=image_id,
                    image_url=image_url or "",
                    emotions={emotion: 0.0 for emotion in self.emotion_classes},
                    dominant_emotion='neutral',
                    confidence=0.0,
                    face_detected=False,
                    num_faces=0,
                    facial_features={},
                    features={},
                    timestamp=datetime.now()
                )
            
            # Preprocess image
            processed_image = self.image_loader.preprocess_image(image)
            
            # Enhance image quality
            enhanced_image = self.image_loader.enhance_image(processed_image)
            
            # Predict emotions
            result = self.emotion_classifier.predict(enhanced_image)
            
            return EmotionResult(
                image_id=image_id,
                image_url=image_url or "",
                emotions=result["emotions"],
                dominant_emotion=result["dominant_emotion"],
                confidence=result["confidence"],
                face_detected=result["face_detected"],
                num_faces=result["num_faces"],
                facial_features=result.get("facial_features", {}),
                features=result.get("features", {}),
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting single image: {str(e)}")
            raise PredictionException(f"Error predicting single image: {str(e)}")

    def predict_batch(self, images: List[np.ndarray], image_ids: List[str] = None, image_urls: List[str] = None) -> BatchEmotionResult:
        """Predict emotions for batch of images"""
        start_time = datetime.now()
        
        try:
            if image_ids is None:
                image_ids = [f"image_{i}_{start_time.timestamp()}" for i in range(len(images))]
            
            if image_urls is None:
                image_urls = [""] * len(images)
            
            results = []
            successful = 0
            failed = 0
            faces_detected = 0
            
            for i, image in enumerate(images):
                try:
                    result = self.predict_single(
                        image, 
                        image_ids[i] if i < len(image_ids) else f"image_{i}",
                        image_urls[i] if i < len(image_urls) else ""
                    )
                    results.append(result)
                    
                    if result.confidence > 0:
                        successful += 1
                    else:
                        failed += 1
                        
                    if result.face_detected:
                        faces_detected += 1
                        
                except Exception as e:
                    logger.error(f"Error predicting image {i}: {str(e)}")
                    failed += 1
                    continue
            
            # Calculate statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
            
            emotion_distribution = {}
            for result in results:
                emotion = result.dominant_emotion
                emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            
            return BatchEmotionResult(
                results=results,
                total_processed=len(images),
                successful_predictions=successful,
                failed_predictions=failed,
                faces_detected=faces_detected,
                average_confidence=avg_confidence,
                emotion_distribution=emotion_distribution,
                processing_time=processing_time,
                batch_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting batch: {str(e)}")
            raise PredictionException(f"Error predicting batch: {str(e)}")

    def process_images_from_supabase(self, limit: int = 20) -> BatchEmotionResult:
        """Process images from Supabase and return results"""
        try:
            # Get images from Supabase
            images_data = self.supabase_client.get_images(limit)
            
            if not images_data:
                logger.warning("No images found in Supabase")
                return BatchEmotionResult(
                    results=[],
                    total_processed=0,
                    successful_predictions=0,
                    failed_predictions=0,
                    faces_detected=0,
                    average_confidence=0,
                    emotion_distribution={},
                    processing_time=0,
                    batch_timestamp=datetime.now()
                )
            
            # Load images from URLs
            images = []
            image_ids = []
            image_urls = []
            
            for image_data in images_data:
                try:
                    image_url = image_data.get('image_url', '')
                    if not image_url:
                        continue
                        
                    # Load image from URL
                    image_array = self.image_loader.load_from_url(image_url)
                    
                    if image_array is not None:
                        images.append(image_array)
                        image_ids.append(str(image_data.get('id', '')))
                        image_urls.append(image_url)
                    else:
                        logger.warning(f"Could not load image from URL: {image_url}")
                        
                except Exception as e:
                    logger.error(f"Error loading image {image_data.get('id', 'unknown')}: {str(e)}")
                    continue
            
            if not images:
                logger.warning("No valid images could be loaded")
                return BatchEmotionResult(
                    results=[],
                    total_processed=0,
                    successful_predictions=0,
                    failed_predictions=0,
                    faces_detected=0,
                    average_confidence=0,
                    emotion_distribution={},
                    processing_time=0,
                    batch_timestamp=datetime.now()
                )
            
            # Predict batch
            batch_result = self.predict_batch(images, image_ids, image_urls)
            
            # Save results to Supabase
            for result in batch_result.results:
                emotion_dict = {
                    'emotions': result.emotions,
                    'dominant_emotion': result.dominant_emotion,
                    'confidence': result.confidence,
                    'face_detected': result.face_detected
                }
                self.supabase_client.save_emotion_result(result.image_id, emotion_dict, 'image')
            
            logger.info(f"Processed {batch_result.total_processed} images from Supabase")
            return batch_result
            
        except Exception as e:
            logger.error(f"Error processing images from Supabase: {str(e)}")
            raise PredictionException(f"Error processing images from Supabase: {str(e)}")

    def predict_from_url(self, image_url: str, image_id: str = None) -> EmotionResult:
        """Predict emotion from image URL"""
        try:
            # Load image from URL
            image_array = self.image_loader.load_from_url(image_url)
            
            if image_array is None:
                raise PredictionException(f"Could not load image from URL: {image_url}")
            
            # Predict emotion
            return self.predict_single(image_array, image_id, image_url)
            
        except Exception as e:
            logger.error(f"Error predicting from URL {image_url}: {str(e)}")
            raise PredictionException(f"Error predicting from URL {image_url}: {str(e)}")