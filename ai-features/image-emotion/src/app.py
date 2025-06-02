from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import logging
import io
import numpy as np
from PIL import Image
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, project_root)

from shared.util.supabase_client import SupabaseClient
from .pipeline.prediction_pipeline import PredictionPipeline
from .logger import setup_logger
from .exceptions import ImageEmotionException

# Setup logging
logger = setup_logger("image_emotion_api")

# Initialize FastAPI app
app = FastAPI(
    title="Image Emotion Detection API",
    description="Analyze emotions in facial images using computer vision",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
prediction_pipeline = PredictionPipeline()
supabase_client = SupabaseClient()

# Pydantic models
class ImageResponse(BaseModel):
    image_id: Optional[str] = None
    emotions: Dict[str, float]
    confidence: float
    dominant_emotion: str
    face_detected: bool

class EmotionResponse(BaseModel):
    emotion_id: str
    image_url: str
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    face_detected: bool
    timestamp: str

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "image-emotion-detection",
        "version": "1.0.0",
        "python_version": "3.10"
    }

@app.post("/analyze-image", response_model=ImageResponse)
async def analyze_image(file: UploadFile = File(...)):
    """Analyze emotion in uploaded image"""
    try:
        logger.info(f"Analyzing uploaded image: {file.filename}")
        
        # Read image file
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Predict emotion
        result = prediction_pipeline.predict_single(image_array)
        
        return ImageResponse(
            image_id=result.image_id,
            emotions=result.emotions,
            confidence=result.confidence,
            dominant_emotion=result.dominant_emotion,
            face_detected=result.face_detected
        )
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-images")
async def process_images():
    """Process images from Supabase"""
    try:
        logger.info("Starting image processing from Supabase...")
        
        # Process images using pipeline (limit to 20 as requested)
        batch_result = prediction_pipeline.process_images_from_supabase(limit=20)
        
        # Convert results for response
        processed_results = []
        for result in batch_result.results:
            processed_results.append({
                "image_id": result.image_id,
                "image_url": result.image_url,
                "emotions": result.emotions,
                "dominant_emotion": result.dominant_emotion,
                "confidence": result.confidence,
                "face_detected": result.face_detected,
                "num_faces": result.num_faces,
                "timestamp": result.timestamp.isoformat()
            })
        
        logger.info(f"Processed {batch_result.total_processed} images successfully")
        
        return {
            "status": "success",
            "processed_count": batch_result.successful_predictions,
            "failed_count": batch_result.failed_predictions,
            "total_count": batch_result.total_processed,
            "faces_detected": batch_result.faces_detected,
            "face_detection_rate": round(batch_result.faces_detected / batch_result.total_processed * 100, 2) if batch_result.total_processed > 0 else 0,
            "average_confidence": batch_result.average_confidence,
            "emotion_distribution": batch_result.emotion_distribution,
            "processing_time": batch_result.processing_time,
            "results": processed_results
        }
        
    except Exception as e:
        logger.error(f"Error processing images: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/emotions")
async def get_processed_emotions():
    """Get processed emotion results"""
    try:
        emotions = supabase_client.get_processed_emotions('image')
        return {
            "status": "success",
            "count": len(emotions),
            "emotions": emotions
        }
    except Exception as e:
        logger.error(f"Error retrieving emotions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
async def get_stats():
    """Get processing statistics"""
    try:
        stats = supabase_client.get_emotion_stats('image')
        return {
            "status": "success",
            "statistics": stats
        }
    except Exception as e:
        logger.error(f"Error retrieving stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch")
async def analyze_batch_images(files: List[UploadFile] = File(...)):
    """Analyze multiple images at once"""
    try:
        logger.info(f"Analyzing batch of {len(files)} images")
        
        images = []
        image_ids = []
        
        for i, file in enumerate(files):
            try:
                # Read and process image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data))
                image_array = np.array(image)
                
                images.append(image_array)
                image_ids.append(f"upload_{i}_{file.filename}")
                
            except Exception as e:
                logger.error(f"Error processing {file.filename}: {str(e)}")
                continue
        
        if not images:
            raise HTTPException(status_code=400, detail="No valid images found")
        
        # Predict batch
        batch_result = prediction_pipeline.predict_batch(images, image_ids)
        
        # Convert results for response
        results = []
        for result in batch_result.results:
            results.append({
                "image_id": result.image_id,
                "emotions": result.emotions,
                "dominant_emotion": result.dominant_emotion,
                "confidence": result.confidence,
                "face_detected": result.face_detected,
                "num_faces": result.num_faces
            })
        
        return {
            "status": "success",
            "processed_count": batch_result.successful_predictions,
            "failed_count": batch_result.failed_predictions,
            "total_count": batch_result.total_processed,
            "faces_detected": batch_result.faces_detected,
            "face_detection_rate": round(batch_result.faces_detected / batch_result.total_processed * 100, 2) if batch_result.total_processed > 0 else 0,
            "average_confidence": batch_result.average_confidence,
            "emotion_distribution": batch_result.emotion_distribution,
            "processing_time": batch_result.processing_time,
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error analyzing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-url")
async def analyze_image_url(image_url: str):
    """Analyze emotion from image URL"""
    try:
        logger.info(f"Analyzing image from URL: {image_url}")
        
        # Predict emotion from URL
        result = prediction_pipeline.predict_from_url(image_url)
        
        return {
            "image_id": result.image_id,
            "image_url": result.image_url,
            "emotions": result.emotions,
            "dominant_emotion": result.dominant_emotion,
            "confidence": result.confidence,
            "face_detected": result.face_detected,
            "num_faces": result.num_faces,
            "timestamp": result.timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error analyzing image URL: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))