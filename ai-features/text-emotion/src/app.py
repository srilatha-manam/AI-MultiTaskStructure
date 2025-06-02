from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any
import logging
import sys
import os

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
sys.path.insert(0, project_root)

from shared.util.supabase_client import SupabaseClient
from .pipeline.prediction_pipeline import PredictionPipeline
from .logger import setup_logger
from .exceptions import TextEmotionException

# Setup logging
logger = setup_logger("text_emotion_api")

# Initialize FastAPI app
app = FastAPI(
    title="Text Emotion Detection API",
    description="Train and analyze emotions in Tenglish text using ML",
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
class TextRequest(BaseModel):
    text: str

class TextResponse(BaseModel):
    text: str
    emotions: Dict[str, float]
    confidence: float
    dominant_emotion: str

@app.get("/")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "text-emotion-detection",
        "version": "1.0.0",
        "python_version": "3.9",
        "model_trained": prediction_pipeline.is_trained
    }

@app.post("/train")
async def train_model():
    """Train model on Tenglish dialogs from Supabase"""
    try:
        logger.info("Starting training on Tenglish dialogs...")
        
        # Train on Tenglish dialogs from Supabase
        training_result = prediction_pipeline.train_on_tenglish_dialogs()
        
        return {
            "status": "success",
            "message": "Model trained on Tenglish dialogs",
            "training_stats": training_result
        }
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-text", response_model=TextResponse)
async def analyze_text(request: TextRequest):
    """Analyze emotion in a single text using trained model"""
    try:
        logger.info(f"Analyzing text: {request.text[:50]}...")
        
        # Predict emotion using trained model
        result = prediction_pipeline.predict_single(request.text)
        
        return TextResponse(
            text=request.text,
            emotions=result.emotions,
            confidence=result.confidence,
            dominant_emotion=result.dominant_emotion
        )
        
    except Exception as e:
        logger.error(f"Error analyzing text: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/process-dialogs")
async def process_dialogs():
    """Process all dialogs from Supabase (NO SAVING)"""
    try:
        logger.info("Starting dialog processing...")
        
        # Process dialogs using trained model (NO SAVING TO SUPABASE)
        batch_result = prediction_pipeline.process_dialogs_from_supabase()
        
        # Convert results for response
        processed_results = []
        for result in batch_result.results[:10]:  # Return first 10 for preview
            processed_results.append({
                "dialog_id": result.text_id,
                "original_text": result.original_text,
                "cleaned_text": result.cleaned_text,
                "emotions": result.emotions,
                "dominant_emotion": result.dominant_emotion,
                "confidence": result.confidence,
                "timestamp": result.timestamp.isoformat()
            })
        
        logger.info(f"Processed {batch_result.total_processed} dialogs successfully (no saving)")
        
        return {
            "status": "success",
            "processed_count": batch_result.successful_predictions,
            "failed_count": batch_result.failed_predictions,
            "total_count": batch_result.total_processed,
            "average_confidence": batch_result.average_confidence,
            "emotion_distribution": batch_result.emotion_distribution,
            "processing_time": batch_result.processing_time,
            "model_used": "trained" if prediction_pipeline.is_trained else "rule-based",
            "results": processed_results,
            "note": "Results not saved to Supabase"
        }
        
    except Exception as e:
        logger.error(f"Error processing dialogs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-batch")
async def analyze_batch(texts: List[str]):
    """Analyze multiple texts at once using trained model"""
    try:
        logger.info(f"Analyzing batch of {len(texts)} texts")
        
        # Predict batch using trained model
        batch_result = prediction_pipeline.predict_batch(texts)
        
        # Convert results for response
        results = []
        for result in batch_result.results:
            results.append({
                "text_id": result.text_id,
                "original_text": result.original_text,
                "emotions": result.emotions,
                "dominant_emotion": result.dominant_emotion,
                "confidence": result.confidence
            })
        
        return {
            "status": "success",
            "processed_count": batch_result.successful_predictions,
            "failed_count": batch_result.failed_predictions,
            "total_count": batch_result.total_processed,
            "average_confidence": batch_result.average_confidence,
            "emotion_distribution": batch_result.emotion_distribution,
            "processing_time": batch_result.processing_time,
            "model_used": "trained" if prediction_pipeline.is_trained else "rule-based",
            "results": results
        }
        
    except Exception as e:
        logger.error(f"Error analyzing batch: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-status")
async def get_model_status():
    """Get current model training status"""
    return {
        "is_trained": prediction_pipeline.is_trained,
        "emotion_classes": prediction_pipeline.emotion_classes,
        "model_type": "Multi-output Naive Bayes with TF-IDF",
        "supports_tenglish": True
    }