from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime
import numpy as np

@dataclass
class DataIngestionArtifact:
    """Data ingestion artifact"""
    images_file_path: str
    total_images: int
    ingestion_timestamp: datetime
    status: str

@dataclass
class ImageProcessingArtifact:
    """Image processing artifact"""
    processed_images_path: str
    total_processed: int
    valid_images: int
    invalid_images: int
    processing_timestamp: datetime
    status: str

@dataclass
class FaceDetectionArtifact:
    """Face detection artifact"""
    face_detection_results_path: str
    total_faces_detected: int
    images_with_faces: int
    images_without_faces: int
    detection_timestamp: datetime
    status: str

@dataclass
class ModelTrainingArtifact:
    """Model training artifact"""
    model_path: str
    training_accuracy: float
    validation_accuracy: float
    training_timestamp: datetime
    status: str

@dataclass
class ModelEvaluationArtifact:
    """Model evaluation artifact"""
    evaluation_report_path: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix_path: str
    evaluation_timestamp: datetime
    status: str

@dataclass
class EmotionResult:
    """Single emotion analysis result"""
    image_id: str
    image_url: str
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    face_detected: bool
    num_faces: int
    facial_features: Dict[str, Any]
    features: Dict[str, Any]
    timestamp: datetime

@dataclass
class BatchEmotionResult:
    """Batch emotion analysis result"""
    results: List[EmotionResult]
    total_processed: int
    successful_predictions: int
    failed_predictions: int
    faces_detected: int
    average_confidence: float
    emotion_distribution: Dict[str, int]
    processing_time: float
    batch_timestamp: datetime