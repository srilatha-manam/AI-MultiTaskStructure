#Defines outputs/results of various stages in the text emotion classification pipeline.
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from datetime import datetime

@dataclass
class DataIngestionArtifact:
    """Data ingestion artifact"""
    dialogs_file_path: str
    total_dialogs: int
    ingestion_timestamp: datetime
    status: str

@dataclass
class DataPreprocessingArtifact:
    """Data preprocessing artifact"""
    preprocessed_data_path: str
    total_processed: int
    valid_texts: int
    invalid_texts: int
    preprocessing_timestamp: datetime
    status: str

@dataclass
class DataTransformationArtifact:
    """Data transformation artifact"""
    transformed_data_path: str
    feature_extraction_path: str
    total_transformed: int
    transformation_timestamp: datetime
    feature_names: List[str]
    status: str

@dataclass
class ModelTrainingArtifact:
    """Model training artifact"""
    model_path: str
    training_accuracy: float
    validation_accuracy: float
    training_loss: float
    validation_loss: float
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
class PredictionArtifact:
    """Prediction artifact"""
    predictions_path: str
    total_predictions: int
    average_confidence: float
    emotion_distribution: Dict[str, int]
    prediction_timestamp: datetime
    status: str

@dataclass
class EmotionResult:
    """Single emotion analysis result"""
    text_id: str
    original_text: str
    cleaned_text: str
    emotions: Dict[str, float]
    dominant_emotion: str
    confidence: float
    features: Dict[str, Any]
    timestamp: datetime

@dataclass
class BatchEmotionResult:
    """Batch emotion analysis result"""
    results: List[EmotionResult]
    total_processed: int
    successful_predictions: int
    failed_predictions: int
    average_confidence: float
    emotion_distribution: Dict[str, int]
    processing_time: float
    batch_timestamp: datetime