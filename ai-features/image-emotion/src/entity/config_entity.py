from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple

@dataclass
class DataIngestionConfig:
    """Data ingestion configuration"""
    supabase_url: str
    supabase_key: str
    batch_size: int = 20
    table_name: str = "images"

@dataclass
class ImageProcessingConfig:
    """Image processing configuration"""
    target_size: Tuple[int, int] = (224, 224)
    max_image_size: int = 2048
    normalize: bool = True
    enhance_contrast: bool = True

@dataclass
class FaceDetectionConfig:
    """Face detection configuration"""
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_size: Tuple[int, int] = (30, 30)
    padding: int = 20

@dataclass
class ModelConfig:
    """Model configuration"""
    emotion_classes: list
    confidence_threshold: float = 0.5
    face_required: bool = True
    use_facial_features: bool = True

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    data_ingestion_config: DataIngestionConfig
    image_processing_config: ImageProcessingConfig
    face_detection_config: FaceDetectionConfig
    model_config: ModelConfig

@dataclass
class PredictionConfig:
    """Prediction configuration"""
    batch_size: int = 10
    timeout: int = 30
    save_features: bool = True