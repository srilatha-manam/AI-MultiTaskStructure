#Defines the sructure for configuration entities used in the text emotion classification pipeline.
#uses model_config.yaml file as input.
from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class DataIngestionConfig:
    """Data ingestion configuration"""
    # URL of the Supabase database
    supabase_url: str
    # API key for the Supabase database
    supabase_key: str
    # Number of records to be ingested at a time
    batch_size: int = 100
    # Name of the table to which the data will be ingested
    table_name: str = "dialogs"

@dataclass
class DataPreprocessingConfig:
    """Data preprocessing configuration"""
    min_text_length: int = 3
    # Minimum length of text to be considered for preprocessing
    max_text_length: int = 5000
    remove_urls: bool = True
    remove_mentions: bool = True
    lowercase: bool = True

@dataclass
class DataTransformationConfig:
    """Data transformation configuration"""
    # Whether to extract features from the data
    extract_features: bool = True
    # Whether to validate the text in the data
    validate_text: bool = True
    # Whether to normalize the features in the data
    normalize_features: bool = True

@dataclass
class ModelConfig:
    """Model configuration"""
    # List of emotion classes
    emotion_classes: list
    # Confidence threshold for emotion detection
    confidence_threshold: float = 0.5
    # Use VADER for emotion detection
    use_vader: bool = True
    # Use TextBlob for emotion detection
    use_textblob: bool = True
    # Use patterns for emotion detection
    use_patterns: bool = True

@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    # Configuration for data ingestion
    data_ingestion_config: DataIngestionConfig
    # Configuration for data preprocessing
    data_preprocessing_config: DataPreprocessingConfig
    # Configuration for data transformation
    data_transformation_config: DataTransformationConfig
    # Configuration for model
    model_config: ModelConfig

@dataclass
class TrainingConfig:
    """Training configuration"""
    # Name of the model to be trained
    model_name: str = "text_emotion_classifier"
    # Number of epochs to train the model
    epochs: int = 10
    # Batch size for training the model
    batch_size: int = 32
    # Learning rate for training the model
    learning_rate: float = 0.001
    # Validation split for training the model
    validation_split: float = 0.2