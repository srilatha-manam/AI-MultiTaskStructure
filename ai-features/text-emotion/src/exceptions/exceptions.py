class TextEmotionException(Exception):
    """Base exception for text emotion detection"""
    pass

class DataIngestionException(TextEmotionException):
    """Exception for data ingestion errors"""
    pass

class DataPreprocessingException(TextEmotionException):
    """Exception for data preprocessing errors"""
    pass

class DataTransformationException(TextEmotionException):
    """Exception for data transformation errors"""
    pass

class ModelTrainingException(TextEmotionException):
    """Exception for model training errors"""
    pass

class ModelEvaluationException(TextEmotionException):
    """Exception for model evaluation errors"""
    pass

class PredictionException(TextEmotionException):
    """Exception for prediction errors"""
    pass

class ConfigurationException(TextEmotionException):
    """Exception for configuration errors"""
    pass