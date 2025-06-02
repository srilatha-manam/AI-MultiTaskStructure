class ImageEmotionException(Exception):
    """Base exception for image emotion detection"""
    pass

class DataIngestionException(ImageEmotionException):
    """Exception for data ingestion errors"""
    pass

class ImageProcessingException(ImageEmotionException):
    """Exception for image processing errors"""
    pass

class FaceDetectionException(ImageEmotionException):
    """Exception for face detection errors"""
    pass

class ModelTrainingException(ImageEmotionException):
    """Exception for model training errors"""
    pass

class ModelEvaluationException(ImageEmotionException):
    """Exception for model evaluation errors"""
    pass

class PredictionException(ImageEmotionException):
    """Exception for prediction errors"""
    pass

class ConfigurationException(ImageEmotionException):
    """Exception for configuration errors"""
    pass