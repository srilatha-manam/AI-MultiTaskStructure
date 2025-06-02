import logging
from typing import List, Dict, Any
from .data_preprocessing import DataPreprocessing

logger = logging.getLogger(__name__)

class DataTransformation:
    def __init__(self):
        """Initialize data transformation"""
        try:
            self.preprocessor = DataPreprocessing()
            logger.info("Data transformation initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing data transformation: {str(e)}")
            raise

    def transform_text(self, text: str) -> Dict[str, Any]:
        """Transform single text"""
        try:
            # Clean the text
            cleaned_text = self.preprocessor.clean_text(text)
            
            # Extract features
            features = self.preprocessor.extract_features(cleaned_text)
            
            # Validate text
            is_valid = self.preprocessor.validate_text(cleaned_text)
            
            return {
                'original_text': text,
                'cleaned_text': cleaned_text,
                'features': features,
                'is_valid': is_valid
            }
            
        except Exception as e:
            logger.error(f"Error transforming text: {str(e)}")
            return {
                'original_text': text,
                'cleaned_text': '',
                'features': {},
                'is_valid': False
            }

    def transform_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Transform batch of texts"""
        try:
            results = []
            for text in texts:
                transformed = self.transform_text(text)
                results.append(transformed)
            
            logger.info(f"Transformed batch of {len(texts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"Error transforming batch: {str(e)}")
            raise

    def get_transformation_stats(self, transformations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about transformations"""
        try:
            valid_count = sum(1 for t in transformations if t['is_valid'])
            total_count = len(transformations)
            
            avg_length = sum(len(t['cleaned_text']) for t in transformations) / total_count if total_count > 0 else 0
            avg_words = sum(t['features'].get('word_count', 0) for t in transformations) / total_count if total_count > 0 else 0
            
            return {
                'total_texts': total_count,
                'valid_texts': valid_count,
                'invalid_texts': total_count - valid_count,
                'validation_rate': round(valid_count / total_count * 100, 2) if total_count > 0 else 0,
                'average_length': round(avg_length, 2),
                'average_words': round(avg_words, 2)
            }
            
        except Exception as e:
            logger.error(f"Error calculating transformation stats: {str(e)}")
            return {
                'total_texts': 0,
                'valid_texts': 0,
                'invalid_texts': 0,
                'validation_rate': 0,
                'average_length': 0,
                'average_words': 0
            }