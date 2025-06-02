import re
import string
import logging
from typing import List, Dict, Any
from textblob import TextBlob
import nltk

logger = logging.getLogger(__name__)

class DataPreprocessing:
    def __init__(self):
        """Initialize text preprocessing"""
        try:
            # Check if punkt tokenizer is available
            try:
                nltk.data.find('tokenizers/punkt') # 
            except LookupError:
                nltk.download('punkt')
                
            logger.info("Data preprocessing initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing data preprocessing: {str(e)}")
            raise

    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        try:
            if not text or not isinstance(text, str):
                return ""
            
            # Convert to lowercase
            text = text.lower()
            
            # Remove URLs
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
            
            # Remove email addresses
            text = re.sub(r'\S+@\S+', '', text)
            
            # Remove mentions and hashtags
            text = re.sub(r'@\w+|#\w+', '', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            # Remove leading/trailing whitespace
            text = text.strip()
            
            # Remove excessive punctuation but keep some for emotion context
            text = re.sub(r'[!]{2,}', '!', text)
            text = re.sub(r'[?]{2,}', '?', text)
            text = re.sub(r'[.]{3,}', '...', text)
            
            return text
            
        except Exception as e:
            logger.error(f"Error cleaning text: {str(e)}")
            return text if text else ""

    def extract_features(self, text: str) -> Dict[str, Any]:
        """Extract features from text for emotion analysis"""
        try:
            if not text:
                return self._empty_features()
            
            blob = TextBlob(text)
            
            features = {
                'text_length': len(text),
                'word_count': len(text.split()),
                'sentence_count': len(blob.sentences),
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity,
                'exclamation_count': text.count('!'),
                'question_count': text.count('?'),
                'uppercase_ratio': sum(1 for c in text if c.isupper()) / len(text) if len(text) > 0 else 0,
                'punctuation_ratio': sum(1 for c in text if c in string.punctuation) / len(text) if len(text) > 0 else 0
            }
            
            # Add word-based features
            words = text.split()
            if words:
                avg_word_length = sum(len(word) for word in words) / len(words)
                features['avg_word_length'] = avg_word_length
            else:
                features['avg_word_length'] = 0
            
            # Emotion indicators
            emotion_keywords = self._get_emotion_keywords()
            for emotion, keywords in emotion_keywords.items():
                count = sum(1 for word in words if word in keywords)
                features[f'{emotion}_keywords'] = count
            
            return features
            
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}")
            return self._empty_features()

    def _empty_features(self) -> Dict[str, Any]:
        """Return empty feature set"""
        return {
            'text_length': 0,
            'word_count': 0,
            'sentence_count': 0,
            'polarity': 0.0,
            'subjectivity': 0.0,
            'exclamation_count': 0,
            'question_count': 0,
            'uppercase_ratio': 0.0,
            'punctuation_ratio': 0.0,
            'avg_word_length': 0.0,
            'joy_keywords': 0,
            'sadness_keywords': 0,
            'anger_keywords': 0,
            'fear_keywords': 0,
            'surprise_keywords': 0,
            'disgust_keywords': 0
        }

    def _get_emotion_keywords(self) -> Dict[str, List[str]]:
        """Get emotion-specific keywords for feature extraction"""
        return {
            'joy': ['happy', 'joy', 'excited', 'cheerful', 'delighted', 'pleased', 'glad', 'elated', 'thrilled', 'awesome', 'amazing', 'wonderful', 'fantastic', 'great', 'excellent', 'love', 'like'],
            'sadness': ['sad', 'depressed', 'unhappy', 'miserable', 'heartbroken', 'disappointed', 'upset', 'down', 'blue', 'melancholy', 'gloomy', 'crying', 'tears', 'sorrow', 'grief'],
            'anger': ['angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 'hate', 'disgusted', 'outraged', 'livid', 'pissed', 'enraged', 'hostile'],
            'fear': ['afraid', 'scared', 'terrified', 'frightened', 'anxious', 'worried', 'nervous', 'panic', 'alarmed', 'concerned', 'uneasy', 'tense', 'stressed'],
            'surprise': ['surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered', 'confused', 'unexpected', 'sudden', 'wow', 'omg', 'unbelievable'],
            'disgust': ['disgusted', 'revolted', 'repulsed', 'sick', 'nauseated', 'appalled', 'horrified', 'gross', 'yuck', 'ew', 'awful', 'terrible']
        }

    def validate_text(self, text: str) -> bool:
        """Validate if text is suitable for emotion analysis"""
        try:
            if not text or not isinstance(text, str):
                return False
            
            # Check minimum length
            if len(text.strip()) < 3:
                return False
            
            # Check if text contains at least one letter
            if not re.search(r'[a-zA-Z]', text):
                return False
            
            # Check if text is not just punctuation or numbers
            cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
            if len(cleaned.strip()) < 2:
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating text: {str(e)}")
            return False

    def preprocess_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Preprocess a batch of texts"""
        try:
            # Initialize an empty list to store the results
            results = []
            # Iterate through each text in the batch
            for text in texts:
                # Clean the text
                cleaned_text = self.clean_text(text)
                # Extract features from the cleaned text
                features = self.extract_features(cleaned_text)
                
                # Append the original text, cleaned text, and features to the results list
                results.append({
                    'original_text': text,
                    'cleaned_text': cleaned_text,
                    'features': features
                })
            
            # Log the number of texts processed
            logger.info(f"Processed batch of {len(texts)} texts")
            # Return the results list
            return results
            
        except Exception as e:
            # Log any errors that occur during preprocessing
            logger.error(f"Error preprocessing batch: {str(e)}")
            # Raise the exception
            raise