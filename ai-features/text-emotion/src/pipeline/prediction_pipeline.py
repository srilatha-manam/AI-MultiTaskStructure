import logging
from typing import Dict, Any, List
from datetime import datetime
import sys
import os
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.multioutput import MultiOutputClassifier
import random

# Add the project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

from shared.util.supabase_client import SupabaseClient
from ..components.data_transformation import DataTransformation
from ..entity.artifact_entity import EmotionResult, BatchEmotionResult
from ..exceptions import PredictionException

logger = logging.getLogger(__name__)

class PredictionPipeline:
    def __init__(self):
        """Initialize prediction pipeline with training capability"""
        try:
            self.supabase_client = SupabaseClient()
            self.data_transformation = DataTransformation()
            self.emotion_classes = ['joy', 'sadness', 'anger', 'fear', 'surprise', 'disgust', 'neutral']
            
            # ML Components for training
            self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
            self.classifier = MultiOutputClassifier(MultinomialNB())
            self.is_trained = False
            self.model_path = "artifacts/text_emotion_model.pkl"
            
            # Try to load existing model
            self._load_model()
            
            logger.info("Text emotion prediction pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing prediction pipeline: {str(e)}")
            raise PredictionException(f"Error initializing prediction pipeline: {str(e)}")

    def train_on_tenglish_dialogs(self) -> Dict[str, Any]:
        """Train model on Tenglish dialogs from Supabase"""
        try:
            logger.info("Starting training on Tenglish dialogs from Supabase...")
            
            # STEP 1: Fetch raw Tenglish dialogs from Supabase
            raw_dialogs = self.supabase_client.get_dialogs(limit=1000)
            
            if not raw_dialogs:
                logger.warning("No Tenglish dialogs found for training")
                return {"status": "failed", "message": "No training data available"}
            
            logger.info(f"Fetched {len(raw_dialogs)} raw Tenglish dialogs from Supabase")
            
            # STEP 2: PREPROCESS THE FETCHED DIALOGS (This was missing!)
            preprocessed_texts = []
            emotion_labels = []
            valid_dialogs = 0
            
            for dialog in raw_dialogs:
                try:
                    # Extract raw text from dialog
                    raw_text = dialog.get('content', '').strip()
                    if len(raw_text) < 3:  # Skip very short texts
                        continue
                    
                    logger.debug(f"Processing raw dialog: {raw_text[:50]}...")
                    
                    # STEP 3: PREPROCESS THE RAW TENGLISH TEXT
                    transformation_result = self.data_transformation.transform_text(raw_text)
                    
                    # Check if preprocessing was successful
                    if not transformation_result['is_valid']:
                        logger.debug(f"Skipping invalid text after preprocessing: {raw_text[:30]}...")
                        continue
                    
                    # Get the cleaned, preprocessed text
                    cleaned_text = transformation_result['cleaned_text']
                    
                    logger.debug(f"Cleaned text: {cleaned_text[:50]}...")
                    
                    # STEP 4: Generate emotion labels for the PREPROCESSED text
                    emotion_labels_for_text = self._generate_emotion_labels(cleaned_text)
                    
                    # Store preprocessed text and labels for training
                    preprocessed_texts.append(cleaned_text)
                    emotion_labels.append(emotion_labels_for_text)
                    valid_dialogs += 1
                    
                except Exception as e:
                    logger.error(f"Error preprocessing dialog {dialog.get('id', 'unknown')}: {str(e)}")
                    continue
            
            if len(preprocessed_texts) < 10:  # Need minimum data for training
                logger.warning("Insufficient valid preprocessed texts for training")
                return {"status": "failed", "message": "Insufficient preprocessed training data"}
            
            logger.info(f"Successfully preprocessed {len(preprocessed_texts)} Tenglish dialogs for training")
            logger.info(f"Example preprocessed text: {preprocessed_texts[0]}")
            
            # STEP 5: Train ML model on PREPROCESSED texts
            X = self.vectorizer.fit_transform(preprocessed_texts)
            y = np.array(emotion_labels)
            
            logger.info(f"Training classifier on {X.shape[0]} preprocessed samples with {X.shape[1]} features")
            
            # Train multi-output classifier
            self.classifier.fit(X, y)
            self.is_trained = True
            
            # Save trained model
            self._save_model()
            
            # Calculate training stats
            training_stats = {
                "status": "success",
                "total_raw_dialogs": len(raw_dialogs),
                "preprocessed_dialogs": len(preprocessed_texts),
                "preprocessing_success_rate": round(len(preprocessed_texts) / len(raw_dialogs) * 100, 2),
                "emotion_classes": self.emotion_classes,
                "training_accuracy": self._calculate_training_accuracy(X, y),
                "model_saved": True,
                "example_preprocessed_text": preprocessed_texts[0] if preprocessed_texts else "",
                "vectorizer_features": X.shape[1]
            }
            
            logger.info(f"Training completed successfully on preprocessed Tenglish data: {training_stats}")
            return training_stats
            
        except Exception as e:
            logger.error(f"Error training on Tenglish dialogs: {str(e)}")
            raise PredictionException(f"Error training on Tenglish dialogs: {str(e)}")

    def _generate_emotion_labels(self, preprocessed_text: str) -> List[float]:
        """Generate emotion labels for training using PREPROCESSED text"""
        # Use rule-based analysis on the CLEANED text
        emotions = self._analyze_emotion_patterns(preprocessed_text)
        
        # Convert to binary/multi-label format for training
        emotion_labels = []
        for emotion in self.emotion_classes:
            # Convert probabilities to binary labels (threshold = 0.3)
            label = 1.0 if emotions.get(emotion, 0) > 0.3 else 0.0
            emotion_labels.append(label)
        
        # Ensure at least one emotion is positive
        if sum(emotion_labels) == 0:
            emotion_labels[self.emotion_classes.index('neutral')] = 1.0
            
        return emotion_labels

    def _analyze_emotion_patterns(self, preprocessed_text: str) -> Dict[str, float]:
        """Analyze emotions using Tenglish-aware patterns on PREPROCESSED text"""
        # Enhanced patterns including some Tenglish expressions
        emotion_patterns = {
            'joy': [
                # English
                'happy', 'joy', 'excited', 'cheerful', 'delighted', 'pleased', 'glad', 
                'elated', 'thrilled', 'awesome', 'amazing', 'wonderful', 'fantastic', 
                'great', 'excellent', 'love', 'like', 'super', 'cool',
                # Common Tenglish expressions (after preprocessing)
                'baaga', 'bagundi', 'manchi', 'nice', 'superb', 'awesome ra'
            ],
            'sadness': [
                # English
                'sad', 'depressed', 'unhappy', 'miserable', 'heartbroken', 'disappointed', 
                'upset', 'down', 'blue', 'melancholy', 'gloomy', 'crying', 'tears', 
                'sorrow', 'grief', 'lonely',
                # Common Tenglish expressions (after preprocessing)
                'badha', 'kastam', 'sad ga', 'bore'
            ],
            'anger': [
                # English
                'angry', 'mad', 'furious', 'irritated', 'annoyed', 'frustrated', 'rage', 
                'hate', 'disgusted', 'outraged', 'livid', 'pissed', 'enraged', 'hostile',
                # Common Tenglish expressions (after preprocessing)
                'kopam', 'tension', 'irritating'
            ],
            'fear': [
                # English
                'afraid', 'scared', 'terrified', 'frightened', 'anxious', 'worried', 
                'nervous', 'panic', 'alarmed', 'concerned', 'uneasy', 'tense', 'stressed',
                # Common Tenglish expressions (after preprocessing)
                'bhayam', 'tension', 'worry'
            ],
            'surprise': [
                # English
                'surprised', 'shocked', 'amazed', 'astonished', 'stunned', 'bewildered', 
                'confused', 'unexpected', 'sudden', 'wow', 'omg', 'unbelievable',
                # Common Tenglish expressions (after preprocessing)
                'shock', 'surprising', 'unexpected ga'
            ],
            'disgust': [
                # English
                'disgusted', 'revolted', 'repulsed', 'sick', 'nauseated', 'appalled', 
                'horrified', 'gross', 'yuck', 'ew', 'awful', 'terrible',
                # Common Tenglish expressions (after preprocessing)
                'chi', 'yuck', 'disgusting'
            ],
            'neutral': [
                'okay', 'fine', 'normal', 'regular', 'usual', 'ordinary', 'standard', 
                'typical', 'common', 'average', 'alright', 'ok'
            ]
        }
        
        emotion_scores = {emotion: 0.0 for emotion in self.emotion_classes}
        text_lower = preprocessed_text.lower()  # Using preprocessed text
        words = text_lower.split()
        
        logger.debug(f"Analyzing emotion patterns in: {text_lower}")
        
        for emotion, keywords in emotion_patterns.items():
            score = 0.0
            matched_keywords = []
            for keyword in keywords:
                if keyword in text_lower:
                    score += 1.0
                    matched_keywords.append(keyword)
            
            # Normalize by text length
            if len(words) > 0:
                emotion_scores[emotion] = min(score / len(words) * 10, 1.0)
            
            if matched_keywords:
                logger.debug(f"Found {emotion} keywords: {matched_keywords}")
        
        return emotion_scores

    def predict_single(self, text: str, text_id: str = None) -> EmotionResult:
        """Predict emotion for single text using trained model"""
        try:
            if text_id is None:
                text_id = f"text_{datetime.now().timestamp()}"
            
            # STEP 1: PREPROCESS the input text (same as training data)
            transformation_result = self.data_transformation.transform_text(text)
            
            if not transformation_result['is_valid']:
                logger.warning(f"Invalid text for prediction: {text[:50]}...")
                return EmotionResult(
                    text_id=text_id,
                    original_text=text,
                    cleaned_text=transformation_result['cleaned_text'],
                    emotions={emotion: 0.0 for emotion in self.emotion_classes},
                    dominant_emotion='neutral',
                    confidence=0.0,
                    features=transformation_result['features'],
                    timestamp=datetime.now()
                )
            
            # STEP 2: Use PREPROCESSED text for prediction
            cleaned_text = transformation_result['cleaned_text']
            
            # Use trained model if available, otherwise fallback to rule-based
            if self.is_trained:
                emotions = self._predict_with_trained_model(cleaned_text)
                logger.info(f"Used trained model for prediction on: {cleaned_text[:50]}")
            else:
                emotions = self._analyze_emotion_patterns(cleaned_text)
                logger.info(f"Used rule-based prediction on: {cleaned_text[:50]}")
            
            dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
            confidence = emotions[dominant_emotion]
            
            return EmotionResult(
                text_id=text_id,
                original_text=text,
                cleaned_text=cleaned_text,
                emotions=emotions,
                dominant_emotion=dominant_emotion,
                confidence=confidence,
                features=transformation_result['features'],
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting single text: {str(e)}")
            raise PredictionException(f"Error predicting single text: {str(e)}")

    def _predict_with_trained_model(self, preprocessed_text: str) -> Dict[str, float]:
        """Predict emotions using trained ML model on PREPROCESSED text"""
        try:
            # Vectorize the PREPROCESSED text (same as training)
            X = self.vectorizer.transform([preprocessed_text])
            
            # Predict probabilities
            predictions = self.classifier.predict_proba(X)[0]
            
            # Convert to emotion dictionary
            emotions = {}
            for i, emotion in enumerate(self.emotion_classes):
                # Get probability of positive class (class 1)
                prob = predictions[i][1] if len(predictions[i]) > 1 else predictions[i][0]
                emotions[emotion] = float(prob)
            
            logger.debug(f"ML model prediction: {emotions}")
            return emotions
            
        except Exception as e:
            logger.error(f"Error predicting with trained model: {str(e)}")
            # Fallback to rule-based on preprocessed text
            return self._analyze_emotion_patterns(preprocessed_text)

    def predict_batch(self, texts: List[str], text_ids: List[str] = None) -> BatchEmotionResult:
        """Predict emotions for batch of texts"""
        start_time = datetime.now()
        
        try:
            if text_ids is None:
                text_ids = [f"text_{i}_{start_time.timestamp()}" for i in range(len(texts))]
            
            results = []
            successful = 0
            failed = 0
            
            for i, text in enumerate(texts):
                try:
                    result = self.predict_single(text, text_ids[i])
                    results.append(result)
                    if result.confidence > 0:
                        successful += 1
                    else:
                        failed += 1
                except Exception as e:
                    logger.error(f"Error predicting text {i}: {str(e)}")
                    failed += 1
                    continue
            
            # Calculate statistics
            processing_time = (datetime.now() - start_time).total_seconds()
            avg_confidence = sum(r.confidence for r in results) / len(results) if results else 0
            
            emotion_distribution = {}
            for result in results:
                emotion = result.dominant_emotion
                emotion_distribution[emotion] = emotion_distribution.get(emotion, 0) + 1
            
            return BatchEmotionResult(
                results=results,
                total_processed=len(texts),
                successful_predictions=successful,
                failed_predictions=failed,
                average_confidence=avg_confidence,
                emotion_distribution=emotion_distribution,
                processing_time=processing_time,
                batch_timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error predicting batch: {str(e)}")
            raise PredictionException(f"Error predicting batch: {str(e)}")

    def process_dialogs_from_supabase(self) -> BatchEmotionResult:
        """Process dialogs from Supabase without saving results"""
        try:
            # Get raw dialogs from Supabase
            raw_dialogs = self.supabase_client.get_dialogs()
            
            if not raw_dialogs:
                logger.warning("No dialogs found in Supabase")
                return BatchEmotionResult(
                    results=[],
                    total_processed=0,
                    successful_predictions=0,
                    failed_predictions=0,
                    average_confidence=0,
                    emotion_distribution={},
                    processing_time=0,
                    batch_timestamp=datetime.now()
                )
            
            # Extract raw texts and IDs
            raw_texts = [dialog.get('content', '') for dialog in raw_dialogs]
            text_ids = [str(dialog.get('id', '')) for dialog in raw_dialogs]
            
            # Predict batch (this will preprocess each text individually)
            batch_result = self.predict_batch(raw_texts, text_ids)
            
            logger.info(f"Processed {batch_result.total_processed} dialogs from Supabase (no saving)")
            return batch_result
            
        except Exception as e:
            logger.error(f"Error processing dialogs from Supabase: {str(e)}")
            raise PredictionException(f"Error processing dialogs from Supabase: {str(e)}")

    def _save_model(self):
        """Save trained model to disk"""
        try:
            os.makedirs("artifacts", exist_ok=True)
            model_data = {
                'vectorizer': self.vectorizer,
                'classifier': self.classifier,
                'emotion_classes': self.emotion_classes,
                'is_trained': self.is_trained
            }
            with open(self.model_path, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {self.model_path}")
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")

    def _load_model(self):
        """Load trained model from disk"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.vectorizer = model_data['vectorizer']
                self.classifier = model_data['classifier']
                self.emotion_classes = model_data['emotion_classes']
                self.is_trained = model_data['is_trained']
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.info("No saved model found, will use rule-based approach")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")

    def _calculate_training_accuracy(self, X, y) -> float:
        """Calculate training accuracy"""
        try:
            predictions = self.classifier.predict(X)
            accuracy = np.mean((predictions == y).all(axis=1))
            return float(accuracy)
        except:
            return 0.0