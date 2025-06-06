import pickle
import json
import logging
import os
from typing import Dict, List
from kserve import Model, ModelServer
import numpy as np

class SentimentModel(Model):
    def __init__(self, name: str):
        super().__init__(name)
        self.name = name
        self.model = None
        self.vectorizer = None
        self.ready = False
        self.load()

    def load(self):
        """Load model and vectorizer from mounted volume"""
        try:
            model_path = "/mnt/models/sentiment_model.pkl"
            vectorizer_path = "/mnt/models/vectorizer.pkl"
            
            # Check if files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            if not os.path.exists(vectorizer_path):
                raise FileNotFoundError(f"Vectorizer file not found: {vectorizer_path}")
            
            # Load model and vectorizer
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            self.ready = True
            logging.info("Model and vectorizer loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise e

    def predict(self, payload: Dict, headers: Dict[str, str] = None) -> Dict:
        """Make sentiment predictions"""
        if not self.ready:
            raise Exception("Model not loaded")
        
        try:
            instances = payload.get("instances", [])
            if not instances:
                raise ValueError("No instances provided")
            
            predictions = []
            
            for instance in instances:
                if isinstance(instance, dict):
                    text = instance.get("text", "")
                elif isinstance(instance, str):
                    text = instance
                else:
                    text = str(instance)
                
                if not text:
                    raise ValueError("Empty text provided")
                
                # Transform text using the fitted vectorizer
                text_vector = self.vectorizer.transform([text])
                
                # Make prediction
                prediction = self.model.predict(text_vector)[0]
                probability = self.model.predict_proba(text_vector)[0]
                
                result = {
                    "prediction": int(prediction),
                    "sentiment": "positive" if prediction == 1 else "negative",
                    "confidence": {
                        "negative": float(probability[0]),
                        "positive": float(probability[1])
                    },
                    "text": text
                }
                predictions.append(result)
            
            return {"predictions": predictions}
            
        except Exception as e:
            logging.error(f"Prediction error: {str(e)}")
            return {"error": str(e)}

if __name__ == "__main__":
    model = SentimentModel("sentiment-classifier")
    ModelServer().start([model])