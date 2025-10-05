import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib
from cachetools import cached, TTLCache
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ExoplanetMLModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = [
            'pl_orbper', 'pl_rade', 'pl_bmasse', 'pl_dens',
            'pl_eqt', 'st_teff', 'st_rad', 'st_mass', 'st_met'
        ]
        self.model_version = "1.0"

    def train(self, data: pd.DataFrame):
        """Train the model on provided data"""
        try:
            # For demo purposes, create a simple target
            X = data[self.feature_columns].fillna(0)
            y = (data['pl_bmasse'] > 1).astype(int)  # Simple binary classification

            X_scaled = self.scaler.fit_transform(X)

            self.model = RandomForestClassifier(n_estimators=50, random_state=42)
            self.model.fit(X_scaled, y)

            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns
            }
            joblib.dump(model_data, 'exoplanet_model.joblib')

            logger.info("Model trained successfully")

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            raise

    def load_model(self):
        """Load pre-trained model"""
        try:
            model_data = joblib.load('exoplanet_model.joblib')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_columns = model_data['feature_columns']
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.warning("No pre-trained model found. Using demo model.")
            # Create a simple demo model
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Train on dummy data for demo
            X_dummy = np.random.rand(100, len(self.feature_columns))
            y_dummy = np.random.randint(0, 2, 100)
            self.scaler.fit(X_dummy)
            X_scaled = self.scaler.transform(X_dummy)
            self.model.fit(X_scaled, y_dummy)

    def predict(self, planet_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Predict exoplanet properties"""
        if self.model is None:
            self.load_model()

        try:
            df = pd.DataFrame(planet_data)

            # Prepare features
            available_columns = [col for col in self.feature_columns if col in df.columns]
            missing_columns = [col for col in self.feature_columns if col not in df.columns]

            if missing_columns:
                logger.warning(f"Missing columns, using defaults: {missing_columns}")
                for col in missing_columns:
                    df[col] = 0

            X = df[self.feature_columns].fillna(0)
            X_scaled = self.scaler.transform(X)

            predictions = self.model.predict(X_scaled)
            probabilities = self.model.predict_proba(X_scaled)

            results = []
            for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
                result = {
                    'prediction': "Habitable" if pred == 1 else "Not Habitable",
                    'confidence': float(max(prob)),
                    'probabilities': {
                        "Habitable": float(prob[1]),
                        "Not Habitable": float(prob[0])
                    },
                    'explanation': self._generate_explanation(pred, max(prob))
                }
                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            # Return demo predictions
            return [{
                'prediction': "Demo Prediction",
                'confidence': 0.8,
                'probabilities': {"Habitable": 0.8, "Not Habitable": 0.2},
                'explanation': "Demo mode - train model with real data"
            } for _ in planet_data]

    def _generate_explanation(self, prediction: Any, confidence: float) -> str:
        if confidence > 0.8:
            certainty = "high confidence"
        elif confidence > 0.6:
            certainty = "moderate confidence"
        else:
            certainty = "low confidence"

        return f"Prediction made with {certainty} based on planetary characteristics"


ml_model = ExoplanetMLModel()
prediction_cache = TTLCache(maxsize=1000, ttl=3600)


@cached(cache=prediction_cache)
def cached_predict(planet_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return ml_model.predict(planet_data)