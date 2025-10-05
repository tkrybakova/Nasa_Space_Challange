import os

class Settings:
    def __init__(self):
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///./exoplanet.db")
        self.ml_model_path = os.getenv("ML_MODEL_PATH", "model.joblib")

settings = Settings()
