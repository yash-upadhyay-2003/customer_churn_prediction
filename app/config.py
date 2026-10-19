"""
Configuration module for Customer Churn Prediction API.
Handles application settings and environment configurations.
"""
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


# Base project root directory
BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """
    Application configuration settings loaded from environment or defaults.
    """
    # API Information
    APP_NAME: str = "Customer Churn Prediction API"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = (
        "Production-grade FastAPI service for real-time customer churn prediction "
        "powered by a trained scikit-learn Decision Tree Pipeline."
    )
    
    # Server Configuration
    HOST: str = "127.0.0.1"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Model Configuration
    # Defaults to saved_model/churn_model.pkl, with fallback check for root churn_model.pkl
    MODEL_PATH: Path = BASE_DIR / "saved_model" / "churn_model.pkl"
    FALLBACK_MODEL_PATH: Path = BASE_DIR / "churn_model.pkl"
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    def get_effective_model_path(self) -> Path:
        """
        Returns the resolved model file path.
        Checks saved_model/churn_model.pkl first, then fallback to root churn_model.pkl.
        """
        if self.MODEL_PATH.exists():
            return self.MODEL_PATH
        if self.FALLBACK_MODEL_PATH.exists():
            return self.FALLBACK_MODEL_PATH
        return self.MODEL_PATH


settings = Settings()
