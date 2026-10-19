"""
Prediction Service for Customer Churn Prediction.
Encapsulates model loading, schema alignment, compatibility patches,
and inference logic for single and batch predictions.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

# Compatibility patch for unpickling scikit-learn pipelines across versions
import sklearn.compose._column_transformer
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    class _RemainderColsList(list):
        """Shim for deserializing ColumnTransformer from scikit-learn <= 1.6."""
        pass
    sklearn.compose._column_transformer._RemainderColsList = _RemainderColsList

import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from app.config import settings
from app.schemas import (
    CustomerInput,
    PredictionResponse,
    BatchPredictionResponse,
    ModelInfoResponse,
)

logger = logging.getLogger("uvicorn.error")


class PredictionService:
    """
    Singleton service managing the trained Machine Learning Pipeline lifecycle and inference.
    """
    _instance: Optional["PredictionService"] = None

    def __init__(self):
        self.model: Optional[Pipeline] = None
        self.feature_names: List[str] = []
        self.numerical_features: List[str] = []
        self.categorical_features: List[str] = []
        self.classes: List[int] = [0, 1]
        self.model_type: str = "Unknown"
        self.pipeline_steps: List[str] = []
        self._target_mapping = {0: "No Churn", 1: "Churn"}

    @classmethod
    def get_instance(cls) -> "PredictionService":
        """Access the singleton instance of the PredictionService."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_model(self) -> None:
        """
        Loads the trained model pipeline from the configured path.
        Applies version compatibility adaptations and extracts feature metadata.
        Raises RuntimeError if the model file is missing or invalid.
        """
        model_path = settings.get_effective_model_path()
        logger.info(f"Attempting to load model from: {model_path}")

        if not model_path.exists():
            error_msg = f"Model file not found at {model_path}. Please ensure churn_model.pkl is placed in saved_model/."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        try:
            loaded_model = joblib.load(model_path)
            if not hasattr(loaded_model, "predict"):
                raise TypeError(f"Loaded object from {model_path} does not have a predict method.")

            # Apply compatibility patch for SimpleImputer attributes across scikit-learn versions
            self._apply_compatibility_patches(loaded_model)

            self.model = loaded_model
            self._extract_metadata()
            logger.info(
                f"Successfully loaded {self.model_type} pipeline with {len(self.feature_names)} features."
            )
        except Exception as e:
            logger.error(f"Failed to load machine learning model: {e}", exc_info=True)
            self.model = None
            raise RuntimeError(f"Failed to load trained model: {str(e)}") from e

    def _apply_compatibility_patches(self, model: Any) -> None:
        """
        Ensures compatibility across scikit-learn versions (e.g. 1.6 vs 1.9+).
        """
        if hasattr(model, "steps"):
            for _, step in model.steps:
                if hasattr(step, "transformers_"):
                    for _, trans, _ in step.transformers_:
                        if hasattr(trans, "named_steps"):
                            for _, substep in trans.named_steps.items():
                                if isinstance(substep, SimpleImputer):
                                    if not hasattr(substep, "_fill_dtype"):
                                        substep._fill_dtype = getattr(substep, "_fit_dtype", None)

    def _extract_metadata(self) -> None:
        """
        Extracts step names, input features, and column subsets from the pipeline.
        """
        if self.model is None:
            return

        # Pipeline steps
        if hasattr(self.model, "steps"):
            self.pipeline_steps = [s[0] for s in self.model.steps]
            # Final classifier
            final_step = self.model.steps[-1][1]
            self.model_type = type(final_step).__name__
            if hasattr(final_step, "classes_"):
                self.classes = [int(c) for c in final_step.classes_]

        # Overall expected input features in exact order
        if hasattr(self.model, "feature_names_in_"):
            self.feature_names = list(self.model.feature_names_in_)
        else:
            # Fallback based on pipeline inspection
            self.feature_names = [
                "Territory", "DeviceType", "Promotion", "HasRenewed", "PlanType",
                "BaseFee", "AdditionalFeatureSpend", "NumberOfPenalties",
                "CurrentPeriodUsage", "PriorPeriodUsage", "ServiceRating",
                "ServiceTickets", "StartMonth", "StartWeek", "CustomerTenure"
            ]

        # Extract numerical vs categorical features from ColumnTransformer
        try:
            preprocessor = self.model.named_steps.get("preprocessor")
            if preprocessor and hasattr(preprocessor, "transformers_"):
                for t_name, _, cols in preprocessor.transformers_:
                    if t_name == "num":
                        self.numerical_features = list(cols)
                    elif t_name == "cat":
                        self.categorical_features = list(cols)
        except Exception as e:
            logger.warning(f"Could not extract sub-feature groups: {e}")

    def is_loaded(self) -> bool:
        """Check if model is successfully loaded and ready for predictions."""
        return self.model is not None

    def predict(self, customer: CustomerInput) -> PredictionResponse:
        """
        Predicts churn outcome for a single customer.
        Converts validated Pydantic model into a DataFrame with exact column names and order.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot process prediction request.")

        # Prepare single-row DataFrame aligned with model's expected feature order
        data_dict = customer.model_dump()
        df = pd.DataFrame([data_dict])[self.feature_names]

        # Inference
        raw_pred = self.model.predict(df)[0]
        pred_class = int(raw_pred)
        label = self._target_mapping.get(pred_class, str(pred_class))

        # Probability extraction if supported
        churn_prob = 0.0
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(df)[0]
            # Classes are typically [0, 1]; find index corresponding to class 1
            if 1 in self.classes:
                idx_1 = self.classes.index(1)
                churn_prob = float(proba[idx_1])
            else:
                churn_prob = float(proba[-1])

        return PredictionResponse(
            prediction=label,
            predicted_class=pred_class,
            churn_probability=round(churn_prob, 4)
        )

    def predict_batch(self, customers: List[CustomerInput]) -> BatchPredictionResponse:
        """
        Performs vectorized batch predictions across multiple customer records.
        """
        if self.model is None:
            raise RuntimeError("Model is not loaded. Cannot process prediction request.")

        # Convert records to DataFrame
        data_list = [c.model_dump() for c in customers]
        df = pd.DataFrame(data_list)[self.feature_names]

        # Vectorized predictions
        raw_preds = self.model.predict(df)
        probas = self.model.predict_proba(df) if hasattr(self.model, "predict_proba") else None

        results: List[PredictionResponse] = []
        idx_1 = self.classes.index(1) if 1 in self.classes else 1

        for i, raw_pred in enumerate(raw_preds):
            pred_class = int(raw_pred)
            label = self._target_mapping.get(pred_class, str(pred_class))
            churn_prob = float(probas[i][idx_1]) if probas is not None else 0.0

            results.append(
                PredictionResponse(
                    prediction=label,
                    predicted_class=pred_class,
                    churn_probability=round(churn_prob, 4)
                )
            )

        return BatchPredictionResponse(
            total_records=len(results),
            predictions=results
        )

    def get_model_info(self) -> ModelInfoResponse:
        """Returns safe metadata regarding the trained model."""
        if not self.is_loaded():
            raise RuntimeError("Model is not loaded.")

        return ModelInfoResponse(
            model_type=self.model_type,
            pipeline_steps=self.pipeline_steps,
            feature_count=len(self.feature_names),
            features=self.feature_names,
            numerical_features=self.numerical_features,
            categorical_features=self.categorical_features,
            classes=self.classes,
            target_mapping=self._target_mapping
        )
