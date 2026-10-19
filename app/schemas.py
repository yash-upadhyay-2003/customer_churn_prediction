"""
Pydantic schemas for data validation and API request/response serialization.
Ensures strict validation matching the trained model's expected feature schema.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field, ConfigDict


class CustomerInput(BaseModel):
    """
    Schema for a single customer record matching the training schema.
    AccountID and Country are intentionally excluded as they were not used in training.
    """
    # Categorical Features
    Territory: str = Field(
        ...,
        description="Customer territory or state code (e.g., 'CA', 'TX', 'NY')",
        examples=["CA"]
    )
    DeviceType: str = Field(
        ...,
        description="Type of mobile device used by customer (e.g., 'iPhone', 'Galaxy', '-')",
        examples=["iPhone"]
    )
    Promotion: str = Field(
        ...,
        description="Indicates whether customer signed up with a promotional offer ('yes' or 'no')",
        examples=["no"]
    )
    HasRenewed: str = Field(
        ...,
        description="Indicates whether the customer has renewed their plan ('yes' or 'no')",
        examples=["yes"]
    )
    PlanType: str = Field(
        ...,
        description="Customer subscription plan (e.g., 'Purple Plan', 'Blue Plan', 'Green Plan', 'Red Plan')",
        examples=["Purple Plan"]
    )
    StartMonth: str = Field(
        ...,
        description="Month the customer subscription started (e.g., 'Jan', 'Feb', 'Mar')",
        examples=["Jan"]
    )
    StartWeek: str = Field(
        ...,
        description="Week identifier when subscription started (e.g., 'W01', 'W02')",
        examples=["W02"]
    )

    # Numerical Features
    BaseFee: float = Field(
        ...,
        ge=0.0,
        description="Base monthly fee for customer plan",
        examples=[50.0]
    )
    AdditionalFeatureSpend: float = Field(
        ...,
        ge=0.0,
        description="Spend on add-on services and features",
        examples=[15.5]
    )
    NumberOfPenalties: int = Field(
        ...,
        ge=0,
        description="Count of penalty fees or violations incurred",
        examples=[0]
    )
    CurrentPeriodUsage: float = Field(
        ...,
        ge=0.0,
        description="Service usage units in the current billing period",
        examples=[120.0]
    )
    PriorPeriodUsage: float = Field(
        ...,
        ge=0.0,
        description="Service usage units in the preceding billing period",
        examples=[115.0]
    )
    ServiceRating: float = Field(
        ...,
        ge=0.0,
        description="Customer satisfaction / service rating score",
        examples=[4.0]
    )
    ServiceTickets: int = Field(
        ...,
        ge=0,
        description="Number of support or service tickets opened by customer",
        examples=[1]
    )
    CustomerTenure: int = Field(
        ...,
        ge=0,
        description="Duration (in months or periods) customer has been subscribed",
        examples=[24]
    )

    model_config = ConfigDict(
        populate_by_name=True,
        str_strip_whitespace=True,
        json_schema_extra={
            "example": {
                "Territory": "CA",
                "DeviceType": "iPhone",
                "Promotion": "no",
                "HasRenewed": "yes",
                "PlanType": "Purple Plan",
                "BaseFee": 50.0,
                "AdditionalFeatureSpend": 15.5,
                "NumberOfPenalties": 0,
                "CurrentPeriodUsage": 120.0,
                "PriorPeriodUsage": 115.0,
                "ServiceRating": 4.0,
                "ServiceTickets": 1,
                "StartMonth": "Jan",
                "StartWeek": "W02",
                "CustomerTenure": 24
            }
        }
    )


class BatchCustomerInput(BaseModel):
    """
    Request payload for bulk/batch customer churn predictions.
    """
    customers: List[CustomerInput] = Field(
        ...,
        min_length=1,
        description="List of customer records to generate predictions for"
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customers": [
                    {
                        "Territory": "CA",
                        "DeviceType": "iPhone",
                        "Promotion": "no",
                        "HasRenewed": "yes",
                        "PlanType": "Purple Plan",
                        "BaseFee": 50.0,
                        "AdditionalFeatureSpend": 15.5,
                        "NumberOfPenalties": 0,
                        "CurrentPeriodUsage": 120.0,
                        "PriorPeriodUsage": 115.0,
                        "ServiceRating": 4.0,
                        "ServiceTickets": 1,
                        "StartMonth": "Jan",
                        "StartWeek": "W02",
                        "CustomerTenure": 24
                    },
                    {
                        "Territory": "TX",
                        "DeviceType": "Galaxy",
                        "Promotion": "no",
                        "HasRenewed": "no",
                        "PlanType": "Red Plan",
                        "BaseFee": 120.0,
                        "AdditionalFeatureSpend": 50.0,
                        "NumberOfPenalties": 3,
                        "CurrentPeriodUsage": 50.0,
                        "PriorPeriodUsage": 350.0,
                        "ServiceRating": 1.0,
                        "ServiceTickets": 7,
                        "StartMonth": "Aug",
                        "StartWeek": "W32",
                        "CustomerTenure": 3
                    }
                ]
            }
        }
    )


class PredictionResponse(BaseModel):
    """
    Prediction outcome response schema.
    Note: churn_probability is the raw model class probability, not a calibrated risk score.
    """
    prediction: str = Field(
        ...,
        description="Predicted label: 'Churn' or 'No Churn'",
        examples=["No Churn"]
    )
    predicted_class: int = Field(
        ...,
        description="Numerical classification target (0 for No Churn, 1 for Churn)",
        examples=[0]
    )
    churn_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Model estimated probability of customer churning (0.0 to 1.0)",
        examples=[0.0]
    )


class BatchPredictionResponse(BaseModel):
    """
    Response schema for batch predictions.
    """
    total_records: int = Field(..., description="Total records evaluated")
    predictions: List[PredictionResponse] = Field(..., description="List of individual prediction results")


class HealthResponse(BaseModel):
    """
    Health check response schema.
    """
    status: str = Field(..., description="Overall service status (e.g. 'healthy' or 'degraded')")
    model_loaded: bool = Field(..., description="True if ML pipeline model is loaded and ready")
    api_version: str = Field(..., description="Application version")


class ModelInfoResponse(BaseModel):
    """
    Safe metadata regarding the loaded model and its expected schema.
    """
    model_type: str = Field(..., description="Type of machine learning model")
    pipeline_steps: List[str] = Field(..., description="Sequential steps in the scikit-learn Pipeline")
    feature_count: int = Field(..., description="Number of expected input features")
    features: List[str] = Field(..., description="Exact required feature names in expected order")
    numerical_features: List[str] = Field(..., description="Numerical feature column names")
    categorical_features: List[str] = Field(..., description="Categorical feature column names")
    classes: List[int] = Field(..., description="Target class labels [0, 1]")
    target_mapping: Dict[int, str] = Field(..., description="Mapping of numeric classes to labels")
