"""
Automated unit and integration tests for FastAPI Customer Churn Prediction backend.
Uses pytest and Starlette/FastAPI TestClient.
"""
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.prediction_service import PredictionService


@pytest.fixture(scope="session")
def client():
    """
    TestClient session fixture that ensures lifespan startup and shutdown execute.
    """
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_no_churn_payload():
    """Realistic customer payload expected to be classified as No Churn."""
    return {
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


@pytest.fixture
def sample_churn_payload():
    """Realistic customer payload expected to be classified as Churn."""
    return {
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


def test_root_endpoint(client: TestClient):
    """GET / should return 200 with confirmation message."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "Customer Churn Prediction API is running" in data["message"]
    assert "version" in data
    assert "docs_url" in data


def test_health_endpoint(client: TestClient):
    """GET /health should return 200 and healthy status when model is loaded."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "api_version" in data


def test_model_info_endpoint(client: TestClient):
    """GET /model-info should return safe model metadata and required features."""
    response = client.get("/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_type"] == "DecisionTreeClassifier"
    assert "preprocessor" in data["pipeline_steps"]
    assert "classifier" in data["pipeline_steps"]
    assert data["feature_count"] == 15
    assert "CustomerTenure" in data["features"]
    assert "BaseFee" in data["features"]
    assert "Territory" in data["features"]
    assert "AccountID" not in data["features"]
    assert "Country" not in data["features"]
    assert data["classes"] == [0, 1]


def test_predict_valid_no_churn(client: TestClient, sample_no_churn_payload: dict):
    """POST /predict with loyal customer data should yield No Churn with valid probability."""
    response = client.post("/predict", json=sample_no_churn_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "No Churn"
    assert data["predicted_class"] == 0
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_valid_churn(client: TestClient, sample_churn_payload: dict):
    """POST /predict with churn-risk customer data should yield Churn prediction."""
    response = client.post("/predict", json=sample_churn_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] == "Churn"
    assert data["predicted_class"] == 1
    assert 0.0 <= data["churn_probability"] <= 1.0


def test_predict_missing_feature(client: TestClient, sample_no_churn_payload: dict):
    """POST /predict missing a required feature should fail with HTTP 422."""
    payload = sample_no_churn_payload.copy()
    del payload["BaseFee"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    # Ensure the error identifies the missing field
    assert any("BaseFee" in str(err.get("loc", [])) for err in data["detail"])


def test_predict_invalid_data_type(client: TestClient, sample_no_churn_payload: dict):
    """POST /predict with invalid data type for numerical field should fail with HTTP 422."""
    payload = sample_no_churn_payload.copy()
    payload["BaseFee"] = "not_a_valid_float"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_negative_value_constraint(client: TestClient, sample_no_churn_payload: dict):
    """POST /predict with negative numerical values where ge=0 should fail with HTTP 422."""
    payload = sample_no_churn_payload.copy()
    payload["ServiceTickets"] = -5
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_unknown_categorical_handling(client: TestClient, sample_no_churn_payload: dict):
    """
    POST /predict with unseen categorical values should succeed because
    the trained preprocessing pipeline uses OneHotEncoder(handle_unknown='ignore').
    """
    payload = sample_no_churn_payload.copy()
    payload["Territory"] = "UNKNOWN_TERRITORY_XYZ"
    payload["DeviceType"] = "NovelSmartwatch"
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Churn", "No Churn"]
    assert data["predicted_class"] in [0, 1]


def test_batch_prediction_endpoint(client: TestClient, sample_no_churn_payload: dict, sample_churn_payload: dict):
    """POST /predict/batch should return predictions for multiple customers."""
    batch_payload = {
        "customers": [
            sample_no_churn_payload,
            sample_churn_payload
        ]
    }
    response = client.post("/predict/batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][0]["prediction"] == "No Churn"
    assert data["predictions"][1]["prediction"] == "Churn"


def test_batch_prediction_empty_list(client: TestClient):
    """POST /predict/batch with empty list should fail with HTTP 422."""
    response = client.post("/predict/batch", json={"customers": []})
    assert response.status_code == 422
