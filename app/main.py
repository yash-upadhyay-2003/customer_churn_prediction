"""
FastAPI application entry point for the Customer Churn Prediction API.
Defines API endpoints, middleware, lifespan lifecycle, and global error handling.
"""
import logging
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.schemas import (
    CustomerInput,
    BatchCustomerInput,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from app.services.prediction_service import PredictionService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("churn_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler. Loads the trained scikit-learn model once on startup.
    Fails cleanly with clear logs if the model is missing or corrupt.
    """
    logger.info("Starting up Customer Churn Prediction API...")
    service = PredictionService.get_instance()
    try:
        service.load_model()
        logger.info("ML model pipeline initialized and ready for inference.")
    except Exception as e:
        logger.error(
            f"Startup Warning: Could not initialize model on startup: {e}. "
            "Inference endpoints will return 503 until resolved."
        )
    yield
    logger.info("Shutting down Customer Churn Prediction API...")


# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=settings.APP_DESCRIPTION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Enable CORS for standard web integrations
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get(
    "/",
    tags=["Root"],
    summary="API Root Status",
    response_description="Confirmation of API availability"
)
async def root() -> Dict[str, Any]:
    """
    Root endpoint confirming that the Customer Churn Prediction API is up and running.
    """
    return {
        "message": "Customer Churn Prediction API is running",
        "version": settings.APP_VERSION,
        "docs_url": "/docs",
        "health_url": "/health"
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="Health Check",
    response_model=HealthResponse,
    response_description="System and model health status"
)
async def health() -> HealthResponse:
    """
    Returns current API health status and confirms whether the ML model is loaded.
    Safely avoids exposing internal server paths or sensitive details.
    """
    service = PredictionService.get_instance()
    is_loaded = service.is_loaded()
    
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        api_version=settings.APP_VERSION
    )


@app.get(
    "/model-info",
    tags=["Model Info"],
    summary="Model Metadata & Expected Schema",
    response_model=ModelInfoResponse,
    response_description="Information about the trained pipeline architecture and feature schema"
)
async def model_info() -> ModelInfoResponse:
    """
    Returns safe metadata about the loaded scikit-learn model pipeline,
    including the required input features, their ordering, and subcategories.
    """
    service = PredictionService.get_instance()
    if not service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded. Please verify the trained model file."
        )
    return service.get_model_info()


@app.post(
    "/predict",
    tags=["Prediction"],
    summary="Predict Customer Churn (Single)",
    response_model=PredictionResponse,
    status_code=status.HTTP_200_OK,
    response_description="Churn prediction result with probability"
)
async def predict_churn(customer: CustomerInput) -> PredictionResponse:
    """
    Accepts one customer's attributes as JSON, validates the schema,
    and returns the predicted churn outcome and probability.
    """
    service = PredictionService.get_instance()
    if not service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is currently unavailable."
        )
    
    try:
        return service.predict(customer)
    except Exception as e:
        logger.error(f"Inference error during single prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating the customer prediction."
        )


@app.post(
    "/predict/batch",
    tags=["Prediction"],
    summary="Predict Customer Churn (Batch)",
    response_model=BatchPredictionResponse,
    status_code=status.HTTP_200_OK,
    response_description="Batch churn prediction results for multiple customers"
)
async def predict_churn_batch(payload: BatchCustomerInput) -> BatchPredictionResponse:
    """
    Accepts multiple customer records in a single payload and performs
    vectorized batch predictions using the loaded ML pipeline.
    """
    service = PredictionService.get_instance()
    if not service.is_loaded():
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Prediction model is currently unavailable."
        )
    
    try:
        return service.predict_batch(payload.customers)
    except Exception as e:
        logger.error(f"Inference error during batch prediction: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while evaluating batch predictions."
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=settings.DEBUG)
