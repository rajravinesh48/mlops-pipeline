"""FastAPI inference service loading a scikit-learn classifier from MLflow Registry.

Environment variables:
    MLFLOW_TRACKING_URI  MLflow server URL (default http://127.0.0.1:5000)
    MODEL_NAME           Registered model name (default housing_rf_model)
    MODEL_VERSION        Optional specific version; otherwise newest READY version
    MODEL_RETRY_SECONDS  How often to retry a failed load (default 15)
"""

import logging
import os
import threading
import time
from contextlib import asynccontextmanager
from typing import List

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("mlops.inference")

MODEL_NAME = os.getenv("MODEL_NAME", "housing_rf_model")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_VERSION = os.getenv("MODEL_VERSION", "").strip()
MODEL_RETRY_SECONDS = max(0, int(os.getenv("MODEL_RETRY_SECONDS", "15")))

FEATURE_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude",
]

model = None
resolved_version = "unknown"
model_load_error = None
last_load_attempt = None
load_lock = threading.Lock()


def load_registered_model(force: bool = False) -> bool:
    """Load a registered classifier, retrying after failures without a restart."""
    global model, resolved_version, model_load_error, last_load_attempt

    if model is not None and not force:
        return True

    with load_lock:
        if model is not None and not force:
            return True

        now = time.monotonic()
        if (
            not force
            and last_load_attempt is not None
            and now - last_load_attempt < MODEL_RETRY_SECONDS
        ):
            return False
        last_load_attempt = now

        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

            if MODEL_VERSION:
                version_meta = client.get_model_version(MODEL_NAME, MODEL_VERSION)
                if str(version_meta.status).upper() != "READY":
                    raise RuntimeError(
                        f"Model '{MODEL_NAME}' version {MODEL_VERSION} "
                        f"is not READY (status: {version_meta.status})."
                    )
            else:
                all_versions = client.search_model_versions(f"name='{MODEL_NAME}'")
                if not all_versions:
                    raise RuntimeError(
                        f"No registered versions of '{MODEL_NAME}' exist on "
                        f"{MLFLOW_TRACKING_URI}. Run train.py first."
                    )
                ready_versions = [
                    version for version in all_versions
                    if str(version.status).upper() == "READY"
                ]
                if not ready_versions:
                    raise RuntimeError(
                        f"'{MODEL_NAME}' has {len(all_versions)} registered "
                        "version(s), but none are READY."
                    )
                version_meta = max(ready_versions, key=lambda v: int(v.version))

            model_uri = f"models:/{MODEL_NAME}/{version_meta.version}"
            logger.info("Loading model %s from %s", model_uri, MLFLOW_TRACKING_URI)
            loaded = mlflow.sklearn.load_model(model_uri)

            if not callable(getattr(loaded, "predict", None)) or not callable(
                getattr(loaded, "predict_proba", None)
            ):
                raise TypeError(
                    "Registered model must implement predict and predict_proba"
                )

            model = loaded
            resolved_version = str(version_meta.version)
            model_load_error = None
            logger.info("Model %s version %s ready", MODEL_NAME, resolved_version)
            return True

        except Exception as exc:
            model = None
            resolved_version = "unknown"
            model_load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Could not load model '%s' from MLflow", MODEL_NAME)
            return False


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Keep the API reachable even if MLflow isn't ready at startup.
    load_registered_model(force=True)
    yield


app = FastAPI(
    title="MLOps Production Inference Service",
    description="FastAPI endpoint serving a registered MLflow classifier",
    version="1.1.1",
    lifespan=lifespan,
)


# Browser navigation uses GET; redirect to interactive Swagger documentation.
# This route must be declared AFTER app = FastAPI(...).
@app.get("/predict", include_in_schema=False)
def predict_browser():
    return RedirectResponse(url="/docs", status_code=302)


class HouseFeatures(BaseModel):
    MedInc: float = Field(default=3.5, description="Median income in block group")
    HouseAge: float = Field(default=25.0, description="Median house age in block group")
    AveRooms: float = Field(default=5.8, description="Average rooms per household")
    AveBedrms: float = Field(default=1.05, description="Average bedrooms per household")
    Population: float = Field(default=1200.0, description="Block group population")
    AveOccup: float = Field(default=2.9, description="Average household members")
    Latitude: float = Field(default=37.88, description="Block group latitude")
    Longitude: float = Field(default=-122.23, description="Block group longitude")


class PredictionRequest(BaseModel):
    instances: List[HouseFeatures] = Field(min_length=1)


class PredictionResponse(BaseModel):
    predictions: List[int]
    probabilities: List[float]
    inference_latency_ms: float
    model_version_loaded: str


@app.get("/health", tags=["Monitoring"])
def health_check():
    if model is None:
        load_registered_model()
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_loaded": model is not None,
        "registry_target": MODEL_NAME,
        "model_version": resolved_version,
        "mlflow_tracking_uri": MLFLOW_TRACKING_URI,
        "last_load_error": model_load_error,
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(payload: PredictionRequest):
    if model is None and not load_registered_model():
        raise HTTPException(
            status_code=503,
            detail=(
                "Model not ready. Check GET /health or FastAPI logs for the "
                "MLflow loading error."
            ),
        )

    started = time.perf_counter()
    input_df = pd.DataFrame(
        [instance.model_dump() for instance in payload.instances],
        columns=FEATURE_COLUMNS,
    )

    try:
        predictions = model.predict(input_df)
        probabilities = model.predict_proba(input_df)

        # This binary classifier uses class 1 as its positive class.
        classes = list(model.classes_)
        positive_index = classes.index(1)
        elapsed_ms = (time.perf_counter() - started) * 1000

        return PredictionResponse(
            predictions=[int(value) for value in predictions],
            probabilities=[float(row[positive_index]) for row in probabilities],
            inference_latency_ms=round(elapsed_ms, 2),
            model_version_loaded=f"{MODEL_NAME}:{resolved_version}",
        )
    except Exception as exc:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=400,
            detail=f"Inference execution failed: {exc}",
        ) from exc
