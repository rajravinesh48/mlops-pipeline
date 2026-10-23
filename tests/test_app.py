"""Offline HTTP and MLflow-registry unit tests for src/app.py.

Run from the repository root with: python -m pytest -v
All model predictions are deterministic fakes: no trained model, running MLflow
server, network connection, or dataset download is needed.
"""

import importlib
import time
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

service = importlib.import_module("src.app")


class DummyClassifier:
    """Small sklearn-shaped classifier with predictable results."""

    classes_ = np.array([0, 1])

    def predict(self, frame):
        assert list(frame.columns) == service.FEATURE_COLUMNS
        return (frame["MedInc"].to_numpy() >= 3.0).astype(int)

    def predict_proba(self, frame):
        p1 = np.where(frame["MedInc"].to_numpy() >= 3.0, 0.85, 0.10)
        return np.column_stack((1 - p1, p1))


class BrokenClassifier(DummyClassifier):
    def predict(self, frame):
        raise ValueError("invalid feature data")


@pytest.fixture(autouse=True)
def reset_state(monkeypatch):
    """Keep tests independent; no real registry access occurs by accident."""
    monkeypatch.setattr(service, "model", None)
    monkeypatch.setattr(service, "resolved_version", "unknown")
    monkeypatch.setattr(service, "model_load_error", None)
    monkeypatch.setattr(service, "last_load_attempt", None)
    monkeypatch.setattr(service, "MODEL_VERSION", "")


@pytest.fixture
def client():
    # Don't start lifespan: individual tests control the model/loading result.
    return TestClient(service.app)


@pytest.fixture
def ready_model(monkeypatch):
    monkeypatch.setattr(service, "model", DummyClassifier())
    monkeypatch.setattr(service, "resolved_version", "2")


def make_features(median_income=3.5):
    return {
        "MedInc": median_income,
        "HouseAge": 25,
        "AveRooms": 5.8,
        "AveBedrms": 1.05,
        "Population": 1200,
        "AveOccup": 2.9,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }


def test_health_with_model(client, ready_model):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "healthy"
    assert body["model_loaded"] is True
    assert body["registry_target"] == "housing_rf_model"
    assert body["model_version"] == "2"
    assert body["last_load_error"] is None


def test_health_when_registry_unavailable(client, monkeypatch):
    monkeypatch.setattr(service, "model_load_error", "ConnectionError: server unavailable")
    monkeypatch.setattr(service, "load_registered_model", lambda force=False: False)
    response = client.get("/health")
    assert response.status_code == 200  # Current endpoint reports degraded in JSON.
    body = response.json()
    assert body["status"] == "degraded"
    assert body["model_loaded"] is False
    assert "ConnectionError" in body["last_load_error"]


def test_predict_one_house(client, ready_model):
    response = client.post("/predict", json={"instances": [make_features()]})
    assert response.status_code == 200
    body = response.json()
    assert body["predictions"] == [1]
    assert body["probabilities"] == pytest.approx([0.85])
    assert body["model_version_loaded"] == "housing_rf_model:2"
    assert isinstance(body["inference_latency_ms"], (float, int))
    assert body["inference_latency_ms"] >= 0


def test_predict_multiple_houses(client, ready_model):
    response = client.post(
        "/predict", json={"instances": [make_features(3.5), make_features(1.5)]}
    )
    assert response.status_code == 200
    assert response.json()["predictions"] == [1, 0]
    assert response.json()["probabilities"] == pytest.approx([0.85, 0.10])


def test_predict_model_not_ready(client, monkeypatch):
    monkeypatch.setattr(service, "load_registered_model", lambda force=False: False)
    response = client.post("/predict", json={"instances": [make_features()]})
    assert response.status_code == 503
    assert "Model not ready" in response.json()["detail"]


def test_predict_empty_instances_is_rejected(client, ready_model):
    response = client.post("/predict", json={"instances": []})
    assert response.status_code == 422


def test_predict_missing_instances_is_rejected(client, ready_model):
    response = client.post("/predict", json={})
    assert response.status_code == 422


def test_predict_invalid_feature_is_rejected(client, ready_model):
    response = client.post(
        "/predict", json={"instances": [{**make_features(), "MedInc": "not-a-number"}]}
    )
    assert response.status_code == 422


def test_predict_uses_feature_defaults(client, ready_model):
    # HouseFeatures declares defaults for all 8 features.
    response = client.post("/predict", json={"instances": [{}]})
    assert response.status_code == 200
    assert response.json()["predictions"] == [1]


def test_prediction_exception_returns_400(client, monkeypatch):
    monkeypatch.setattr(service, "model", BrokenClassifier())
    response = client.post("/predict", json={"instances": [make_features()]})
    assert response.status_code == 400
    assert "invalid feature data" in response.json()["detail"]


def test_get_predict_redirects_to_docs(client):
    # The browser GET route added to src/app.py should redirect to Swagger.
    response = client.get("/predict", follow_redirects=False)
    assert response.status_code in (302, 307, 308)
    assert response.headers["location"] == "/docs"


def test_swagger_docs_available(client):
    assert client.get("/docs").status_code == 200


def test_lifespan_attempts_model_load(monkeypatch):
    calls = []

    def fake_loader(force=False):
        calls.append(force)
        return True

    monkeypatch.setattr(service, "load_registered_model", fake_loader)
    with TestClient(service.app) as client:
        assert client.get("/docs").status_code == 200
    assert calls == [True]


def test_load_registered_model_picks_latest_ready(monkeypatch):
    versions = [
        SimpleNamespace(version="1", status="READY"),
        SimpleNamespace(version="2", status="READY"),
        SimpleNamespace(version="3", status="PENDING_REGISTRATION"),
    ]

    class FakeRegistryClient:
        def __init__(self, tracking_uri):
            assert tracking_uri == service.MLFLOW_TRACKING_URI

        def search_model_versions(self, filter_string):
            assert "housing_rf_model" in filter_string
            return versions

    loaded_uris = []

    def fake_model_loader(uri):
        loaded_uris.append(uri)
        return DummyClassifier()

    monkeypatch.setattr(service, "MlflowClient", FakeRegistryClient)
    monkeypatch.setattr(service.mlflow, "set_tracking_uri", lambda uri: None)
    monkeypatch.setattr(service.mlflow.sklearn, "load_model", fake_model_loader)

    assert service.load_registered_model() is True
    assert loaded_uris == ["models:/housing_rf_model/2"]
    assert service.resolved_version == "2"
    assert isinstance(service.model, DummyClassifier)


def test_load_registered_model_rejects_unready_version(monkeypatch):
    monkeypatch.setattr(service, "MODEL_VERSION", "9")

    class FakeRegistryClient:
        def __init__(self, tracking_uri):
            pass

        def get_model_version(self, name, version):
            assert (name, version) == ("housing_rf_model", "9")
            return SimpleNamespace(version="9", status="PENDING_REGISTRATION")

    monkeypatch.setattr(service, "MlflowClient", FakeRegistryClient)
    monkeypatch.setattr(service.mlflow, "set_tracking_uri", lambda uri: None)
    assert service.load_registered_model() is False
    assert "not READY" in service.model_load_error
    assert service.model is None


def test_retry_is_throttled(monkeypatch):
    monkeypatch.setattr(service, "last_load_attempt", time.monotonic())
    monkeypatch.setattr(service, "MODEL_RETRY_SECONDS", 60)

    def unexpected_registry_call(*args, **kwargs):
        raise AssertionError("Should not contact registry before retry interval")

    monkeypatch.setattr(service, "MlflowClient", unexpected_registry_call)
    assert service.load_registered_model() is False
