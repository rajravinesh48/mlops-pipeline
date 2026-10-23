"""Prometheus instrumentation tests, independent of the MLflow server."""
import importlib

from fastapi.testclient import TestClient

service = importlib.import_module("src.app")


def test_metrics_exposes_model_readiness(monkeypatch):
    monkeypatch.setattr(service, "model", object())
    response = TestClient(service.app).get("/metrics")
    assert response.status_code == 200
    assert "mlops_model_loaded 1.0" in response.text
    assert "mlops_http_requests_total" in response.text or "mlops_http_request_duration_seconds" in response.text


def test_metrics_reports_model_unavailable(monkeypatch):
    monkeypatch.setattr(service, "model", None)
    response = TestClient(service.app).get("/metrics")
    assert response.status_code == 200
    assert "mlops_model_loaded 0.0" in response.text


def test_http_metrics_capture_health_and_latency(monkeypatch):
    monkeypatch.setattr(service, "model", object())
    client = TestClient(service.app)
    assert client.get("/health").status_code == 200
    body = client.get("/metrics").text
    assert 'route="/health"' in body
    assert 'status_code="200"' in body
    assert "mlops_http_request_duration_seconds_bucket" in body


def test_unknown_routes_do_not_create_unbounded_metric_labels(monkeypatch):
    monkeypatch.setattr(service, "model", object())
    client = TestClient(service.app)
    assert client.get("/some-user-specific-identifier-abc").status_code == 404
    metrics = client.get("/metrics").text
    assert 'route="unmatched"' in metrics
    assert 'route="/some-user-specific-identifier-abc"' not in metrics
