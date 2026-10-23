"""Prometheus metrics for the existing FastAPI service.

Avoids user IDs, raw paths, request bodies, and other high-cardinality labels.
Instrument a SINGLE Uvicorn worker (the Codespaces development setup).
"""

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

HTTP_REQUESTS = Counter(
    "mlops_http_requests_total",
    "Total FastAPI HTTP requests, grouped by HTTP method, route, and status code.",
    ("method", "route", "status_code"),
)

HTTP_LATENCY = Histogram(
    "mlops_http_request_duration_seconds",
    "Elapsed time in seconds for FastAPI HTTP requests.",
    ("method", "route"),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)

MODEL_LOADED = Gauge(
    "mlops_model_loaded",
    "Whether the MLflow classifier is loaded and available in FastAPI memory.",
)


def setup_monitoring(app: FastAPI, model_is_loaded: Callable[[], bool]) -> None:
    """Add low-cardinality metrics and an undocumented Prometheus scrape endpoint."""

    @app.middleware("http")
    async def record_requests(request: Request, call_next):
        if request.url.path == "/metrics":
            return await call_next(request)

        started = time.perf_counter()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            # A matched route has a stable template, not a unique URL/user input.
            route_obj = request.scope.get("route")
            route = getattr(route_obj, "path", "unmatched")
            method = request.method
            HTTP_REQUESTS.labels(method, route, str(status)).inc()
            HTTP_LATENCY.labels(method, route).observe(time.perf_counter() - started)

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        MODEL_LOADED.set(1 if model_is_loaded() else 0)
        return Response(
            content=generate_latest(),
            media_type=CONTENT_TYPE_LATEST,
            headers={"Cache-Control": "no-store"},
        )
