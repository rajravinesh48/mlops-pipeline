# MLOps API monitoring (FastAPI + Prometheus + Grafana)

This add-on monitors the **existing** FastAPI process in GitHub Codespaces. It does not reinstall, change, or retrain your MLflow model. It keeps the original `src/app.py` business logic and the existing `tests/test_app.py`.

## Included files

- `src/monitoring.py`: request count and latency, plus MLflow model-ready gauge and `/metrics` endpoint.
- `setup_monitoring.py`: safely adds one instrumentation call **after** `app = FastAPI(...)`; updates requirements and `.gitignore`. Safe to run twice. **Review changes with `git diff`.**
- `monitoring/docker-compose.yml`: Prometheus + Grafana, ports **9090** and **3000**, bound to loopback only.
- `monitoring/prometheus.yml`: scrape `host.docker.internal:8000/metrics` every 15 seconds.
- `monitoring/grafana/...`: provisioned Prometheus data source and a 10-panel MLOps dashboard.
- `tests/test_monitoring.py`: four new tests; preserves your original 16 tests.

## 1. Upload and enable (Codespaces project root)

```bash
unzip -o mlops_monitoring.zip -d .
python setup_monitoring.py
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -v tests/
git diff -- src/app.py requirements.txt .gitignore
```

The patch tool inserts the following code immediately after the top-level `app = FastAPI(...)` assignment:

```python
from .monitoring import setup_monitoring
setup_monitoring(app, lambda: model is not None)
```

It does not overwrite your API implementation and requires your current `src/app.py` layout (`app = FastAPI(...)` at module scope).

## 2. Run MLflow and FastAPI in Codespaces

Keep the existing MLflow setup you already use. If the Codespace restarted, open separate terminals:

Terminal A:

```bash
source venv/bin/activate
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db
```

Terminal B:

```bash
source venv/bin/activate
export MLFLOW_TRACKING_URI=http://127.0.0.1:5000
python -m uvicorn src.app:app --host 0.0.0.0 --port 8000
```

If FastAPI is **already running**, stop and restart it after the monitoring update. No need to retrain if the registered model still exists.

Check metrics:

```bash
curl -s http://127.0.0.1:8000/metrics | grep -E 'mlops_model_loaded|mlops_http_requests_total' | head
```

## 3. Start Prometheus + Grafana with Docker Compose

Codespaces often has Docker support. First check:

```bash
docker compose version
```

Create a password file, **ignored by Git**, using a cryptographically random password:

```bash
python -c "import secrets; print('GRAFANA_ADMIN_PASSWORD=' + secrets.token_urlsafe(24))" > .env.monitoring
chmod 600 .env.monitoring
```

To display your password to use for login, you can open `.env.monitoring` locally in Codespaces. Do not share or commit this file.

```bash
docker compose --env-file .env.monitoring -f monitoring/docker-compose.yml config --quiet
docker compose --env-file .env.monitoring -f monitoring/docker-compose.yml up -d
docker compose --env-file .env.monitoring -f monitoring/docker-compose.yml ps
```

**In Codespaces PORTS tab**, forward ports **3000** (Grafana) and **9090** (Prometheus) and set them **Private**. Port **8000** (FastAPI) should also be Private.

- Port `9090`: open Prometheus > **Status / Targets** and look for `fastapi` = **UP**.
- Port `3000`: log in to Grafana with user `admin` and the generated password.
- Grafana > **Dashboards** > **MLOps** > **MLOps FastAPI Performance**.

The dashboard is created automatically by provisioning. It has request rate, 5xx ratio, p95 and mean request latency, health/model readiness, memory and CPU consumption.

Generate real prediction requests using `/docs` or your existing tests against the live Codespaces API. Metrics from unit tests run inside pytest are **not** expected to appear in the separate FastAPI server's Grafana dashboard.

## Troubleshooting

- **Docker unavailable:** Check `docker version` and `docker compose version`. If the daemon is not available in your Codespace, use a Codespace with Docker enabled or install/run Prometheus and Grafana via a different environment. You can still test `/metrics` without Docker.
- **Target DOWN:** FastAPI must bind `0.0.0.0:8000`; verify `curl http://127.0.0.1:8000/metrics`, then `docker compose ... logs prometheus`. `host.docker.internal:host-gateway` is required because Prometheus is a container and FastAPI runs on the Codespace host.
- **Dashboard blank:** Wait for two scrape intervals and send a few POST requests. Rolling `rate(...[5m])` needs more than one sample; status errors remain absent if no errors occurred.
- **Grafana password:** Password environment values affect **initial** Grafana admin creation when data volume is empty. Changing `.env.monitoring` later does not automatically reset an existing Grafana database.
- **Codespaces stops:** Containers/processes stop along with Codespace. Git tracked code persists; metrics persist only as long as the Codespace and its Docker volumes remain.

## Security and scale

- Keep ports 3000, 9090, and 8000 **Private** in Codespaces. Never commit `.env.monitoring`.
- Grafana uses authenticated admin login; Prometheus has no authentication in this sample. Do not expose either publicly.
- `/metrics` must be protected behind a network boundary/authenticated proxy if you deploy an otherwise-public FastAPI API.
- This development setup assumes **one Uvicorn worker**. Multiworker production monitoring requires Prometheus Python client's multiprocess configuration or per-worker scraping.
- This is HTTP request/inference-service monitoring, not model accuracy/drift monitoring. Add those separately for production.

## Save to GitHub

```bash
git add src/app.py src/monitoring.py setup_monitoring.py monitoring/ tests/test_monitoring.py requirements.txt .gitignore MONITORING_SETUP.md .env.monitoring.example
git status
git commit -m "Add FastAPI Prometheus and Grafana monitoring"
git push origin main
```

Do **not** stage `mlops_monitoring.zip`, `.env.monitoring`, or generated `mlflow.db`/`mlruns` files.
