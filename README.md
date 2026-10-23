# Automated API tests for `mlops-pipeline`

Copy these files into the root of your existing repository. Keep your current
`src/app.py`, `src/train.py`, and `requirements.txt` unchanged.

```
mlops-pipeline/
├── src/
│   ├── app.py
│   └── train.py
├── tests/
│   └── test_app.py
├── .github/
│   └── workflows/
│       └── api-tests.yml
├── requirements.txt
└── requirements-dev.txt
```

## Run in GitHub Codespaces

```bash
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -v tests/
```

If the main project dependencies have not yet been installed:

```bash
python -m pip install -r requirements.txt -r requirements-dev.txt
```

## Run automatically on GitHub

Commit/push the added files:

```bash
git add tests/test_app.py requirements-dev.txt .github/workflows/api-tests.yml
git commit -m "Add automated FastAPI tests"
git push origin main
```

Then check the **Actions** tab in your repository for the **FastAPI automated tests** workflow.
The workflow also runs on pull requests into `main` and supports manual run.

## What is tested?

- Health endpoint healthy/degraded responses
- One- and multi-record predictions and probabilities
- 503 when model cannot be loaded
- 422 for missing/invalid request inputs
- 400 on prediction exception
- Browser `GET /predict` redirect to Swagger
- Swagger documentation route
- Startup lifespan attempts to load model
- MLflow registry chooses latest READY version
- Explicit unready versions are rejected
- Model loading retry interval

The tests mock the classifier/registry, so they work without a running MLflow server,
without an MLflow database, and without downloading the housing dataset.
They do not replace integration tests against a real MLflow server and trained model.

Expected existing app features: `src/app.py` contains a hidden `GET /predict` route
redirecting to `/docs` and supports `MODEL_RETRY_SECONDS` / `MODEL_VERSION`.
