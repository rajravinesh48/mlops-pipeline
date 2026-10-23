"""Train and register the California Housing binary classifier in MLflow.

Uses the same MLFLOW_TRACKING_URI and MODEL_NAME as app.py. The California
Housing dataset may need internet access on its first download.
"""

import os

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MODEL_NAME = os.getenv("MODEL_NAME", "housing_rf_model")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "housing-classification-production")


def run_training():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    print(f"Connecting to MLflow: {MLFLOW_TRACKING_URI}")
    mlflow.set_experiment(EXPERIMENT_NAME)

    print("Loading California Housing dataset...")
    data = fetch_california_housing(as_frame=True)
    features = data.frame.drop(columns=["MedHouseVal"])
    labels = (data.frame["MedHouseVal"] > data.frame["MedHouseVal"].median()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels,
    )

    n_estimators = 100
    max_depth = 10

    with mlflow.start_run(run_name="rf_baseline_run") as run:
        mlflow.log_params({
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "features_count": features.shape[1],
        })

        print("Training Random Forest classifier...")
        classifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
        )
        classifier.fit(X_train, y_train)

        predictions = classifier.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions),
            "precision": precision_score(y_test, predictions, zero_division=0),
            "recall": recall_score(y_test, predictions, zero_division=0),
            "f1_score": f1_score(y_test, predictions, zero_division=0),
        }
        mlflow.log_metrics(metrics)
        print(f"Accuracy: {metrics['accuracy']:.4f}; F1: {metrics['f1_score']:.4f}")

        # Pin cloudpickle because newer MLflow versions can otherwise use
        # skops by default, and the inference endpoint uses sklearn.load_model.
        sample = X_train.iloc[:20]
        signature = mlflow.models.infer_signature(sample, classifier.predict(sample))
        mlflow.sklearn.log_model(
            sk_model=classifier,
            artifact_path="model",  # compatible with MLflow 2.x and 3.x
            registered_model_name=MODEL_NAME,
            signature=signature,
            input_example=X_train.iloc[:2],
            serialization_format="cloudpickle",
            await_registration_for=300,
        )

        versions = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI).search_model_versions(
            f"name='{MODEL_NAME}'"
        )
        matching = [v for v in versions if getattr(v, "run_id", None) == run.info.run_id]
        if matching:
            registered_version = max(matching, key=lambda v: int(v.version)).version
            print(f"Successfully registered {MODEL_NAME}, version {registered_version}")
        else:
            print(f"Successfully logged and requested registration for {MODEL_NAME}")
        print("Training Run ID:", run.info.run_id)
        print("Start/retry the FastAPI endpoint to load the registered model.")


if __name__ == "__main__":
    run_training()
