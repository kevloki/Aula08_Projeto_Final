"""
Etapa 5 — Treinamento de modelos e rastreamento com MLflow/DagsHub.
Treina 3 modelos, registra métricas/artefatos e promove o melhor ao Model Registry.
"""
import os
import sys
import tempfile
import pathlib
import dagshub
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from dotenv import load_dotenv
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix,
)
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

import yaml

PROCESSED_PATH = (
    pathlib.Path(__file__).parent.parent / "data" / "processed" / "fetal_health_processed.parquet"
)
PARAMS_PATH = pathlib.Path(__file__).parent.parent / "params.yaml"
TARGET_COL = "fetal_health"
EXPERIMENT_NAME = "fetal-health-classification"
MODEL_NAME = "fetal-health-best-model"

_params = yaml.safe_load(PARAMS_PATH.read_text())["train"]
RANDOM_STATE = _params["random_state"]
TEST_SIZE = _params["test_size"]
CV_FOLDS = _params["cv_folds"]


def setup_mlflow():
    dagshub.init(
        repo_owner=os.environ["DAGSHUB_USERNAME"],
        repo_name=os.environ["DAGSHUB_REPO"].split("/")[-1],
        mlflow=True,
    )
    mlflow.set_experiment(EXPERIMENT_NAME)


def load_data():
    df = pd.read_parquet(PROCESSED_PATH)
    X = df.drop(columns=[TARGET_COL])
    # Converte target para 0-indexado para compatibilidade com roc_auc_score ovr
    y = df[TARGET_COL].astype(int) - 1
    return X, y


def compute_metrics(y_true, y_pred, y_prob):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc_ovr": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
    }


def train_model(name: str, pipeline: Pipeline, X_train, X_test, y_train, y_test, params: dict):
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_type", name)
        mlflow.log_params(params)

        # Cross-validation
        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        mlflow.log_metric("cv_f1_macro_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_macro_std", cv_scores.std())

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        metrics = compute_metrics(y_test, y_pred, y_prob)
        mlflow.log_metrics(metrics)

        # Salva relatório de classificação como artefato texto
        report = classification_report(y_test, y_pred, target_names=["Normal", "Suspect", "Pathological"])
        report_path = pathlib.Path(tempfile.gettempdir()) / f"{name}_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(report)
        mlflow.log_artifact(str(report_path), artifact_path="reports")

        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            registered_model_name=None,
        )

        run_id = mlflow.active_run().info.run_id
        print(f"  [{name}] accuracy={metrics['accuracy']:.4f} | f1_macro={metrics['f1_macro']:.4f} | roc_auc={metrics['roc_auc_ovr']:.4f}")
        return run_id, metrics["f1_macro"], model_info.model_uri


def run():
    print("Configurando MLflow + DagsHub...")
    setup_mlflow()

    print("Carregando dados processados...")
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train)} | Test: {len(X_test)}")

    class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    cw_dict = {i: w for i, w in enumerate(class_weights)}

    models = [
        (
            "RandomForest",
            Pipeline([
                ("clf", RandomForestClassifier(
                    n_estimators=200, max_depth=15,
                    class_weight=cw_dict, random_state=RANDOM_STATE, n_jobs=-1
                ))
            ]),
            {"n_estimators": 200, "max_depth": 15, "class_weight": "balanced"},
        ),
        (
            "GradientBoosting",
            Pipeline([
                ("clf", GradientBoostingClassifier(
                    n_estimators=150, learning_rate=0.1, max_depth=5,
                    random_state=RANDOM_STATE
                ))
            ]),
            {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5},
        ),
        (
            "LogisticRegression",
            Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(
                    max_iter=1000, class_weight=cw_dict,
                    random_state=RANDOM_STATE, multi_class="multinomial"
                ))
            ]),
            {"max_iter": 1000, "solver": "lbfgs", "class_weight": "balanced"},
        ),
    ]

    results = []
    print("\nTreinando modelos:")
    for name, pipeline, params in models:
        run_id, f1, model_uri = train_model(name, pipeline, X_train, X_test, y_train, y_test, params)
        results.append((run_id, f1, name, model_uri))

    # Seleciona o melhor modelo e registra no Model Registry
    best_run_id, best_f1, best_name, best_model_uri = max(results, key=lambda x: x[1])
    print(f"\nMelhor modelo: {best_name} (f1_macro={best_f1:.4f})")

    mv = mlflow.register_model(best_model_uri, MODEL_NAME)
    print(f"Modelo registrado: {MODEL_NAME} v{mv.version}")

    # Promove para Production
    client = mlflow.tracking.MlflowClient()
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=mv.version,
        stage="Production",
        archive_existing_versions=True,
    )
    print(f"Modelo promovido para Production.")


if __name__ == "__main__":
    run()
