import os
import tempfile
import pathlib
import yaml
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
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
import warnings
warnings.filterwarnings("ignore")

load_dotenv()

PROCESSED_PATH = pathlib.Path(__file__).parent.parent / "data" / "processed" / "fetal_health_processed.parquet"
PARAMS_PATH    = pathlib.Path(__file__).parent.parent / "params.yaml"
TARGET_COL     = "fetal_health"
EXPERIMENT     = "fetal-health-classification"
MODEL_NAME     = "fetal-health-best-model"

# parametros versionados no params.yaml pra garantir reproducibilidade
_p           = yaml.safe_load(PARAMS_PATH.read_text())["train"]
RANDOM_STATE = _p["random_state"]
TEST_SIZE    = _p["test_size"]
CV_FOLDS     = _p["cv_folds"]


def setup_mlflow():
    dagshub.init(
        repo_owner=os.environ["DAGSHUB_USERNAME"],
        repo_name=os.environ["DAGSHUB_REPO"].split("/")[-1],
        mlflow=True,
    )
    mlflow.set_experiment(EXPERIMENT)


def load_data():
    df = pd.read_parquet(PROCESSED_PATH)
    X = df.drop(columns=[TARGET_COL])
    # sklearn espera classes 0-indexadas, o dataset usa 1/2/3
    y = df[TARGET_COL].astype(int) - 1
    return X, y


def metricas(y_true, y_pred, y_prob):
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "f1_macro":    f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "roc_auc_ovr": roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro"),
    }


def treinar(name, pipeline, X_train, X_test, y_train, y_test, params):
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_type", name)
        mlflow.log_params(params)

        cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1_macro")
        mlflow.log_metric("cv_f1_macro_mean", cv_scores.mean())
        mlflow.log_metric("cv_f1_macro_std",  cv_scores.std())

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)

        m = metricas(y_test, y_pred, y_prob)
        mlflow.log_metrics(m)

        report = classification_report(y_test, y_pred, target_names=["Normal", "Suspect", "Pathological"])
        rpath = pathlib.Path(tempfile.gettempdir()) / f"{name}_report.txt"
        rpath.write_text(report, encoding="utf-8")
        mlflow.log_artifact(str(rpath), artifact_path="reports")

        info = mlflow.sklearn.log_model(pipeline, name="model", registered_model_name=None)

        print(f"  [{name}] accuracy={m['accuracy']:.4f} | f1_macro={m['f1_macro']:.4f} | roc_auc={m['roc_auc_ovr']:.4f}")
        return mlflow.active_run().info.run_id, m["f1_macro"], info.model_uri


def run():
    print("Conectando MLflow ao DagsHub...")
    setup_mlflow()

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train: {len(X_train)} | Test: {len(X_test)}")

    # class_weight balanceado porque temos 77% Normal e so 8% Pathological
    cw = dict(enumerate(compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)))

    modelos = [
        ("RandomForest",
         Pipeline([("clf", RandomForestClassifier(n_estimators=200, max_depth=15, class_weight=cw, random_state=RANDOM_STATE, n_jobs=-1))]),
         {"n_estimators": 200, "max_depth": 15, "class_weight": "balanced"}),

        ("GradientBoosting",
         Pipeline([("clf", GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5, random_state=RANDOM_STATE))]),
         {"n_estimators": 150, "learning_rate": 0.1, "max_depth": 5}),

        ("LogisticRegression",
         Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=1000, class_weight=cw, random_state=RANDOM_STATE, multi_class="multinomial"))]),
         {"max_iter": 1000, "solver": "lbfgs", "class_weight": "balanced"}),
    ]

    resultados = []
    print("\nTreinando:")
    for name, pipeline, params in modelos:
        run_id, f1, uri = treinar(name, pipeline, X_train, X_test, y_train, y_test, params)
        resultados.append((run_id, f1, name, uri))

    _, best_f1, best_name, best_uri = max(resultados, key=lambda x: x[1])
    print(f"\nMelhor: {best_name} (f1_macro={best_f1:.4f})")

    mv = mlflow.register_model(best_uri, MODEL_NAME)
    mlflow.tracking.MlflowClient().transition_model_version_stage(
        name=MODEL_NAME, version=mv.version, stage="Production", archive_existing_versions=True
    )
    print(f"Registrado como {MODEL_NAME} v{mv.version} em Production.")


if __name__ == "__main__":
    run()
