"""
Função de inferência reutilizável — carrega o modelo Production do MLflow
e retorna a predição com probabilidades para um input arbitrário.
"""
import os
import mlflow
import mlflow.sklearn
import pandas as pd
import dagshub
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "fetal-health-best-model"
LABEL_MAP = {0: "Normal", 1: "Suspect", 2: "Pathological"}
FEATURE_COLS = [
    "baseline_value", "accelerations", "fetal_movement", "uterine_contractions",
    "light_decelerations", "severe_decelerations", "prolongued_decelerations",
    "abnormal_short_term_variability", "mean_value_of_short_term_variability",
    "percentage_of_time_with_abnormal_long_term_variability",
    "mean_value_of_long_term_variability", "histogram_width", "histogram_min",
    "histogram_max", "histogram_number_of_peaks", "histogram_number_of_zeroes",
    "histogram_mode", "histogram_mean", "histogram_median", "histogram_variance",
    "histogram_tendency", "accel_decel_ratio", "histogram_range", "histogram_skew_proxy",
]

_model_cache = None


def _setup_dagshub():
    dagshub.init(
        repo_owner=os.environ["DAGSHUB_USERNAME"],
        repo_name=os.environ["DAGSHUB_REPO"].split("/")[-1],
        mlflow=True,
    )


def load_production_model():
    global _model_cache
    if _model_cache is None:
        _setup_dagshub()
        _model_cache = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/Production")
    return _model_cache


def predict(input_dict: dict) -> dict:
    """
    Recebe um dicionário com as features brutas (mesmas do CSV original)
    e retorna {'label': str, 'class_id': int, 'probabilities': dict}.
    """
    # Calcula features derivadas
    decels = (
        input_dict.get("light_decelerations", 0)
        + input_dict.get("severe_decelerations", 0)
        + input_dict.get("prolongued_decelerations", 0)
    )
    input_dict["accel_decel_ratio"] = (
        input_dict.get("accelerations", 0) / decels if decels > 0 else 0
    )
    input_dict["histogram_range"] = (
        input_dict.get("histogram_max", 0) - input_dict.get("histogram_min", 0)
    )
    input_dict["histogram_skew_proxy"] = (
        input_dict.get("histogram_mean", 0) - input_dict.get("histogram_median", 0)
    )

    df = pd.DataFrame([input_dict])[FEATURE_COLS]
    model = load_production_model()

    class_id = int(model.predict(df)[0])
    probs = model.predict_proba(df)[0]

    return {
        "label": LABEL_MAP[class_id],
        "class_id": class_id,
        "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
    }
