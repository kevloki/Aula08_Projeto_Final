import os
import mlflow
import mlflow.sklearn
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = "fetal-health-best-model"
LABEL_MAP  = {0: "Normal", 1: "Suspect", 2: "Pathological"}

# mesma ordem usada no treino — importante nao mudar
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

_cache = None


def load_model():
    global _cache
    if _cache is None:
        username = os.environ["DAGSHUB_USERNAME"]
        token    = os.environ["DAGSHUB_TOKEN"]
        repo     = os.environ["DAGSHUB_REPO"].split("/")[-1]
        mlflow.set_tracking_uri(f"https://dagshub.com/{username}/{repo}.mlflow")
        os.environ["MLFLOW_TRACKING_USERNAME"] = username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token
        _cache = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/1")
    return _cache


def predict(inputs: dict) -> dict:
    # calculo as features derivadas antes de montar o DataFrame
    decels = inputs.get("light_decelerations", 0) + inputs.get("severe_decelerations", 0) + inputs.get("prolongued_decelerations", 0)
    inputs["accel_decel_ratio"]     = inputs.get("accelerations", 0) / decels if decels > 0 else 0
    inputs["histogram_range"]       = inputs.get("histogram_max", 0) - inputs.get("histogram_min", 0)
    inputs["histogram_skew_proxy"]  = inputs.get("histogram_mean", 0) - inputs.get("histogram_median", 0)

    df       = pd.DataFrame([inputs])[FEATURE_COLS]
    model    = load_model()
    class_id = int(model.predict(df)[0])
    probs    = model.predict_proba(df)[0]

    return {
        "label":         LABEL_MAP[class_id],
        "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
    }
