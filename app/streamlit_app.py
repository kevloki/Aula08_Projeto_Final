import sys
import os
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(page_title="Saúde Fetal — CTG", page_icon="🫀", layout="wide")

LABEL_COLORS = {"Normal": "#2ecc71", "Suspect": "#f39c12", "Pathological": "#e74c3c"}

FEATURES = {
    "baseline_value": ("Baseline FHR (bpm)", 106.0, 160.0, 1.0, 120.0),
    "accelerations": ("Acelerações/s", 0.0, 0.02, 0.001, 0.0),
    "fetal_movement": ("Movimentos fetais/s", 0.0, 0.5, 0.001, 0.0),
    "uterine_contractions": ("Contrações uterinas/s", 0.0, 0.015, 0.001, 0.004),
    "light_decelerations": ("Desacel. leves/s", 0.0, 0.015, 0.001, 0.0),
    "severe_decelerations": ("Desacel. severas/s", 0.0, 0.002, 0.0001, 0.0),
    "prolongued_decelerations": ("Desacel. prolongadas/s", 0.0, 0.005, 0.0001, 0.0),
    "abnormal_short_term_variability": ("Variab. anormal curto prazo (%)", 0.0, 100.0, 1.0, 73.0),
    "mean_value_of_short_term_variability": ("Média variab. curto prazo", 0.0, 7.0, 0.1, 0.5),
    "percentage_of_time_with_abnormal_long_term_variability": ("% tempo variab. anormal longo prazo", 0.0, 100.0, 1.0, 43.0),
    "mean_value_of_long_term_variability": ("Média variab. longo prazo", 0.0, 50.0, 0.1, 4.0),
    "histogram_width": ("Largura histograma", 0.0, 180.0, 1.0, 64.0),
    "histogram_min": ("Mínimo histograma", 50.0, 159.0, 1.0, 62.0),
    "histogram_max": ("Máximo histograma", 122.0, 238.0, 1.0, 126.0),
    "histogram_number_of_peaks": ("Nº picos", 0.0, 18.0, 1.0, 2.0),
    "histogram_number_of_zeroes": ("Nº zeros", 0.0, 10.0, 1.0, 0.0),
    "histogram_mode": ("Moda histograma", 60.0, 187.0, 1.0, 120.0),
    "histogram_mean": ("Média histograma", 73.0, 182.0, 1.0, 137.0),
    "histogram_median": ("Mediana histograma", 77.0, 186.0, 1.0, 138.0),
    "histogram_variance": ("Variância histograma", 0.0, 270.0, 1.0, 3.0),
    "histogram_tendency": ("Tendência histograma", -1.0, 1.0, 1.0, 0.0),
}


@st.cache_resource(show_spinner="Carregando modelo... (pode levar ~30s na primeira vez)")
def load_model():
    import mlflow, mlflow.sklearn
    username = os.environ["DAGSHUB_USERNAME"]
    token = os.environ["DAGSHUB_TOKEN"]
    repo = os.environ["DAGSHUB_REPO"].split("/")[-1]
    os.environ["MLFLOW_TRACKING_URI"] = f"https://dagshub.com/{username}/{repo}.mlflow"
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = token
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    return mlflow.sklearn.load_model("models:/fetal-health-best-model/1")


@st.cache_data(show_spinner="Carregando dados do Supabase...")
def load_eda_data():
    from supabase import create_client
    client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])
    rows = []
    offset = 0
    while True:
        batch = client.table("fetal_health").select("*").range(offset, offset + 999).execute().data
        if not batch:
            break
        rows.extend(batch)
        offset += 1000
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows)
    df = df.rename(columns={"baseline value": "baseline_value"})
    df["fetal_health"] = df["fetal_health"].astype(int)
    df["label"] = df["fetal_health"].map({1: "Normal", 2: "Suspect", 3: "Pathological"})
    return df


def run_predict(model, input_dict):
    decels = input_dict.get("light_decelerations", 0) + input_dict.get("severe_decelerations", 0) + input_dict.get("prolongued_decelerations", 0)
    input_dict["accel_decel_ratio"] = input_dict.get("accelerations", 0) / decels if decels > 0 else 0
    input_dict["histogram_range"] = input_dict.get("histogram_max", 0) - input_dict.get("histogram_min", 0)
    input_dict["histogram_skew_proxy"] = input_dict.get("histogram_mean", 0) - input_dict.get("histogram_median", 0)

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
    row = pd.DataFrame([input_dict])[FEATURE_COLS]
    class_id = int(model.predict(row)[0])
    probs = model.predict_proba(row)[0]
    return {
        "label": LABEL_MAP[class_id],
        "probabilities": {LABEL_MAP[i]: float(p) for i, p in enumerate(probs)},
    }


# ── Pré-carrega modelo ao iniciar ─────────────────────────────────────────
model = load_model()

# ── Sidebar ───────────────────────────────────────────────────────────────
st.sidebar.title("Parâmetros CTG")
input_vals = {}
for key, (label, mn, mx, step, default) in FEATURES.items():
    input_vals[key] = st.sidebar.number_input(label, min_value=mn, max_value=mx, value=default, step=step, format="%.4f")

predict_btn = st.sidebar.button("Classificar", use_container_width=True, type="primary")

# ── Main ──────────────────────────────────────────────────────────────────
st.title("Classificador de Saude Fetal")
st.caption("Pipeline MLOps · Supabase → DuckDB → MLflow → Streamlit")

tab_pred, tab_eda, tab_info = st.tabs(["Predicao", "Exploracao de Dados", "Sobre o Projeto"])

with tab_pred:
    if predict_btn:
        try:
            result = run_predict(model, dict(input_vals))
            label = result["label"]
            color = LABEL_COLORS[label]
            st.markdown(f"<h2 style='color:{color}; text-align:center;'>Resultado: {label}</h2>", unsafe_allow_html=True)

            cols = st.columns(3)
            for col, (lbl, prob) in zip(cols, result["probabilities"].items()):
                fig = go.Figure(go.Indicator(
                    mode="gauge+number", value=prob * 100,
                    number={"suffix": "%"},
                    title={"text": lbl},
                    gauge={"axis": {"range": [0, 100]}, "bar": {"color": LABEL_COLORS[lbl]}},
                ))
                fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
                col.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Erro: {e}")
    else:
        st.info("Preencha os parametros no painel esquerdo e clique em **Classificar**.")
        st.markdown("""
        **Classes:**
        - Normal — saude fetal saudavel
        - Suspect — requer acompanhamento
        - Pathological — requer intervencao
        """)

with tab_eda:
    st.subheader("Exploracao dos Dados")
    try:
        df = load_eda_data()
        c1, c2, c3 = st.columns(3)
        c1.metric("Registros", len(df))
        c2.metric("Features", 21)
        c3.metric("Classes", 3)

        fig_pie = px.pie(df, names="label", title="Distribuicao das Classes",
                         color="label", color_discrete_map=LABEL_COLORS)
        st.plotly_chart(fig_pie, use_container_width=True)

        num_cols = [c for c in df.columns if c not in ("id", "label", "fetal_health")]
        feat = st.selectbox("Feature:", num_cols)
        fig_box = px.box(df, x="label", y=feat, color="label",
                         color_discrete_map=LABEL_COLORS,
                         title=f"{feat} por classe")
        st.plotly_chart(fig_box, use_container_width=True)

        st.subheader("Estatisticas Descritivas")
        st.dataframe(df[num_cols].describe().T.style.format("{:.3f}"))
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")

with tab_info:
    st.markdown("""
    ## Sobre o Projeto

    Pipeline completo de MLOps para classificacao de saude fetal via Cardiotocografia (CTG).

    **Dataset:** [Fetal Health Classification — Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/) · 2.126 registros · 21 features

    **Arquitetura:**
    ```
    Kaggle CSV → Supabase → DuckDB → Feature Engineering
                    ↓
              DagsHub / DVC (versionamento)
                    ↓
              MLflow (experimentacao — 3 modelos)
                    ↓
              Docker → Render → Streamlit (esta app)
    ```

    **Modelos comparados:**

    | Modelo | F1-Macro | ROC-AUC |
    |--------|----------|---------|
    | Random Forest | 93.6% | 98.9% |
    | Gradient Boosting | 91.5% | 98.4% |
    | Logistic Regression | 79.6% | 96.9% |

    O melhor modelo (Random Forest) esta em producao via MLflow Model Registry.

    **Stack:** Python · scikit-learn · DuckDB · Supabase · MLflow · DVC · DagsHub · Docker · Render · Streamlit
    """)
