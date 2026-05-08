"""
Etapa 6 — Interface Streamlit para classificação de saúde fetal.
"""
import sys
import pathlib
sys.path.append(str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Saúde Fetal — Classificador CTG",
    page_icon="🫀",
    layout="wide",
)

LABEL_COLORS = {
    "Normal": "#2ecc71",
    "Suspect": "#f39c12",
    "Pathological": "#e74c3c",
}

FEATURE_DESCRIPTIONS = {
    "baseline_value": ("Baseline FHR (bpm)", 106, 160, 1, 120.0),
    "accelerations": ("Acelerações por segundo", 0.0, 0.02, 0.001, 0.0),
    "fetal_movement": ("Movimentos fetais por segundo", 0.0, 0.5, 0.001, 0.0),
    "uterine_contractions": ("Contrações uterinas por segundo", 0.0, 0.015, 0.001, 0.004),
    "light_decelerations": ("Desacelerações leves por segundo", 0.0, 0.015, 0.001, 0.0),
    "severe_decelerations": ("Desacelerações severas por segundo", 0.0, 0.002, 0.0001, 0.0),
    "prolongued_decelerations": ("Desacelerações prolongadas por segundo", 0.0, 0.005, 0.0001, 0.0),
    "abnormal_short_term_variability": ("Variabilidade anormal de curto prazo (%)", 0, 100, 1, 73.0),
    "mean_value_of_short_term_variability": ("Média variabilidade curto prazo", 0.0, 7.0, 0.1, 0.5),
    "percentage_of_time_with_abnormal_long_term_variability": ("% tempo variab. anormal longo prazo", 0, 100, 1, 43.0),
    "mean_value_of_long_term_variability": ("Média variabilidade longo prazo", 0.0, 50.0, 0.1, 4.0),
    "histogram_width": ("Largura do histograma", 0, 180, 1, 64.0),
    "histogram_min": ("Mínimo do histograma", 50, 159, 1, 62.0),
    "histogram_max": ("Máximo do histograma", 122, 238, 1, 126.0),
    "histogram_number_of_peaks": ("Nº de picos no histograma", 0, 18, 1, 2.0),
    "histogram_number_of_zeroes": ("Nº de zeros no histograma", 0, 10, 1, 0.0),
    "histogram_mode": ("Moda do histograma", 60, 187, 1, 120.0),
    "histogram_mean": ("Média do histograma", 73, 182, 1, 137.0),
    "histogram_median": ("Mediana do histograma", 77, 186, 1, 138.0),
    "histogram_tendency": ("Tendência do histograma", -1, 1, 1, 0),
}


@st.cache_resource(show_spinner="Carregando modelo do MLflow...")
def get_model():
    from src.predict import load_production_model
    return load_production_model()


def make_gauge(probability: float, label: str):
    color = LABEL_COLORS.get(label, "#3498db")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        number={"suffix": "%", "font": {"size": 28}},
        title={"text": label, "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "gray",
        },
    ))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=40, b=10))
    return fig


# ── Sidebar: inputs ─────────────────────────────────────────────────────────
st.sidebar.title("Parâmetros CTG")
st.sidebar.caption("Cardiotocografia fetal — insira os valores do exame")

input_vals = {}
for key, (label, min_v, max_v, step, default) in FEATURE_DESCRIPTIONS.items():
    input_vals[key] = st.sidebar.number_input(
        label, min_value=float(min_v), max_value=float(max_v),
        value=float(default), step=float(step), format="%.4f",
    )

predict_btn = st.sidebar.button("🔍 Classificar", use_container_width=True, type="primary")

# ── Main area ────────────────────────────────────────────────────────────────
st.title("🫀 Classificador de Saúde Fetal")
st.caption("Pipeline MLOps · Supabase → DuckDB → MLflow → Streamlit")

tab_pred, tab_eda, tab_info = st.tabs(["Predição", "Exploração de Dados", "Sobre o Projeto"])

with tab_pred:
    if predict_btn:
        try:
            from src.predict import predict
            with st.spinner("Calculando..."):
                result = predict(dict(input_vals))

            label = result["label"]
            color = LABEL_COLORS[label]

            st.markdown(
                f"<h2 style='color:{color}; text-align:center;'>Resultado: {label}</h2>",
                unsafe_allow_html=True,
            )

            cols = st.columns(3)
            for col, (lbl, prob) in zip(cols, result["probabilities"].items()):
                col.plotly_chart(make_gauge(prob, lbl), use_container_width=True)

            with st.expander("Detalhes da predição"):
                prob_df = pd.DataFrame(
                    list(result["probabilities"].items()),
                    columns=["Classe", "Probabilidade"],
                ).sort_values("Probabilidade", ascending=False)
                st.dataframe(prob_df.style.format({"Probabilidade": "{:.2%}"}), hide_index=True)

        except Exception as e:
            st.error(f"Erro ao carregar o modelo: {e}")
            st.info("Verifique as variáveis de ambiente (DAGSHUB_USERNAME, DAGSHUB_TOKEN, etc.) no arquivo .env")
    else:
        st.info("Configure os parâmetros no painel lateral e clique em **Classificar**.")

        st.markdown("""
        ### Como usar
        1. Insira os valores obtidos do exame de cardiotocografia (CTG) no painel à esquerda.
        2. Clique em **Classificar** para obter a predição do modelo.
        3. O resultado indica uma de três categorias:
           - 🟢 **Normal** — saúde fetal saudável
           - 🟡 **Suspect** — requer acompanhamento
           - 🔴 **Pathological** — requer intervenção
        """)

with tab_eda:
    st.subheader("Exploração dos Dados de Treinamento")
    try:
        processed_path = pathlib.Path(__file__).parent.parent / "data" / "processed" / "fetal_health_processed.parquet"
        if processed_path.exists():
            df = pd.read_parquet(processed_path)
            df["fetal_health_label"] = df["fetal_health"].map({1: "Normal", 2: "Suspect", 3: "Pathological"})

            col1, col2, col3 = st.columns(3)
            col1.metric("Total de registros", len(df))
            col2.metric("Features", len(df.columns) - 2)
            col3.metric("Classes", df["fetal_health"].nunique())

            fig_dist = px.pie(
                df, names="fetal_health_label",
                title="Distribuição das Classes",
                color="fetal_health_label",
                color_discrete_map=LABEL_COLORS,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            feature_sel = st.selectbox(
                "Selecione uma feature para visualizar distribuição:",
                [c for c in df.columns if c not in ("fetal_health", "fetal_health_label")],
            )
            fig_box = px.box(
                df, x="fetal_health_label", y=feature_sel,
                color="fetal_health_label",
                color_discrete_map=LABEL_COLORS,
                title=f"Distribuição de '{feature_sel}' por classe",
            )
            st.plotly_chart(fig_box, use_container_width=True)

            st.subheader("Estatísticas Descritivas")
            st.dataframe(df.drop(columns=["fetal_health_label"]).describe().T.style.format("{:.3f}"))
        else:
            st.warning("Dados processados não encontrados. Execute `python src/preprocessing.py` primeiro.")
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")

with tab_info:
    st.markdown("""
    ## Sobre o Projeto

    Este projeto implementa um pipeline completo de MLOps para classificação de saúde fetal
    a partir de dados de Cardiotocografia (CTG).

    ### Dataset
    - **Fonte**: [Fetal Health Classification — Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/)
    - **Registros**: ~2.126 exames de CTG
    - **Target**: Normal (1), Suspect (2), Pathological (3)

    ### Arquitetura
    ```
    Kaggle CSV → Supabase (PostgreSQL) → DuckDB (Feature Eng.)
                        ↓
              DagsHub/DVC (Versionamento)
                        ↓
              MLflow (Experimentação)
                        ↓
              Docker → Render (Deploy)
                        ↓
              Streamlit (Esta aplicação)
    ```

    ### Modelos Comparados
    | Modelo | Tipo |
    |--------|------|
    | Random Forest | Ensemble Bagging |
    | Gradient Boosting | Ensemble Boosting |
    | Logistic Regression | Linear |

    ### Stack Técnica
    `Python` · `scikit-learn` · `DuckDB` · `Supabase` · `MLflow` · `DVC` · `DagsHub` · `Docker` · `Render` · `Streamlit`
    """)
