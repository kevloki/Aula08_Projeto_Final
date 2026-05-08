# Fetal Health Classification — Pipeline MLOps End-to-End

Pipeline completo de Machine Learning para classificação de saúde fetal a partir de dados de Cardiotocografia (CTG).

## Problema

Exames de cardiotocografia medem a frequência cardíaca fetal e contrações uterinas durante a gestação. O objetivo é classificar automaticamente o estado de saúde fetal em:

- **Normal (1)** — sem indicadores de risco
- **Suspect (2)** — requer acompanhamento especializado
- **Pathological (3)** — requer intervenção imediata

**Dataset**: [Fetal Health Classification — Kaggle](https://www.kaggle.com/datasets/andrewmvd/fetal-health-classification/) · ~2.126 registros · 21 features

## Arquitetura

```
[Kaggle CSV] → [Supabase (PostgreSQL)] → [DuckDB (Feature Eng.)]
                                                    ↓
                              [DagsHub/DVC] ← [Versionamento]
                                    ↓
                             [MLflow Tracking]
                                    ↓
                           [Docker Container]
                                    ↓
                          [Render Deploy] → [Streamlit App]
```

## Estrutura do Projeto

```
projeto-final/
├── data/
│   ├── raw/                  # CSV bruto (versionado via DVC)
│   └── processed/            # Parquet processado (versionado via DVC)
├── notebooks/                # EDA exploratória
├── src/
│   ├── ingestion.py          # Supabase → DataFrame → CSV
│   ├── preprocessing.py      # Feature engineering com DuckDB
│   ├── train.py              # Treinamento + MLflow tracking
│   └── predict.py            # Função de inferência
├── app/
│   └── streamlit_app.py      # Interface do usuário
├── Dockerfile
├── requirements.txt
├── dvc.yaml                  # Pipeline DVC
├── params.yaml               # Hiperparâmetros versionados
└── .env.example              # Template de credenciais
```

## Setup

### 1. Clone e ambiente

```bash
git clone https://github.com/<SEU_USUARIO>/fetal-health-mlops.git
cd fetal-health-mlops
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Credenciais

```bash
cp .env.example .env
# Edite .env com suas credenciais reais
```

### 3. Supabase — upload do dataset

1. Crie um projeto em [supabase.com](https://supabase.com)
2. No SQL Editor, execute:

```sql
CREATE TABLE fetal_health (
    baseline_value FLOAT,
    accelerations FLOAT,
    fetal_movement FLOAT,
    uterine_contractions FLOAT,
    light_decelerations FLOAT,
    severe_decelerations FLOAT,
    prolongued_decelerations FLOAT,
    abnormal_short_term_variability FLOAT,
    mean_value_of_short_term_variability FLOAT,
    percentage_of_time_with_abnormal_long_term_variability FLOAT,
    mean_value_of_long_term_variability FLOAT,
    histogram_width FLOAT,
    histogram_min FLOAT,
    histogram_max FLOAT,
    histogram_number_of_peaks FLOAT,
    histogram_number_of_zeroes FLOAT,
    histogram_mode FLOAT,
    histogram_mean FLOAT,
    histogram_median FLOAT,
    histogram_tendency FLOAT,
    fetal_health FLOAT
);
```

3. Importe o CSV via **Table Editor → Import Data** ou via Supabase CLI.

### 4. DagsHub + DVC

```bash
dvc init
dvc remote add origin https://dagshub.com/<USUARIO>/<REPO>.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <USUARIO>
dvc remote modify origin --local password <TOKEN>
```

### 5. Reproduzir o pipeline

```bash
dvc repro
```

Ou etapa por etapa:

```bash
python src/ingestion.py      # baixa dados do Supabase
python src/preprocessing.py  # feature engineering com DuckDB
python src/train.py          # treina 3 modelos e registra no MLflow
```

### 6. Rodar a aplicação localmente

```bash
streamlit run app/streamlit_app.py
```

### 7. Docker

```bash
docker build -t fetal-health-app .
docker run -p 8501:8501 --env-file .env -e PORT=8501 fetal-health-app
```

## Modelos Comparados

| Modelo | Estratégia | Tratamento Desbalanceamento |
|--------|-----------|----------------------------|
| Random Forest (200 trees) | Bagging | `class_weight=balanced` |
| Gradient Boosting (150 est.) | Boosting | — |
| Logistic Regression | Linear + StandardScaler | `class_weight=balanced` |

Métrica principal: **F1-Macro** (adequada para classes desbalanceadas).

## Deploy no Render

1. Conecte o repositório GitHub ao [Render](https://render.com)
2. Crie um **Web Service** → Docker
3. Configure as variáveis de ambiente (seção *Environment*):
   - `SUPABASE_URL`, `SUPABASE_KEY`
   - `DAGSHUB_USERNAME`, `DAGSHUB_TOKEN`, `DAGSHUB_REPO`
4. Deploy automático a cada push na branch `main`.

## Decisões Técnicas

- **DuckDB**: escolhido pela integração nativa com pandas e execução SQL in-process, sem servidor dedicado.
- **F1-Macro**: métrica principal porque as classes são desbalanceadas (~77% Normal).
- **class_weight=balanced**: compensa o desbalanceamento sem necessidade de oversampling.
- **MLflow Model Registry**: facilita o carregamento do modelo em produção com `models:/nome/Production`.
