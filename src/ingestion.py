import os
import pathlib
import pandas as pd
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

RAW_DIR = pathlib.Path(__file__).parent.parent / "data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)
TABLE_NAME = "fetal_health"
OUTPUT_PATH = RAW_DIR / "fetal_health.csv"


def get_supabase_client():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_KEY"])


def fetch_all_rows(client, table):
    # o Supabase limita 1000 linhas por request, entao preciso paginar
    page_size = 1000
    offset = 0
    rows = []
    while True:
        batch = client.table(table).select("*").range(offset, offset + page_size - 1).execute().data
        if not batch:
            break
        rows.extend(batch)
        offset += page_size
        if len(batch) < page_size:
            break
    return pd.DataFrame(rows)


def run():
    print("Conectando ao Supabase...")
    client = get_supabase_client()

    print(f"Baixando tabela '{TABLE_NAME}'...")
    df = fetch_all_rows(client, TABLE_NAME)
    print(f"  {len(df)} registros, {len(df.columns)} colunas.")

    # o kaggle chama de 'baseline value' com espaco, padronizo aqui
    if "baseline value" in df.columns:
        df = df.rename(columns={"baseline value": "baseline_value"})

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Salvo em: {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    run()
