import pathlib
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RAW_PATH = pathlib.Path(__file__).parent.parent / "data" / "raw" / "fetal_health.csv"
PROCESSED_DIR = pathlib.Path(__file__).parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PATH = PROCESSED_DIR / "fetal_health_processed.parquet"

TARGET_COL = "fetal_health"


def run():
    print(f"Lendo: {RAW_PATH}")
    raw_df = pd.read_csv(RAW_PATH)

    con = duckdb.connect()
    con.register("raw", raw_df)

    # uso DuckDB pra fazer tudo em SQL mesmo — mais limpo que pandas pra esse tipo de coisa
    # as 3 features novas que criei tentam capturar relacoes que o dataset original nao tem
    query = """
        SELECT
            baseline_value,
            accelerations,
            fetal_movement,
            uterine_contractions,
            light_decelerations,
            severe_decelerations,
            prolongued_decelerations,
            abnormal_short_term_variability,
            mean_value_of_short_term_variability,
            percentage_of_time_with_abnormal_long_term_variability,
            mean_value_of_long_term_variability,
            histogram_width,
            histogram_min,
            histogram_max,
            histogram_number_of_peaks,
            histogram_number_of_zeroes,
            histogram_mode,
            histogram_mean,
            histogram_median,
            histogram_variance,
            histogram_tendency,
            CASE
                WHEN (light_decelerations + severe_decelerations + prolongued_decelerations) = 0 THEN 0
                ELSE accelerations / (light_decelerations + severe_decelerations + prolongued_decelerations)
            END AS accel_decel_ratio,
            histogram_max - histogram_min AS histogram_range,
            histogram_mean - histogram_median AS histogram_skew_proxy,
            CAST(fetal_health AS INTEGER) AS fetal_health
        FROM (SELECT DISTINCT * FROM raw) deduped
        WHERE fetal_health IS NOT NULL
    """

    df = con.execute(query).df()
    con.close()

    print(f"  {len(df)} registros apos limpeza.")
    print(f"  Classes:\n{df[TARGET_COL].value_counts().to_string()}")

    df.to_parquet(OUTPUT_PATH, index=False)
    print(f"Parquet salvo em: {OUTPUT_PATH}")
    return df


if __name__ == "__main__":
    run()
