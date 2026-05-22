"""
Treina modelos preliminares para o Acompanhamento 2 e salva o modelo final.

Saidas geradas:
- modelo.pkl
- data/resultados_modelos.csv
- data/resumo_acompanhamento_2.txt
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_log_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

from src.feature_engineering import create_features


RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
COL_CONFIG_PATH = ROOT / "src" / "col_config.json"
MODEL_PATH = ROOT / "modelo.pkl"
RESULTS_PATH = DATA_DIR / "resultados_modelos.csv"
SUMMARY_PATH = DATA_DIR / "resumo_acompanhamento_2.txt"


def load_column_config() -> dict:
    with COL_CONFIG_PATH.open(encoding="utf-8") as file:
        return json.load(file)


def build_preprocessor() -> ColumnTransformer:
    cfg = load_column_config()

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    cat_none_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    cat_real_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    ord_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])

    return ColumnTransformer(
        [
            ("num", num_pipe, cfg["num_cols"]),
            ("cat_none", cat_none_pipe, cfg["cat_none_cols"]),
            ("cat_real", cat_real_pipe, cfg["cat_real_cols"]),
            ("ordinal", ord_pipe, cfg["already_encoded"]),
        ],
        remainder="drop",
    )


def make_pipeline(model) -> Pipeline:
    regressor = TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
    )
    return Pipeline(
        [
            ("prep", build_preprocessor()),
            ("model", regressor),
        ]
    )


def rmsle(y_true, y_pred) -> float:
    y_pred = np.maximum(y_pred, 0)
    return float(np.sqrt(mean_squared_log_error(y_true, y_pred)))


def evaluate_model(name: str, pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict:
    start = time.perf_counter()
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=-1)
    elapsed = time.perf_counter() - start
    y_pred = np.maximum(y_pred, 0)

    return {
        "modelo": name,
        "RMSLE": rmsle(y, y_pred),
        "MAE": mean_absolute_error(y, y_pred),
        "R2": r2_score(y, y_pred),
        "tempo_cv_seg": elapsed,
    }


def main() -> None:
    train_raw = pd.read_csv(DATA_DIR / "treino.csv")
    outlier_mask = (train_raw["GrLivArea"] > 4000) & (train_raw["SalePrice"] < 300_000)
    train_raw = train_raw.loc[~outlier_mask].copy()

    X = create_features(train_raw.drop(columns=["SalePrice"]))
    y = train_raw["SalePrice"].copy()

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    models = {
        "Regressao Linear": LinearRegression(),
        "Ridge": Ridge(alpha=10.0, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=250,
            max_features="sqrt",
            min_samples_leaf=2,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "XGBoost": XGBRegressor(
            n_estimators=450,
            learning_rate=0.04,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        ),
    }

    rows = []
    print("Avaliando modelos com validacao cruzada 5-fold...\n")
    for name, model in models.items():
        pipe = make_pipeline(model)
        result = evaluate_model(name, pipe, X, y, cv)
        rows.append(result)
        print(
            f"{name:16s} RMSLE={result['RMSLE']:.5f} "
            f"MAE=${result['MAE']:,.0f} R2={result['R2']:.4f} "
            f"tempo={result['tempo_cv_seg']:.1f}s"
        )

    results = pd.DataFrame(rows).sort_values("RMSLE")
    results.to_csv(RESULTS_PATH, index=False)

    best_name = results.iloc[0]["modelo"]
    best_model = models[best_name]
    final_pipeline = make_pipeline(best_model)
    final_pipeline.fit(X, y)
    joblib.dump(final_pipeline, MODEL_PATH)

    best_row = results.iloc[0]
    summary = (
        "RESUMO ACOMPANHAMENTO 2\n"
        "========================\n"
        f"Amostras de treino usadas: {len(X)}\n"
        "Validacao: KFold 5-fold com random_state=42\n"
        f"Modelo escolhido: {best_name}\n"
        f"RMSLE: {best_row['RMSLE']:.5f}\n"
        f"MAE: ${best_row['MAE']:,.0f}\n"
        f"R2: {best_row['R2']:.4f}\n"
        f"Modelo salvo em: {MODEL_PATH.name}\n"
        f"Metricas salvas em: {RESULTS_PATH}\n"
    )
    SUMMARY_PATH.write_text(summary, encoding="utf-8")

    print("\nRanking:")
    print(results.to_string(index=False, formatters={"MAE": "${:,.0f}".format}))
    print(f"\nModelo final salvo em: {MODEL_PATH}")
    print(f"Metricas salvas em: {RESULTS_PATH}")
    print(f"Resumo salvo em: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
