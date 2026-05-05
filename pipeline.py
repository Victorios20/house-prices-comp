"""
pipeline.py — script de predição avaliado pelo professor.

Uso:
    python pipeline.py                       # roda com data/teste_publico.csv
    python pipeline.py caminho/para/test.csv # roda com arquivo custom
"""

import sys
import pathlib
import numpy as np
import pandas as pd
import joblib

MODEL_PATH = pathlib.Path(__file__).parent / 'modelo.pkl'
SRC_PATH   = pathlib.Path(__file__).parent / 'src'

# Garante que src/ está no path para importar feature_engineering
if str(SRC_PATH.parent) not in sys.path:
    sys.path.insert(0, str(SRC_PATH.parent))

from src.feature_engineering import create_features


def predict(df: pd.DataFrame) -> np.ndarray:
    """
    Recebe um DataFrame com as colunas brutas do dataset (sem SalePrice)
    e retorna um array com os preços preditos em dólares.
    """
    model = joblib.load(MODEL_PATH)
    X = create_features(df)
    preds = model.predict(X)
    return np.maximum(preds, 0)  # garante sem valores negativos


if __name__ == '__main__':
    test_path = sys.argv[1] if len(sys.argv) > 1 else 'data/teste_publico.csv'

    print(f'Carregando dados de: {test_path}')
    test_df = pd.read_csv(test_path)

    # Remove SalePrice se presente (modo de avaliação interna)
    has_truth = 'SalePrice' in test_df.columns
    if has_truth:
        y_true = test_df.pop('SalePrice')

    predictions = predict(test_df)

    output = test_df[['Id']].copy() if 'Id' in test_df.columns else pd.DataFrame()
    output['SalePrice'] = predictions

    print(f'\nPredições geradas: {len(predictions)} amostras')
    print(f'Faixa de preços  : ${predictions.min():,.0f} — ${predictions.max():,.0f}')
    print(f'Mediana          : ${np.median(predictions):,.0f}')

    if has_truth:
        from sklearn.metrics import mean_squared_log_error
        rmsle = np.sqrt(mean_squared_log_error(y_true, predictions))
        print(f'\nRMSLE vs valores reais: {rmsle:.4f}')

    out_path = 'predicoes.csv'
    output.to_csv(out_path, index=False)
    print(f'\nPredições salvas em: {out_path}')
