# House Prices — Competição ML

Previsão de preços de imóveis em Ames, Iowa. Métrica oficial: **RMSLE**.

## Estrutura

```
house-prices-comp/
├── data/
│   ├── treino.csv
│   ├── teste_publico.csv
│   └── data_description.txt
├── notebooks/
│   ├── 01_EDA.ipynb            ← Análise exploratória
│   ├── 02_Preprocessamento.ipynb ← Pipeline de features
│   └── 03_Modelagem.ipynb      ← Treinamento e tuning
├── src/
│   ├── feature_engineering.py  ← create_features()
│   └── col_config.json         ← configuração de colunas (gerado pelo nb02)
├── pipeline.py                 ← script de predição (avaliado)
├── modelo.pkl                  ← modelo treinado (gerado pelo nb03)
└── requirements.txt
```

## Setup

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Executar predições

```bash
python pipeline.py                        # usa data/teste_publico.csv
python pipeline.py caminho/para/dados.csv # arquivo custom
```

## Fluxo dos notebooks

1. **01_EDA** — exploração, faltantes, correlações, outliers, estratégia de limpeza
2. **02_Preprocessamento** — monta `ColumnTransformer`, testa e salva `col_config.json`
3. **03_Modelagem** — compara modelos, tuning XGBoost, salva `modelo.pkl`

Execute sempre na ordem 01 → 02 → 03.
