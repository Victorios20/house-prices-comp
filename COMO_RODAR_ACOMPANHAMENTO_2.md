# Como rodar para o Acompanhamento 2

Abra o PowerShell na pasta do projeto:

```powershell
cd "C:\dev\codigos UNIFOR\Ciências de Dados - Tulio\AV2 - Ciencias de dados\house-prices-comp"
```

## 1. Treinar os modelos e gerar o modelo final

```powershell
.\run_python.ps1 train_model.py
```

Esse comando gera:

- `modelo.pkl`
- `data/resultados_modelos.csv`
- `data/resumo_acompanhamento_2.txt`

## 2. Testar o pipeline no teste público

```powershell
.\run_python.ps1 pipeline.py data\teste_publico.csv
```

Esse comando deve imprimir a quantidade de predições, a faixa de preços e gerar:

- `predicoes.csv`

## 3. 

- O arquivo `pipeline.py` rodando com `data\teste_publico.csv`.
- O arquivo `modelo.pkl` existente na raiz do projeto.
- A tabela `data/resultados_modelos.csv`.
- O resumo `data/resumo_acompanhamento_2.txt`.

Resultados atuais:

| Modelo | RMSLE | MAE | R2 |
|---|---:|---:|---:|
| Ridge | 0.11295 | $13,657 | 0.9333 |
| XGBoost | 0.11603 | $14,204 | 0.9109 |
| Regressao Linear | 0.13181 | $15,334 | 0.9008 |
| Random Forest | 0.13696 | $16,958 | 0.8572 |
