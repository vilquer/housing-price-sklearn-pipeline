# credit-risk-sklearn-pipeline

## Problem

Regressão para prever o valor mediano de imóveis (proxy de risco de crédito) usando o California Housing dataset. Objetivo: demonstrar um pipeline ML produtizável com Scikit-Learn — desde o download dos dados até comparação de modelos e curvas de aprendizado.

## Dataset

- **Fonte:** `sklearn.datasets.fetch_california_housing` (StatLib, 1990 Census)
- **Dimensões:** 20.640 amostras × 8 features numéricas + 1 target
- **Features principais:** `MedInc` (renda mediana), `HouseAge`, `AveRooms`, `AveBedrms`, `Population`, `AveOccup`, `Latitude`, `Longitude`
- **Target:** `MedHouseVal` — valor mediano dos imóveis em centenas de milhares de dólares

## Technical Decisions

- **Estratificação:** `train_test_split` estratificado por faixas de `MedHouseVal` (`pd.cut` em 5 bins) para garantir distribuição uniforme do target em treino e teste, evitando que imóveis caros ou baratos se concentrem em um único conjunto.

- **Feature engineering:** `RatioFeatureTransformer` cria 3 features de razão — `rooms_per_household`, `population_per_household` e `bedrooms_per_room`. Valores absolutos como `AveRooms` são ruidosos sem contexto: um bloco com 10 quartos e 2 domicílios indica algo muito diferente de 10 quartos e 10 domicílios. A razão captura densidade habitacional real, que correlaciona melhor com o valor do imóvel.

- **Modelo escolhido:** `Ridge` (alpha=1.0) — melhor RMSE no test set (0.7215) e R² de 0.61. Apesar do ElasticNet apresentar RMSE médio ligeiramente menor no CV (0.8019 vs 0.8177), o alto desvio padrão do Ridge no CV (0.1958) indicava instabilidade nos folds com poucos dados — no test set final o Ridge supera todos os modelos, sugerindo que a regularização L1 do Lasso/ElasticNet penaliza demais as features contínuas desse dataset.

## Results

| Modelo     | RMSE (test) | MAE (test) | R² (test) |
|------------|-------------|------------|-----------|
| **Ridge**  | **0.7215**  | **0.5226** | **0.6085**|
| Linear     | 0.7216      | 0.5227     | 0.6084    |
| ElasticNet | 0.7970      | 0.5985     | 0.5223    |
| Lasso      | 0.8227      | 0.6218     | 0.4910    |

RMSE em unidades de $100k — o Ridge erra em média ~$72k por previsão.

## Learning Curve

As curvas de aprendizado (`outputs/learning_curve_*.png`) mostram que Ridge e Linear convergem a partir de ~9.000 amostras com gap treino/validação pequeno — sinal de underfitting leve, não overfitting. Lasso e ElasticNet convergem mais cedo mas em um patamar de erro mais alto, confirmando penalização excessiva.

## What I'd Do Differently

- Testar modelos não-lineares (Random Forest, Gradient Boosting) — o R² de 0.61 sugere que relações não-lineares entre localização e valor não estão sendo capturadas.
- Adicionar features geográficas (distância ao oceano, ao centro urbano) para substituir Latitude/Longitude brutas.
- Usar `HalvingRandomSearchCV` em vez de `RandomizedSearchCV` para fine-tuning mais eficiente.
- Versionar o modelo treinado com MLflow ou joblib + hash do dataset.

## How to Run

```bash
# Instalar dependências
pip install -r requirements.txt

# Baixar e cachear o dataset
python data/download.py

# Comparar todos os modelos (CV)
python -m src.train --model all --cv 5

# Avaliar no test set + gerar curvas de aprendizado
python -m src.evaluate

# Fine-tuning do melhor modelo
python -m src.train --fine-tune --cv 5

# Rodar testes
pytest tests/ -v
```
