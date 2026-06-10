"""
Ponto de entrada para treino e comparação de modelos do pipeline de risco de crédito.

Responsabilidades:
- Definir o catálogo de modelos candidatos
- Treinar cada modelo com validação cruzada
- Comparar e salvar os resultados
"""

import argparse
import time

import pandas as pd
from scipy.stats import uniform
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from data.download import load_data
from src.pipeline import build_full_pipeline

MODELS = {
    "linear":     LinearRegression(),
    "ridge":      Ridge(alpha=1.0),
    "lasso":      Lasso(alpha=0.1),
    "elasticnet": ElasticNet(alpha=0.1, l1_ratio=0.5),
}


def train_and_evaluate(model_name: str, cv_folds: int = 5) -> dict:
    """Treina um modelo com validação cruzada e retorna suas métricas.

    Parâmetros
    ----------
    model_name : str
        Chave do modelo em MODELS.
    cv_folds : int
        Número de folds para cross_val_score. Padrão: 5.

    Retorna
    -------
    dict com: model_name, rmse_mean, rmse_std, fit_time (segundos)
    """
    df = load_data(save=True)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    # Estratificação por faixas de target evita que splits aleatórios
    # concentrem imóveis baratos ou caros em apenas um conjunto
    faixas = pd.cut(y, bins=5, labels=False)
    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=faixas
    )

    full_pipeline = Pipeline([
        ("preprocessor", build_full_pipeline()),
        ("model", MODELS[model_name]),
    ])

    inicio = time.perf_counter()
    scores = cross_val_score(
        full_pipeline, X_train, y_train,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
    )
    fit_time = time.perf_counter() - inicio

    # cross_val_score retorna valores negativos; invertemos para RMSE positivo
    rmse_scores = -scores

    return {
        "model_name": model_name,
        "rmse_mean":  round(rmse_scores.mean(), 4),
        "rmse_std":   round(rmse_scores.std(), 4),
        "fit_time":   round(fit_time, 2),
    }


def compare_all_models(cv_folds: int = 5) -> None:
    """Avalia todos os modelos, imprime tabela comparativa e salva CSV.

    Parâmetros
    ----------
    cv_folds : int
        Número de folds repassado a train_and_evaluate. Padrão: 5.
    """
    resultados = []
    for name in MODELS:
        print(f"Avaliando {name}...")
        resultados.append(train_and_evaluate(name, cv_folds=cv_folds))

    df_resultados = pd.DataFrame(resultados).sort_values("rmse_mean")

    print("\n" + "=" * 52)
    print(f"{'Modelo':<14} {'RMSE médio':>12} {'RMSE std':>10} {'Tempo (s)':>10}")
    print("-" * 52)
    for row in df_resultados.itertuples(index=False):
        print(f"{row.model_name:<14} {row.rmse_mean:>12.4f} {row.rmse_std:>10.4f} {row.fit_time:>10.2f}")
    print("=" * 52)

    caminho = "outputs/model_comparison.csv"
    df_resultados.to_csv(caminho, index=False)
    print(f"\nResultados salvos em {caminho}")


def fine_tune_best_model(cv_folds: int = 5) -> None:
    """Identifica o melhor modelo e o otimiza com RandomizedSearchCV.

    Fluxo
    -----
    1. Roda train_and_evaluate para todos os modelos e escolhe o de menor RMSE
    2. Se for Ridge ou ElasticNet, executa RandomizedSearchCV com distribuições scipy
    3. Imprime best_params e best_cv_rmse
    4. Reavalia o melhor estimador no test set via evaluate_on_test

    Parâmetros
    ----------
    cv_folds : int
        Número de folds para o RandomizedSearchCV. Padrão: 5.
    """
    # Import adiado para evitar import circular (evaluate.py importa MODELS daqui)
    from src.evaluate import evaluate_on_test

    print("Comparando todos os modelos para identificar o melhor...\n")
    resultados = [train_and_evaluate(name, cv_folds=cv_folds) for name in MODELS]
    df_res = pd.DataFrame(resultados).sort_values("rmse_mean")
    best_name = df_res.iloc[0]["model_name"]
    print(f"\nMelhor modelo: {best_name}  (RMSE médio CV: {df_res.iloc[0]['rmse_mean']:.4f})")

    if best_name not in ("ridge", "elasticnet"):
        print(f"Fine-tuning não implementado para '{best_name}' — sem hiperparâmetros contínuos a ajustar.")
        return

    df = load_data(save=True)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    faixas = pd.cut(y, bins=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=faixas
    )

    # Distribuições e n_iter variam conforme o modelo vencedor
    if best_name == "ridge":
        model = Ridge()
        param_dist = {"model__alpha": uniform(loc=0.01, scale=10)}
        n_iter = 20
    else:
        model = ElasticNet()
        param_dist = {
            "model__alpha":    uniform(loc=0.01, scale=5),
            "model__l1_ratio": uniform(loc=0.1,  scale=0.9),
        }
        n_iter = 30

    full_pipeline = Pipeline([
        ("preprocessor", build_full_pipeline()),
        ("model", model),
    ])

    print(f"\nRodando RandomizedSearchCV ({n_iter} iterações, {cv_folds} folds)...")
    search = RandomizedSearchCV(
        full_pipeline,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv_folds,
        random_state=42,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)

    best_cv_rmse = -search.best_score_
    print(f"\nbest_params : {search.best_params_}")
    print(f"best_cv_rmse: {best_cv_rmse:.4f}")

    print("\nAvaliação no conjunto de teste com melhor estimador:")
    evaluate_on_test(
        search.best_estimator_,
        X_train, y_train, X_test, y_test,
        model_name=f"{best_name} (tuned)",
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Treina e compara modelos de regressão.")
    parser.add_argument(
        "--model",
        choices=[*MODELS.keys(), "all"],
        default="all",
        help="Modelo a treinar (padrão: all)",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Número de folds para validação cruzada (padrão: 5)",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Identifica o melhor modelo e roda RandomizedSearchCV para otimizá-lo",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.fine_tune:
        fine_tune_best_model(cv_folds=args.cv)
    elif args.model == "all":
        compare_all_models(cv_folds=args.cv)
    else:
        resultado = train_and_evaluate(args.model, cv_folds=args.cv)
        print(f"\n{resultado['model_name']}: RMSE {resultado['rmse_mean']:.4f} "
              f"± {resultado['rmse_std']:.4f}  ({resultado['fit_time']:.2f}s)")
