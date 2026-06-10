"""
Avaliação e visualização de modelos do pipeline de risco de crédito.

Responsabilidades:
- Gerar curvas de aprendizado por modelo
- Calcular métricas finais no conjunto de teste
- Exportar tabela consolidada de resultados
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import learning_curve, train_test_split
from sklearn.pipeline import Pipeline

from data.download import load_data
from src.pipeline import build_full_pipeline
from src.train import MODELS

plt.style.use("seaborn-v0_8")


def plot_learning_curve(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    model_name: str,
    save: bool = True,
):
    """Plota a curva de aprendizado de um modelo com intervalo de confiança.

    Parâmetros
    ----------
    model : estimador sklearn
        Pipeline completo (pré-processamento + modelo) já instanciado.
    X : pd.DataFrame
        Features de treino.
    y : pd.Series
        Target de treino.
    model_name : str
        Nome usado no título e no nome do arquivo salvo.
    save : bool
        Se True, salva a figura em outputs/learning_curve_{model_name}.png.

    Retorna
    -------
    fig, ax : matplotlib Figure e Axes
    """
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y,
        train_sizes=np.linspace(0.1, 1.0, 10),
        cv=5,
        scoring="neg_root_mean_squared_error",
        n_jobs=-1,
    )

    # learning_curve retorna valores negativos; invertemos para RMSE positivo
    train_rmse = -train_scores
    val_rmse   = -val_scores

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_sizes, train_rmse.mean(axis=1), color="steelblue",
            label="Treino", linewidth=2)
    ax.fill_between(train_sizes,
                    train_rmse.mean(axis=1) - train_rmse.std(axis=1),
                    train_rmse.mean(axis=1) + train_rmse.std(axis=1),
                    alpha=0.2, color="steelblue")

    ax.plot(train_sizes, val_rmse.mean(axis=1), color="darkorange",
            label="Validação", linewidth=2)
    ax.fill_between(train_sizes,
                    val_rmse.mean(axis=1) - val_rmse.std(axis=1),
                    val_rmse.mean(axis=1) + val_rmse.std(axis=1),
                    alpha=0.2, color="darkorange")

    ax.set_title(f"Curva de Aprendizado — {model_name}", fontsize=14)
    ax.set_xlabel("Training Set Size", fontsize=12)
    ax.set_ylabel("RMSE", fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True)
    fig.tight_layout()

    if save:
        caminho = f"outputs/learning_curve_{model_name}.png"
        fig.savefig(caminho, dpi=150)
        print(f"  Curva salva em {caminho}")

    return fig, ax


def evaluate_on_test(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
) -> dict:
    """Treina o pipeline em X_train e avalia RMSE, MAE e R² em X_test.

    Parâmetros
    ----------
    pipeline : Pipeline
        Pipeline completo (pré-processamento + modelo), ainda não treinado.
    X_train, y_train : dados de treino.
    X_test, y_test   : dados de teste.
    model_name : str  : nome para exibição.

    Retorna
    -------
    dict com: model_name, rmse, mae, r2
    """
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    rmse = root_mean_squared_error(y_test, y_pred)
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)

    print(f"\n  {model_name}")
    print(f"    RMSE : {rmse:.4f}")
    print(f"    MAE  : {mae:.4f}")
    print(f"    R²   : {r2:.4f}")

    return {"model_name": model_name, "rmse": round(rmse, 4),
            "mae": round(mae, 4), "r2": round(r2, 4)}


def run_full_evaluation() -> None:
    """Avalia todos os modelos no conjunto de teste e salva os resultados.

    Fluxo
    -----
    1. Carrega dados e faz train/test split estratificado
    2. Para cada modelo: plota curva de aprendizado e avalia no test set
    3. Salva tabela consolidada em outputs/final_results.csv
    """
    df = load_data(save=True)
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    faixas = pd.cut(y, bins=5, labels=False)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=faixas
    )

    resultados = []
    for name, model in MODELS.items():
        print(f"\n[{name}]")

        full_pipeline = Pipeline([
            ("preprocessor", build_full_pipeline()),
            ("model", model),
        ])

        print("  Gerando curva de aprendizado...")
        plot_learning_curve(full_pipeline, X_train, y_train, model_name=name)

        # Reinstancia o pipeline para treinar do zero na avaliação final
        full_pipeline = Pipeline([
            ("preprocessor", build_full_pipeline()),
            ("model", model),
        ])
        metricas = evaluate_on_test(
            full_pipeline, X_train, y_train, X_test, y_test, model_name=name
        )
        resultados.append(metricas)

    df_final = pd.DataFrame(resultados).sort_values("rmse")

    print("\n" + "=" * 54)
    print(f"{'Modelo':<14} {'RMSE':>8} {'MAE':>8} {'R²':>8}")
    print("-" * 54)
    for row in df_final.itertuples(index=False):
        print(f"{row.model_name:<14} {row.rmse:>8.4f} {row.mae:>8.4f} {row.r2:>8.4f}")
    print("=" * 54)

    caminho = "outputs/final_results.csv"
    df_final.to_csv(caminho, index=False)
    print(f"\nResultados finais salvos em {caminho}")


if __name__ == "__main__":
    run_full_evaluation()
