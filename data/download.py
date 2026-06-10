"""Baixa e prepara o dataset para o pipeline de risco de crédito."""

import os
import pandas as pd
from sklearn.datasets import fetch_california_housing


def load_data(save: bool = True, path: str = "data/housing.csv") -> pd.DataFrame:
    """Carrega o dataset California Housing em um DataFrame pandas.

    Parâmetros
    ----------
    save : bool
        Se True, salva o dataset em `path` como CSV no primeiro download.
        Chamadas subsequentes leem do arquivo em cache sem baixar novamente.
        Padrão: True.
    path : str
        Caminho do arquivo CSV para salvar ou ler.
        Padrão: "data/housing.csv".

    Retorna
    -------
    pd.DataFrame
        DataFrame com as colunas de features mais a coluna 'target'
        (valor médio dos imóveis em centenas de milhares de dólares).
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"Carregado do cache: {path}")
    else:
        dataset = fetch_california_housing(as_frame=True)
        df = dataset.frame
        if save:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            df.to_csv(path, index=False)
            print(f"Salvo em: {path}")

    print(f"Shape: {df.shape}")
    print(df.head())
    return df


if __name__ == "__main__":
    load_data()
