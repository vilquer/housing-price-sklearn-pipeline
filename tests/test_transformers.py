"""Testes pytest para RatioFeatureTransformer e build_full_pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.pipeline import RatioFeatureTransformer, build_full_pipeline

RNG = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def X_8():
    """Array 100x8 com valores positivos — simula entrada numérica genérica."""
    return RNG.uniform(low=0.1, high=10.0, size=(100, 8))


@pytest.fixture
def X_households_zero():
    """Array 100x8 com coluna households (índice 3) completamente zerada."""
    X = RNG.uniform(low=0.1, high=10.0, size=(100, 8))
    X[:, 3] = 0.0
    return X


@pytest.fixture
def X_sample():
    """DataFrame com as 8 colunas numéricas do California Housing e 200 amostras sintéticas.

    Não inclui ocean_proximity — build_full_pipeline() opera apenas sobre colunas
    numéricas desde que fetch_california_housing (sklearn) não fornece essa coluna.
    """
    n = 200
    return pd.DataFrame({
        "MedInc":     RNG.uniform(0.5, 15.0, n),
        "HouseAge":   RNG.uniform(1.0, 52.0, n),
        "AveRooms":   RNG.uniform(1.0, 15.0, n),
        "AveBedrms":  RNG.uniform(0.5, 5.0, n),
        "Population": RNG.uniform(3.0, 3000.0, n),
        "AveOccup":   RNG.uniform(1.0, 10.0, n),
        "Latitude":   RNG.uniform(32.0, 42.0, n),
        "Longitude":  RNG.uniform(-124.0, -114.0, n),
    })


# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

def test_output_shape(X_8):
    """Array 100x8 deve produzir 100x11 com add_bedrooms_per_room=True."""
    t = RatioFeatureTransformer(add_bedrooms_per_room=True)
    saida = t.fit_transform(X_8)
    assert saida.shape == (100, 11)


def test_output_shape_no_bedrooms(X_8):
    """Array 100x8 deve produzir 100x10 com add_bedrooms_per_room=False."""
    t = RatioFeatureTransformer(add_bedrooms_per_room=False)
    saida = t.fit_transform(X_8)
    assert saida.shape == (100, 10)


def test_no_inf_values(X_households_zero):
    """Divisão por zero em households não deve gerar inf na saída."""
    t = RatioFeatureTransformer(add_bedrooms_per_room=True)
    saida = t.fit_transform(X_households_zero)
    assert not np.isinf(saida).any()
    assert not np.isnan(saida).any()


def test_fit_returns_self(X_8):
    """fit() deve retornar a própria instância do transformer."""
    t = RatioFeatureTransformer()
    resultado = t.fit(X_8)
    assert resultado is t


def test_pipeline_integration(X_sample):
    """build_full_pipeline().fit_transform() deve rodar sem erros em dados sintéticos."""
    pipeline = build_full_pipeline()
    saida = pipeline.fit_transform(X_sample)
    assert isinstance(saida, np.ndarray)
    assert saida.shape[0] == len(X_sample)
    assert not np.isnan(saida).any()
    assert not np.isinf(saida).any()
