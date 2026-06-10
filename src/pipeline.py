"""
Definição do pipeline scikit-learn para o modelo de risco de crédito.

Responsabilidades:
- Criar features de razão que capturam relações entre variáveis do imóvel
- Montar o pipeline completo de pré-processamento (imputação, encoding, escala)
- Expor a função build_full_pipeline() como ponto de entrada
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class RatioFeatureTransformer(BaseEstimator, TransformerMixin):
    """Cria features de razão a partir de colunas numéricas do dataset.

    Herdar de BaseEstimator garante get_params()/set_params() automáticos,
    o que habilita GridSearchCV sem código extra. Herdar de TransformerMixin
    fornece fit_transform() gratuitamente.

    Parâmetros
    ----------
    rooms_ix : int
        Índice da coluna AveRooms na matriz de entrada.
    bedrooms_ix : int
        Índice da coluna AveBedrms na matriz de entrada.
    population_ix : int
        Índice da coluna Population na matriz de entrada.
    households_ix : int
        Índice da coluna AveOccup / Households na matriz de entrada.
    add_bedrooms_per_room : bool
        Se True, adiciona a feature bedrooms_per_room (razão quartos/cômodos).
        Útil para testar via GridSearchCV se essa feature agrega valor.
        Padrão: True.
    """

    def __init__(
        self,
        rooms_ix: int = 0,
        bedrooms_ix: int = 1,
        population_ix: int = 2,
        households_ix: int = 3,
        add_bedrooms_per_room: bool = True,
    ):
        self.rooms_ix = rooms_ix
        self.bedrooms_ix = bedrooms_ix
        self.population_ix = population_ix
        self.households_ix = households_ix
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        """Sem ajuste necessário; retorna self para compatibilidade com o pipeline."""
        return self

    def transform(self, X) -> np.ndarray:
        """Acrescenta features de razão à direita da matriz X original.

        Parâmetros
        ----------
        X : array-like de shape (n_amostras, n_features)
            Matriz de entrada já imputada (sem NaN).

        Retorna
        -------
        np.ndarray de shape (n_amostras, n_features + 2 ou + 3)
            X original concatenado com as novas features de razão.
        """
        rooms = X[:, self.rooms_ix]
        bedrooms = X[:, self.bedrooms_ix]
        population = X[:, self.population_ix]
        households = X[:, self.households_ix]

        # np.where evita divisão por zero substituindo por 0 quando denominador é nulo
        rooms_per_household = np.where(households != 0, rooms / households, 0)
        population_per_household = np.where(households != 0, population / households, 0)

        new_features = [rooms_per_household, population_per_household]

        if self.add_bedrooms_per_room:
            bedrooms_per_room = np.where(rooms != 0, bedrooms / rooms, 0)
            new_features.append(bedrooms_per_room)

        # np.c_ concatena ao longo do eixo das colunas
        return np.c_[X, *new_features]


# Índices das colunas de razão dentro do subconjunto numérico definido em build_full_pipeline.
# Declarados como constantes para facilitar testes unitários do transformer isolado.
_ROOMS_IX = 3       # AveRooms
_BEDROOMS_IX = 4    # AveBedrms
_POPULATION_IX = 5  # Population
_HOUSEHOLDS_IX = 6  # AveOccup


def build_full_pipeline() -> Pipeline:
    """Constrói o pipeline completo de pré-processamento para o California Housing.

    O dataset do sklearn (fetch_california_housing) contém apenas colunas numéricas,
    portanto não há etapa de encoding categórico.

    Arquitetura
    -----------
    Pipeline
    ├── SimpleImputer(strategy="median")   # robusto a outliers
    ├── RatioFeatureTransformer            # engenharia de features
    └── StandardScaler                     # necessário para modelos lineares e KNN

    Retorna
    -------
    Pipeline
        Pipeline pronto para receber um DataFrame ou array e retornar uma matriz numpy.
    """
    return Pipeline([
        # Mediana é preferível à média porque é resistente a outliers extremos
        ("imputer", SimpleImputer(strategy="median")),
        ("ratio_features", RatioFeatureTransformer(
            rooms_ix=_ROOMS_IX,
            bedrooms_ix=_BEDROOMS_IX,
            population_ix=_POPULATION_IX,
            households_ix=_HOUSEHOLDS_IX,
            add_bedrooms_per_room=True,
        )),
        ("scaler", StandardScaler()),
    ])
