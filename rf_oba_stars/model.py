"""Construção, validação cruzada e treino do classificador Random Forest.

O modelo é uma floresta aleatória com 101 árvores, profundidade máxima
25, critério de Gini e ``class_weight='balanced'`` — este último
compensa o forte desbalanceamento entre as classes O, B e A da amostra.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator, cross_val_score

from . import config

PathLike = Union[str, Path]


def build_random_forest(
    params: Optional[Dict[str, Any]] = None,
    **overrides: Any,
) -> RandomForestClassifier:
    """Instancia o classificador Random Forest do projeto.

    :param params: Dicionário de hiperparâmetros. Se ``None``, usa
        :data:`rf_oba_stars.config.RANDOM_FOREST_PARAMS`.
    :type params: dict or None
    :param overrides: Hiperparâmetros individuais que sobrescrevem os
        valores de ``params`` (por exemplo ``max_depth=15``).
    :type overrides: Any
    :returns: Classificador ainda não treinado.
    :rtype: sklearn.ensemble.RandomForestClassifier

    .. rubric:: Exemplo

    .. code-block:: python

        >>> modelo = build_random_forest(n_estimators=200)
        >>> modelo.n_estimators
        200
    """
    final_params = dict(config.RANDOM_FOREST_PARAMS if params is None else params)
    final_params.update(overrides)
    return RandomForestClassifier(**final_params)


def cross_validate_model(
    model: RandomForestClassifier,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    scoring: str = config.CV_SCORING,
    n_jobs: Optional[int] = None,
) -> np.ndarray:
    """Avalia o modelo por validação cruzada, sem alterá-lo.

    O ``cross_val_score`` treina clones internos do estimador; o objeto
    passado em ``model`` continua não treinado após a chamada.

    :param model: Classificador a ser avaliado.
    :type model: sklearn.ensemble.RandomForestClassifier
    :param X: Atributos do conjunto de treino.
    :type X: pandas.DataFrame
    :param y: Classes espectrais do conjunto de treino.
    :type y: pandas.Series
    :param cv: Esquema de validação cruzada, tipicamente o retornado por
        :func:`rf_oba_stars.splitting.build_cv`.
    :type cv: sklearn.model_selection.BaseCrossValidator
    :param scoring: Nome da métrica do scikit-learn a ser usada.
    :type scoring: str
    :param n_jobs: Número de processos paralelos. ``None`` usa um
        processo; ``-1`` usa todos os núcleos disponíveis.
    :type n_jobs: int or None
    :returns: Vetor com a pontuação obtida em cada dobra.
    :rtype: numpy.ndarray
    """
    return cross_val_score(model, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)


def summarize_scores(scores: np.ndarray) -> Dict[str, float]:
    """Resume as pontuações da validação cruzada.

    :param scores: Vetor de pontuações, uma por dobra.
    :type scores: numpy.ndarray
    :returns: Dicionário com as chaves ``'media'``, ``'desvio_padrao'``,
        ``'minimo'`` e ``'maximo'``.
    :rtype: dict[str, float]
    """
    scores = np.asarray(scores, dtype=float)
    return {
        "media": float(scores.mean()),
        "desvio_padrao": float(scores.std()),
        "minimo": float(scores.min()),
        "maximo": float(scores.max()),
    }


def fit_model(
    model: RandomForestClassifier,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> RandomForestClassifier:
    """Treina o classificador no conjunto de treino.

    :param model: Classificador ainda não treinado.
    :type model: sklearn.ensemble.RandomForestClassifier
    :param X_train: Atributos do conjunto de treino.
    :type X_train: pandas.DataFrame
    :param y_train: Classes espectrais do conjunto de treino.
    :type y_train: pandas.Series
    :returns: O mesmo objeto, já ajustado aos dados.
    :rtype: sklearn.ensemble.RandomForestClassifier
    """
    model.fit(X_train, y_train)
    return model


def predict(model: RandomForestClassifier, X: pd.DataFrame) -> np.ndarray:
    """Prediz a classe espectral de um conjunto de estrelas.

    :param model: Classificador já treinado.
    :type model: sklearn.ensemble.RandomForestClassifier
    :param X: Atributos das estrelas a classificar.
    :type X: pandas.DataFrame
    :returns: Vetor de classes previstas (``'O'``, ``'B'`` ou ``'A'``).
    :rtype: numpy.ndarray
    :raises sklearn.exceptions.NotFittedError: Se o modelo ainda não
        tiver sido treinado.
    """
    return model.predict(X)


def predict_proba(
    model: RandomForestClassifier,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Retorna as probabilidades previstas para cada classe.

    :param model: Classificador já treinado.
    :type model: sklearn.ensemble.RandomForestClassifier
    :param X: Atributos das estrelas a classificar.
    :type X: pandas.DataFrame
    :returns: Tabela com uma coluna por classe espectral e uma linha por
        estrela, preservando o índice de ``X``.
    :rtype: pandas.DataFrame
    :raises sklearn.exceptions.NotFittedError: Se o modelo ainda não
        tiver sido treinado.
    """
    probas = model.predict_proba(X)
    return pd.DataFrame(probas, columns=model.classes_, index=X.index)


def feature_importances(
    model: RandomForestClassifier,
    feature_names: Optional[Sequence[str]] = None,
) -> pd.Series:
    """Extrai a importância de cada atributo, em ordem decrescente.

    :param model: Classificador já treinado.
    :type model: sklearn.ensemble.RandomForestClassifier
    :param feature_names: Nomes dos atributos. Se ``None``, usa os nomes
        registrados durante o treino (``feature_names_in_``).
    :type feature_names: collections.abc.Sequence[str] or None
    :returns: Série indexada pelo nome do atributo, ordenada da maior
        para a menor importância.
    :rtype: pandas.Series
    :raises AttributeError: Se o modelo não tiver sido treinado ou não
        expuser os nomes dos atributos.
    """
    if feature_names is None:
        feature_names = getattr(model, "feature_names_in_", None)
    if feature_names is None:
        raise AttributeError(
            "Nomes de atributos indisponíveis; passe 'feature_names' "
            "explicitamente ou treine o modelo com um DataFrame."
        )

    return pd.Series(
        model.feature_importances_,
        index=list(feature_names),
    ).sort_values(ascending=False)


def save_model(model: RandomForestClassifier, path: PathLike) -> Path:
    """Salva o modelo treinado em disco no formato ``joblib``.

    :param model: Classificador treinado.
    :type model: sklearn.ensemble.RandomForestClassifier
    :param path: Caminho do arquivo de destino (``.joblib``).
    :type path: str or pathlib.Path
    :returns: Caminho do arquivo gravado.
    :rtype: pathlib.Path
    """
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: PathLike) -> RandomForestClassifier:
    """Carrega um modelo previamente salvo com :func:`save_model`.

    :param path: Caminho do arquivo ``.joblib``.
    :type path: str or pathlib.Path
    :returns: Classificador treinado.
    :rtype: sklearn.ensemble.RandomForestClassifier
    :raises FileNotFoundError: Se o arquivo não existir.
    """
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    return joblib.load(path)
