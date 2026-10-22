"""Divisão estratificada dos dados em treino, validação e teste.

A divisão é feita em dois estágios, como no notebook original:

1. separa-se 15% do conjunto completo para teste;
2. o restante (85%) é dividido novamente, reservando 17,65% para
   validação — o que corresponde a aproximadamente 15% do total.

Ambas as divisões são estratificadas pela classe espectral, preservando
a proporção entre estrelas O, B e A em todos os subconjuntos.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from . import config


@dataclass
class DataSplits:
    """Contêiner com os seis subconjuntos gerados pela divisão dos dados.

    :ivar X_train: Atributos do conjunto de treino.
    :vartype X_train: pandas.DataFrame
    :ivar X_val: Atributos do conjunto de validação.
    :vartype X_val: pandas.DataFrame
    :ivar X_test: Atributos do conjunto de teste.
    :vartype X_test: pandas.DataFrame
    :ivar y_train: Classes espectrais do conjunto de treino.
    :vartype y_train: pandas.Series
    :ivar y_val: Classes espectrais do conjunto de validação.
    :vartype y_val: pandas.Series
    :ivar y_test: Classes espectrais do conjunto de teste.
    :vartype y_test: pandas.Series
    """

    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    @property
    def sizes(self) -> Dict[str, int]:
        """Número absoluto de amostras em cada subconjunto.

        :returns: Dicionário com as chaves ``'treino'``, ``'validacao'``
            e ``'teste'``.
        :rtype: dict[str, int]
        """
        return {
            "treino": len(self.X_train),
            "validacao": len(self.X_val),
            "teste": len(self.X_test),
        }

    @property
    def proportions(self) -> Dict[str, float]:
        """Percentual de amostras em cada subconjunto.

        :returns: Dicionário com as chaves ``'treino'``, ``'validacao'``
            e ``'teste'``, em porcentagem do total.
        :rtype: dict[str, float]
        """
        total = sum(self.sizes.values())
        return {nome: 100.0 * n / total for nome, n in self.sizes.items()}


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.TEST_SIZE,
    val_size_of_temp: float = config.VAL_SIZE_OF_TEMP,
    random_state: int = config.RANDOM_STATE_SPLIT,
) -> DataSplits:
    """Divide os dados em treino, validação e teste de forma estratificada.

    :param X: Matriz de atributos.
    :type X: pandas.DataFrame
    :param y: Vetor de classes espectrais, usado também para a
        estratificação.
    :type y: pandas.Series
    :param test_size: Fração do conjunto completo reservada para teste.
    :type test_size: float
    :param val_size_of_temp: Fração do conjunto temporário
        (treino + validação) reservada para validação.
    :type val_size_of_temp: float
    :param random_state: Semente aleatória aplicada às duas divisões.
    :type random_state: int
    :returns: Objeto com os seis subconjuntos resultantes.
    :rtype: DataSplits
    :raises ValueError: Se ``X`` e ``y`` tiverem tamanhos diferentes ou se
        alguma classe tiver amostras insuficientes para a estratificação.

    .. rubric:: Exemplo

    .. code-block:: python

        >>> splits = train_val_test_split(X, y)
        >>> splits.sizes
        {'treino': 5763, 'validacao': 1012, 'teste': 1196}
    """
    if len(X) != len(y):
        raise ValueError(
            f"X e y têm tamanhos incompatíveis: {len(X)} e {len(y)}."
        )

    X_temp, X_test, y_temp, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_of_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    return DataSplits(
        X_train=X_train,
        X_val=X_val,
        X_test=X_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def build_cv(
    n_splits: int = config.N_SPLITS,
    random_state: int = config.RANDOM_STATE_SPLIT,
    shuffle: bool = True,
) -> StratifiedKFold:
    """Constrói o esquema de validação cruzada estratificada.

    :param n_splits: Número de dobras.
    :type n_splits: int
    :param random_state: Semente aleatória do embaralhamento.
    :type random_state: int
    :param shuffle: Se ``True``, embaralha as amostras antes de dividir.
    :type shuffle: bool
    :returns: Objeto de validação cruzada pronto para uso em
        :func:`sklearn.model_selection.cross_val_score`.
    :rtype: sklearn.model_selection.StratifiedKFold
    """
    return StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state if shuffle else None,
    )
