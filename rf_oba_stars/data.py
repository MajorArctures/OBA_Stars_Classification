"""Carregamento, inspeção e preparação da tabela de atributos.

O módulo trabalha sobre o arquivo ``oba_stars_features.csv``, que contém
um registro por estrela com os índices espectrais medidos a partir dos
espectros XP do Gaia (``he_ii``, ``he_i``, ``hbeta_deep``, ``C2``,
``C3``), a cor fotométrica ``BP-RP``, o identificador ``source_id`` e a
classe espectral de referência ``spectral_type``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import pandas as pd

from . import config

PathLike = Union[str, Path]


def load_features(
    path: PathLike,
    target_column: str = config.TARGET_COLUMN,
    dropna: bool = True,
) -> pd.DataFrame:
    """Carrega a tabela de atributos das estrelas OBA.

    :param path: Caminho para o arquivo CSV de atributos
        (por exemplo ``'oba_stars_features.csv'``).
    :type path: str or pathlib.Path
    :param target_column: Nome da coluna que contém a classe espectral.
    :type target_column: str
    :param dropna: Se ``True``, remove as linhas que possuem qualquer
        valor ausente. A Random Forest do scikit-learn não aceita
        ``NaN`` por padrão.
    :type dropna: bool
    :returns: Tabela com os atributos e a classe espectral.
    :rtype: pandas.DataFrame
    :raises FileNotFoundError: Se o arquivo informado não existir.
    :raises KeyError: Se a coluna alvo não estiver presente no arquivo.

    .. rubric:: Exemplo

    .. code-block:: python

        >>> df = load_features("oba_stars_features.csv")
        >>> df.shape
        (7971, 8)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de atributos não encontrado: {path}")

    df = pd.read_csv(path)

    if target_column not in df.columns:
        raise KeyError(
            f"Coluna alvo '{target_column}' ausente. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    if dropna:
        df = df.dropna().reset_index(drop=True)

    return df


def class_distribution(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    normalize: bool = False,
    class_order: Optional[Sequence[str]] = config.CLASS_ORDER,
) -> pd.Series:
    """Conta quantas estrelas existem em cada classe espectral.

    Substitui as contagens manuais ``len(df[df['spectral_type'] == 'O'])``
    do notebook original.

    :param df: Tabela contendo a coluna alvo.
    :type df: pandas.DataFrame
    :param target_column: Nome da coluna que contém a classe espectral.
    :type target_column: str
    :param normalize: Se ``True``, retorna frações em vez de contagens
        absolutas.
    :type normalize: bool
    :param class_order: Ordem em que as classes devem aparecer no
        resultado. Classes ausentes recebem ``0``. Use ``None`` para
        manter a ordem de frequência devolvida pelo pandas.
    :type class_order: collections.abc.Sequence[str] or None
    :returns: Série indexada pela classe espectral com as contagens
        (ou frações) correspondentes.
    :rtype: pandas.Series
    :raises KeyError: Se a coluna alvo não existir na tabela.
    """
    if target_column not in df.columns:
        raise KeyError(f"Coluna alvo '{target_column}' ausente na tabela.")

    counts = df[target_column].value_counts(normalize=normalize)

    if class_order is not None:
        counts = counts.reindex(list(class_order), fill_value=0)

    counts.index.name = target_column
    return counts


def describe_dataset(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
) -> Dict[str, object]:
    """Resume o conjunto de dados em um dicionário.

    :param df: Tabela de atributos já carregada.
    :type df: pandas.DataFrame
    :param target_column: Nome da coluna que contém a classe espectral.
    :type target_column: str
    :returns: Dicionário com as chaves ``'n_amostras'``, ``'n_colunas'``,
        ``'colunas'``, ``'distribuicao'`` (``dict`` classe → contagem) e
        ``'valores_ausentes'`` (``dict`` coluna → contagem).
    :rtype: dict
    """
    return {
        "n_amostras": int(len(df)),
        "n_colunas": int(df.shape[1]),
        "colunas": list(df.columns),
        "distribuicao": class_distribution(df, target_column).to_dict(),
        "valores_ausentes": df.isna().sum().to_dict(),
    }


def split_features_target(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    id_columns: Iterable[str] = config.ID_COLUMNS,
    drop_id: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Separa a matriz de atributos ``X`` do vetor de classes ``y``.

    :param df: Tabela completa de atributos.
    :type df: pandas.DataFrame
    :param target_column: Nome da coluna que contém a classe espectral.
    :type target_column: str
    :param id_columns: Colunas identificadoras a serem descartadas quando
        ``drop_id`` for ``True``. Colunas inexistentes são ignoradas.
    :type id_columns: collections.abc.Iterable[str]
    :param drop_id: Se ``True`` (padrão), remove ``source_id`` da matriz
        de atributos. Identificadores não carregam informação física e
        podem induzir o modelo a memorizar amostras. Use ``False``
        apenas para reproduzir literalmente o notebook original.
    :type drop_id: bool
    :returns: Par ``(X, y)`` com a matriz de atributos e a série de
        classes espectrais.
    :rtype: tuple[pandas.DataFrame, pandas.Series]
    :raises KeyError: Se a coluna alvo não existir na tabela.
    """
    if target_column not in df.columns:
        raise KeyError(f"Coluna alvo '{target_column}' ausente na tabela.")

    to_drop = [target_column]
    if drop_id:
        to_drop += [col for col in id_columns if col in df.columns]

    X = df.drop(columns=to_drop)
    y = df[target_column]
    return X, y
