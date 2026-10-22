"""Pré-processamento da tabela de atributos.

Cobre a remoção de outliers pela regra de Tukey e o escalonamento das
colunas numéricas.

.. warning::
   Tanto a remoção de outliers quanto o escalonamento devem ser
   *ajustados apenas no conjunto de treino* e depois aplicados aos
   demais. Ajustá-los na tabela inteira, antes da divisão, faz
   informação do teste vazar para o treino — foi o que ocorreu no
   notebook original, onde o ``MinMaxScaler`` era ajustado sobre todas
   as linhas. As funções deste módulo separam explicitamente as etapas
   de ajuste e de aplicação para evitar isso.

.. note::
   Florestas aleatórias são invariantes a transformações monotônicas de
   cada atributo, de modo que o escalonamento não altera o resultado do
   Random Forest. Ele continua sendo útil para comparações com modelos
   sensíveis à escala (SVM, k-NN, PCA) e para a visualização conjunta
   dos atributos.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from . import config


def iqr_bounds(
    series: pd.Series,
    factor: float = config.IQR_FACTOR,
) -> Tuple[float, float]:
    """Calcula os limites inferior e superior da regra de Tukey.

    :param series: Coluna numérica a analisar.
    :type series: pandas.Series
    :param factor: Multiplicador do intervalo interquartil. O valor
        clássico ``1.5`` marca outliers moderados; ``3.0`` marca
        extremos.
    :type factor: float
    :returns: Par ``(limite_inferior, limite_superior)``.
    :rtype: tuple[float, float]
    """
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr


def fit_outlier_bounds(
    df: pd.DataFrame,
    columns: Iterable[str] = config.OUTLIER_COLUMNS,
    factor: float = config.IQR_FACTOR,
) -> Dict[str, Tuple[float, float]]:
    """Determina os limites de outlier de cada coluna, sem remover nada.

    :param df: Tabela de referência — em geral, apenas o conjunto de
        treino.
    :type df: pandas.DataFrame
    :param columns: Colunas sobre as quais calcular os limites.
    :type columns: collections.abc.Iterable[str]
    :param factor: Multiplicador do intervalo interquartil.
    :type factor: float
    :returns: Dicionário ``coluna -> (limite_inferior, limite_superior)``.
    :rtype: dict[str, tuple[float, float]]
    :raises KeyError: Se alguma coluna não existir na tabela.
    """
    faltando = [coluna for coluna in columns if coluna not in df.columns]
    if faltando:
        raise KeyError(f"Colunas ausentes na tabela: {faltando}")

    return {coluna: iqr_bounds(df[coluna], factor) for coluna in columns}


def apply_outlier_bounds(
    df: pd.DataFrame,
    bounds: Dict[str, Tuple[float, float]],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Remove as linhas que ficam fora dos limites informados.

    :param df: Tabela a filtrar.
    :type df: pandas.DataFrame
    :param bounds: Dicionário devolvido por :func:`fit_outlier_bounds`.
    :type bounds: dict[str, tuple[float, float]]
    :returns: Par ``(tabela_filtrada, relatorio)``. O relatório traz uma
        linha por coluna com ``limite_inferior``, ``limite_superior``,
        ``n_removidas`` e ``pct_removidas``.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    mask = pd.Series(True, index=df.index)
    registros = []

    for coluna, (inferior, superior) in bounds.items():
        dentro = df[coluna].between(inferior, superior)
        registros.append(
            {
                "coluna": coluna,
                "limite_inferior": inferior,
                "limite_superior": superior,
                "n_removidas": int((~dentro).sum()),
                "pct_removidas": 100.0 * float((~dentro).mean()),
            }
        )
        mask &= dentro

    return df.loc[mask].reset_index(drop=True), pd.DataFrame(registros)


def remove_outliers(
    df: pd.DataFrame,
    columns: Iterable[str] = config.OUTLIER_COLUMNS,
    factor: float = config.IQR_FACTOR,
    target_column: str = config.TARGET_COLUMN,
    by_class: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Ajusta e aplica a remoção de outliers na mesma tabela.

    Conveniência que encadeia :func:`fit_outlier_bounds` e
    :func:`apply_outlier_bounds`. O relatório inclui o impacto por
    classe espectral, acessível em ``relatorio.attrs['por_classe']``.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Colunas sobre as quais aplicar a regra.
    :type columns: collections.abc.Iterable[str]
    :param factor: Multiplicador do intervalo interquartil.
    :type factor: float
    :param target_column: Nome da coluna de classe, usada no relatório
        por classe e, quando ``by_class`` for ``True``, no agrupamento.
    :type target_column: str
    :param by_class: Se ``True``, calcula os limites separadamente
        dentro de cada classe espectral. Sobre a amostra global, C2 e C3
        medem inclinação do contínuo e, portanto, temperatura: as
        estrelas O e B ocupam legitimamente a cauda da distribuição
        dominada por estrelas A, e o corte global as descarta como se
        fossem erros de medida. O corte por classe preserva essa
        estrutura física.
    :type by_class: bool
    :returns: Par ``(tabela_filtrada, relatorio)``.
    :rtype: tuple[pandas.DataFrame, pandas.DataFrame]
    """
    if by_class and target_column in df.columns:
        partes, registros = [], []
        for classe, bloco in df.groupby(target_column, sort=False):
            bounds = fit_outlier_bounds(bloco, columns, factor)
            filtrado, relatorio_classe = apply_outlier_bounds(bloco, bounds)
            relatorio_classe.insert(0, target_column, classe)
            partes.append(filtrado)
            registros.append(relatorio_classe)

        filtrada = pd.concat(partes, ignore_index=True)
        relatorio = pd.concat(registros, ignore_index=True)
    else:
        bounds = fit_outlier_bounds(df, columns, factor)
        filtrada, relatorio = apply_outlier_bounds(df, bounds)

    if target_column in df.columns:
        antes = df[target_column].value_counts()
        depois = filtrada[target_column].value_counts()
        por_classe = pd.DataFrame({"antes": antes, "depois": depois}).fillna(0)
        por_classe["removidas"] = por_classe["antes"] - por_classe["depois"]
        por_classe["pct_removidas"] = (
            100.0 * por_classe["removidas"] / por_classe["antes"]
        )
        relatorio.attrs["por_classe"] = por_classe.astype(
            {"antes": int, "depois": int, "removidas": int}
        )

    return filtrada, relatorio


def fit_scaler(
    df: pd.DataFrame,
    columns: Sequence[str],
    method: str = "minmax",
):
    """Ajusta um escalonador nas colunas indicadas.

    :param df: Tabela de referência — em geral, apenas o conjunto de
        treino.
    :type df: pandas.DataFrame
    :param columns: Colunas numéricas a escalonar.
    :type columns: collections.abc.Sequence[str]
    :param method: ``'minmax'`` mapeia cada coluna para ``[0, 1]``;
        ``'standard'`` centra em zero com desvio padrão unitário.
    :type method: str
    :returns: Escalonador já ajustado.
    :rtype: sklearn.base.TransformerMixin
    :raises ValueError: Se ``method`` não for um dos valores aceitos.
    """
    if method == "minmax":
        scaler = MinMaxScaler()
    elif method == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"method='{method}' inválido; use 'minmax' ou 'standard'.")

    scaler.fit(df[list(columns)])
    return scaler


def apply_scaler(
    df: pd.DataFrame,
    scaler,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Aplica um escalonador já ajustado, devolvendo uma cópia da tabela.

    :param df: Tabela a transformar.
    :type df: pandas.DataFrame
    :param scaler: Escalonador devolvido por :func:`fit_scaler`.
    :type scaler: sklearn.base.TransformerMixin
    :param columns: Colunas a transformar, na mesma ordem usada no
        ajuste.
    :type columns: collections.abc.Sequence[str]
    :returns: Nova tabela com as colunas indicadas transformadas.
    :rtype: pandas.DataFrame
    """
    saida = df.copy()
    saida[list(columns)] = scaler.transform(df[list(columns)])
    return saida


def numeric_feature_columns(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    id_columns: Iterable[str] = config.ID_COLUMNS,
) -> list:
    """Lista as colunas numéricas que servem como atributos.

    Exclui a coluna alvo e os identificadores.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param target_column: Nome da coluna alvo.
    :type target_column: str
    :param id_columns: Nomes das colunas identificadoras.
    :type id_columns: collections.abc.Iterable[str]
    :returns: Lista de nomes de colunas, na ordem em que aparecem.
    :rtype: list[str]
    """
    excluir = {target_column, *id_columns}
    return [
        coluna
        for coluna in df.columns
        if coluna not in excluir and pd.api.types.is_numeric_dtype(df[coluna])
    ]


#: Amplitude mínima que um atributo precisa ter para que as árvores do
#: scikit-learn consigam dividi-lo. O ``Splitter`` do módulo
#: :mod:`sklearn.tree` trata como constante todo atributo cuja amplitude
#: total seja menor ou igual a ``FEATURE_THRESHOLD``.
SKLEARN_FEATURE_THRESHOLD: float = 1e-7


def check_feature_scales(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    threshold: float = SKLEARN_FEATURE_THRESHOLD,
    warn: bool = True,
) -> pd.DataFrame:
    """Detecta atributos pequenos demais para as árvores do scikit-learn.

    O divisor das árvores do scikit-learn considera constante qualquer
    atributo cuja amplitude total não ultrapasse ``1e-7`` e nunca abre um
    nó por ele. Como os fluxos XP do Gaia são da ordem de
    :math:`10^{-13}\\ \\mathrm{W\\,m^{-2}\\,nm^{-1}}`, qualquer atributo
    obtido de diferenças de fluxo bruto — como ``blue_red_diff`` ou
    ``slope`` — cai silenciosamente nessa faixa e é ignorado pelo modelo,
    sem erro nem aviso.

    A saída é uma razão adimensional (``blue_red_ratio``, ``color_mag``)
    ou um escalonamento explícito dos atributos afetados.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Colunas a verificar. Se ``None``, usa todas as
        numéricas exceto alvo e identificadores.
    :type columns: collections.abc.Sequence[str] or None
    :param threshold: Amplitude mínima exigida.
    :type threshold: float
    :param warn: Se ``True``, emite um :class:`UserWarning` listando os
        atributos problemáticos.
    :type warn: bool
    :returns: Tabela com as colunas ``atributo``, ``amplitude`` e
        ``invisivel_para_arvores``.
    :rtype: pandas.DataFrame
    """
    import warnings

    if columns is None:
        columns = numeric_feature_columns(df)

    linhas = []
    for coluna in columns:
        valores = df[coluna].to_numpy(dtype=float)
        amplitude = float(np.nanmax(valores) - np.nanmin(valores))
        linhas.append(
            {
                "atributo": coluna,
                "amplitude": amplitude,
                "invisivel_para_arvores": amplitude <= threshold,
            }
        )

    relatorio = pd.DataFrame(linhas).set_index("atributo")
    problematicos = relatorio.index[relatorio["invisivel_para_arvores"]].tolist()

    if warn and problematicos:
        warnings.warn(
            f"Atributos com amplitude ≤ {threshold:g} serão tratados como "
            f"constantes pelas árvores do scikit-learn e ignorados pelo "
            f"modelo: {problematicos}. Use uma razão adimensional ou "
            f"escalone essas colunas.",
            UserWarning,
            stacklevel=2,
        )

    return relatorio


def clean_feature_table(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    verbose: bool = True,
) -> pd.DataFrame:
    """Remove linhas com valores não finitos e normaliza os tipos.

    :param df: Tabela de atributos recém-construída.
    :type df: pandas.DataFrame
    :param target_column: Nome da coluna alvo.
    :type target_column: str
    :param verbose: Se ``True``, informa quantas linhas foram removidas.
    :type verbose: bool
    :returns: Tabela limpa, com índice reiniciado.
    :rtype: pandas.DataFrame
    """
    n_inicial = len(df)

    numericas = numeric_feature_columns(df, target_column)
    finitas = np.isfinite(df[numericas].to_numpy()).all(axis=1)
    limpa = df.loc[finitas].reset_index(drop=True)

    if "source_id" in limpa.columns:
        limpa["source_id"] = limpa["source_id"].astype(np.int64)
    limpa[target_column] = limpa[target_column].astype(str)

    if verbose and n_inicial != len(limpa):
        print(
            f"Removidas {n_inicial - len(limpa)} linhas com valores "
            f"não finitos ({100 * (n_inicial - len(limpa)) / n_inicial:.2f}%)."
        )

    return limpa


#: Limiar interno do divisor de árvores do scikit-learn. Atributos cuja
#: amplitude total fica abaixo deste valor são tratados como constantes
#: e nunca entram em uma divisão.
SKLEARN_FEATURE_THRESHOLD: float = 1e-7


def check_feature_scale(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    threshold: float = SKLEARN_FEATURE_THRESHOLD,
) -> pd.DataFrame:
    """Detecta atributos pequenos demais para o divisor de árvores.

    O ``DecisionTreeClassifier`` do scikit-learn converte os dados para
    ``float32`` e considera dois valores iguais quando diferem por menos
    de ``1e-7``. Um atributo cuja amplitude inteira caiba abaixo desse
    limiar — como um fluxo absoluto do Gaia, da ordem de
    :math:`10^{-14}` — é visto como constante e recebe importância zero,
    sem qualquer aviso.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Colunas a verificar. Se ``None``, usa todas as
        numéricas exceto identificadores e alvo.
    :type columns: collections.abc.Sequence[str] or None
    :param threshold: Amplitude mínima considerada segura.
    :type threshold: float
    :returns: Tabela com as colunas ``amplitude``, ``menor_diferenca`` e
        ``invisivel_para_arvore``, uma linha por atributo verificado.
    :rtype: pandas.DataFrame
    """
    if columns is None:
        columns = numeric_feature_columns(df)

    linhas = []
    for coluna in columns:
        valores = df[coluna].dropna().to_numpy(dtype=float)
        amplitude = float(valores.max() - valores.min()) if valores.size else 0.0
        distintos = np.unique(valores)
        menor = float(np.min(np.diff(distintos))) if distintos.size > 1 else 0.0
        linhas.append(
            {
                "atributo": coluna,
                "amplitude": amplitude,
                "menor_diferenca": menor,
                "invisivel_para_arvore": amplitude < threshold,
            }
        )

    return pd.DataFrame(linhas).set_index("atributo")
