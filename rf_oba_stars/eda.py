"""Análise exploratória: estatísticas, testes e visualizações.

Reúne as etapas descritivas do projeto — estatísticas por atributo,
verificação de normalidade, correlações, desbalanceamento das classes e
os gráficos de espectros e de distribuições.

Todas as funções de gráfico devolvem os eixos do matplotlib, permitindo
ajustes posteriores pelo chamador, e aceitam ``save_path`` para gravar a
figura.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import config
from .spectra import SpectraSet

PathLike = Union[str, Path]

#: Cores associadas a cada classe espectral nos gráficos.
CLASS_COLORS: Dict[str, str] = {"O": "royalblue", "B": "seagreen", "A": "darkorange"}


def _finish(fig, ax, save_path: Optional[PathLike], show: bool):
    """Finaliza uma figura: ajusta o layout, salva e exibe.

    :param fig: Figura do matplotlib.
    :type fig: matplotlib.figure.Figure
    :param ax: Eixo ou conjunto de eixos a devolver.
    :type ax: matplotlib.axes.Axes
    :param save_path: Caminho para gravar a figura, ou ``None``.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: O mesmo objeto recebido em ``ax``.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig.tight_layout()
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return ax


# ---------------------------------------------------------------------------
# Estatísticas
# ---------------------------------------------------------------------------


def descriptive_statistics(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """Calcula estatísticas descritivas das colunas numéricas.

    Além das medidas usuais, inclui assimetria e curtose, úteis para
    julgar o afastamento da normalidade antes de aplicar testes.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Colunas a resumir. Se ``None``, usa todas as
        numéricas exceto identificadores.
    :type columns: collections.abc.Sequence[str] or None
    :returns: Tabela com uma linha por estatística e uma coluna por
        atributo.
    :rtype: pandas.DataFrame
    """
    if columns is None:
        columns = [
            coluna
            for coluna in df.select_dtypes(include=np.number).columns
            if coluna not in config.ID_COLUMNS
        ]

    return df[list(columns)].agg(
        ["count", "mean", "median", "std", "min", "max", "skew", "kurt"]
    )


def class_balance(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    class_order: Sequence[str] = config.CLASS_ORDER,
) -> pd.DataFrame:
    """Quantifica o desbalanceamento entre as classes espectrais.

    :param df: Tabela contendo a coluna alvo.
    :type df: pandas.DataFrame
    :param target_column: Nome da coluna de classe.
    :type target_column: str
    :param class_order: Ordem em que as classes aparecem no resultado.
    :type class_order: collections.abc.Sequence[str]
    :returns: Tabela com as colunas ``n``, ``percentual`` e
        ``razao_para_maior`` (fração em relação à classe majoritária).
    :rtype: pandas.DataFrame
    """
    contagem = df[target_column].value_counts().reindex(list(class_order), fill_value=0)

    tabela = pd.DataFrame({"n": contagem})
    tabela["percentual"] = 100.0 * tabela["n"] / tabela["n"].sum()
    tabela["razao_para_maior"] = tabela["n"] / tabela["n"].max()
    tabela.index.name = target_column
    return tabela


def normality_tests(
    df: pd.DataFrame,
    columns: Sequence[str],
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Aplica os testes de D'Agostino-Pearson e Anderson-Darling.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Colunas a testar.
    :type columns: collections.abc.Sequence[str]
    :param alpha: Nível de significância usado na conclusão do teste de
        D'Agostino-Pearson.
    :type alpha: float
    :returns: Tabela com uma linha por atributo e as colunas
        ``dagostino_stat``, ``dagostino_p``, ``normal_dagostino``,
        ``anderson_stat`` e ``anderson_critico_5pct``.
    :rtype: pandas.DataFrame

    .. note::
       Com milhares de amostras, testes de normalidade rejeitam a
       hipótese nula diante de desvios minúsculos e sem importância
       prática. Interprete o resultado junto com os histogramas e os
       gráficos Q-Q, não isoladamente.
    """
    from scipy.stats import anderson, normaltest

    linhas = []
    for coluna in columns:
        valores = df[coluna].dropna().to_numpy()
        estatistica, p = normaltest(valores)
        resultado_ad = anderson(valores, dist="norm")
        i_5pct = int(np.argmin(np.abs(resultado_ad.significance_level - 5.0)))

        linhas.append(
            {
                "atributo": coluna,
                "dagostino_stat": float(estatistica),
                "dagostino_p": float(p),
                "normal_dagostino": bool(p > alpha),
                "anderson_stat": float(resultado_ad.statistic),
                "anderson_critico_5pct": float(resultado_ad.critical_values[i_5pct]),
            }
        )

    return pd.DataFrame(linhas).set_index("atributo")


def correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Calcula a matriz de correlação entre os atributos.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Colunas a correlacionar. Se ``None``, usa todas as
        numéricas exceto identificadores.
    :type columns: collections.abc.Sequence[str] or None
    :param method: ``'pearson'`` (linear), ``'spearman'`` (monotônica)
        ou ``'kendall'``.
    :type method: str
    :returns: Matriz de correlação quadrada.
    :rtype: pandas.DataFrame
    """
    if columns is None:
        columns = [
            coluna
            for coluna in df.select_dtypes(include=np.number).columns
            if coluna not in config.ID_COLUMNS
        ]

    return df[list(columns)].corr(method=method)


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------


def plot_spectrum(
    spectra: SpectraSet,
    index: int = 0,
    show_error: bool = True,
    lines: Optional[Iterable[str]] = None,
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Plota o espectro de uma estrela, opcionalmente com as linhas marcadas.

    :param spectra: Conjunto de espectros.
    :type spectra: SpectraSet
    :param index: Posição da estrela dentro do conjunto.
    :type index: int
    :param show_error: Se ``True``, sombreia a faixa de incerteza.
    :type show_error: bool
    :param lines: Nomes de linhas do catálogo a marcar com linhas
        verticais. Se ``None``, nenhuma é marcada.
    :type lines: collections.abc.Iterable[str] or None
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Eixo do matplotlib com o gráfico.
    :rtype: matplotlib.axes.Axes
    :raises IndexError: Se ``index`` estiver fora do conjunto.
    """
    import matplotlib.pyplot as plt

    if not 0 <= index < len(spectra):
        raise IndexError(f"index={index} fora do intervalo [0, {len(spectra)}).")

    flux = spectra.flux[index]
    erro = spectra.flux_error[index]
    meta = spectra.meta.iloc[index]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(spectra.wavelength, flux, color="black", lw=1.2, label="Fluxo")

    if show_error:
        ax.fill_between(
            spectra.wavelength,
            flux - erro,
            flux + erro,
            alpha=0.3,
            color="gray",
            label="Incerteza",
        )

    if lines:
        for nome in lines:
            lam = config.LINE_CATALOG_NM[nome]
            ax.axvline(lam, color="crimson", ls="--", lw=0.8, alpha=0.7)
            ax.text(
                lam, ax.get_ylim()[1], nome, rotation=90,
                va="top", ha="right", fontsize=7, color="crimson",
            )

    ax.set_xlabel("Comprimento de onda (nm)")
    ax.set_ylabel("Fluxo (W m⁻² nm⁻¹)")
    ax.set_title(
        f"Gaia DR3 {meta['source_id']} — classe {meta[config.TARGET_COLUMN]}, "
        f"G = {meta['magnitude']:.2f}"
    )
    ax.legend()
    ax.grid(alpha=0.3)

    return _finish(fig, ax, save_path, show)


def plot_mean_spectra_by_class(
    spectra: SpectraSet,
    flux: Optional[np.ndarray] = None,
    normalize_each: bool = True,
    lines: Optional[Iterable[str]] = None,
    xlim: Optional[Tuple[float, float]] = None,
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Compara os espectros médios de cada classe espectral.

    :param spectra: Conjunto de espectros com os metadados de classe.
    :type spectra: SpectraSet
    :param flux: Matriz de fluxos a promediar. Se ``None``, usa
        ``spectra.flux``. Passe aqui os fluxos normalizados pelo
        contínuo para comparar as linhas em vez do contínuo.
    :type flux: numpy.ndarray or None
    :param normalize_each: Se ``True``, divide cada espectro pela sua
        própria mediana antes de promediar, o que remove diferenças de
        brilho entre estrelas.
    :type normalize_each: bool
    :param lines: Linhas do catálogo a marcar.
    :type lines: collections.abc.Iterable[str] or None
    :param xlim: Faixa de comprimento de onda a exibir, em nanômetros.
    :type xlim: tuple[float, float] or None
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Eixo do matplotlib com o gráfico.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    flux = spectra.flux if flux is None else flux
    classes = spectra.meta[config.TARGET_COLUMN].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.5))

    for classe in config.CLASS_ORDER:
        mask = classes == classe
        if not mask.any():
            continue

        bloco = flux[mask]
        if normalize_each:
            mediana = np.median(bloco, axis=1, keepdims=True)
            bloco = np.divide(
                bloco, mediana, out=np.full_like(bloco, np.nan), where=mediana != 0
            )

        media = np.nanmean(bloco, axis=0)
        ax.plot(
            spectra.wavelength,
            media,
            label=f"{classe} (n = {int(mask.sum())})",
            color=CLASS_COLORS.get(classe),
            lw=1.5,
        )

    if lines:
        for nome in lines:
            ax.axvline(
                config.LINE_CATALOG_NM[nome], color="gray", ls="--", lw=0.8, alpha=0.6
            )

    if xlim:
        ax.set_xlim(*xlim)

    ax.set_xlabel("Comprimento de onda (nm)")
    ax.set_ylabel("Fluxo normalizado")
    ax.set_title("Espectro médio por classe espectral")
    ax.legend(title="Classe")
    ax.grid(alpha=0.3)

    return _finish(fig, ax, save_path, show)


def plot_feature_distributions(
    df: pd.DataFrame,
    columns: Sequence[str],
    target_column: Optional[str] = config.TARGET_COLUMN,
    bins: int = 50,
    ncols: int = 3,
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Plota o histograma de cada atributo, separado por classe.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Atributos a plotar, um painel por atributo.
    :type columns: collections.abc.Sequence[str]
    :param target_column: Coluna de classe usada para separar as
        distribuições. Se ``None``, plota a distribuição global.
    :type target_column: str or None
    :param bins: Número de intervalos de cada histograma.
    :type bins: int
    :param ncols: Número de colunas na grade de painéis.
    :type ncols: int
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Vetor de eixos do matplotlib.
    :rtype: numpy.ndarray
    """
    import matplotlib.pyplot as plt

    colunas = list(columns)
    nrows = int(np.ceil(len(colunas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax, coluna in zip(axes, colunas):
        if target_column is not None:
            for classe in config.CLASS_ORDER:
                valores = df.loc[df[target_column] == classe, coluna].dropna()
                if len(valores):
                    ax.hist(
                        valores,
                        bins=bins,
                        alpha=0.55,
                        label=classe,
                        color=CLASS_COLORS.get(classe),
                        density=True,
                    )
            ax.legend(fontsize=7)
        else:
            ax.hist(df[coluna].dropna(), bins=bins, color="steelblue", alpha=0.8)

        ax.set_xlabel(coluna)
        ax.set_ylabel("Densidade")
        ax.grid(alpha=0.25)

    for ax in axes[len(colunas):]:
        ax.set_visible(False)

    return _finish(fig, axes, save_path, show)


def plot_qq(
    df: pd.DataFrame,
    columns: Sequence[str],
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Plota gráficos Q-Q contra a distribuição normal.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Atributos a avaliar, um painel por atributo.
    :type columns: collections.abc.Sequence[str]
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Vetor de eixos do matplotlib.
    :rtype: numpy.ndarray
    """
    import matplotlib.pyplot as plt
    from scipy.stats import probplot

    colunas = list(columns)
    fig, axes = plt.subplots(1, len(colunas), figsize=(4.2 * len(colunas), 4))
    axes = np.atleast_1d(axes).ravel()

    for ax, coluna in zip(axes, colunas):
        probplot(df[coluna].dropna(), dist="norm", plot=ax)
        ax.set_title(f"Q-Q — {coluna}")
        ax.grid(alpha=0.25)

    return _finish(fig, axes, save_path, show)


def plot_feature_scatter(
    df: pd.DataFrame,
    x: str,
    y: str,
    target_column: str = config.TARGET_COLUMN,
    alpha: float = 0.5,
    size: float = 10.0,
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Dispersão de dois atributos, colorida por classe espectral.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param x: Nome do atributo no eixo horizontal.
    :type x: str
    :param y: Nome do atributo no eixo vertical.
    :type y: str
    :param target_column: Coluna de classe usada para colorir os pontos.
    :type target_column: str
    :param alpha: Transparência dos pontos.
    :type alpha: float
    :param size: Tamanho dos pontos.
    :type size: float
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Eixo do matplotlib com o gráfico.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5.5))

    for classe in config.CLASS_ORDER:
        subconjunto = df[df[target_column] == classe]
        if subconjunto.empty:
            continue
        ax.scatter(
            subconjunto[x],
            subconjunto[y],
            s=size,
            alpha=alpha,
            label=classe,
            color=CLASS_COLORS.get(classe),
            edgecolors="none",
        )

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_title(f"{x} vs. {y} por classe espectral")
    ax.axhline(0, color="k", alpha=0.25, lw=0.8)
    ax.axvline(0, color="k", alpha=0.25, lw=0.8)
    ax.legend(title="Classe")
    ax.grid(alpha=0.3)

    return _finish(fig, ax, save_path, show)


def plot_class_balance(
    df: pd.DataFrame,
    target_column: str = config.TARGET_COLUMN,
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Plota a contagem de estrelas por classe espectral.

    :param df: Tabela contendo a coluna alvo.
    :type df: pandas.DataFrame
    :param target_column: Nome da coluna de classe.
    :type target_column: str
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Eixo do matplotlib com o gráfico.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    tabela = class_balance(df, target_column)

    fig, ax = plt.subplots(figsize=(6, 4))
    barras = ax.bar(
        tabela.index,
        tabela["n"],
        color=[CLASS_COLORS.get(c, "gray") for c in tabela.index],
    )

    for barra, (_, linha) in zip(barras, tabela.iterrows()):
        ax.text(
            barra.get_x() + barra.get_width() / 2,
            barra.get_height(),
            f"{int(linha['n'])}\n{linha['percentual']:.1f}%",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Classe espectral")
    ax.set_ylabel("Número de estrelas")
    ax.set_title("Distribuição das classes")
    ax.margins(y=0.15)
    ax.grid(alpha=0.25, axis="y")

    return _finish(fig, ax, save_path, show)


def plot_snr_distribution(
    snr_mean: np.ndarray,
    threshold: Optional[float] = config.SNR_THRESHOLD,
    bins: int = 40,
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Plota a distribuição da razão sinal-ruído média dos espectros.

    :param snr_mean: Vetor de SNR médio, um valor por estrela.
    :type snr_mean: numpy.ndarray
    :param threshold: Limiar de corte a marcar com uma linha vertical.
        Use ``None`` para omitir.
    :type threshold: float or None
    :param bins: Número de intervalos do histograma.
    :type bins: int
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Eixo do matplotlib com o gráfico.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(snr_mean, bins=bins, color="steelblue", alpha=0.85, edgecolor="black")

    if threshold is not None:
        aprovadas = int(np.sum(snr_mean >= threshold))
        ax.axvline(threshold, color="crimson", ls="--", lw=1.5)
        ax.text(
            threshold,
            ax.get_ylim()[1] * 0.95,
            f" corte = {threshold:g}\n {aprovadas}/{len(snr_mean)} aprovadas",
            color="crimson",
            va="top",
            fontsize=9,
        )

    ax.set_xlabel("SNR médio")
    ax.set_ylabel("Número de estrelas")
    ax.set_title("Distribuição da razão sinal-ruído")
    ax.grid(alpha=0.25)

    return _finish(fig, ax, save_path, show)


def plot_correlation_heatmap(
    df: pd.DataFrame,
    columns: Optional[Sequence[str]] = None,
    method: str = "spearman",
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Plota a matriz de correlação como mapa de calor anotado.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Colunas a incluir. Se ``None``, usa todas as
        numéricas exceto identificadores.
    :type columns: collections.abc.Sequence[str] or None
    :param method: Método de correlação repassado a
        :func:`correlation_matrix`.
    :type method: str
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Eixo do matplotlib com o gráfico.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    matriz = correlation_matrix(df, columns, method)

    fig, ax = plt.subplots(figsize=(1.0 * len(matriz) + 2, 0.9 * len(matriz) + 1.5))
    imagem = ax.imshow(matriz, cmap="RdBu_r", vmin=-1, vmax=1)

    ax.set_xticks(range(len(matriz)))
    ax.set_xticklabels(matriz.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(matriz)))
    ax.set_yticklabels(matriz.index, fontsize=8)

    for i in range(len(matriz)):
        for j in range(len(matriz)):
            valor = matriz.iloc[i, j]
            ax.text(
                j, i, f"{valor:.2f}",
                ha="center", va="center", fontsize=7,
                color="white" if abs(valor) > 0.6 else "black",
            )

    fig.colorbar(imagem, ax=ax, shrink=0.8, label=f"correlação ({method})")
    ax.set_title(f"Matriz de correlação — {method}")

    return _finish(fig, ax, save_path, show)


def plot_feature_importances(
    importances: pd.Series,
    top_n: Optional[int] = None,
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Plota a importância dos atributos como barras horizontais.

    :param importances: Série devolvida por
        :func:`rf_oba_stars.model.feature_importances`.
    :type importances: pandas.Series
    :param top_n: Número de atributos a exibir. Se ``None``, exibe todos.
    :type top_n: int or None
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Eixo do matplotlib com o gráfico.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    dados = importances if top_n is None else importances.head(top_n)
    dados = dados.iloc[::-1]

    fig, ax = plt.subplots(figsize=(7, 0.45 * len(dados) + 1.8))
    ax.barh(dados.index, dados.to_numpy(), color="steelblue")

    for i, valor in enumerate(dados.to_numpy()):
        ax.text(valor, i, f" {valor:.3f}", va="center", fontsize=8)

    ax.set_xlabel("Importância (redução média de impureza)")
    ax.set_title("Importância dos atributos")
    ax.margins(x=0.15)
    ax.grid(alpha=0.25, axis="x")

    return _finish(fig, ax, save_path, show)
