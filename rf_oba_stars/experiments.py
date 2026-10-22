"""Experimentos comparativos, curvas de aprendizado e registro em disco.

Reúne as ferramentas de comparação sistemática entre configurações do
modelo: varredura de conjuntos de atributos, varredura de
hiperparâmetros, curva de aprendizado e o registro cumulativo dos
resultados em ``experiments.csv``.

.. rubric:: Fluxo típico

.. code-block:: python

    resultados = run_experiments(df, {"linhas": FISICO, "completo": TODOS})
    append_experiments(resultados, "experiments.csv")
    plot_experiment_comparison(resultados)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import config, model as model_module, splitting

PathLike = Union[str, Path]

#: Colunas gravadas em ``experiments.csv``, nesta ordem.
EXPERIMENT_COLUMNS: Sequence[str] = (
    "id",
    "data_hora",
    "descricao",
    "tabela",
    "n_amostras",
    "n_atributos",
    "atributos",
    "n_estimators",
    "max_depth",
    "min_samples_leaf",
    "max_features",
    "class_weight",
    "cv_folds",
    "acuracia",
    "acuracia_dp",
    "f1_macro",
    "f1_macro_dp",
)


def evaluate_configuration(
    X: pd.DataFrame,
    y: pd.Series,
    rf_params: Optional[Dict[str, Any]] = None,
    cv=None,
    scorings: Sequence[str] = ("accuracy", "f1_macro"),
    n_jobs: Optional[int] = None,
) -> Dict[str, float]:
    """Avalia uma configuração por validação cruzada, sem treinar em disco.

    :param X: Matriz de atributos do experimento.
    :type X: pandas.DataFrame
    :param y: Vetor de classes espectrais.
    :type y: pandas.Series
    :param rf_params: Hiperparâmetros do Random Forest. Se ``None``, usa
        :data:`rf_oba_stars.config.RANDOM_FOREST_PARAMS`.
    :type rf_params: dict or None
    :param cv: Esquema de validação cruzada. Se ``None``, usa
        :func:`rf_oba_stars.splitting.build_cv`.
    :type cv: sklearn.model_selection.BaseCrossValidator or None
    :param scorings: Métricas do scikit-learn a calcular.
    :type scorings: collections.abc.Sequence[str]
    :param n_jobs: Número de processos paralelos.
    :type n_jobs: int or None
    :returns: Dicionário com ``'<metrica>'`` e ``'<metrica>_dp'`` para
        cada métrica pedida.
    :rtype: dict[str, float]

    .. note::
       As métricas são calculadas em uma única passagem de
       :func:`sklearn.model_selection.cross_validate`, e não com uma
       chamada por métrica: o modelo é ajustado uma vez por dobra em vez
       de uma vez por dobra e por métrica.
    """
    from sklearn.model_selection import cross_validate

    cv = splitting.build_cv() if cv is None else cv
    rf = model_module.build_random_forest(rf_params)

    saida = cross_validate(
        rf, X, y, cv=cv, scoring=list(scorings), n_jobs=n_jobs
    )

    resultado: Dict[str, float] = {}
    for metrica in scorings:
        scores = saida[f"test_{metrica}"]
        resultado[metrica] = float(scores.mean())
        resultado[f"{metrica}_dp"] = float(scores.std())

    return resultado


def run_experiments(
    df: pd.DataFrame,
    feature_sets: Mapping[str, Sequence[str]],
    rf_params: Optional[Dict[str, Any]] = None,
    target_column: str = config.TARGET_COLUMN,
    table_name: str = "",
    id_prefix: str = "E",
    start_id: int = 1,
    cv=None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compara vários conjuntos de atributos sobre a mesma tabela.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param feature_sets: Mapa ``descrição -> lista de colunas``.
    :type feature_sets: collections.abc.Mapping[str, collections.abc.Sequence[str]]
    :param rf_params: Hiperparâmetros comuns a todos os experimentos.
    :type rf_params: dict or None
    :param target_column: Nome da coluna alvo.
    :type target_column: str
    :param table_name: Rótulo da tabela de origem, gravado no registro.
    :type table_name: str
    :param id_prefix: Prefixo dos identificadores gerados.
    :type id_prefix: str
    :param start_id: Número do primeiro experimento.
    :type start_id: int
    :param cv: Esquema de validação cruzada.
    :type cv: sklearn.model_selection.BaseCrossValidator or None
    :param n_jobs: Número de processos paralelos.
    :type n_jobs: int or None
    :param verbose: Se ``True``, imprime cada resultado assim que sai.
    :type verbose: bool
    :returns: Tabela com uma linha por experimento, nas colunas de
        :data:`EXPERIMENT_COLUMNS`.
    :rtype: pandas.DataFrame
    :raises KeyError: Se algum conjunto citar colunas ausentes na tabela.
    """
    cv = splitting.build_cv() if cv is None else cv
    y = df[target_column]
    referencia = model_module.build_random_forest(rf_params)

    linhas = []
    for i, (descricao, colunas) in enumerate(feature_sets.items(), start=start_id):
        faltando = [c for c in colunas if c not in df.columns]
        if faltando:
            raise KeyError(f"Colunas ausentes para '{descricao}': {faltando}")

        metricas = evaluate_configuration(
            df[list(colunas)], y, rf_params, cv=cv, n_jobs=n_jobs
        )

        linhas.append(
            {
                "id": f"{id_prefix}{i:02d}",
                "data_hora": datetime.now().isoformat(timespec="seconds"),
                "descricao": descricao,
                "tabela": table_name,
                "n_amostras": len(df),
                "n_atributos": len(colunas),
                "atributos": " ".join(colunas),
                "n_estimators": referencia.n_estimators,
                "max_depth": referencia.max_depth,
                "min_samples_leaf": referencia.min_samples_leaf,
                "max_features": str(referencia.max_features),
                "class_weight": str(referencia.class_weight),
                "cv_folds": cv.get_n_splits(),
                "acuracia": round(metricas["accuracy"], 4),
                "acuracia_dp": round(metricas["accuracy_dp"], 4),
                "f1_macro": round(metricas["f1_macro"], 4),
                "f1_macro_dp": round(metricas["f1_macro_dp"], 4),
            }
        )

        if verbose:
            print(
                f"{linhas[-1]['id']} {descricao:42.42} "
                f"acc {metricas['accuracy']:.4f}  "
                f"F1m {metricas['f1_macro']:.4f} ± {metricas['f1_macro_dp']:.4f}",
                flush=True,
            )

    return pd.DataFrame(linhas, columns=list(EXPERIMENT_COLUMNS))


def run_hyperparameter_sweep(
    df: pd.DataFrame,
    columns: Sequence[str],
    grid: Mapping[str, Iterable[Any]],
    target_column: str = config.TARGET_COLUMN,
    base_params: Optional[Dict[str, Any]] = None,
    cv=None,
    n_jobs: Optional[int] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Varre um hiperparâmetro por vez, mantendo os demais fixos.

    Diferente de uma busca em grade completa: cada chave de ``grid`` é
    variada isoladamente sobre a configuração base, o que responde
    diretamente às perguntas "qual o efeito do número de árvores?" e
    "qual o efeito da profundidade?" sem o custo combinatório.

    :param df: Tabela de atributos.
    :type df: pandas.DataFrame
    :param columns: Atributos usados em todos os ajustes.
    :type columns: collections.abc.Sequence[str]
    :param grid: Mapa ``hiperparâmetro -> valores a testar``.
    :type grid: collections.abc.Mapping[str, collections.abc.Iterable]
    :param target_column: Nome da coluna alvo.
    :type target_column: str
    :param base_params: Configuração base. Se ``None``, usa
        :data:`rf_oba_stars.config.RANDOM_FOREST_PARAMS`.
    :type base_params: dict or None
    :param cv: Esquema de validação cruzada.
    :type cv: sklearn.model_selection.BaseCrossValidator or None
    :param n_jobs: Número de processos paralelos.
    :type n_jobs: int or None
    :param verbose: Se ``True``, imprime cada resultado.
    :type verbose: bool
    :returns: Tabela com as colunas ``hiperparametro``, ``valor``,
        ``acuracia``, ``acuracia_dp``, ``f1_macro`` e ``f1_macro_dp``.
    :rtype: pandas.DataFrame
    """
    cv = splitting.build_cv() if cv is None else cv
    base = dict(config.RANDOM_FOREST_PARAMS if base_params is None else base_params)
    X, y = df[list(columns)], df[target_column]

    linhas = []
    for nome, valores in grid.items():
        for valor in valores:
            params = dict(base)
            params[nome] = valor

            metricas = evaluate_configuration(X, y, params, cv=cv, n_jobs=n_jobs)
            linhas.append(
                {
                    "hiperparametro": nome,
                    "valor": valor,
                    "acuracia": round(metricas["accuracy"], 4),
                    "acuracia_dp": round(metricas["accuracy_dp"], 4),
                    "f1_macro": round(metricas["f1_macro"], 4),
                    "f1_macro_dp": round(metricas["f1_macro_dp"], 4),
                }
            )

            if verbose:
                print(
                    f"{nome:18} = {str(valor):>6}   "
                    f"acc {metricas['accuracy']:.4f}   "
                    f"F1m {metricas['f1_macro']:.4f} ± {metricas['f1_macro_dp']:.4f}",
                    flush=True,
                )

    return pd.DataFrame(linhas)


def append_experiments(
    resultados: pd.DataFrame,
    path: PathLike = "experiments.csv",
) -> Path:
    """Acrescenta os resultados ao registro cumulativo de experimentos.

    Se o arquivo já existir, as linhas são anexadas preservando o
    histórico — é o registro de laboratório do projeto, não um relatório
    que se sobrescreve.

    :param resultados: Tabela devolvida por :func:`run_experiments`.
    :type resultados: pandas.DataFrame
    :param path: Caminho do arquivo CSV cumulativo.
    :type path: str or pathlib.Path
    :returns: Caminho do arquivo gravado.
    :rtype: pathlib.Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        anterior = pd.read_csv(path)
        resultados = pd.concat([anterior, resultados], ignore_index=True)

    resultados.to_csv(path, index=False)
    return path


def learning_curve_data(
    X: pd.DataFrame,
    y: pd.Series,
    train_sizes: Sequence[float] = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0),
    rf_params: Optional[Dict[str, Any]] = None,
    cv=None,
    scoring: str = "f1_macro",
    n_jobs: Optional[int] = None,
) -> pd.DataFrame:
    """Calcula a curva de aprendizado do modelo.

    Mostra como o desempenho em treino e em validação evolui conforme
    mais estrelas entram no ajuste. A distância entre as duas curvas
    indica sobreajuste; o achatamento da curva de validação indica que
    mais dados do mesmo tipo não trarão ganho.

    :param X: Matriz de atributos.
    :type X: pandas.DataFrame
    :param y: Vetor de classes espectrais.
    :type y: pandas.Series
    :param train_sizes: Frações do conjunto de treino a avaliar.
    :type train_sizes: collections.abc.Sequence[float]
    :param rf_params: Hiperparâmetros do modelo.
    :type rf_params: dict or None
    :param cv: Esquema de validação cruzada.
    :type cv: sklearn.model_selection.BaseCrossValidator or None
    :param scoring: Métrica do scikit-learn a acompanhar.
    :type scoring: str
    :param n_jobs: Número de processos paralelos.
    :type n_jobs: int or None
    :returns: Tabela com as colunas ``n_amostras``, ``treino_media``,
        ``treino_dp``, ``validacao_media`` e ``validacao_dp``.
    :rtype: pandas.DataFrame
    """
    from sklearn.model_selection import learning_curve

    cv = splitting.build_cv() if cv is None else cv
    rf = model_module.build_random_forest(rf_params)

    tamanhos, treino, validacao = learning_curve(
        rf,
        X,
        y,
        train_sizes=np.asarray(train_sizes, dtype=float),
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        shuffle=True,
        random_state=config.RANDOM_STATE_SPLIT,
    )

    return pd.DataFrame(
        {
            "n_amostras": tamanhos,
            "treino_media": treino.mean(axis=1),
            "treino_dp": treino.std(axis=1),
            "validacao_media": validacao.mean(axis=1),
            "validacao_dp": validacao.std(axis=1),
        }
    )


def plot_learning_curve(
    curva: pd.DataFrame,
    scoring: str = "f1_macro",
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Plota a curva de aprendizado com as faixas de desvio padrão.

    :param curva: Tabela devolvida por :func:`learning_curve_data`.
    :type curva: pandas.DataFrame
    :param scoring: Nome da métrica, usado no rótulo do eixo.
    :type scoring: str
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Eixo do matplotlib com o gráfico.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 5))

    for prefixo, rotulo, cor in [
        ("treino", "Treino", "steelblue"),
        ("validacao", "Validação cruzada", "darkorange"),
    ]:
        media = curva[f"{prefixo}_media"]
        desvio = curva[f"{prefixo}_dp"]
        ax.plot(curva["n_amostras"], media, "o-", color=cor, label=rotulo)
        ax.fill_between(
            curva["n_amostras"], media - desvio, media + desvio, alpha=0.2, color=cor
        )

    ax.set_xlabel("Número de estrelas no treino")
    ax.set_ylabel(scoring)
    ax.set_title("Curva de aprendizado")
    ax.legend()
    ax.grid(alpha=0.3)
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


def plot_experiment_comparison(
    resultados: pd.DataFrame,
    metric: str = "f1_macro",
    label_column: str = "descricao",
    save_path: Optional[PathLike] = None,
    show: bool = True,
):
    """Compara os experimentos em um gráfico de barras com barras de erro.

    :param resultados: Tabela devolvida por :func:`run_experiments` ou
        lida de ``experiments.csv``.
    :type resultados: pandas.DataFrame
    :param metric: Coluna da métrica a comparar. A coluna de desvio
        correspondente deve chamar-se ``'<metric>_dp'``.
    :type metric: str
    :param label_column: Coluna usada como rótulo de cada barra.
    :type label_column: str
    :param save_path: Caminho opcional para gravar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura.
    :type show: bool
    :returns: Eixo do matplotlib com o gráfico.
    :rtype: matplotlib.axes.Axes
    :raises KeyError: Se a métrica pedida não estiver na tabela.
    """
    import matplotlib.pyplot as plt

    if metric not in resultados.columns:
        raise KeyError(f"Métrica '{metric}' ausente. Colunas: {list(resultados.columns)}")

    dados = resultados.sort_values(metric)
    erro = dados.get(f"{metric}_dp")

    fig, ax = plt.subplots(figsize=(9, 0.45 * len(dados) + 2))
    ax.barh(
        dados[label_column],
        dados[metric],
        xerr=erro if erro is not None else None,
        color="steelblue",
        error_kw={"ecolor": "black", "capsize": 3, "lw": 1},
    )

    for i, valor in enumerate(dados[metric]):
        ax.text(valor, i, f"  {valor:.3f}", va="center", fontsize=8)

    ax.set_xlabel(metric)
    ax.set_title("Comparação entre experimentos")
    ax.margins(x=0.18)
    ax.grid(alpha=0.25, axis="x")
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


def error_analysis(
    y_true: pd.Series,
    y_pred: Sequence[str],
    probabilities: Optional[pd.DataFrame] = None,
    source_ids: Optional[Sequence[int]] = None,
) -> pd.DataFrame:
    """Lista os erros de classificação, ordenados por confiança.

    Os erros cometidos com alta confiança são os mais informativos:
    costumam apontar rótulos duvidosos no catálogo de referência ou
    estrelas fisicamente atípicas, não apenas ruído de fronteira.

    :param y_true: Classes espectrais verdadeiras.
    :type y_true: pandas.Series
    :param y_pred: Classes previstas pelo modelo.
    :type y_pred: collections.abc.Sequence[str]
    :param probabilities: Tabela de probabilidades devolvida por
        :func:`rf_oba_stars.model.predict_proba`. Se informada, a
        confiança de cada erro entra no resultado.
    :type probabilities: pandas.DataFrame or None
    :param source_ids: Identificadores Gaia das estrelas, para permitir
        a inspeção individual dos casos.
    :type source_ids: collections.abc.Sequence[int] or None
    :returns: Tabela com ``verdadeira``, ``prevista``, opcionalmente
        ``source_id`` e ``confianca``, ordenada da maior para a menor
        confiança.
    :rtype: pandas.DataFrame
    """
    y_pred = np.asarray(y_pred)
    y_true_arr = np.asarray(y_true)
    erros = y_true_arr != y_pred

    tabela = pd.DataFrame(
        {"verdadeira": y_true_arr[erros], "prevista": y_pred[erros]}
    )

    if source_ids is not None:
        tabela.insert(0, "source_id", np.asarray(source_ids)[erros])

    if probabilities is not None:
        confianca = probabilities.to_numpy().max(axis=1)[erros]
        tabela["confianca"] = confianca
        tabela = tabela.sort_values("confianca", ascending=False)

    return tabela.reset_index(drop=True)


def confusion_pairs(
    y_true: pd.Series,
    y_pred: Sequence[str],
    class_order: Sequence[str] = config.CLASS_ORDER,
) -> pd.DataFrame:
    """Tabula os pares de confusão mais frequentes entre classes.

    :param y_true: Classes espectrais verdadeiras.
    :type y_true: pandas.Series
    :param y_pred: Classes previstas pelo modelo.
    :type y_pred: collections.abc.Sequence[str]
    :param class_order: Ordem canônica das classes.
    :type class_order: collections.abc.Sequence[str]
    :returns: Tabela com ``verdadeira``, ``prevista``, ``n`` e
        ``pct_da_classe``, ordenada por ``n`` decrescente e sem os
        acertos.
    :rtype: pandas.DataFrame
    """
    from .evaluation import compute_confusion_matrix

    matriz = compute_confusion_matrix(y_true, y_pred, labels=list(class_order))
    totais = matriz.sum(axis=1)

    linhas = []
    for i, verdadeira in enumerate(class_order):
        for j, prevista in enumerate(class_order):
            if i == j or matriz[i, j] == 0:
                continue
            linhas.append(
                {
                    "verdadeira": verdadeira,
                    "prevista": prevista,
                    "n": int(matriz[i, j]),
                    "pct_da_classe": round(100.0 * matriz[i, j] / totais[i], 2),
                }
            )

    return (
        pd.DataFrame(linhas)
        .sort_values("n", ascending=False)
        .reset_index(drop=True)
    )
