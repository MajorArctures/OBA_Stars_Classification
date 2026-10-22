"""Métricas de desempenho e visualização dos resultados.

Reúne acurácia, relatório de classificação e matriz de confusão para os
conjuntos de validação e teste.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from . import model as model_module

PathLike = Union[str, Path]
ArrayLike = Union[np.ndarray, pd.Series, Sequence[str]]


def compute_accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Calcula a acurácia global das previsões.

    :param y_true: Classes espectrais verdadeiras.
    :type y_true: numpy.ndarray or pandas.Series
    :param y_pred: Classes espectrais previstas pelo modelo.
    :type y_pred: numpy.ndarray or pandas.Series
    :returns: Fração de acertos, entre ``0.0`` e ``1.0``.
    :rtype: float
    """
    return float(accuracy_score(y_true, y_pred))


def build_classification_report(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence[str]] = None,
    as_dict: bool = False,
    digits: int = 4,
) -> Union[str, Dict[str, Any]]:
    """Gera o relatório de precisão, revocação e F1 por classe.

    :param y_true: Classes espectrais verdadeiras.
    :type y_true: numpy.ndarray or pandas.Series
    :param y_pred: Classes espectrais previstas pelo modelo.
    :type y_pred: numpy.ndarray or pandas.Series
    :param labels: Ordem das classes no relatório. Se ``None``, usa a
        ordem alfabética determinada pelo scikit-learn.
    :type labels: collections.abc.Sequence[str] or None
    :param as_dict: Se ``True``, retorna um dicionário estruturado em vez
        do texto formatado.
    :type as_dict: bool
    :param digits: Número de casas decimais no relatório em texto.
    :type digits: int
    :returns: Relatório formatado como texto ou como dicionário.
    :rtype: str or dict
    """
    return classification_report(
        y_true,
        y_pred,
        labels=list(labels) if labels is not None else None,
        output_dict=as_dict,
        digits=digits,
        zero_division=0,
    )


def compute_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence[str]] = None,
    normalize: Optional[str] = None,
) -> np.ndarray:
    """Calcula a matriz de confusão.

    :param y_true: Classes espectrais verdadeiras.
    :type y_true: numpy.ndarray or pandas.Series
    :param y_pred: Classes espectrais previstas pelo modelo.
    :type y_pred: numpy.ndarray or pandas.Series
    :param labels: Ordem das classes nas linhas e colunas.
    :type labels: collections.abc.Sequence[str] or None
    :param normalize: ``'true'`` normaliza por linha (revocação),
        ``'pred'`` por coluna, ``'all'`` pelo total e ``None`` mantém as
        contagens absolutas.
    :type normalize: str or None
    :returns: Matriz de confusão quadrada.
    :rtype: numpy.ndarray
    """
    return confusion_matrix(
        y_true,
        y_pred,
        labels=list(labels) if labels is not None else None,
        normalize=normalize,
    )


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    labels: Optional[Sequence[str]] = None,
    title: str = "Matriz de Confusão",
    normalize: Optional[str] = None,
    cmap: str = "plasma",
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> "matplotlib.axes.Axes":  # noqa: F821
    """Desenha a matriz de confusão como figura do matplotlib.

    :param y_true: Classes espectrais verdadeiras.
    :type y_true: numpy.ndarray or pandas.Series
    :param y_pred: Classes espectrais previstas pelo modelo.
    :type y_pred: numpy.ndarray or pandas.Series
    :param labels: Ordem das classes nos eixos.
    :type labels: collections.abc.Sequence[str] or None
    :param title: Título da figura.
    :type title: str
    :param normalize: Mesmo significado de
        :func:`compute_confusion_matrix`.
    :type normalize: str or None
    :param cmap: Mapa de cores do matplotlib.
    :type cmap: str
    :param save_path: Se informado, salva a figura nesse caminho com
        200 dpi.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, chama ``plt.show()`` ao final — útil em
        notebooks, dispensável em scripts.
    :type show: bool
    :returns: Eixo do matplotlib contendo a figura, permitindo ajustes
        adicionais pelo chamador.
    :rtype: matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    cm = compute_confusion_matrix(y_true, y_pred, labels=labels, normalize=normalize)

    display_labels = list(labels) if labels is not None else np.unique(y_true)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    disp.plot(ax=ax, cmap=cmap, values_format=".2f" if normalize else "d")
    ax.set_title(title)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    fig.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()

    return ax


def evaluate_model(
    trained_model: RandomForestClassifier,
    X: pd.DataFrame,
    y_true: pd.Series,
    set_name: str = "Validação",
    labels: Optional[Sequence[str]] = None,
    plot: bool = True,
    save_path: Optional[PathLike] = None,
    show: bool = True,
) -> Dict[str, Any]:
    """Avalia o modelo em um subconjunto e reúne todas as métricas.

    :param trained_model: Classificador já treinado.
    :type trained_model: sklearn.ensemble.RandomForestClassifier
    :param X: Atributos do subconjunto avaliado.
    :type X: pandas.DataFrame
    :param y_true: Classes espectrais verdadeiras do subconjunto.
    :type y_true: pandas.Series
    :param set_name: Nome do subconjunto, usado no título da figura
        (por exemplo ``'Validação'`` ou ``'Teste'``).
    :type set_name: str
    :param labels: Ordem das classes nas métricas e na figura. Se
        ``None``, usa ``trained_model.classes_``.
    :type labels: collections.abc.Sequence[str] or None
    :param plot: Se ``True``, desenha a matriz de confusão.
    :type plot: bool
    :param save_path: Caminho opcional para salvar a figura.
    :type save_path: str or pathlib.Path or None
    :param show: Se ``True``, exibe a figura gerada.
    :type show: bool
    :returns: Dicionário com as chaves ``'conjunto'``, ``'acuracia'``,
        ``'relatorio'`` (dicionário por classe), ``'matriz_confusao'``,
        ``'classes'`` e ``'y_pred'``.
    :rtype: dict
    """
    if labels is None:
        labels = list(trained_model.classes_)

    y_pred = model_module.predict(trained_model, X)

    resultados: Dict[str, Any] = {
        "conjunto": set_name,
        "acuracia": compute_accuracy(y_true, y_pred),
        "relatorio": build_classification_report(
            y_true, y_pred, labels=labels, as_dict=True
        ),
        "matriz_confusao": compute_confusion_matrix(y_true, y_pred, labels=labels),
        "classes": list(labels),
        "y_pred": y_pred,
    }

    if plot:
        plot_confusion_matrix(
            y_true,
            y_pred,
            labels=labels,
            title=f"Matriz de Confusão - Conjunto de {set_name}",
            save_path=save_path,
            show=show,
        )

    return resultados


def print_evaluation(resultados: Dict[str, Any]) -> None:
    """Imprime de forma legível o resultado de :func:`evaluate_model`.

    :param resultados: Dicionário devolvido por :func:`evaluate_model`.
    :type resultados: dict
    :returns: Nada; a função apenas escreve na saída padrão.
    :rtype: None
    """
    nome = resultados["conjunto"]
    print(f"Acurácia ({nome}): {resultados['acuracia']:.4f}\n")

    relatorio = resultados["relatorio"]
    print(f"{'classe':<12}{'precisão':>10}{'revocação':>12}{'f1':>8}{'suporte':>10}")
    for classe in resultados["classes"]:
        m = relatorio[str(classe)]
        print(
            f"{classe:<12}{m['precision']:>10.4f}{m['recall']:>12.4f}"
            f"{m['f1-score']:>8.4f}{int(m['support']):>10d}"
        )
    print(f"\n{'acurácia':<12}{relatorio['accuracy']:>10.4f}")
