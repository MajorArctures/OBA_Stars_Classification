"""Orquestração de ponta a ponta do experimento de classificação.

Encadeia as etapas dos demais módulos — carregamento, divisão,
validação cruzada, treino e avaliação — em uma única função,
:func:`run_pipeline`, e expõe uma interface de linha de comando por meio
de :func:`main`.

.. rubric:: Uso como script

.. code-block:: bash

    python -m rf_oba_stars.pipeline --csv oba_stars_features.csv --no-plot
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional, Union

from . import config, data, evaluation, model as model_module, splitting

PathLike = Union[str, Path]


def run_pipeline(
    csv_path: PathLike,
    drop_id: bool = True,
    rf_params: Optional[Dict[str, Any]] = None,
    run_cv: bool = True,
    plot: bool = True,
    show: bool = True,
    figures_dir: Optional[PathLike] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Executa o experimento completo de classificação espectral.

    Etapas executadas, nesta ordem:

    1. carregamento da tabela de atributos;
    2. separação entre atributos e classe;
    3. divisão estratificada em treino, validação e teste;
    4. validação cruzada estratificada no conjunto de treino (opcional);
    5. treino do modelo;
    6. avaliação em validação e em teste.

    :param csv_path: Caminho do arquivo ``oba_stars_features.csv``.
    :type csv_path: str or pathlib.Path
    :param drop_id: Se ``True``, descarta ``source_id`` da matriz de
        atributos.
    :type drop_id: bool
    :param rf_params: Hiperparâmetros do Random Forest. Se ``None``, usa
        :data:`rf_oba_stars.config.RANDOM_FOREST_PARAMS`.
    :type rf_params: dict or None
    :param run_cv: Se ``True``, executa a validação cruzada antes do
        treino final.
    :type run_cv: bool
    :param plot: Se ``True``, gera as matrizes de confusão.
    :type plot: bool
    :param show: Se ``True``, exibe as figuras geradas.
    :type show: bool
    :param figures_dir: Diretório onde salvar as figuras. Se ``None``,
        as figuras não são salvas em disco.
    :type figures_dir: str or pathlib.Path or None
    :param verbose: Se ``True``, imprime o progresso e as métricas.
    :type verbose: bool
    :returns: Dicionário com as chaves ``'dataframe'``, ``'splits'``,
        ``'modelo'``, ``'cv_scores'``, ``'cv_resumo'``,
        ``'importancias'``, ``'validacao'`` e ``'teste'``.
    :rtype: dict
    :raises FileNotFoundError: Se o arquivo CSV não existir.

    .. rubric:: Exemplo

    .. code-block:: python

        >>> resultado = run_pipeline("oba_stars_features.csv", plot=False)
        >>> resultado["teste"]["acuracia"]
        0.9...
    """
    figures_dir = Path(figures_dir) if figures_dir is not None else None

    # 1. Carregamento
    df = data.load_features(csv_path)
    if verbose:
        resumo = data.describe_dataset(df)
        print(f"Amostras: {resumo['n_amostras']}")
        print(f"Distribuição por classe: {resumo['distribuicao']}\n")

    # 2. Atributos e alvo
    X, y = data.split_features_target(df, drop_id=drop_id)
    if verbose:
        print(f"Atributos usados ({X.shape[1]}): {list(X.columns)}\n")

    # 3. Divisão treino / validação / teste
    splits = splitting.train_val_test_split(X, y)
    if verbose:
        for nome, n in splits.sizes.items():
            print(f"{nome.capitalize():<11}: {n:>6}  ({splits.proportions[nome]:.2f}%)")
        print()

    # 4. Validação cruzada
    rf = model_module.build_random_forest(rf_params)
    cv_scores = None
    cv_resumo = None
    if run_cv:
        cv = splitting.build_cv()
        cv_scores = model_module.cross_validate_model(
            rf, splits.X_train, splits.y_train, cv=cv
        )
        cv_resumo = model_module.summarize_scores(cv_scores)
        if verbose:
            print(
                f"Validação cruzada ({config.N_SPLITS} dobras): "
                f"{cv_resumo['media']:.4f} ± {cv_resumo['desvio_padrao']:.4f}\n"
            )

    # 5. Treino
    rf = model_module.fit_model(rf, splits.X_train, splits.y_train)
    importancias = model_module.feature_importances(rf)
    if verbose:
        print("Importância dos atributos:")
        for nome, valor in importancias.items():
            print(f"  {nome:<12} {valor:.4f}")
        print()

    # 6. Avaliação
    resultado_val = evaluation.evaluate_model(
        rf,
        splits.X_val,
        splits.y_val,
        set_name="Validação",
        plot=plot,
        show=show,
        save_path=(figures_dir / "matriz_confusao_validacao.png") if figures_dir else None,
    )
    resultado_test = evaluation.evaluate_model(
        rf,
        splits.X_test,
        splits.y_test,
        set_name="Teste",
        plot=plot,
        show=show,
        save_path=(figures_dir / "matriz_confusao_teste.png") if figures_dir else None,
    )

    if verbose:
        evaluation.print_evaluation(resultado_val)
        print()
        evaluation.print_evaluation(resultado_test)

    return {
        "dataframe": df,
        "splits": splits,
        "modelo": rf,
        "cv_scores": cv_scores,
        "cv_resumo": cv_resumo,
        "importancias": importancias,
        "validacao": resultado_val,
        "teste": resultado_test,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Constrói o analisador de argumentos da linha de comando.

    :returns: Analisador configurado com as opções do pipeline.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Classificação espectral de estrelas OBA com Random Forest.",
    )
    parser.add_argument(
        "--csv",
        default="oba_stars_features.csv",
        help="Caminho do arquivo CSV de atributos.",
    )
    parser.add_argument(
        "--keep-id",
        action="store_true",
        help="Mantém 'source_id' como atributo (reproduz o notebook original).",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Pula a validação cruzada.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Não gera as matrizes de confusão.",
    )
    parser.add_argument(
        "--figures-dir",
        default=None,
        help="Diretório onde salvar as figuras geradas.",
    )
    parser.add_argument(
        "--save-model",
        default=None,
        help="Caminho .joblib para salvar o modelo treinado.",
    )
    return parser


def main(argv: Optional[list] = None) -> Dict[str, Any]:
    """Ponto de entrada da execução por linha de comando.

    :param argv: Lista de argumentos. Se ``None``, usa ``sys.argv[1:]``.
    :type argv: list[str] or None
    :returns: Mesmo dicionário devolvido por :func:`run_pipeline`.
    :rtype: dict
    """
    args = build_arg_parser().parse_args(argv)

    resultado = run_pipeline(
        csv_path=args.csv,
        drop_id=not args.keep_id,
        run_cv=not args.no_cv,
        plot=not args.no_plot,
        show=False,
        figures_dir=args.figures_dir,
        verbose=True,
    )

    if args.save_model:
        caminho = model_module.save_model(resultado["modelo"], args.save_model)
        print(f"\nModelo salvo em: {caminho}")

    return resultado


if __name__ == "__main__":
    main()
