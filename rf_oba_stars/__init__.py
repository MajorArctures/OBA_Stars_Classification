"""Classificação espectral de estrelas OBA com Random Forest.

Pacote que organiza, em módulos independentes, o experimento completo de
classificação das classes espectrais O, B e A a partir dos espectros XP
do Gaia DR3 — da leitura dos espectros brutos à avaliação do
classificador.

.. rubric:: Organização

===========================================  =========================================
Módulo                                       Responsabilidade
===========================================  =========================================
:mod:`rf_oba_stars.config`                   Constantes, linhas espectrais, hiperparâmetros
:mod:`rf_oba_stars.spectra`                  Leitura dos espectros, SNR, grade de onda
:mod:`rf_oba_stars.correction`               Correção sistemática, C2, C3 e confiabilidade
:mod:`rf_oba_stars.features`                 Normalização pelo contínuo e índices espectrais
:mod:`rf_oba_stars.preprocessing`            Outliers, escalonamento, limpeza
:mod:`rf_oba_stars.eda`                      Estatísticas, testes e visualizações
:mod:`rf_oba_stars.feature_pipeline`         Espectros → tabela de atributos
:mod:`rf_oba_stars.data`                     Carregamento da tabela de atributos
:mod:`rf_oba_stars.splitting`                Divisões treino/validação/teste e K-Fold
:mod:`rf_oba_stars.model`                    Construção, treino e persistência do modelo
:mod:`rf_oba_stars.evaluation`               Métricas e matrizes de confusão
:mod:`rf_oba_stars.pipeline`                 Atributos → modelo treinado e avaliado
:mod:`rf_oba_stars.experiments`              Comparações, curvas de aprendizado, registro
===========================================  =========================================

.. rubric:: Uso rápido

.. code-block:: python

    from rf_oba_stars import run_feature_pipeline, run_pipeline

    # Etapa 1: espectros XP -> tabela de atributos
    extracao = run_feature_pipeline(
        "oba_stars.fits",
        correction_package_dir="./GaiaXPcorrection-main",
        output_csv="oba_stars_features.csv",
    )

    # Etapa 2: tabela de atributos -> modelo treinado e avaliado
    resultado = run_pipeline("oba_stars_features.csv")
    print(resultado["teste"]["acuracia"])
"""

from __future__ import annotations

from . import (
    config,
    correction,
    data,
    eda,
    evaluation,
    experiments,
    feature_pipeline,
    features,
    model,
    pipeline,
    preprocessing,
    splitting,
    spectra,
)
from .correction import apply_caution_filter, correct_spectra, summarize_caution
from .data import (
    class_distribution,
    describe_dataset,
    load_features,
    split_features_target,
)
from .eda import (
    class_balance,
    correlation_matrix,
    descriptive_statistics,
    normality_tests,
    plot_class_balance,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_feature_importances,
    plot_feature_scatter,
    plot_mean_spectra_by_class,
    plot_qq,
    plot_snr_distribution,
    plot_spectrum,
)
from .evaluation import (
    build_classification_report,
    compute_accuracy,
    compute_confusion_matrix,
    evaluate_model,
    plot_confusion_matrix,
    print_evaluation,
)
from .experiments import (
    append_experiments,
    confusion_pairs,
    error_analysis,
    learning_curve_data,
    plot_experiment_comparison,
    plot_learning_curve,
    run_experiments,
    run_hyperparameter_sweep,
)
from .feature_pipeline import run_feature_pipeline
from .features import (
    audit_legacy_windows,
    band_shift_scan,
    compute_line_indices,
    line_index,
    pseudo_continuum,
    broadband_features,
    build_feature_table,
    compute_legacy_indices,
    compute_line_features,
    line_depth,
    normalize_continuum,
)
from .model import (
    build_random_forest,
    cross_validate_model,
    feature_importances,
    fit_model,
    load_model,
    predict,
    predict_proba,
    save_model,
    summarize_scores,
)
from .pipeline import run_pipeline
from .preprocessing import (
    check_feature_scales,
    clean_feature_table,
    numeric_feature_columns,
    remove_outliers,
)
from .spectra import SpectraSet, apply_snr_cut, compute_snr, load_spectra
from .splitting import DataSplits, build_cv, train_val_test_split

__version__ = "0.5.0"

__all__ = [
    # módulos
    "config",
    "spectra",
    "correction",
    "features",
    "preprocessing",
    "eda",
    "feature_pipeline",
    "experiments",
    "data",
    "splitting",
    "model",
    "evaluation",
    "pipeline",
    # espectros
    "SpectraSet",
    "load_spectra",
    "compute_snr",
    "apply_snr_cut",
    # correção
    "correct_spectra",
    "summarize_caution",
    "apply_caution_filter",
    # atributos
    "normalize_continuum",
    "line_depth",
    "compute_line_features",
    "compute_line_indices",
    "line_index",
    "pseudo_continuum",
    "band_shift_scan",
    "compute_legacy_indices",
    "audit_legacy_windows",
    "broadband_features",
    "build_feature_table",
    "run_feature_pipeline",
    # pré-processamento
    "remove_outliers",
    "clean_feature_table",
    "check_feature_scales",
    "numeric_feature_columns",
    # análise exploratória
    "descriptive_statistics",
    "class_balance",
    "normality_tests",
    "correlation_matrix",
    "plot_spectrum",
    "plot_mean_spectra_by_class",
    "plot_feature_distributions",
    "plot_qq",
    "plot_feature_scatter",
    "plot_class_balance",
    "plot_snr_distribution",
    "plot_correlation_heatmap",
    "plot_feature_importances",
    # tabela de atributos e modelo
    "load_features",
    "class_distribution",
    "describe_dataset",
    "split_features_target",
    "DataSplits",
    "train_val_test_split",
    "build_cv",
    "build_random_forest",
    "cross_validate_model",
    "summarize_scores",
    "fit_model",
    "predict",
    "predict_proba",
    "feature_importances",
    "save_model",
    "load_model",
    "compute_accuracy",
    "build_classification_report",
    "compute_confusion_matrix",
    "plot_confusion_matrix",
    "evaluate_model",
    "print_evaluation",
    "run_pipeline",
    "run_experiments",
    "run_hyperparameter_sweep",
    "append_experiments",
    "learning_curve_data",
    "plot_learning_curve",
    "plot_experiment_comparison",
    "error_analysis",
    "confusion_pairs",
    "__version__",
]
