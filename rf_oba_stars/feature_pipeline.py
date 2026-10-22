"""Pipeline de extração de atributos: dos espectros XP à tabela final.

Encadeia as etapas do notebook ``OBA_Stars_Pipeline`` — leitura,
correção sistemática, corte de qualidade, normalização pelo contínuo,
extração de índices, remoção de outliers e escalonamento — em uma única
função, :func:`run_feature_pipeline`.

.. rubric:: Ordem das etapas

A ordem importa e difere do notebook original em dois pontos:

1. a correção é aplicada **antes** do corte de SNR, para que ``C2`` e
   ``C3`` sejam calculados para todas as estrelas e depois filtrados
   junto com os espectros — no notebook, os vetores eram filtrados
   separadamente, o que arriscava desalinhá-los;
2. o corte por ``caution`` entra logo após a correção, descartando as
   estrelas cujos coeficientes são extrapolações não confiáveis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Union

import numpy as np
import pandas as pd

from . import config, correction, eda, features, preprocessing, spectra as spectra_mod

PathLike = Union[str, Path]


def run_feature_pipeline(
    spectra_path: PathLike,
    correction_package_dir: Optional[PathLike] = None,
    correction_cache: Optional[PathLike] = None,
    use_corrected_flux: bool = True,
    snr_threshold: float = config.SNR_THRESHOLD,
    filter_caution: bool = True,
    continuum_method: str = "specutils",
    feature_mode: str = "ew",
    lines: Iterable[str] = config.DEFAULT_LINES,
    include_broadband: Sequence[str] = ("color_mag", "blue_red_norm", "slope_norm"),
    remove_outliers: bool = False,
    outlier_columns: Iterable[str] = config.OUTLIER_COLUMNS,
    outliers_by_class: bool = True,
    scale: bool = False,
    scale_method: str = "minmax",
    output_csv: Optional[PathLike] = None,
    figures_dir: Optional[PathLike] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Constrói a tabela de atributos a partir dos espectros XP brutos.

    :param spectra_path: Caminho do arquivo ``oba_stars.fits`` ou
        ``oba_stars.csv``.
    :type spectra_path: str or pathlib.Path
    :param correction_package_dir: Diretório do pacote
        ``GaiaDR3XPspectracorrectionV1``. Se ``None``, a correção é
        pulada e as colunas ``C2`` e ``C3`` ficam ausentes.
    :type correction_package_dir: str or pathlib.Path or None
    :param correction_cache: Caminho de um arquivo ``.npz`` usado como
        cache da correção. Se o arquivo existir, é lido em vez de
        recalcular; caso contrário, o resultado é gravado nele. A
        correção leva vários minutos para milhares de espectros.
    :type correction_cache: str or pathlib.Path or None
    :param use_corrected_flux: Se ``True``, todas as etapas seguintes
        usam o espectro **corrigido**. No notebook original o fluxo
        corrigido era calculado, salvo e depois ignorado: os índices
        eram medidos sobre o espectro bruto, de modo que a correção
        sistemática só entrava indiretamente, via ``C2`` e ``C3``.
    :type use_corrected_flux: bool
    :param snr_threshold: SNR médio mínimo. Use ``0`` para não filtrar.
    :type snr_threshold: float
    :param filter_caution: Se ``True``, descarta estrelas cuja correção
        de C2 ou C3 seja considerada não confiável.
    :type filter_caution: bool
    :param continuum_method: Método de normalização repassado a
        :func:`rf_oba_stars.features.normalize_continuum`.
    :type continuum_method: str
    :param feature_mode: ``'ew'``, ``'physical'``, ``'legacy'``,
        ``'both'`` ou ``'all'``, repassado a
        :func:`rf_oba_stars.features.build_feature_table`.
    :type feature_mode: str
    :param lines: Linhas a medir no modo ``'physical'``.
    :type lines: collections.abc.Iterable[str]
    :param include_broadband: Índices de banda larga a manter.
    :type include_broadband: collections.abc.Sequence[str]
    :param remove_outliers: Se ``True``, aplica a regra de Tukey às
        colunas de ``outlier_columns``. O padrão é ``False``: sobre esta
        amostra, o corte global descartava 24% das estrelas O e 18% das
        B contra 3% das A, sem ganho mensurável de desempenho.
    :type remove_outliers: bool
    :param outlier_columns: Colunas sobre as quais remover outliers.
    :type outlier_columns: collections.abc.Iterable[str]
    :param outliers_by_class: Se ``True``, calcula os limites dentro de
        cada classe espectral, preservando a estrutura física de C2 e
        C3. Só tem efeito quando ``remove_outliers`` for ``True``.
    :type outliers_by_class: bool
    :param scale: Se ``True``, escalona os atributos numéricos. Não é
        necessário para Random Forest e introduz risco de vazamento se
        aplicado antes da divisão treino/teste — mantido como opção
        desligada por padrão.
    :type scale: bool
    :param scale_method: ``'minmax'`` ou ``'standard'``.
    :type scale_method: str
    :param output_csv: Caminho para gravar a tabela final. Se ``None``,
        nada é gravado.
    :type output_csv: str or pathlib.Path or None
    :param figures_dir: Diretório onde salvar as figuras da análise
        exploratória. Se ``None``, nenhuma figura é gerada.
    :type figures_dir: str or pathlib.Path or None
    :param verbose: Se ``True``, imprime o progresso de cada etapa.
    :type verbose: bool
    :returns: Dicionário com as chaves ``'features'`` (a tabela final),
        ``'spectra'``, ``'flux_norm'``, ``'diagnostico'``,
        ``'relatorio_outliers'``, ``'scaler'`` e ``'historico'`` (número
        de estrelas após cada etapa).
    :rtype: dict
    :raises FileNotFoundError: Se o arquivo de espectros não existir.

    .. rubric:: Exemplo

    .. code-block:: python

        >>> saida = run_feature_pipeline(
        ...     "oba_stars.fits",
        ...     correction_package_dir="./GaiaXPcorrection-main",
        ...     output_csv="oba_stars_features.csv",
        ... )
        >>> saida["features"].shape
        (7000, 11)
    """
    figures_dir = Path(figures_dir) if figures_dir is not None else None
    historico: Dict[str, int] = {}

    def log(mensagem: str) -> None:
        if verbose:
            print(mensagem)

    # 1. Leitura -----------------------------------------------------------
    conjunto = spectra_mod.load_spectra(spectra_path)
    historico["carregadas"] = len(conjunto)
    if verbose:
        resumo = spectra_mod.summarize_spectra(conjunto)
        log(f"1. Leitura: {resumo['n_estrelas']} estrelas, "
            f"{resumo['n_pontos']} pontos, "
            f"{resumo['faixa_nm'][0]:.0f}–{resumo['faixa_nm'][1]:.0f} nm")
        log(f"   distribuição: {resumo['distribuicao']}")
        if resumo["fluxos_invalidos"]:
            log(f"   atenção: {resumo['fluxos_invalidos']} espectros com "
                "valores não finitos")

    # 2. Correção sistemática ---------------------------------------------
    diagnostico = None
    cache = Path(correction_cache) if correction_cache is not None else None

    if cache is not None and cache.exists():
        log(f"2. Correção sistemática: lida do cache {cache}")
        flux_corrigido, diagnostico = correction.load_correction(cache)
        if len(diagnostico) != len(conjunto):
            raise ValueError(
                f"O cache tem {len(diagnostico)} linhas, mas o arquivo de "
                f"espectros tem {len(conjunto)}. Apague o cache e refaça a "
                "correção."
            )
    elif correction_package_dir is not None:
        log("2. Correção sistemática (Huang et al. 2024)...")
        flux_corrigido, diagnostico = correction.correct_spectra(
            conjunto, correction_package_dir, verbose=verbose
        )
        if cache is not None:
            correction.save_correction(flux_corrigido, diagnostico, cache)
            log(f"   resultado gravado em {cache}")
    else:
        flux_corrigido = None
        log("2. Correção sistemática: pulada (pacote não informado)")

    if diagnostico is not None:
        confiaveis = int(diagnostico["correcao_confiavel"].sum())
        log(f"   correções confiáveis: {confiaveis}/{len(diagnostico)} "
            f"({100 * confiaveis / len(diagnostico):.1f}%)")

        if use_corrected_flux:
            conjunto.flux = flux_corrigido
            log("   espectros corrigidos adotados nas etapas seguintes")

        if filter_caution:
            conjunto, diagnostico, flux_corrigido = correction.apply_caution_filter(
                conjunto, diagnostico, flux_corrigido
            )
            historico["apos_caution"] = len(conjunto)
            log(f"   após filtro de confiabilidade: {len(conjunto)} estrelas")

    # 3. Corte de qualidade -----------------------------------------------
    _, snr_mean = spectra_mod.compute_snr(conjunto.flux, conjunto.flux_error)
    if figures_dir is not None:
        eda.plot_snr_distribution(
            snr_mean,
            threshold=snr_threshold,
            save_path=figures_dir / "snr_distribuicao.png",
            show=False,
        )

    if snr_threshold > 0:
        mask_snr = snr_mean >= snr_threshold
        conjunto.add_meta(snr_mean=snr_mean)
        conjunto = conjunto.select(mask_snr)
        if diagnostico is not None:
            diagnostico = diagnostico.loc[mask_snr].reset_index(drop=True)
        historico["apos_snr"] = len(conjunto)
        log(f"3. Corte de SNR ≥ {snr_threshold:g}: {len(conjunto)} estrelas")
    else:
        conjunto.add_meta(snr_mean=snr_mean)
        log("3. Corte de SNR: desativado")

    # 4. Normalização pelo contínuo ---------------------------------------
    log(f"4. Normalização pelo contínuo (método '{continuum_method}')...")
    flux_norm = features.normalize_continuum(
        conjunto.flux,
        conjunto.wavelength,
        method=continuum_method,
        verbose=verbose,
    )

    # 5. Extração de atributos --------------------------------------------
    tabela = features.build_feature_table(
        conjunto,
        flux_norm,
        diagnostico=diagnostico,
        mode=feature_mode,
        lines=lines,
        include_broadband=include_broadband,
    )
    tabela = preprocessing.clean_feature_table(tabela, verbose=verbose)

    escalas = preprocessing.check_feature_scale(tabela)
    invisiveis = escalas.index[escalas["invisivel_para_arvore"]].tolist()
    if invisiveis:
        log(f"   AVISO: atributos com amplitude abaixo de "
            f"{preprocessing.SKLEARN_FEATURE_THRESHOLD:g} são tratados como "
            f"constantes pelo divisor de árvores do scikit-learn e recebem "
            f"importância zero: {invisiveis}")
    escalas = preprocessing.check_feature_scales(tabela, warn=verbose)
    historico["apos_extracao"] = len(tabela)
    colunas = preprocessing.numeric_feature_columns(tabela)
    log(f"5. Atributos extraídos ({len(colunas)}): {colunas}")

    # 6. Outliers ----------------------------------------------------------
    relatorio_outliers = None
    if remove_outliers:
        alvo = [c for c in outlier_columns if c in tabela.columns]
        if alvo:
            tabela, relatorio_outliers = preprocessing.remove_outliers(
                tabela, columns=alvo, by_class=outliers_by_class
            )
            historico["apos_outliers"] = len(tabela)
            if verbose:
                for _, linha in relatorio_outliers.iterrows():
                    log(f"6. Outliers em {linha['coluna']}: "
                        f"{int(linha['n_removidas'])} removidas "
                        f"({linha['pct_removidas']:.2f}%)")
        else:
            log("6. Outliers: nenhuma das colunas indicadas está presente")

    # 7. Escalonamento -----------------------------------------------------
    scaler = None
    if scale:
        colunas = preprocessing.numeric_feature_columns(tabela)
        scaler = preprocessing.fit_scaler(tabela, colunas, method=scale_method)
        tabela = preprocessing.apply_scaler(tabela, scaler, colunas)
        log(f"7. Escalonamento '{scale_method}' aplicado a {len(colunas)} colunas")

    # 8. Figuras exploratórias --------------------------------------------
    if figures_dir is not None:
        colunas = preprocessing.numeric_feature_columns(tabela)
        eda.plot_class_balance(
            tabela, save_path=figures_dir / "distribuicao_classes.png", show=False
        )
        eda.plot_feature_distributions(
            tabela, colunas,
            save_path=figures_dir / "distribuicao_atributos.png", show=False,
        )
        eda.plot_correlation_heatmap(
            tabela, colunas,
            save_path=figures_dir / "correlacao_atributos.png", show=False,
        )
        eda.plot_mean_spectra_by_class(
            conjunto, flux_norm,
            lines=lines, xlim=(360.0, 700.0),
            save_path=figures_dir / "espectros_medios_por_classe.png", show=False,
        )
        log(f"8. Figuras gravadas em {figures_dir}/")

    # 9. Gravação ----------------------------------------------------------
    if output_csv is not None:
        output_csv = Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        tabela.to_csv(output_csv, index=False)
        log(f"9. Tabela gravada em {output_csv} ({len(tabela)} linhas)")

    return {
        "features": tabela,
        "spectra": conjunto,
        "flux_norm": flux_norm,
        "diagnostico": diagnostico,
        "relatorio_outliers": relatorio_outliers,
        "scaler": scaler,
        "historico": historico,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """Constrói o analisador de argumentos da linha de comando.

    :returns: Analisador configurado com as opções do pipeline de
        atributos.
    :rtype: argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(
        description="Extrai atributos espectrais dos espectros XP do Gaia DR3.",
    )
    parser.add_argument("--spectra", default="oba_stars.fits",
                        help="Arquivo FITS ou CSV com os espectros.")
    parser.add_argument("--correction-dir", default=None,
                        help="Diretório do pacote GaiaDR3XPspectracorrectionV1.")
    parser.add_argument("--correction-cache", default=None,
                        help="Arquivo .npz de cache da correção sistemática.")
    parser.add_argument("--raw-flux", action="store_true",
                        help="Mede os índices no espectro bruto, não no corrigido.")
    parser.add_argument("--snr", type=float, default=config.SNR_THRESHOLD,
                        help="Limiar de SNR médio (0 desativa o corte).")
    parser.add_argument("--no-caution-filter", action="store_true",
                        help="Não descarta correções marcadas como não confiáveis.")
    parser.add_argument("--continuum", default="specutils",
                        choices=["specutils", "median_filter"],
                        help="Método de normalização pelo contínuo.")
    parser.add_argument("--mode", default="ew",
                        choices=["ew", "physical", "legacy", "both", "all"],
                        help="Conjunto de índices espectrais a extrair.")
    parser.add_argument("--remove-outliers", action="store_true",
                        help="Aplica a regra de Tukey em C2 e C3 (desligado por padrão).")
    parser.add_argument("--outliers-global", action="store_true",
                        help="Calcula os limites de Tukey sobre toda a amostra, "
                             "em vez de dentro de cada classe.")
    parser.add_argument("--scale", action="store_true",
                        help="Escalona os atributos numéricos.")
    parser.add_argument("--output", default="oba_stars_features.csv",
                        help="Caminho do CSV de saída.")
    parser.add_argument("--figures-dir", default=None,
                        help="Diretório onde salvar as figuras exploratórias.")
    return parser


def main(argv: Optional[list] = None) -> Dict[str, Any]:
    """Ponto de entrada da execução por linha de comando.

    :param argv: Lista de argumentos. Se ``None``, usa ``sys.argv[1:]``.
    :type argv: list[str] or None
    :returns: Mesmo dicionário devolvido por :func:`run_feature_pipeline`.
    :rtype: dict
    """
    args = build_arg_parser().parse_args(argv)

    return run_feature_pipeline(
        spectra_path=args.spectra,
        correction_package_dir=args.correction_dir,
        correction_cache=args.correction_cache,
        use_corrected_flux=not args.raw_flux,
        snr_threshold=args.snr,
        filter_caution=not args.no_caution_filter,
        continuum_method=args.continuum,
        feature_mode=args.mode,
        remove_outliers=args.remove_outliers,
        outliers_by_class=not args.outliers_global,
        scale=args.scale,
        output_csv=args.output,
        figures_dir=args.figures_dir,
        verbose=True,
    )


if __name__ == "__main__":
    main()
