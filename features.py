"""Normalização pelo contínuo e extração dos atributos espectrais.

Concentra a construção da tabela de atributos usada no aprendizado de
máquina: profundidades de linha medidas sobre o espectro normalizado
pelo contínuo, índices de banda larga (cor e inclinação) e os
coeficientes C2 e C3 herdados da correção sistemática.

.. warning::
   Na resolução dos espectros XP amostrados — 2 nm por ponto, com o
   perfil instrumental cobrindo várias dezenas de ångströms — nenhuma
   linha aparece isolada. Os índices aqui medem *regiões espectrais*
   dominadas pela linha nomeada, não a linha em si.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import config
from .spectra import SpectraSet, window_indices

# ---------------------------------------------------------------------------
# Normalização pelo contínuo
# ---------------------------------------------------------------------------


def normalize_continuum(
    flux: np.ndarray,
    wavelength: Optional[np.ndarray] = None,
    method: str = "specutils",
    median_window: int = 51,
    verbose: bool = False,
) -> np.ndarray:
    """Divide cada espectro pelo seu contínuo ajustado.

    Após a normalização, o contínuo fica em torno de 1 e as linhas de
    absorção aparecem como depressões — o que permite medir
    profundidades de linha comparáveis entre estrelas de brilhos
    diferentes.

    :param flux: Matriz de fluxos de forma ``(n_estrelas, n_pontos)``.
    :type flux: numpy.ndarray
    :param wavelength: Grade de comprimentos de onda em nanômetros. Se
        ``None``, usa :func:`rf_oba_stars.config.wavelength_grid`.
    :type wavelength: numpy.ndarray or None
    :param method: ``'specutils'`` reproduz o notebook original,
        ajustando o contínuo com
        :func:`specutils.fitting.fit_generic_continuum`;
        ``'median_filter'`` usa um filtro de mediana móvel, cerca de
        vinte vezes mais rápido e sem dependências extras.
    :type method: str
    :param median_window: Largura da janela de mediana, em número de
        pontos. Deve ser bem maior que a largura típica das linhas.
    :type median_window: int
    :param verbose: Se ``True``, imprime o progresso a cada 1000
        espectros no modo ``'specutils'``.
    :type verbose: bool
    :returns: Matriz de fluxos normalizados, mesma forma da entrada.
    :rtype: numpy.ndarray
    :raises ValueError: Se ``method`` não for um dos valores aceitos.
    """
    if wavelength is None:
        wavelength = config.wavelength_grid()

    if method == "median_filter":
        from scipy.ndimage import median_filter

        continuo = median_filter(flux, size=(1, median_window), mode="nearest")
        continuo = np.where(continuo == 0, np.nan, continuo)
        return flux / continuo

    if method == "specutils":
        from astropy import units as u
        from specutils import Spectrum
        from specutils.fitting import fit_generic_continuum

        unidade = u.W / (u.m**2 * u.nm)
        eixo = wavelength * u.nm
        normalizado = np.empty_like(flux, dtype=float)

        for i in range(flux.shape[0]):
            espectro = Spectrum(spectral_axis=eixo, flux=flux[i] * unidade)
            modelo = fit_generic_continuum(espectro, median_window=median_window)
            normalizado[i] = (espectro / modelo(espectro.spectral_axis)).flux.value

            if verbose and (i + 1) % 1000 == 0:
                print(f"  normalizados {i + 1}/{flux.shape[0]} espectros")

        return normalizado

    raise ValueError(
        f"method='{method}' inválido; use 'specutils' ou 'median_filter'."
    )


# ---------------------------------------------------------------------------
# Medidas elementares
# ---------------------------------------------------------------------------


def band_mean(flux: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Média do fluxo dentro de uma janela de índices.

    :param flux: Matriz de fluxos de forma ``(n_estrelas, n_pontos)``.
    :type flux: numpy.ndarray
    :param start: Índice inicial da janela (inclusivo).
    :type start: int
    :param stop: Índice final da janela (exclusivo).
    :type stop: int
    :returns: Vetor com uma média por estrela.
    :rtype: numpy.ndarray
    :raises ValueError: Se a janela for vazia ou extrapolar o espectro.
    """
    if not 0 <= start < stop <= flux.shape[1]:
        raise ValueError(
            f"Janela ({start}, {stop}) inválida para espectros com "
            f"{flux.shape[1]} pontos."
        )
    return np.mean(flux[:, start:stop], axis=1)


def line_depth(
    flux_norm: np.ndarray,
    center_nm: float,
    half_width_nm: float = config.LINE_HALF_WIDTH_NM,
    wavelength: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Profundidade de uma linha no espectro normalizado pelo contínuo.

    Definida como :math:`1 - \\overline{F_{norm}}` na janela da linha.
    Valores positivos indicam absorção; negativos, emissão.

    :param flux_norm: Matriz de fluxos já normalizados pelo contínuo.
    :type flux_norm: numpy.ndarray
    :param center_nm: Comprimento de onda central da linha, em
        nanômetros.
    :type center_nm: float
    :param half_width_nm: Semilargura da janela de integração, em
        nanômetros.
    :type half_width_nm: float
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :returns: Vetor de profundidades, uma por estrela.
    :rtype: numpy.ndarray
    """
    inicio, fim = window_indices(center_nm, half_width_nm, wavelength)
    return 1.0 - band_mean(flux_norm, inicio, fim)


def band_ratio(
    flux: np.ndarray,
    numerator: Tuple[int, int],
    denominator: Tuple[int, int],
) -> np.ndarray:
    """Razão entre as médias de fluxo de duas janelas de índices.

    :param flux: Matriz de fluxos de forma ``(n_estrelas, n_pontos)``.
    :type flux: numpy.ndarray
    :param numerator: Par ``(inicio, fim)`` da janela do numerador.
    :type numerator: tuple[int, int]
    :param denominator: Par ``(inicio, fim)`` da janela do denominador.
    :type denominator: tuple[int, int]
    :returns: Vetor de razões, uma por estrela.
    :rtype: numpy.ndarray
    """
    num = band_mean(flux, *numerator)
    den = band_mean(flux, *denominator)
    return np.divide(num, den, out=np.full_like(num, np.nan), where=den != 0)


# ---------------------------------------------------------------------------
# Índice de três bandas (pseudo-largura equivalente)
# ---------------------------------------------------------------------------


def _band_slice(
    banda: Tuple[float, float],
    wavelength: np.ndarray,
) -> slice:
    """Converte uma banda em nanômetros na fatia de índices correspondente.

    :param banda: Par ``(inicio_nm, fim_nm)``, ambos inclusivos.
    :type banda: tuple[float, float]
    :param wavelength: Grade de comprimentos de onda, em nanômetros.
    :type wavelength: numpy.ndarray
    :returns: Fatia que seleciona os pontos dentro da banda.
    :rtype: slice
    :raises ValueError: Se a banda não contiver nenhum ponto da grade.
    """
    inicio = int(np.searchsorted(wavelength, banda[0], side="left"))
    fim = int(np.searchsorted(wavelength, banda[1], side="right"))

    if fim <= inicio:
        raise ValueError(
            f"A banda {banda[0]}–{banda[1]} nm não contém nenhum ponto da grade "
            f"({wavelength[0]:.0f}–{wavelength[-1]:.0f} nm)."
        )
    return slice(inicio, fim)


def line_index(
    flux: np.ndarray,
    bands: Dict[str, Tuple[float, float]],
    wavelength: Optional[np.ndarray] = None,
    in_angstrom: bool = True,
) -> np.ndarray:
    """Mede a pseudo-largura equivalente de uma feição, no estilo Lick/IDS.

    O pseudo-contínuo é definido **localmente**, por uma reta que liga os
    fluxos médios das duas bandas laterais, avaliados nos comprimentos de
    onda médios de cada banda. O índice é

    .. math::

       I = \\int_{\\lambda_1}^{\\lambda_2}
           \\left( 1 - \\frac{F(\\lambda)}{F_C(\\lambda)} \\right) d\\lambda

    integrado sobre a banda central. Valores positivos indicam absorção;
    negativos, emissão.

    Diferente de :func:`line_depth`, este índice **não** exige a
    normalização global pelo contínuo: ele traz o próprio contínuo de
    referência, o que o torna imune a erros do ajuste global — relevante
    nos espectros XP, onde a representação por funções de Hermite produz
    oscilações espúrias, sobretudo nas bordas da faixa.

    Quando apenas uma banda lateral é informada, o pseudo-contínuo é
    constante e igual à média dessa banda. Isso é necessário na região
    azul do espectro de estrelas OBA, onde a série de Balmer não deixa
    contínuo limpo dos dois lados.

    :param flux: Matriz de fluxos de forma ``(n_estrelas, n_pontos)``,
        em unidades arbitrárias mas consistentes. Use o fluxo
        **não** normalizado.
    :type flux: numpy.ndarray
    :param bands: Dicionário com a chave obrigatória ``'center'`` e ao
        menos uma das chaves ``'blue'`` e ``'red'``, cada valor sendo um
        par ``(inicio_nm, fim_nm)``.
    :type bands: dict[str, tuple[float, float]]
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :param in_angstrom: Se ``True``, devolve o índice em ångströms, como
        é convenção na literatura de índices espectrais; se ``False``,
        em nanômetros.
    :type in_angstrom: bool
    :returns: Vetor de pseudo-larguras equivalentes, uma por estrela.
    :rtype: numpy.ndarray
    :raises KeyError: Se ``'center'`` estiver ausente ou se nenhuma
        banda lateral for informada.
    :raises ValueError: Se alguma banda cair fora da grade.

    .. rubric:: Exemplo

    .. code-block:: python

        >>> ew = line_index(espectros.flux, config.LINE_BANDS["h_beta"])
        >>> ew[:3].round(2)
        array([ 8.14, 11.90, 13.02])
    """
    if wavelength is None:
        wavelength = config.wavelength_grid()

    if "center" not in bands:
        raise KeyError("A banda 'center' é obrigatória.")
    if "blue" not in bands and "red" not in bands:
        raise KeyError("Informe ao menos uma banda lateral ('blue' ou 'red').")

    fatia_central = _band_slice(bands["center"], wavelength)
    lam = wavelength[fatia_central]
    delta = float(np.mean(np.diff(wavelength)))

    laterais = []
    for lado in ("blue", "red"):
        if lado in bands:
            fatia = _band_slice(bands[lado], wavelength)
            laterais.append(
                (float(np.mean(wavelength[fatia])), np.mean(flux[:, fatia], axis=1))
            )

    if len(laterais) == 2:
        (lam_azul, f_azul), (lam_verm, f_verm) = laterais
        inclinacao = (f_verm - f_azul) / (lam_verm - lam_azul)
        continuo = (
            f_azul[:, None] + inclinacao[:, None] * (lam[None, :] - lam_azul)
        )
    else:
        continuo = np.repeat(laterais[0][1][:, None], lam.size, axis=1)

    razao = np.divide(
        flux[:, fatia_central],
        continuo,
        out=np.full_like(continuo, np.nan),
        where=continuo != 0,
    )
    indice = np.sum(1.0 - razao, axis=1) * delta

    return indice * 10.0 if in_angstrom else indice


def compute_line_indices(
    flux: np.ndarray,
    lines: Iterable[str] = config.DEFAULT_LINES,
    wavelength: Optional[np.ndarray] = None,
    bands: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
    in_angstrom: bool = True,
) -> pd.DataFrame:
    """Calcula o índice de três bandas de cada linha listada.

    :param flux: Matriz de fluxos **não** normalizados.
    :type flux: numpy.ndarray
    :param lines: Nomes das linhas a medir, presentes em ``bands``.
    :type lines: collections.abc.Iterable[str]
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :param bands: Dicionário de bandas. Se ``None``, usa
        :data:`rf_oba_stars.config.LINE_BANDS`.
    :type bands: dict or None
    :param in_angstrom: Unidade do resultado, repassada a
        :func:`line_index`.
    :type in_angstrom: bool
    :returns: Tabela com uma coluna ``'ew_<linha>'`` por linha pedida.
    :rtype: pandas.DataFrame
    :raises KeyError: Se alguma linha não tiver bandas definidas.
    """
    bands = config.LINE_BANDS if bands is None else bands

    colunas = {}
    for nome in lines:
        if nome not in bands:
            raise KeyError(
                f"Linha '{nome}' sem bandas definidas. Disponíveis: {sorted(bands)}"
            )
        colunas[f"ew_{nome}"] = line_index(
            flux, bands[nome], wavelength, in_angstrom=in_angstrom
        )

    return pd.DataFrame(colunas)


def balmer_jump(
    flux: np.ndarray,
    wavelength: Optional[np.ndarray] = None,
    bands: Optional[Dict[str, Tuple[float, float]]] = None,
) -> np.ndarray:
    """Mede o salto de Balmer, em magnitudes.

    A descontinuidade em 364,6 nm, produzida pela ionização do hidrogênio
    a partir do nível :math:`n = 2`, é o diagnóstico clássico de
    temperatura efetiva para estrelas OBA. Ao contrário das linhas
    individuais, é perfeitamente resolvida na amostragem do XP: a razão
    de fluxo entre as bandas cresce de forma monótona de O para A.

    .. math::

       D_B = 2{,}5 \\log_{10}
             \\left( \\overline{F_{vermelho}} / \\overline{F_{azul}} \\right)

    :param flux: Matriz de fluxos **não** normalizados.
    :type flux: numpy.ndarray
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :param bands: Dicionário com as chaves ``'blue'`` e ``'red'``. Se
        ``None``, usa
        :data:`rf_oba_stars.config.BALMER_JUMP_BANDS`.
    :type bands: dict[str, tuple[float, float]] or None
    :returns: Vetor com o salto de Balmer de cada estrela, em
        magnitudes.
    :rtype: numpy.ndarray
    """
    if wavelength is None:
        wavelength = config.wavelength_grid()
    bands = config.BALMER_JUMP_BANDS if bands is None else bands

    azul = np.mean(flux[:, _band_slice(bands["blue"], wavelength)], axis=1)
    vermelho = np.mean(flux[:, _band_slice(bands["red"], wavelength)], axis=1)

    razao = np.divide(
        vermelho, azul, out=np.full_like(azul, np.nan), where=azul > 0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        return 2.5 * np.log10(np.where(razao > 0, razao, np.nan))


def band_shift_scan(
    flux: np.ndarray,
    y: Sequence[str],
    line: str,
    shifts_nm: Sequence[float] = (-12, -10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12),
    wavelength: Optional[np.ndarray] = None,
    bands: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
) -> pd.DataFrame:
    """Testa a sensibilidade de um índice ao deslocamento da banda central.

    Ferramenta de validação empírica das bandas. Desloca a banda central
    em torno da posição nominal, mantendo as laterais fixas, e mede em
    cada posição o poder de separação entre as classes espectrais pela
    estatística F da análise de variância.

    Uma banda bem posicionada produz um **máximo local** de separação no
    deslocamento zero. Se a separação cresce monotonicamente até a borda
    da varredura, a banda está fora do lugar — foi exatamente o que
    ocorreu com as janelas herdadas do notebook original.

    :param flux: Matriz de fluxos não normalizados.
    :type flux: numpy.ndarray
    :param y: Classes espectrais, uma por estrela.
    :type y: collections.abc.Sequence[str]
    :param line: Nome da linha a testar.
    :type line: str
    :param shifts_nm: Deslocamentos a aplicar à banda central, em
        nanômetros.
    :type shifts_nm: collections.abc.Sequence[float]
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :param bands: Dicionário de bandas. Se ``None``, usa
        :data:`rf_oba_stars.config.LINE_BANDS`.
    :type bands: dict or None
    :returns: Tabela com as colunas ``deslocamento_nm``,
        ``centro_nm``, ``f_anova`` e a média do índice em cada classe.
    :rtype: pandas.DataFrame
    """
    from sklearn.feature_selection import f_classif

    if wavelength is None:
        wavelength = config.wavelength_grid()
    bands = config.LINE_BANDS if bands is None else bands

    base = bands[line]
    y = np.asarray(y)

    linhas = []
    for shift in shifts_nm:
        deslocada = dict(base)
        deslocada["center"] = (
            base["center"][0] + shift,
            base["center"][1] + shift,
        )
        try:
            valores = line_index(flux, deslocada, wavelength)
        except ValueError:
            continue

        finitos = np.isfinite(valores)
        f_stat = float(f_classif(valores[finitos, None], y[finitos])[0][0])

        registro = {
            "deslocamento_nm": float(shift),
            "centro_nm": float(np.mean(deslocada["center"])),
            "f_anova": f_stat,
        }
        for classe in config.CLASS_ORDER:
            mask = finitos & (y == classe)
            registro[f"media_{classe}"] = (
                float(np.mean(valores[mask])) if mask.any() else np.nan
            )
        linhas.append(registro)

    return pd.DataFrame(linhas)


# ---------------------------------------------------------------------------
# Conjuntos de atributos
# ---------------------------------------------------------------------------


def compute_line_features(
    flux_norm: np.ndarray,
    lines: Iterable[str] = config.DEFAULT_LINES,
    half_width_nm: float = config.LINE_HALF_WIDTH_NM,
    wavelength: Optional[np.ndarray] = None,
    catalog: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Calcula a profundidade das linhas listadas, uma coluna por linha.

    :param flux_norm: Matriz de fluxos normalizados pelo contínuo.
    :type flux_norm: numpy.ndarray
    :param lines: Nomes das linhas a medir, presentes em ``catalog``.
    :type lines: collections.abc.Iterable[str]
    :param half_width_nm: Semilargura das janelas, em nanômetros.
    :type half_width_nm: float
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :param catalog: Dicionário ``nome -> comprimento de onda (nm)``. Se
        ``None``, usa
        :data:`rf_oba_stars.config.LINE_CATALOG_NM`.
    :type catalog: dict[str, float] or None
    :returns: Tabela com uma coluna ``'depth_<linha>'`` por linha pedida.
    :rtype: pandas.DataFrame
    :raises KeyError: Se alguma linha não constar do catálogo.
    """
    catalog = config.LINE_CATALOG_NM if catalog is None else catalog

    colunas = {}
    for nome in lines:
        if nome not in catalog:
            raise KeyError(
                f"Linha '{nome}' ausente do catálogo. "
                f"Disponíveis: {sorted(catalog)}"
            )
        colunas[f"depth_{nome}"] = line_depth(
            flux_norm, catalog[nome], half_width_nm, wavelength
        )

    return pd.DataFrame(colunas)


def compute_legacy_indices(
    flux_norm: np.ndarray,
    windows: Optional[Dict[str, Dict[str, Tuple[int, int]]]] = None,
) -> pd.DataFrame:
    """Reproduz os índices do notebook original, com as janelas fixas.

    Mantido para permitir a comparação direta com os resultados
    anteriores. Os rótulos ``he_ii``, ``he_i`` e ``hbeta_deep`` **não**
    correspondem aos comprimentos de onda dessas linhas — use
    :func:`audit_legacy_windows` para verificar.

    :param flux_norm: Matriz de fluxos normalizados pelo contínuo.
    :type flux_norm: numpy.ndarray
    :param windows: Dicionário de janelas no formato de
        :data:`rf_oba_stars.config.LEGACY_INDEX_WINDOWS`. Se ``None``,
        usa o padrão.
    :type windows: dict or None
    :returns: Tabela com as colunas ``he_ii``, ``he_i``, ``mg_ii`` e
        ``hbeta_deep``.
    :rtype: pandas.DataFrame
    """
    windows = config.LEGACY_INDEX_WINDOWS if windows is None else windows

    colunas = {}
    for nome, janelas in windows.items():
        razao = band_ratio(flux_norm, janelas["numerador"], janelas["denominador"])
        # 'hbeta_deep' era definido no notebook como profundidade (1 - razão).
        colunas[nome] = 1.0 - razao if nome.endswith("_deep") else razao

    return pd.DataFrame(colunas)


def audit_legacy_windows(
    windows: Optional[Dict[str, Dict[str, Tuple[int, int]]]] = None,
    wavelength: Optional[np.ndarray] = None,
    catalog: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Traduz as janelas de índices herdadas para comprimentos de onda.

    Ferramenta de diagnóstico: para cada janela usada no notebook
    original, informa a faixa em nanômetros que ela realmente cobre e a
    linha do catálogo mais próxima do seu centro.

    :param windows: Dicionário de janelas. Se ``None``, usa
        :data:`rf_oba_stars.config.LEGACY_INDEX_WINDOWS`.
    :type windows: dict or None
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :param catalog: Catálogo de linhas. Se ``None``, usa
        :data:`rf_oba_stars.config.LINE_CATALOG_NM`.
    :type catalog: dict[str, float] or None
    :returns: Tabela com as colunas ``indice``, ``janela``,
        ``inicio_nm``, ``fim_nm``, ``centro_nm``, ``linha_mais_proxima``
        e ``distancia_nm``.
    :rtype: pandas.DataFrame
    """
    windows = config.LEGACY_INDEX_WINDOWS if windows is None else windows
    catalog = config.LINE_CATALOG_NM if catalog is None else catalog
    if wavelength is None:
        wavelength = config.wavelength_grid()

    nomes = np.array(list(catalog))
    valores = np.array(list(catalog.values()))

    linhas = []
    for indice, janelas in windows.items():
        for papel, (inicio, fim) in janelas.items():
            centro = float(np.mean(wavelength[inicio:fim]))
            distancias = np.abs(valores - centro)
            j = int(np.argmin(distancias))
            linhas.append(
                {
                    "indice": indice,
                    "janela": papel,
                    "inicio_nm": float(wavelength[inicio]),
                    "fim_nm": float(wavelength[fim - 1]),
                    "centro_nm": centro,
                    "linha_mais_proxima": nomes[j],
                    "distancia_nm": float(distancias[j]),
                }
            )

    return pd.DataFrame(linhas)


def broadband_features(
    flux: np.ndarray,
    wavelength: Optional[np.ndarray] = None,
    blue_band_nm: Tuple[float, float] = config.BLUE_BAND_NM,
    red_band_nm: Tuple[float, float] = config.RED_BAND_NM,
) -> pd.DataFrame:
    """Calcula os índices de banda larga: cor e inclinação do contínuo.

    Todos são medidos sobre o fluxo **não** normalizado, já que a
    normalização pelo contínuo destruiria justamente a informação de cor.

    :param flux: Matriz de fluxos de forma ``(n_estrelas, n_pontos)``.
    :type flux: numpy.ndarray
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :param blue_band_nm: Par ``(inicio, fim)``, em nanômetros, da faixa
        azul.
    :type blue_band_nm: tuple[float, float]
    :param red_band_nm: Par ``(inicio, fim)``, em nanômetros, da faixa
        vermelha.
    :type red_band_nm: tuple[float, float]
    :returns: Tabela com as colunas ``flux_blue``, ``flux_red``,
        ``blue_red_ratio`` (razão azul/vermelho), ``blue_red_diff``
        (diferença, equivalente ao ``'BP-RP'`` do notebook original),
        ``blue_red_norm`` (diferença normalizada,
        :math:`(azul-verm)/(azul+verm)`), ``color_mag`` (a razão em
        magnitudes, :math:`-2{,}5\\log_{10}(azul/verm)`), ``slope`` e
        ``slope_norm`` (inclinação dividida pelo fluxo médio).
    :rtype: pandas.DataFrame

    .. note::
       ``blue_red_diff`` e ``slope`` são grandezas **absolutas**: seus
       valores acompanham o brilho aparente da estrela e, em unidades do
       Gaia, ficam na casa de :math:`10^{-14}`. Duas consequências:
       carregam mais informação de distância e extinção do que de
       temperatura, e caem abaixo do limiar interno do divisor de
       árvores do scikit-learn (``FEATURE_THRESHOLD = 1e-7``), que passa
       a tratá-las como colunas constantes. Prefira ``color_mag``,
       ``blue_red_norm`` e ``slope_norm``, que são adimensionais.
    """
    if wavelength is None:
        wavelength = config.wavelength_grid()

    azul_ini = int(np.searchsorted(wavelength, blue_band_nm[0], side="left"))
    azul_fim = int(np.searchsorted(wavelength, blue_band_nm[1], side="left"))
    verm_ini = int(np.searchsorted(wavelength, red_band_nm[0], side="left"))
    verm_fim = int(np.searchsorted(wavelength, red_band_nm[1], side="right"))

    blue = band_mean(flux, azul_ini, azul_fim)
    red = band_mean(flux, verm_ini, verm_fim)

    razao = np.divide(blue, red, out=np.full_like(blue, np.nan), where=red != 0)
    with np.errstate(divide="ignore", invalid="ignore"):
        color_mag = -2.5 * np.log10(np.where(razao > 0, razao, np.nan))

    soma = blue + red
    blue_red_norm = np.divide(
        blue - red, soma, out=np.full_like(blue, np.nan), where=soma != 0
    )

    slope = (flux[:, -1] - flux[:, 0]) / (wavelength[-1] - wavelength[0])
    escala = np.mean(np.abs(flux), axis=1)
    slope_norm = np.divide(
        slope, escala, out=np.full_like(slope, np.nan), where=escala != 0
    ) * (wavelength[-1] - wavelength[0])

    return pd.DataFrame(
        {
            "flux_blue": blue,
            "flux_red": red,
            "blue_red_ratio": razao,
            "blue_red_diff": blue - red,
            "blue_red_norm": blue_red_norm,
            "color_mag": color_mag,
            "slope": slope,
            "slope_norm": slope_norm,
        }
    )


def build_feature_table(
    spectra: SpectraSet,
    flux_norm: np.ndarray,
    diagnostico: Optional[pd.DataFrame] = None,
    mode: str = "ew",
    lines: Iterable[str] = config.DEFAULT_LINES,
    only_reliable_ew: bool = False,
    half_width_nm: float = config.LINE_HALF_WIDTH_NM,
    include_broadband: Sequence[str] = ("color_mag", "blue_red_ratio"),
) -> pd.DataFrame:
    """Monta a tabela final de atributos, pronta para o aprendizado.

    :param spectra: Conjunto de espectros já filtrado por qualidade.
    :type spectra: SpectraSet
    :param flux_norm: Matriz de fluxos normalizados pelo contínuo,
        alinhada com ``spectra``.
    :type flux_norm: numpy.ndarray
    :param diagnostico: Tabela com ``C2`` e ``C3`` devolvida por
        :func:`rf_oba_stars.correction.correct_spectra`. Se ``None``,
        essas colunas não são incluídas.
    :type diagnostico: pandas.DataFrame or None
    :param mode: ``'ew'`` (recomendado) mede pseudo-larguras
        equivalentes pelo índice de três bandas sobre o fluxo calibrado;
        ``'physical'`` mede profundidades de linha sobre o espectro
        normalizado pelo contínuo; ``'legacy'`` reproduz as janelas fixas
        do notebook original; ``'both'`` combina ``'physical'`` e
        ``'legacy'``; ``'all'`` inclui os três conjuntos.
    :type mode: str
    :param only_reliable_ew: No modo ``'ew'``, restringe aos índices cujas
        bandas laterais amostram contínuo aceitável.
    :type only_reliable_ew: bool
    :param lines: Linhas a medir no modo ``'physical'``.
    :type lines: collections.abc.Iterable[str]
    :param half_width_nm: Semilargura das janelas de linha, em
        nanômetros.
    :type half_width_nm: float
    :param include_broadband: Colunas de :func:`broadband_features` a
        manter na tabela final.
    :type include_broadband: collections.abc.Sequence[str]
    :returns: Tabela com ``source_id``, os atributos escolhidos e a
        coluna alvo ``spectral_type`` na última posição.
    :rtype: pandas.DataFrame
    :raises ValueError: Se ``mode`` for inválido ou se ``flux_norm`` e
        ``spectra`` tiverem tamanhos diferentes.
    """
    if len(flux_norm) != len(spectra):
        raise ValueError(
            f"flux_norm tem {len(flux_norm)} linhas e spectra tem {len(spectra)}."
        )
    modos_validos = {"ew", "physical", "legacy", "both", "all"}
    if mode not in modos_validos:
        raise ValueError(f"mode='{mode}' inválido; use um de {sorted(modos_validos)}.")

    partes = [pd.DataFrame({"source_id": spectra.meta["source_id"].to_numpy()})]

    if mode in {"ew", "all"}:
        partes.append(
            compute_line_indices(
                spectra.flux,
                spectra.wavelength,
                only_reliable=only_reliable_ew,
            )
        )
    if mode in {"physical", "both", "all"}:
        partes.append(
            compute_line_features(
                flux_norm, lines, half_width_nm, spectra.wavelength
            )
        )
    if mode in {"legacy", "both", "all"}:
        partes.append(compute_legacy_indices(flux_norm))

    largas = broadband_features(spectra.flux, spectra.wavelength)
    partes.append(largas[list(include_broadband)])

    if diagnostico is not None:
        if len(diagnostico) != len(spectra):
            raise ValueError(
                f"diagnostico tem {len(diagnostico)} linhas e spectra tem "
                f"{len(spectra)}."
            )
        partes.append(diagnostico[["C2", "C3"]].reset_index(drop=True))

    partes.append(
        pd.DataFrame(
            {
                config.TARGET_COLUMN: spectra.meta[config.TARGET_COLUMN].to_numpy(),
            }
        )
    )

    tabela = pd.concat(partes, axis=1)
    tabela["source_id"] = tabela["source_id"].astype(np.int64)
    return tabela


# ---------------------------------------------------------------------------
# Índices de três bandas (pseudo-largura equivalente)
# ---------------------------------------------------------------------------


def _band_slice(
    band_nm: Tuple[float, float],
    wavelength: np.ndarray,
) -> slice:
    """Converte uma banda em nanômetros na fatia de índices correspondente.

    :param band_nm: Par ``(inicio, fim)`` em nanômetros, com ``fim``
        inclusivo.
    :type band_nm: tuple[float, float]
    :param wavelength: Grade de comprimentos de onda.
    :type wavelength: numpy.ndarray
    :returns: Fatia utilizável para indexar o eixo espectral.
    :rtype: slice
    :raises ValueError: Se a banda não contiver nenhum ponto da grade.
    """
    inicio = int(np.searchsorted(wavelength, band_nm[0], side="left"))
    fim = int(np.searchsorted(wavelength, band_nm[1], side="right"))

    if fim <= inicio:
        raise ValueError(
            f"Banda {band_nm} nm não contém pontos da grade "
            f"({wavelength[0]:.0f}–{wavelength[-1]:.0f} nm)."
        )
    return slice(inicio, fim)


def pseudo_continuum(
    flux: np.ndarray,
    wavelength: np.ndarray,
    blue_band_nm: Tuple[float, float],
    red_band_nm: Tuple[float, float],
    target_wavelength: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Traça o pseudo-contínuo local entre duas bandas laterais.

    O contínuo é a reta que liga o ponto médio de fluxo da banda azul ao
    da banda vermelha, avaliada nos comprimentos de onda pedidos — a
    definição do sistema Lick/IDS (Worthey et al. 1994; Trager et al.
    1998), também adotada nos índices de linha do LAMOST para estrelas OB.

    Por ser local, dispensa a normalização global pelo contínuo e fica
    imune aos erros desse ajuste, que nos espectros XP são reais nas
    bordas da faixa espectral.

    :param flux: Matriz de fluxos de forma ``(n_estrelas, n_pontos)``.
    :type flux: numpy.ndarray
    :param wavelength: Grade de comprimentos de onda, em nanômetros.
    :type wavelength: numpy.ndarray
    :param blue_band_nm: Banda lateral azul, em nanômetros.
    :type blue_band_nm: tuple[float, float]
    :param red_band_nm: Banda lateral vermelha, em nanômetros.
    :type red_band_nm: tuple[float, float]
    :param target_wavelength: Comprimentos de onda em que avaliar a reta.
        Se ``None``, usa a grade inteira.
    :type target_wavelength: numpy.ndarray or None
    :returns: Matriz de forma ``(n_estrelas, len(target_wavelength))`` com
        o pseudo-contínuo de cada estrela.
    :rtype: numpy.ndarray
    """
    fatia_azul = _band_slice(blue_band_nm, wavelength)
    fatia_verm = _band_slice(red_band_nm, wavelength)

    lam_azul = float(np.mean(wavelength[fatia_azul]))
    lam_verm = float(np.mean(wavelength[fatia_verm]))

    f_azul = np.mean(flux[:, fatia_azul], axis=1)[:, None]
    f_verm = np.mean(flux[:, fatia_verm], axis=1)[:, None]

    alvo = wavelength if target_wavelength is None else target_wavelength
    peso = (np.asarray(alvo, dtype=float) - lam_azul) / (lam_verm - lam_azul)

    return f_azul + (f_verm - f_azul) * peso[None, :]


def line_index(
    flux: np.ndarray,
    wavelength: Optional[np.ndarray] = None,
    central_band_nm: Tuple[float, float] = (478.0, 494.0),
    blue_band_nm: Tuple[float, float] = (462.0, 472.0),
    red_band_nm: Tuple[float, float] = (500.0, 512.0),
    unit: str = "angstrom",
) -> np.ndarray:
    """Mede a pseudo-largura equivalente de uma feição espectral.

    Implementa o índice de três bandas do sistema Lick/IDS:

    .. math::

       W = \\int_{\\lambda_1}^{\\lambda_2}
           \\left[1 - \\frac{F(\\lambda)}{F_C(\\lambda)}\\right] d\\lambda

    onde :math:`F_C` é o pseudo-contínuo de :func:`pseudo_continuum`.
    Valores positivos indicam absorção; negativos, emissão.

    Ao contrário de :func:`line_depth`, este índice **não** requer que o
    espectro tenha sido normalizado pelo contínuo — o contínuo é definido
    localmente pelas próprias bandas laterais. Passe o fluxo calibrado.

    :param flux: Matriz de fluxos de forma ``(n_estrelas, n_pontos)``.
    :type flux: numpy.ndarray
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :param central_band_nm: Banda central sobre a feição, em nanômetros.
    :type central_band_nm: tuple[float, float]
    :param blue_band_nm: Banda lateral azul, em nanômetros.
    :type blue_band_nm: tuple[float, float]
    :param red_band_nm: Banda lateral vermelha, em nanômetros.
    :type red_band_nm: tuple[float, float]
    :param unit: ``'angstrom'`` (padrão, convenção da literatura) ou
        ``'nm'``.
    :type unit: str
    :returns: Vetor de pseudo-larguras equivalentes, uma por estrela.
    :rtype: numpy.ndarray
    :raises ValueError: Se ``unit`` for inválido ou alguma banda estiver
        fora da grade.

    .. rubric:: Exemplo

    .. code-block:: python

        >>> w = line_index(flux, central_band_nm=(478, 494),
        ...                blue_band_nm=(462, 472),
        ...                red_band_nm=(500, 512))
    """
    if unit not in {"angstrom", "nm"}:
        raise ValueError(f"unit='{unit}' inválido; use 'angstrom' ou 'nm'.")

    if wavelength is None:
        wavelength = config.wavelength_grid()

    fatia = _band_slice(central_band_nm, wavelength)
    lam = wavelength[fatia]

    continuo = pseudo_continuum(
        flux, wavelength, blue_band_nm, red_band_nm, target_wavelength=lam
    )

    razao = np.divide(
        flux[:, fatia],
        continuo,
        out=np.full_like(continuo, np.nan),
        where=continuo != 0,
    )

    passo = float(np.mean(np.diff(wavelength)))
    largura = np.sum(1.0 - razao, axis=1) * passo

    return largura * 10.0 if unit == "angstrom" else largura


def compute_line_indices(
    flux: np.ndarray,
    wavelength: Optional[np.ndarray] = None,
    bands: Optional[Dict[str, Dict[str, object]]] = None,
    only_reliable: bool = False,
    unit: str = "angstrom",
    prefix: str = "ew_",
) -> pd.DataFrame:
    """Calcula a pseudo-largura equivalente de várias feições de uma vez.

    :param flux: Matriz de fluxos calibrados, **sem** normalização pelo
        contínuo.
    :type flux: numpy.ndarray
    :param wavelength: Grade de comprimentos de onda. Se ``None``, usa a
        grade padrão do Gaia.
    :type wavelength: numpy.ndarray or None
    :param bands: Definição das bandas, no formato de
        :data:`rf_oba_stars.config.LINE_BANDS`. Se ``None``, usa o padrão.
    :type bands: dict or None
    :param only_reliable: Se ``True``, mede apenas os índices marcados
        como confiáveis, isto é, aqueles cujas bandas laterais amostram
        contínuo aceitável.
    :type only_reliable: bool
    :param unit: Unidade das larguras, repassada a :func:`line_index`.
    :type unit: str
    :param prefix: Prefixo dos nomes das colunas.
    :type prefix: str
    :returns: Tabela com uma coluna ``'<prefix><feição>'`` por índice.
    :rtype: pandas.DataFrame
    """
    bands = config.LINE_BANDS if bands is None else bands
    if wavelength is None:
        wavelength = config.wavelength_grid()

    colunas = {}
    for nome, definicao in bands.items():
        if only_reliable and not definicao.get("confiavel", True):
            continue
        colunas[f"{prefix}{nome}"] = line_index(
            flux,
            wavelength,
            central_band_nm=definicao["central"],
            blue_band_nm=definicao["azul"],
            red_band_nm=definicao["vermelha"],
            unit=unit,
        )

    return pd.DataFrame(colunas)


def band_shift_scan(
    flux: np.ndarray,
    labels: Sequence[str],
    band_name: str,
    shifts_nm: Sequence[float] = (-8.0, -6.0, -4.0, -2.0, 0.0, 2.0, 4.0, 6.0, 8.0),
    wavelength: Optional[np.ndarray] = None,
    bands: Optional[Dict[str, Dict[str, object]]] = None,
    shift_sidebands: bool = False,
) -> pd.DataFrame:
    """Testa a sensibilidade de um índice ao deslocamento da banda central.

    Diagnóstico proposto para validar o posicionamento das bandas sem
    depender de inspeção visual. Um índice bem centrado sobre a feição
    apresenta **máximo local** de profundidade em deslocamento zero; um
    índice mal posicionado varia monotonicamente, sinal de que está
    medindo a asa de outra linha ou o contínuo.

    A coluna ``contraste`` traz a diferença entre a maior e a menor média
    de classe — o diagnóstico de centragem propriamente dito, que atinge
    o máximo quando a banda cobre a feição. A coluna ``separacao``
    normaliza essa diferença pelo desvio padrão combinado das classes
    extremas; ela mede poder discriminante, mas **não** serve para
    centrar a banda, porque cresce quando a dispersão cai mesmo com o
    índice medindo cada vez menos sinal.

    :param flux: Matriz de fluxos calibrados.
    :type flux: numpy.ndarray
    :param labels: Classe espectral de cada estrela.
    :type labels: collections.abc.Sequence[str]
    :param band_name: Nome do índice em ``bands``.
    :type band_name: str
    :param shifts_nm: Deslocamentos a aplicar, em nanômetros.
    :type shifts_nm: collections.abc.Sequence[float]
    :param wavelength: Grade de comprimentos de onda.
    :type wavelength: numpy.ndarray or None
    :param bands: Definição das bandas. Se ``None``, usa o padrão.
    :type bands: dict or None
    :param shift_sidebands: Se ``True``, desloca também as bandas
        laterais, mantendo a geometria do índice; se ``False``, move
        apenas a banda central.
    :type shift_sidebands: bool
    :returns: Tabela com ``deslocamento_nm``, ``centro_nm``, a média do
        índice por classe, ``contraste`` e ``separacao``.
    :rtype: pandas.DataFrame
    :raises KeyError: Se ``band_name`` não constar de ``bands``.
    """
    bands = config.LINE_BANDS if bands is None else bands
    if band_name not in bands:
        raise KeyError(f"Índice '{band_name}' ausente. Disponíveis: {sorted(bands)}")

    if wavelength is None:
        wavelength = config.wavelength_grid()

    definicao = bands[band_name]
    labels = np.asarray(labels)

    linhas = []
    for deslocamento in shifts_nm:
        central = tuple(v + deslocamento for v in definicao["central"])
        azul = definicao["azul"]
        vermelha = definicao["vermelha"]
        if shift_sidebands:
            azul = tuple(v + deslocamento for v in azul)
            vermelha = tuple(v + deslocamento for v in vermelha)

        valores = line_index(
            flux, wavelength,
            central_band_nm=central, blue_band_nm=azul, red_band_nm=vermelha,
        )

        registro = {
            "deslocamento_nm": deslocamento,
            "centro_nm": float(np.mean(central)),
        }
        for classe in config.CLASS_ORDER:
            mask = labels == classe
            registro[f"media_{classe}"] = (
                float(np.nanmean(valores[mask])) if mask.any() else np.nan
            )

        medias = [
            registro[f"media_{c}"]
            for c in config.CLASS_ORDER
            if not np.isnan(registro[f"media_{c}"])
        ]
        registro["contraste"] = (
            float(max(medias) - min(medias)) if len(medias) > 1 else np.nan
        )

        mask_a, mask_o = labels == "A", labels == "O"
        if mask_a.any() and mask_o.any():
            dp = np.sqrt(
                0.5 * (np.nanvar(valores[mask_a]) + np.nanvar(valores[mask_o]))
            )
            registro["separacao"] = float(
                abs(np.nanmean(valores[mask_a]) - np.nanmean(valores[mask_o])) / dp
            ) if dp > 0 else np.nan

        linhas.append(registro)

    return pd.DataFrame(linhas)
