"""Constantes, hiperparâmetros e definições espectrais do projeto.

Este módulo concentra todos os valores fixos usados na classificação
espectral de estrelas OBA — da grade de comprimentos de onda dos
espectros XP até os hiperparâmetros da floresta aleatória — de modo que
qualquer alteração seja feita em um único lugar e a reprodutibilidade
dos resultados fique explícita.

.. note::
   As sementes aleatórias reproduzem exatamente os notebooks originais:
   ``57`` para as divisões dos dados e ``71`` para a floresta aleatória.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Grade espectral do Gaia DR3 (xp_sampled_mean_spectrum)
# ---------------------------------------------------------------------------

#: Comprimento de onda inicial da grade amostrada, em nanômetros.
WAVELENGTH_START_NM: float = 336.0

#: Passo da grade amostrada, em nanômetros.
WAVELENGTH_STEP_NM: float = 2.0

#: Número de pontos da grade amostrada.
N_WAVELENGTH_POINTS: int = 343


def wavelength_grid() -> np.ndarray:
    """Retorna a grade de comprimentos de onda dos espectros XP amostrados.

    Conforme a documentação do Gaia DR3, todos os espectros médios são
    amostrados nas mesmas 343 posições absolutas, de 336 a 1020 nm com
    passo de 2 nm.

    :returns: Vetor de 343 comprimentos de onda, em nanômetros.
    :rtype: numpy.ndarray
    """
    return (
        WAVELENGTH_START_NM
        + WAVELENGTH_STEP_NM * np.arange(N_WAVELENGTH_POINTS, dtype=float)
    )


# ---------------------------------------------------------------------------
# Qualidade dos espectros
# ---------------------------------------------------------------------------

#: Razão sinal-ruído média mínima para que um espectro seja aceito.
SNR_THRESHOLD: float = 20.0

#: Valores de ``caution`` (itens 1 e 2) considerados confiáveis pelo
#: pacote de correção de Huang et al. (2024). O item 3 (termo G) não
#: restringe a validade de C2 e C3.
CAUTION_OK_VALUES: Tuple[int, ...] = (0, 1)

# ---------------------------------------------------------------------------
# Bandas largas para cor e inclinação
# ---------------------------------------------------------------------------

#: Faixa azul usada no índice de cor, em nanômetros (índices 0:110).
BLUE_BAND_NM: Tuple[float, float] = (336.0, 554.0)

#: Faixa vermelha usada no índice de cor, em nanômetros (índices 233:343).
RED_BAND_NM: Tuple[float, float] = (802.0, 1020.0)

# ---------------------------------------------------------------------------
# Linhas espectrais de interesse (comprimento de onda de repouso, no ar)
# ---------------------------------------------------------------------------

#: Comprimentos de onda de repouso, em nanômetros, das linhas usadas na
#: classificação O/B/A. Na resolução dos espectros XP amostrados
#: (2 nm por ponto) essas linhas aparecem misturadas ao contínuo e às
#: linhas vizinhas: os índices medem *regiões*, não linhas isoladas.
LINE_CATALOG_NM: Dict[str, float] = {
    "ca_ii_k": 393.4,
    "h_delta": 410.2,
    "h_gamma": 434.0,
    "he_i_4471": 447.1,
    "he_ii_4686": 468.6,
    "h_beta": 486.1,
    "he_i_5876": 587.6,
    "h_alpha": 656.3,
    "he_i_6678": 667.8,
}

#: Semilargura padrão, em nanômetros, das janelas centradas nas linhas.
LINE_HALF_WIDTH_NM: float = 6.0

#: Conjunto padrão de linhas usadas para construir os atributos.
DEFAULT_LINES: Tuple[str, ...] = (
    "h_delta",
    "h_gamma",
    "he_i_4471",
    "he_ii_4686",
    "h_beta",
    "he_i_5876",
    "h_alpha",
)

#: Bandas do índice do salto de Balmer, em nanômetros. A descontinuidade
#: em 364,6 nm é o diagnóstico clássico de temperatura para estrelas OBA
#: e, ao contrário das linhas individuais, é perfeitamente resolvida na
#: amostragem do XP. Medida como
#: :math:`D_B = 2{,}5 \log_{10}(F_{vermelho} / F_{azul})`.
BALMER_JUMP_BANDS: Dict[str, Tuple[float, float]] = {
    "blue": (344.0, 360.0),
    "red": (400.0, 416.0),
}

#: Bandas dos índices de três bandas, em nanômetros, no estilo Lick/IDS
#: (Faber et al. 1985; Worthey et al. 1994; Trager et al. 1998), adaptadas
#: à resolução dos espectros XP (:math:`R \\sim 20`–70, ou seja, de 7 a
#: 24 nm por elemento de resolução em torno de H\\ :math:`\\beta`).
#:
#: Cada entrada traz a banda ``'center'``, sobre a feição, e uma ou duas
#: bandas laterais (``'blue'`` e ``'red'``) que definem o
#: pseudo-contínuo. Linhas na região azul, congestionada pela série de
#: Balmer, usam apenas a lateral vermelha: não existe contínuo limpo do
#: lado azul em estrelas OBA.
#:
#: .. warning::
#:    O índice ``he_ii_4686`` mede um **blend**, não a linha isolada. Na
#:    resolução do XP a linha de He II 468,6 nm não é individualmente
#:    detectável (Vincent et al. 2024, A&A 682, A5); a banda central
#:    cobre a mistura de He II com a asa azul de H\\ :math:`\\beta` e a
#:    forma local do contínuo.
LINE_BANDS: Dict[str, Dict[str, Tuple[float, float]]] = {
    "h_delta": {
        "center": (404.0, 416.0),
        "blue": (396.0, 402.0),
        "red": (418.0, 426.0),
    },
    "h_gamma": {
        "center": (428.0, 440.0),
        "blue": (416.0, 424.0),
        "red": (452.0, 460.0),
    },
    "he_i_4471": {
        "center": (441.0, 453.0),
        "blue": (418.0, 426.0),
        "red": (456.0, 462.0),
    },
    "he_ii_4686": {
        "center": (462.0, 476.0),
        "blue": (454.0, 460.0),
        "red": (500.0, 512.0),
    },
    "h_beta": {
        "center": (478.0, 494.0),
        "blue": (462.0, 472.0),
        "red": (500.0, 512.0),
    },
    "he_i_5876": {
        "center": (580.0, 594.0),
        "blue": (564.0, 574.0),
        "red": (600.0, 610.0),
    },
    "h_alpha": {
        "center": (648.0, 666.0),
        "blue": (620.0, 636.0),
        "red": (690.0, 704.0),
    },
}

#: Janelas de índices usadas no notebook original ``OBA_Stars_Pipeline``.
#: Preservadas apenas para reprodutibilidade — os rótulos **não**
#: correspondem aos comprimentos de onda das linhas homônimas
#: (ver :func:`rf_oba_stars.features.audit_legacy_windows`).
LEGACY_INDEX_WINDOWS: Dict[str, Dict[str, Tuple[int, int]]] = {
    "he_ii": {"numerador": (116, 120), "denominador": (122, 126)},
    "he_i": {"numerador": (109, 113), "denominador": (114, 118)},
    "mg_ii": {"numerador": (114, 118), "denominador": (119, 123)},
    "hbeta_deep": {"numerador": (146, 156), "denominador": (160, 170)},
}

# ---------------------------------------------------------------------------
# Tabela de atributos
# ---------------------------------------------------------------------------

#: Nome da coluna alvo (classe espectral: ``'O'``, ``'B'`` ou ``'A'``).
TARGET_COLUMN: str = "spectral_type"

#: Colunas identificadoras que não devem ser usadas como atributos.
ID_COLUMNS: Tuple[str, ...] = ("source_id",)

#: Ordem canônica das classes espectrais (do mais quente para o mais frio).
CLASS_ORDER: Tuple[str, ...] = ("O", "B", "A")

#: Colunas sobre as quais a remoção de outliers por IQR é aplicada.
OUTLIER_COLUMNS: Tuple[str, ...] = ("C2", "C3")

#: Multiplicador do intervalo interquartil na regra de Tukey.
IQR_FACTOR: float = 1.5

# ---------------------------------------------------------------------------
# Divisão dos dados e modelo
# ---------------------------------------------------------------------------

#: Semente usada em todas as divisões de dados (treino/validação/teste e K-Fold).
RANDOM_STATE_SPLIT: int = 57

#: Semente usada na inicialização do :class:`~sklearn.ensemble.RandomForestClassifier`.
RANDOM_STATE_MODEL: int = 71

#: Fração do conjunto completo reservada para teste.
TEST_SIZE: float = 0.15

#: Fração do conjunto temporário (treino + validação) reservada para validação.
#: O valor ``0.1765`` resulta em aproximadamente 15% do conjunto original.
VAL_SIZE_OF_TEMP: float = 0.1765

#: Número de dobras da validação cruzada estratificada.
N_SPLITS: int = 7

#: Métrica usada na validação cruzada.
CV_SCORING: str = "accuracy"

#: Hiperparâmetros do classificador Random Forest.
RANDOM_FOREST_PARAMS: Dict[str, Any] = {
    "n_estimators": 101,
    "criterion": "gini",
    "max_features": "sqrt",
    "max_depth": 25,
    "class_weight": "balanced",
    "random_state": RANDOM_STATE_MODEL,
}


# ---------------------------------------------------------------------------
# Índices espectrais de três bandas (sistema Lick/IDS adaptado)
# ---------------------------------------------------------------------------

#: Definição das bandas de cada índice, em nanômetros. Cada entrada traz a
#: banda lateral azul, a banda central sobre a feição, a banda lateral
#: vermelha e um sinalizador de confiabilidade.
#:
#: As larguras seguem a resolução dos espectros XP (:math:`R \\sim 20`–70,
#: ou seja, 7 a 24 nm por elemento de resolução em torno de H\\ :sub:`β`):
#: as bandas centrais têm 14 a 18 nm e as laterais ficam afastadas por
#: cerca de 1,5 vezes a largura à meia altura. Bandas do sistema Lick
#: original ou do LAMOST, com 2 a 5 nm, seriam estreitas demais.
#:
#: ``confiavel=False`` marca índices cujas bandas laterais caem sobre
#: outras linhas — inevitável no azul de estrelas OBA, onde a série de
#: Balmer não deixa contínuo verdadeiro. Eles continuam sendo medidas
#: instrumentais úteis, mas não são larguras equivalentes físicas.
LINE_BANDS: Dict[str, Dict[str, Any]] = {
    "h_alpha": {
        "azul": (620.0, 636.0),
        "central": (648.0, 666.0),
        "vermelha": (690.0, 704.0),
        "confiavel": True,
        "nota": "região de superposição BP/RP; melhor amostrada no RP",
    },
    "h_beta": {
        "azul": (462.0, 472.0),
        "central": (478.0, 494.0),
        "vermelha": (500.0, 512.0),
        "confiavel": True,
        "nota": "",
    },
    "he_i_5876": {
        "azul": (564.0, 574.0),
        "central": (580.0, 594.0),
        "vermelha": (600.0, 610.0),
        "confiavel": True,
        "nota": "",
    },
    "h_gamma": {
        "azul": (416.0, 424.0),
        "central": (426.0, 442.0),
        "vermelha": (452.0, 460.0),
        "confiavel": False,
        "nota": "lateral azul contaminada por H-delta, vermelha por He I 4471",
    },
    "ca_ii_k": {
        "azul": (376.0, 384.0),
        "central": (386.0, 400.0),
        "vermelha": (402.0, 408.0),
        "confiavel": False,
        "nota": "blend de Ca II H+K com H-epsilon nesta resolução",
    },
    "he_i_4471": {
        "azul": (428.0, 436.0),
        "central": (440.0, 454.0),
        "vermelha": (458.0, 466.0),
        "confiavel": False,
        "nota": "lateral azul cai sobre H-gamma",
    },
    "blend_4686": {
        "azul": (450.0, 458.0),
        "central": (462.0, 476.0),
        "vermelha": (496.0, 508.0),
        "confiavel": False,
        "nota": (
            "He II 4686 é invisível na resolução XP; o índice mede o blend "
            "com a asa azul de H-beta e a forma do contínuo. A lateral "
            "vermelha fica além de H-beta"
        ),
    },
}

#: Índices cujas bandas laterais amostram contínuo aceitável.
RELIABLE_LINE_INDICES: Tuple[str, ...] = tuple(
    nome for nome, banda in LINE_BANDS.items() if banda["confiavel"]
)


# ---------------------------------------------------------------------------
# Conjuntos de atributos
# ---------------------------------------------------------------------------

#: Conjunto principal, usado na classificação de referência do projeto.
#:
#: A composição segue os critérios de classificação MK, restringindo-se a
#: diagnósticos com justificativa na literatura de classificação de
#: estrelas OBA (ver ``atributos_e_fontes.md``):
#:
#: * ``ew_blend_4686`` — região de He II 4686, critério clássico da classe O;
#: * ``ew_he_i_4471`` — He I 4471, critério de temperatura de O e de B;
#: * ``ew_h_beta`` — H\\ :sub:`β`, série de Balmer, máxima em torno de A2;
#: * ``color_mag`` — índice de cor, indicador de temperatura efetiva;
#: * ``slope_norm`` — inclinação do contínuo, indicador de temperatura;
#: * ``C2`` e ``C3`` — inclinações do contínuo no azul e no verde.
FEATURES_PRINCIPAL: Tuple[str, ...] = (
    "ew_blend_4686",
    "ew_he_i_4471",
    "ew_h_beta",
    "color_mag",
    "slope_norm",
    "C2",
    "C3",
)

#: Atributos adicionais, reservados para experimentos posteriores à
#: classificação principal. Cada um tem fundamento físico, mas foi mantido
#: fora do conjunto de referência para preservar a correspondência estrita
#: com os critérios MK.
FEATURES_EXTRA: Tuple[str, ...] = (
    "ew_h_alpha",
    "ew_h_gamma",
    "ew_ca_ii_k",
    "ew_he_i_5876",
    "blue_red_norm",
)

#: Conjunto principal acrescido dos atributos de teste.
FEATURES_ESTENDIDO: Tuple[str, ...] = FEATURES_PRINCIPAL + FEATURES_EXTRA

#: Índices necessários para montar o conjunto principal.
LINES_PRINCIPAL: Tuple[str, ...] = ("blend_4686", "he_i_4471", "h_beta")

#: Conjuntos nomeados, para uso direto em
#: :func:`rf_oba_stars.experiments.run_experiments`.
FEATURE_SETS: Dict[str, Tuple[str, ...]] = {
    "principal": FEATURES_PRINCIPAL,
    "extra": FEATURES_EXTRA,
    "estendido": FEATURES_ESTENDIDO,
}
