"""Correção sistemática dos espectros XP e extração de C2 e C3.

Envelopa o pacote ``GaiaDR3XPspectracorrectionV1`` de Huang et al.
(2024), que corrige os erros sistemáticos dos espectros XP do Gaia DR3 —
mais severos no ultravioleta próximo, entre 336 e 400 nm.

.. rubric:: O que são C2 e C3

São inclinações do espectro normalizado, calculadas pelo próprio pacote
de correção a partir do espectro **original**:

* ``C2`` — diferença de fluxo normalizado entre as faixas de 390–434 nm
  e 434–478 nm, dividida pela separação em micrômetros;
* ``C3`` — o mesmo entre 478–540 nm e 540–602 nm.

Ambas são, portanto, medidas de inclinação do contínuo no azul e no
verde, fisicamente ligadas à temperatura efetiva — o que justifica seu
uso como atributos na classificação O/B/A.

.. rubric:: Confiabilidade

O pacote devolve um vetor ``caution`` com três itens, referentes às
correções dos termos C2, C3 e G. Valores ``0`` e ``1`` indicam correção
dentro da faixa aceitável; ``2`` indica extrapolação não confiável. As
correções são calibradas para :math:`-0.5 < BP-RP < 2`,
:math:`3 < G < 17{,}5` e :math:`E(B-V) < 0{,}8` — faixa que muitas
estrelas O avermelhadas ultrapassam, tornando a verificação de
``caution`` especialmente relevante neste projeto.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from . import config
from .spectra import SpectraSet

PathLike = Union[str, Path]


def _first_index(indices) -> int:
    """Devolve o primeiro elemento de um vetor de índices como inteiro.

    Substitui as chamadas ``int(np.where(...)[0])`` do pacote original,
    que dependiam de uma conversão implícita removida no NumPy 2.

    :param indices: Vetor de índices, tipicamente vindo de
        :func:`numpy.where`.
    :type indices: numpy.ndarray
    :returns: Primeiro índice do vetor.
    :rtype: int
    :raises IndexError: Se o vetor estiver vazio.
    """
    achatado = np.asarray(indices).ravel()
    if achatado.size == 0:
        raise IndexError("Nenhum índice encontrado na tabela de correção.")
    return int(achatado[0])


def load_correction_module(
    package_dir: Optional[PathLike] = None,
    numpy2_compat: bool = True,
):
    """Importa o módulo ``Gaia_Correction_V1``, opcionalmente de um diretório.

    O pacote publicado por Huang et al. (2024) foi escrito para NumPy 1
    e usa ``int(np.where(...)[0])`` em quinze pontos. No NumPy 2 essa
    conversão implícita foi removida e levanta ``TypeError: only
    0-dimensional arrays can be converted to Python scalars``. Com
    ``numpy2_compat=True``, o módulo é carregado a partir do código-fonte
    com essas chamadas reescritas em memória — os arquivos originais não
    são alterados.

    :param package_dir: Diretório que contém a pasta
        ``GaiaDR3XPspectracorrectionV1``. Se ``None``, o pacote precisa
        estar no ``PYTHONPATH``.
    :type package_dir: str or pathlib.Path or None
    :param numpy2_compat: Se ``True`` e a versão instalada do NumPy for
        2 ou superior, aplica a correção de compatibilidade descrita
        acima.
    :type numpy2_compat: bool
    :returns: O módulo de correção, pronto para uso.
    :rtype: types.ModuleType
    :raises ImportError: Se o pacote não puder ser importado ou
        localizado, com instruções na mensagem.
    """
    if package_dir is not None:
        raiz = str(Path(package_dir).resolve())
        if raiz not in sys.path:
            sys.path.insert(0, raiz)

    precisa_patch = numpy2_compat and int(np.__version__.split(".")[0]) >= 2

    if not precisa_patch:
        try:
            from GaiaDR3XPspectracorrectionV1 import Gaia_Correction_V1
        except ImportError as erro:  # pragma: no cover - depende do ambiente
            raise ImportError(
                "Pacote 'GaiaDR3XPspectracorrectionV1' não encontrado. "
                "Baixe-o em https://github.com/HiromonGON/GaiaXPcorrection e "
                "informe o diretório em 'package_dir'."
            ) from erro
        return Gaia_Correction_V1

    return _load_patched_module(package_dir)


def _load_patched_module(package_dir: Optional[PathLike]):
    """Carrega ``Gaia_Correction_V1`` com o remendo de compatibilidade.

    :param package_dir: Diretório que contém a pasta do pacote, ou
        ``None`` para procurá-lo no ``sys.path``.
    :type package_dir: str or pathlib.Path or None
    :returns: Módulo de correção carregado a partir do código corrigido.
    :rtype: types.ModuleType
    :raises ImportError: Se o arquivo ``Gaia_Correction_V1.py`` não for
        localizado.
    """
    import importlib.util
    import types

    candidatos = []
    if package_dir is not None:
        candidatos.append(Path(package_dir))
    candidatos += [Path(p) for p in sys.path if p]

    caminho = None
    for base in candidatos:
        tentativa = base / "GaiaDR3XPspectracorrectionV1" / "Gaia_Correction_V1.py"
        if tentativa.exists():
            caminho = tentativa
            break

    if caminho is None:
        raise ImportError(
            "Arquivo 'Gaia_Correction_V1.py' não encontrado. Baixe o pacote em "
            "https://github.com/HiromonGON/GaiaXPcorrection e informe o "
            "diretório que contém 'GaiaDR3XPspectracorrectionV1' em "
            "'package_dir'."
        )

    codigo = caminho.read_text(encoding="utf-8", errors="replace")
    codigo = codigo.replace("int(np.where(", "_first_index(np.where(")

    modulo = types.ModuleType("Gaia_Correction_V1_patched")
    modulo.__file__ = str(caminho)
    modulo._first_index = _first_index
    exec(compile(codigo, str(caminho), "exec"), modulo.__dict__)

    return modulo


def correct_spectra(
    spectra: SpectraSet,
    package_dir: Optional[PathLike] = None,
    truncation: bool = False,
    absolute_correction: bool = True,
    verbose: bool = True,
    log_every: int = 1000,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Aplica a correção sistemática a todos os espectros do conjunto.

    :param spectra: Conjunto de espectros a corrigir.
    :type spectra: SpectraSet
    :param package_dir: Diretório do pacote de correção, repassado a
        :func:`load_correction_module`.
    :type package_dir: str or pathlib.Path or None
    :param truncation: Repassado ao pacote; indica se os espectros foram
        gerados com truncamento no GaiaXPy.
    :type truncation: bool
    :param absolute_correction: Se ``True``, aplica também a calibração
        absoluta de fluxo.
    :type absolute_correction: bool
    :param verbose: Se ``True``, imprime o progresso periodicamente.
    :type verbose: bool
    :param log_every: Intervalo, em número de espectros, entre as
        mensagens de progresso.
    :type log_every: int
    :returns: Par ``(flux_corrigido, diagnostico)``. O primeiro é a
        matriz de espectros corrigidos, de mesma forma que
        ``spectra.flux``; o segundo é uma tabela com as colunas ``C2``,
        ``C3``, ``caution_C2``, ``caution_C3``, ``caution_G`` e
        ``correcao_confiavel``.
    :rtype: tuple[numpy.ndarray, pandas.DataFrame]

    .. rubric:: Exemplo

    .. code-block:: python

        >>> flux_cor, diag = correct_spectra(espectros, package_dir="./GaiaXPcorrection-main")
        >>> diag["correcao_confiavel"].mean()
        0.97
    """
    modulo = load_correction_module(package_dir)

    n = len(spectra)
    flux_corrigido = np.empty_like(spectra.flux, dtype=float)
    registros = []

    magnitudes = spectra.meta["magnitude"].to_numpy()

    for i in range(n):
        saida, caution, c2, c3 = modulo.correction(
            spectra.flux[i],
            magnitudes[i],
            spectra.flux_error[i],
            Truncation=truncation,
            absolute_correction=absolute_correction,
        )
        flux_corrigido[i] = np.asarray(saida, dtype=float)
        caution = np.asarray(caution, dtype=int).ravel()
        registros.append(
            {
                "C2": float(c2),
                "C3": float(c3),
                "caution_C2": int(caution[0]),
                "caution_C3": int(caution[1]),
                "caution_G": int(caution[2]),
            }
        )

        if verbose and (i + 1) % log_every == 0:
            print(f"  corrigidos {i + 1}/{n} espectros", flush=True)

    diagnostico = pd.DataFrame(registros)
    diagnostico["correcao_confiavel"] = diagnostico["caution_C2"].isin(
        config.CAUTION_OK_VALUES
    ) & diagnostico["caution_C3"].isin(config.CAUTION_OK_VALUES)

    return flux_corrigido, diagnostico


def save_correction(
    flux_corrigido: np.ndarray,
    diagnostico: pd.DataFrame,
    path: PathLike,
) -> Path:
    """Grava em disco o resultado da correção.

    A correção leva alguns minutos para milhares de espectros; guardar o
    resultado evita repeti-la a cada execução do pipeline.

    :param flux_corrigido: Matriz de espectros corrigidos.
    :type flux_corrigido: numpy.ndarray
    :param diagnostico: Tabela devolvida por :func:`correct_spectra`.
    :type diagnostico: pandas.DataFrame
    :param path: Caminho do arquivo ``.npz`` de destino.
    :type path: str or pathlib.Path
    :returns: Caminho do arquivo gravado.
    :rtype: pathlib.Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        flux_corrigido=flux_corrigido,
        colunas=np.array(list(diagnostico.columns), dtype=object),
        valores=diagnostico.to_numpy(),
    )
    return path


def load_correction(path: PathLike) -> Tuple[np.ndarray, pd.DataFrame]:
    """Recarrega um resultado de correção gravado por :func:`save_correction`.

    :param path: Caminho do arquivo ``.npz``.
    :type path: str or pathlib.Path
    :returns: Par ``(flux_corrigido, diagnostico)``.
    :rtype: tuple[numpy.ndarray, pandas.DataFrame]
    :raises FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cache de correção não encontrado: {path}")

    with np.load(path, allow_pickle=True) as dados:
        flux_corrigido = dados["flux_corrigido"]
        diagnostico = pd.DataFrame(
            dados["valores"], columns=list(dados["colunas"])
        )

    inteiros = ["caution_C2", "caution_C3", "caution_G"]
    diagnostico[inteiros] = diagnostico[inteiros].astype(int)
    diagnostico[["C2", "C3"]] = diagnostico[["C2", "C3"]].astype(float)
    diagnostico["correcao_confiavel"] = diagnostico["correcao_confiavel"].astype(bool)

    return flux_corrigido, diagnostico


def summarize_caution(diagnostico: pd.DataFrame) -> pd.DataFrame:
    """Tabula a distribuição dos códigos de confiabilidade da correção.

    :param diagnostico: Tabela devolvida por :func:`correct_spectra`.
    :type diagnostico: pandas.DataFrame
    :returns: Tabela com uma linha por código (0, 1, 2, 3) e uma coluna
        por termo corrigido (C2, C3 e G), contendo as contagens.
    :rtype: pandas.DataFrame
    """
    colunas = ["caution_C2", "caution_C3", "caution_G"]
    resumo = pd.DataFrame(
        {
            coluna: diagnostico[coluna].value_counts()
            for coluna in colunas
            if coluna in diagnostico
        }
    )
    resumo = resumo.reindex(range(4), fill_value=0)
    resumo.index.name = "codigo_caution"
    return resumo.fillna(0).astype(int)


def apply_caution_filter(
    spectra: SpectraSet,
    diagnostico: pd.DataFrame,
    flux_corrigido: Optional[np.ndarray] = None,
) -> Tuple[SpectraSet, pd.DataFrame, Optional[np.ndarray]]:
    """Descarta estrelas cuja correção de C2 ou C3 é considerada não confiável.

    Esta etapa não existia no notebook original: o vetor ``caution`` era
    calculado, mas nunca usado para filtrar a amostra.

    :param spectra: Conjunto de espectros.
    :type spectra: SpectraSet
    :param diagnostico: Tabela devolvida por :func:`correct_spectra`,
        alinhada linha a linha com ``spectra``.
    :type diagnostico: pandas.DataFrame
    :param flux_corrigido: Matriz de espectros corrigidos, filtrada em
        conjunto quando informada.
    :type flux_corrigido: numpy.ndarray or None
    :returns: Tripla ``(spectra, diagnostico, flux_corrigido)`` já
        filtrada de forma consistente. O terceiro elemento é ``None`` se
        nada tiver sido passado em ``flux_corrigido``.
    :rtype: tuple[SpectraSet, pandas.DataFrame, numpy.ndarray or None]
    :raises ValueError: Se ``diagnostico`` não tiver o mesmo número de
        linhas que ``spectra``.
    """
    if len(diagnostico) != len(spectra):
        raise ValueError(
            f"diagnostico tem {len(diagnostico)} linhas e spectra tem {len(spectra)}."
        )

    mask = diagnostico["correcao_confiavel"].to_numpy()

    return (
        spectra.select(mask),
        diagnostico.loc[mask].reset_index(drop=True),
        flux_corrigido[mask] if flux_corrigido is not None else None,
    )
