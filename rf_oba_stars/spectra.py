"""Leitura, validação e controle de qualidade dos espectros XP.

Trata da entrada bruta do projeto — o arquivo ``oba_stars.fits`` (ou o
CSV equivalente) produzido pelo *crossmatch* entre Gaia DR3, GOSC e
SIMBAD — em que cada linha contém o espectro amostrado de uma estrela
armazenado como *string* no formato ``'[v1,v2,...]'``.

.. rubric:: Fluxo típico

.. code-block:: python

    espectros = load_spectra("oba_stars.fits")
    espectros = apply_snr_cut(espectros)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from . import config

PathLike = Union[str, Path]


@dataclass
class SpectraSet:
    """Conjunto de espectros e metadados alinhados linha a linha.

    :ivar flux: Matriz de fluxos de forma ``(n_estrelas, 343)``.
    :vartype flux: numpy.ndarray
    :ivar flux_error: Matriz de incertezas, mesma forma de ``flux``.
    :vartype flux_error: numpy.ndarray
    :ivar meta: Tabela de metadados com as colunas ``source_id``,
        ``spectral_type`` e ``magnitude``, na mesma ordem das linhas de
        ``flux``.
    :vartype meta: pandas.DataFrame
    :ivar wavelength: Grade de comprimentos de onda, em nanômetros.
    :vartype wavelength: numpy.ndarray
    """

    flux: np.ndarray
    flux_error: np.ndarray
    meta: pd.DataFrame
    wavelength: np.ndarray = field(default_factory=config.wavelength_grid)

    def __post_init__(self) -> None:
        """Valida a consistência entre fluxos, incertezas e metadados.

        :raises ValueError: Se as três estruturas não tiverem o mesmo
            número de linhas ou se o número de pontos espectrais não
            coincidir com o tamanho da grade de comprimentos de onda.
        """
        n_flux, n_pontos = self.flux.shape
        if self.flux_error.shape != self.flux.shape:
            raise ValueError(
                f"flux {self.flux.shape} e flux_error {self.flux_error.shape} "
                "têm formas diferentes."
            )
        if len(self.meta) != n_flux:
            raise ValueError(
                f"meta tem {len(self.meta)} linhas, mas flux tem {n_flux}."
            )
        if n_pontos != len(self.wavelength):
            raise ValueError(
                f"Espectros com {n_pontos} pontos, mas a grade tem "
                f"{len(self.wavelength)} comprimentos de onda."
            )

    def __len__(self) -> int:
        """Número de estrelas no conjunto.

        :returns: Quantidade de espectros armazenados.
        :rtype: int
        """
        return self.flux.shape[0]

    def select(self, mask: np.ndarray) -> "SpectraSet":
        """Devolve um novo conjunto contendo apenas as linhas selecionadas.

        Aplica a mesma máscara a fluxos, incertezas e metadados, o que
        elimina a principal fonte de erro do notebook original — filtrar
        cada estrutura separadamente e correr o risco de desalinhá-las.

        :param mask: Máscara booleana ou vetor de índices inteiros.
        :type mask: numpy.ndarray
        :returns: Novo conjunto com o subconjunto de estrelas.
        :rtype: SpectraSet
        """
        mask = np.asarray(mask)
        return SpectraSet(
            flux=self.flux[mask],
            flux_error=self.flux_error[mask],
            meta=self.meta.loc[mask].reset_index(drop=True)
            if mask.dtype == bool
            else self.meta.iloc[mask].reset_index(drop=True),
            wavelength=self.wavelength,
        )

    def add_meta(self, **colunas: Sequence) -> "SpectraSet":
        """Acrescenta colunas à tabela de metadados, no lugar.

        :param colunas: Pares ``nome=valores``, cada vetor com uma
            entrada por estrela.
        :type colunas: collections.abc.Sequence
        :returns: O próprio conjunto, permitindo encadeamento.
        :rtype: SpectraSet
        :raises ValueError: Se algum vetor tiver tamanho diferente do
            número de estrelas.
        """
        for nome, valores in colunas.items():
            if len(valores) != len(self):
                raise ValueError(
                    f"Coluna '{nome}' tem {len(valores)} valores, "
                    f"mas há {len(self)} estrelas."
                )
            self.meta[nome] = np.asarray(valores)
        return self


def parse_array_column(values: Sequence[str]) -> np.ndarray:
    """Converte uma coluna de *strings* ``'[v1,v2,...]'`` em matriz numérica.

    Usa :func:`numpy.fromstring`, que é uma ordem de grandeza mais rápido
    que ``ast.literal_eval`` para milhares de espectros e não executa
    código arbitrário.

    :param values: Sequência de *strings*, cada uma representando um
        vetor de fluxos.
    :type values: collections.abc.Sequence[str]
    :returns: Matriz de forma ``(n_linhas, n_pontos)`` do tipo
        ``float64``.
    :rtype: numpy.ndarray
    :raises ValueError: Se as linhas não tiverem todas o mesmo número de
        pontos.
    """
    linhas = [
        np.fromstring(str(v).strip().strip("[]"), sep=",", dtype=float)
        for v in values
    ]

    tamanhos = {len(linha) for linha in linhas}
    if len(tamanhos) > 1:
        raise ValueError(
            f"Espectros com números de pontos diferentes: {sorted(tamanhos)}."
        )

    return np.vstack(linhas)


def load_spectra(
    path: PathLike,
    hdu: int = 1,
    flux_column: str = "flux",
    error_column: str = "flux_error",
    id_column: str = "source_id",
    class_column: str = "classe",
    magnitude_column: str = "phot_g_mean_mag",
) -> SpectraSet:
    """Carrega os espectros XP a partir de um arquivo FITS ou CSV.

    O formato é detectado pela extensão: ``.fits`` e ``.fit`` são lidos
    com :mod:`astropy.io.fits`; qualquer outra extensão é tratada como
    CSV.

    :param path: Caminho do arquivo de entrada (``oba_stars.fits`` ou
        ``oba_stars.csv``).
    :type path: str or pathlib.Path
    :param hdu: Índice da extensão do FITS que contém a tabela.
    :type hdu: int
    :param flux_column: Nome da coluna com o espectro.
    :type flux_column: str
    :param error_column: Nome da coluna com as incertezas do espectro.
    :type error_column: str
    :param id_column: Nome da coluna com o identificador Gaia.
    :type id_column: str
    :param class_column: Nome da coluna com a classe espectral de
        referência.
    :type class_column: str
    :param magnitude_column: Nome da coluna com a magnitude G, necessária
        para a correção sistemática.
    :type magnitude_column: str
    :returns: Conjunto de espectros com metadados alinhados.
    :rtype: SpectraSet
    :raises FileNotFoundError: Se o arquivo não existir.
    :raises KeyError: Se alguma das colunas esperadas estiver ausente.

    .. rubric:: Exemplo

    .. code-block:: python

        >>> espectros = load_spectra("oba_stars.fits")
        >>> len(espectros)
        8862
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de espectros não encontrado: {path}")

    if path.suffix.lower() in {".fits", ".fit"}:
        from astropy.io import fits

        with fits.open(path) as hdul:
            tabela = hdul[hdu].data
            disponiveis = list(tabela.columns.names)
            colunas = {
                flux_column,
                error_column,
                id_column,
                class_column,
                magnitude_column,
            }
            faltando = colunas - set(disponiveis)
            if faltando:
                raise KeyError(
                    f"Colunas ausentes no FITS: {sorted(faltando)}. "
                    f"Disponíveis: {disponiveis}"
                )
            flux = parse_array_column(tabela[flux_column])
            flux_error = parse_array_column(tabela[error_column])
            meta = pd.DataFrame(
                {
                    "source_id": np.asarray(tabela[id_column], dtype=np.int64),
                    config.TARGET_COLUMN: np.asarray(tabela[class_column], dtype=str),
                    "magnitude": np.asarray(tabela[magnitude_column], dtype=float),
                }
            )
    else:
        tabela = pd.read_csv(path)
        faltando = {
            flux_column,
            error_column,
            id_column,
            class_column,
            magnitude_column,
        } - set(tabela.columns)
        if faltando:
            raise KeyError(
                f"Colunas ausentes no CSV: {sorted(faltando)}. "
                f"Disponíveis: {list(tabela.columns)}"
            )
        flux = parse_array_column(tabela[flux_column].to_numpy())
        flux_error = parse_array_column(tabela[error_column].to_numpy())
        meta = pd.DataFrame(
            {
                "source_id": tabela[id_column].astype(np.int64).to_numpy(),
                config.TARGET_COLUMN: tabela[class_column].astype(str).to_numpy(),
                "magnitude": tabela[magnitude_column].astype(float).to_numpy(),
            }
        )

    return SpectraSet(flux=flux, flux_error=flux_error, meta=meta)


def compute_snr(
    flux: np.ndarray,
    flux_error: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calcula a razão sinal-ruído ponto a ponto e a média por estrela.

    Pontos com incerteza nula ou negativa recebem SNR zero, evitando
    divisão por zero.

    :param flux: Matriz de fluxos de forma ``(n_estrelas, n_pontos)``.
    :type flux: numpy.ndarray
    :param flux_error: Matriz de incertezas, mesma forma de ``flux``.
    :type flux_error: numpy.ndarray
    :returns: Par ``(snr, snr_medio)``, com a matriz de SNR ponto a ponto
        e o vetor de SNR médio por estrela.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    snr = np.divide(
        flux,
        flux_error,
        out=np.zeros_like(flux, dtype=float),
        where=flux_error > 0,
    )
    return snr, np.mean(snr, axis=1)


def apply_snr_cut(
    spectra: SpectraSet,
    threshold: float = config.SNR_THRESHOLD,
    store_snr: bool = True,
) -> SpectraSet:
    """Remove espectros com razão sinal-ruído média abaixo do limiar.

    :param spectra: Conjunto de espectros a filtrar.
    :type spectra: SpectraSet
    :param threshold: SNR médio mínimo aceitável. O valor padrão de 20
        corresponde a uma qualidade considerada aceitável para os
        espectros XP.
    :type threshold: float
    :param store_snr: Se ``True``, grava o SNR médio na coluna
        ``'snr_mean'`` dos metadados antes de filtrar.
    :type store_snr: bool
    :returns: Novo conjunto contendo apenas os espectros aprovados.
    :rtype: SpectraSet
    """
    _, snr_mean = compute_snr(spectra.flux, spectra.flux_error)

    if store_snr:
        spectra.add_meta(snr_mean=snr_mean)

    return spectra.select(snr_mean >= threshold)


def summarize_spectra(spectra: SpectraSet) -> Dict[str, object]:
    """Resume o conjunto de espectros em um dicionário.

    :param spectra: Conjunto de espectros já carregado.
    :type spectra: SpectraSet
    :returns: Dicionário com as chaves ``'n_estrelas'``, ``'n_pontos'``,
        ``'faixa_nm'``, ``'distribuicao'``, ``'magnitude'`` e
        ``'fluxos_invalidos'`` (número de espectros com valores não
        finitos).
    :rtype: dict
    """
    classes = spectra.meta[config.TARGET_COLUMN].value_counts()
    classes = classes.reindex(list(config.CLASS_ORDER), fill_value=0)

    return {
        "n_estrelas": len(spectra),
        "n_pontos": spectra.flux.shape[1],
        "faixa_nm": (float(spectra.wavelength[0]), float(spectra.wavelength[-1])),
        "distribuicao": classes.to_dict(),
        "magnitude": {
            "min": float(spectra.meta["magnitude"].min()),
            "max": float(spectra.meta["magnitude"].max()),
            "mediana": float(spectra.meta["magnitude"].median()),
        },
        "fluxos_invalidos": int((~np.isfinite(spectra.flux)).any(axis=1).sum()),
    }


def wavelength_to_index(
    wavelength_nm: float,
    grid: Optional[np.ndarray] = None,
) -> int:
    """Converte um comprimento de onda no índice mais próximo da grade.

    :param wavelength_nm: Comprimento de onda desejado, em nanômetros.
    :type wavelength_nm: float
    :param grid: Grade de comprimentos de onda. Se ``None``, usa
        :func:`rf_oba_stars.config.wavelength_grid`.
    :type grid: numpy.ndarray or None
    :returns: Índice do ponto da grade mais próximo.
    :rtype: int

    .. rubric:: Exemplo

    .. code-block:: python

        >>> wavelength_to_index(486.1)   # H-beta
        75
    """
    if grid is None:
        grid = config.wavelength_grid()
    return int(np.argmin(np.abs(grid - wavelength_nm)))


def window_indices(
    center_nm: float,
    half_width_nm: float = config.LINE_HALF_WIDTH_NM,
    grid: Optional[np.ndarray] = None,
) -> Tuple[int, int]:
    """Converte uma janela em comprimento de onda em um par de índices.

    :param center_nm: Centro da janela, em nanômetros.
    :type center_nm: float
    :param half_width_nm: Semilargura da janela, em nanômetros.
    :type half_width_nm: float
    :param grid: Grade de comprimentos de onda. Se ``None``, usa
        :func:`rf_oba_stars.config.wavelength_grid`.
    :type grid: numpy.ndarray or None
    :returns: Par ``(inicio, fim)`` de índices, no estilo ``slice``
        (``fim`` exclusivo).
    :rtype: tuple[int, int]
    """
    if grid is None:
        grid = config.wavelength_grid()

    inicio = wavelength_to_index(center_nm - half_width_nm, grid)
    fim = wavelength_to_index(center_nm + half_width_nm, grid) + 1
    return inicio, fim
