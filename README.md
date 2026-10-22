# Automatic Classification of O-, B-, and A-Type Stars Using Low-Resolution Spectra and Machine Learning

[🇧🇷 Português](#português) | [🇬🇧 English](#english)

---

<a id="português"></a>

## Classificação Automática de Estrelas O, B e A com Espectros de Baixa Resolução e Aprendizado de Máquina

## Sobre o projeto

Este projeto investiga o uso de técnicas de aprendizado de máquina (*Machine Learning*) na classificação automática de estrelas dos tipos espectrais O, B e A, utilizando espectros ópticos de baixa resolução disponibilizados pela missão **Gaia, da Agência Espacial Europeia (ESA), Data Release 3 (DR3)**.

O trabalho propõe um pipeline computacional para processamento de dados astronômicos, extração de atributos espectrais e classificação supervisionada, explorando informações presentes nas linhas de absorção de hidrogênio e hélio e na distribuição global do fluxo espectral.

O objetivo é contribuir para o desenvolvimento de métodos reprodutíveis de classificação espectral, aplicáveis a levantamentos astronômicos de grande escala.

## Objetivos

- Desenvolver um método computacional para a classificação automática de estrelas dos tipos O, B e A.
- Construir uma amostra rotulada por meio do cruzamento de dados do Gaia DR3 com catálogos astronômicos de referência.
- Investigar atributos espectrais associados a linhas de hidrogênio e hélio e à distribuição global do fluxo.
- Avaliar o desempenho de um classificador supervisionado baseado em Random Forest.
- Documentar as etapas de processamento e modelagem para favorecer a reprodutibilidade científica.

## Dados utilizados

A pesquisa utiliza o produto `xp_sampled_mean_spectrum` do Gaia DR3, que disponibiliza espectros médios amostrados na faixa óptica e no infravermelho próximo.

| Característica | Descrição |
|---|---|
| Missão | Gaia — ESA |
| Liberação de dados | Gaia DR3 |
| Produto espectral | `xp_sampled_mean_spectrum` |
| Faixa espectral aproximada | 336–1020 nm |
| Passo de amostragem | 2 nm |
| Catálogos de referência | GOSC e SIMBAD |
| Amostra final | 8.686 espectros |
| Critério de qualidade | SNR ≥ 20 |

A identificação e a rotulagem das fontes foram realizadas pelo cruzamento de coordenadas celestes com o *Galactic O-Star Catalog* (GOSC) e o SIMBAD Astronomical Database.

## Metodologia

O pipeline contempla as seguintes etapas:

1. **Obtenção e integração dos dados:** seleção de espectros do Gaia DR3 e cruzamento com catálogos astronômicos para a composição da amostra rotulada.
2. **Controle de qualidade:** aplicação de um filtro baseado na razão sinal-ruído (*Signal-to-Noise Ratio*, SNR ≥ 20).
3. **Correção espectral:** aplicação da correção sistemática descrita por Huang et al. (2024), com derivação dos coeficientes C2 e C3.
4. **Engenharia de atributos:** extração de pseudo-larguras equivalentes em regiões associadas às linhas Hβ, He II 4686 e He I 4471, além de descritores da distribuição global do fluxo, como cor e inclinação do contínuo.
5. **Pré-processamento:** normalização das variáveis por meio do Min-Max Scaler.
6. **Treinamento e avaliação:** aplicação do algoritmo Random Forest, com separação dos dados em conjuntos de treinamento, ajuste e teste, além de validação cruzada estratificada.

A validação cruzada estratificada foi utilizada para avaliar a estabilidade do modelo e preservar, tanto quanto possível, a distribuição das classes entre os folds.

## Modelo de aprendizado de máquina

O classificador escolhido foi o **Random Forest**, um método supervisionado baseado na combinação de múltiplas árvores de decisão.

Sua utilização permite investigar relações não lineares entre os atributos espectrais e as classes de interesse, oferecendo uma abordagem adequada à avaliação de características espectrais combinadas.

## Resultados

Na validação cruzada estratificada com sete folds (*k* = 7), o modelo apresentou os seguintes resultados médios:

| Métrica | Resultado |
|---|---:|
| Acurácia média | 87% |
| F1-score macro | 0,76 |
| Número de folds | 7 |
| Amostra analisada | 8.686 espectros |

O F1-score macro atribui o mesmo peso a cada classe no cálculo da média, permitindo uma avaliação complementar à acurácia em cenários com distribuição desigual entre classes.

**Observação:** os valores apresentados correspondem à validação cruzada descrita no resumo do trabalho. Eles não devem ser interpretados como métricas finais do conjunto de teste independente.

## Tecnologias e ferramentas

- **Python** — linguagem de programação.
- **Pandas e NumPy** — manipulação e processamento de dados.
- **Scikit-learn** — pré-processamento, modelagem e avaliação.
- **Astropy e SciPy** — ferramentas para computação científica e análise de dados.
- **Matplotlib** — visualização de dados e resultados.
- **Gaia Archive** — acesso aos dados astronômicos.

A lista final deverá ser ajustada às bibliotecas efetivamente utilizadas na versão publicada do código.

## Reprodutibilidade científica

A proposta do projeto é documentar as principais etapas de processamento, construção de atributos e avaliação do modelo, facilitando a compreensão da metodologia e sua eventual reprodução por outros pesquisadores.

Para uma reprodução completa, devem ser registrados também os parâmetros do modelo, as sementes aleatórias, os critérios de cruzamento dos catálogos, as versões das bibliotecas e a estratégia utilizada para evitar vazamento de dados entre treinamento e avaliação.

## Referências e fontes de dados

- **European Space Agency (ESA).** Gaia Archive. Disponível em: https://gea.esac.esa.int/archive/
- **Gaia Collaboration et al. (2022).** Gaia Data Release 3: The Gaia mission. *Astronomy & Astrophysics*.
- **Galactic O-Star Catalog (GOSC).** Disponível em: https://gosc.cab.inta-csic.es/
- **SIMBAD Astronomical Database.** Disponível em: https://simbad.u-strasbg.fr/simbad/
- **Huang et al. (2024).** Referência do método de correção espectral utilizado — completar os dados bibliográficos a partir do artigo efetivamente consultado.

## Contexto acadêmico

Trabalho de conclusão de curso dedicado à aplicação de aprendizado de máquina à classificação espectral estelar, integrando processamento de dados, engenharia de atributos e computação científica em astronomia.

---

<a id="english"></a>

## Automatic Classification of O-, B-, and A-Type Stars Using Low-Resolution Spectra and Machine Learning

## Project overview

This project investigates the application of machine learning techniques to the automatic classification of O-, B-, and A-type stars using low-resolution optical spectra provided by the **European Space Agency's Gaia mission, Data Release 3 (DR3)**.

The study proposes a computational pipeline for astronomical data processing, spectral feature engineering, and supervised classification. It explores information contained in hydrogen and helium absorption lines, as well as the overall distribution of spectral flux.

The primary goal is to contribute to the development of reproducible spectral classification methods suitable for large-scale astronomical surveys.

## Objectives

- Develop a computational method for automatically classifying O-, B-, and A-type stars.
- Build a labeled sample by cross-matching Gaia DR3 sources with reference astronomical catalogs.
- Investigate spectral features associated with hydrogen and helium lines and the overall flux distribution.
- Evaluate a supervised classifier based on Random Forest.
- Document data processing and modeling procedures to support scientific reproducibility.

## Data sources

The study uses the Gaia DR3 `xp_sampled_mean_spectrum` product, which provides sampled mean spectra across the optical and near-infrared wavelength range.

| Characteristic | Description |
|---|---|
| Mission | Gaia — ESA |
| Data release | Gaia DR3 |
| Spectral product | `xp_sampled_mean_spectrum` |
| Approximate wavelength range | 336–1020 nm |
| Sampling step | 2 nm |
| Reference catalogs | GOSC and SIMBAD |
| Final sample | 8,686 spectra |
| Quality criterion | SNR ≥ 20 |

Source identification and labeling were performed by cross-matching celestial coordinates with the Galactic O-Star Catalog (GOSC) and the SIMBAD Astronomical Database.

## Methodology

The computational pipeline consists of the following stages:

1. **Data acquisition and integration:** selection of Gaia DR3 spectra and cross-matching with astronomical catalogs to construct the labeled sample.
2. **Quality control:** application of a signal-to-noise ratio threshold (SNR ≥ 20).
3. **Spectral correction:** application of the systematic correction described by Huang et al. (2024), including the derivation of coefficients C2 and C3.
4. **Feature engineering:** extraction of pseudo-equivalent widths from spectral regions associated with Hβ, He II 4686, and He I 4471, together with global flux descriptors such as color and continuum slope.
5. **Preprocessing:** normalization of variables using Min-Max Scaler.
6. **Model training and evaluation:** implementation of Random Forest, with separate training, tuning, and test subsets, alongside stratified cross-validation.

Stratified cross-validation was used to assess model performance while preserving, as far as possible, the class distribution across folds.

## Machine learning model

The selected classifier is **Random Forest**, a supervised learning algorithm that combines multiple decision trees.

This approach can capture nonlinear relationships between spectral features and target classes, making it suitable for investigating combinations of spectral characteristics.

## Results

Using seven-fold stratified cross-validation (*k* = 7), the model achieved the following mean results:

| Metric | Result |
|---|---:|
| Mean accuracy | 87% |
| Macro F1-score | 0.76 |
| Number of folds | 7 |
| Sample size | 8,686 spectra |

The macro F1-score assigns equal weight to each class when computing the average, providing a complementary assessment of accuracy when class distributions are imbalanced.

**Note:** these values refer to the cross-validation results reported in the study summary. They should not be interpreted as final performance metrics on an independent test set.

## Technologies and tools

- **Python** — programming language.
- **Pandas and NumPy** — data manipulation and processing.
- **Scikit-learn** — preprocessing, model training, and evaluation.
- **Astropy and SciPy** — scientific computing and data analysis.
- **Matplotlib** — data visualization.
- **Gaia Archive** — access to astronomical data.

The final list should reflect the libraries actually used in the published version of the code.

## Scientific reproducibility

The project aims to document the main data processing, feature engineering, and model evaluation stages, enabling other researchers to understand the methodology and potentially reproduce the experiments.

For full reproducibility, additional details should include model hyperparameters, random seeds, catalog cross-matching criteria, software versions, and procedures to prevent data leakage between training and evaluation.

## References and data sources

- **European Space Agency (ESA).** Gaia Archive. Available at: https://gea.esac.esa.int/archive/
- **Gaia Collaboration et al. (2022).** Gaia Data Release 3: The Gaia mission. *Astronomy & Astrophysics*.
- **Galactic O-Star Catalog (GOSC).** Available at: https://gosc.cab.inta-csic.es/
- **SIMBAD Astronomical Database.** Available at: https://simbad.u-strasbg.fr/simbad/
- **Huang et al. (2024).** Reference for the spectral correction method used in this study — complete the bibliographic details using the original paper.

## Academic context

Undergraduate thesis research applying machine learning to stellar spectral classification, combining data processing, feature engineering, and scientific computing in astronomy.
