# Atributos: justificativa física e fontes

Documento de referência para cada atributo usado na classificação espectral
O/B/A a partir dos espectros XP do Gaia DR3. Para cada um: o que mede, por
que é um diagnóstico legítimo, a fonte que sustenta essa afirmação e as
ressalvas impostas pela resolução do instrumento.

A separação entre conjunto **principal** e **atributos de teste** segue a
decisão de manter na classificação de referência apenas diagnósticos com
correspondência direta aos critérios MK, deixando os demais para
experimentos posteriores.

---

## 1. Conjunto principal

| Atributo | Grandeza | Diagnóstico de |
|---|---|---|
| `ew_blend_4686` | pseudo-largura equivalente, 462–476 nm | classe O |
| `ew_he_i_4471` | pseudo-largura equivalente, 440–454 nm | temperatura, O e B |
| `ew_h_beta` | pseudo-largura equivalente, 478–494 nm | série de Balmer, máxima em A |
| `color_mag` | −2,5·log₁₀(F_azul/F_verm) | temperatura efetiva |
| `slope_norm` | inclinação do contínuo normalizada | temperatura efetiva |
| `C2` | inclinação normalizada, 390–478 nm | temperatura efetiva |
| `C3` | inclinação normalizada, 478–602 nm | temperatura efetiva |

---

### 1.1 `ew_blend_4686` — região de He II 4686

**O que mede.** Pseudo-largura equivalente na banda 462–476 nm, com
laterais em 450–458 e 496–508 nm.

**Justificativa.** A presença de linhas de hélio ionizado é o que define
operacionalmente a classe O: estrelas B foram historicamente definidas como
aquelas que exibem He I mas não He II. He II 4686 é a linha de He II mais
acessível no óptico e figura entre os critérios canônicos do sistema MK
desde Morgan, Keenan & Kellman (1943).

**Fontes.**
- Morgan, W. W., Keenan, P. C. & Kellman, E. (1943). *An Atlas of Stellar
  Spectra*. University of Chicago Press. — definição do sistema MK.
- Walborn, N. R. (1971). ApJS 23, 257. — He II 4686 em emissão como critério
  de classe de luminosidade em estrelas O (sufixo `f`).
- Mathys, G. (1988). A&AS 76, 427. — quantificação: EW(4686) = 0,56 Å como
  fronteira entre gigantes e anãs para tipos O7 e anteriores.
- Sota, A. et al. (2011). ApJS 193, 24 (GOSSS). — critérios modernos de
  classificação de estrelas O.

**Ressalva obrigatória.** A linha de He II 4686 **não é resolvida** nos
espectros XP. Vincent et al. (2024, A&A) registram explicitamente que, entre
as linhas identificáveis nesses espectros, He II 4686 — tipicamente usada
para classificação — não aparece. O índice mede o *blend* da linha com a asa
azul de Hβ e com a forma local do contínuo. O nome do atributo deve refletir
isso, e o texto final precisa de uma frase explícita a respeito. Manter o
rótulo `he_ii_4686` seria uma afirmação que o instrumento não sustenta.

---

### 1.2 `ew_he_i_4471` — He I 4471

**O que mede.** Pseudo-largura equivalente na banda 440–454 nm, com laterais
em 428–436 e 458–466 nm.

**Justificativa.** É a linha de hélio neutro mais usada em classificação
espectral, e serve a dois propósitos distintos ao longo da sequência OBA:

- **Estrelas O.** O critério quantitativo clássico de tipo espectral é a
  razão entre as larguras equivalentes de He I 4471 e He II 4542. Em estrelas
  O9 a linha de He I é mais forte; a razão muda gradualmente até que He II
  4542 passa a dominar em O6 e tipos anteriores, e o logaritmo dessa razão
  determina o subtipo de forma unívoca.
- **Estrelas B.** A razão He I 4471 / Mg II 4481 é o indicador usual de tipo
  espectral entre B1 e B tardias. A intensidade do He I cresce até um máximo
  em B2 e depois enfraquece progressivamente, desaparecendo em torno de A0.

**Fontes.**
- Conti, P. S. & Alschuler, W. R. (1971). ApJ 170, 325. — critério
  quantitativo original, razão EW(He I 4471)/EW(He II 4542).
- Mathys, G. (1988, 1989). A&AS 76, 427; A&AS 81, 237. — calibração dos
  intervalos de razão por subtipo, de O3 a O9.7.
- Gray, R. O. & Corbally, C. J. (2009). *Stellar Spectral Classification*.
  Princeton University Press; e o *Digital Spectral Classification Atlas*
  do mesmo autor — razão He I 4471 / Mg II 4481 para estrelas B.
- Negueruela, I. et al. (2024). IACOB XII, A&A. — comportamento do espectro
  de He I ao longo da sequência B: cresce até B2, enfraquece e desaparece
  perto de A0.

**Ressalva.** Nesta resolução, He I 4471 e Mg II 4481 formam um blend único;
a separação que o critério MK exige não está disponível. Além disso, a banda
lateral azul do índice (428–436 nm) fica sobre a asa de Hγ, o que deprime o
pseudo-contínuo. O efeito é visível nos valores: em estrelas A o índice fica
negativo (−6,8 Å), o que não significa emissão, e sim que a banda central
está acima de um pseudo-contínuo rebaixado pelas asas de Balmer. O índice
continua discriminando fortemente as classes — é o atributo mais importante
do conjunto principal —, mas sua interpretação como largura equivalente de
He I é inválida sem calibração contra modelos.

---

### 1.3 `ew_h_beta` — Hβ

**O que mede.** Pseudo-largura equivalente na banda 478–494 nm, com laterais
em 462–472 e 500–512 nm.

**Justificativa.** As linhas de Balmer são o indicador de temperatura efetiva
mais robusto na faixa OBA. Sua intensidade cresce monotonicamente de O para
A em estrelas anãs, atingindo o máximo próximo a A2. A largura das asas de
Balmer é, adicionalmente, o critério principal de classe de luminosidade em
estrelas B.

**Fontes.**
- Morgan, Keenan & Kellman (1943), como acima.
- Negueruela, I. et al. (2024). IACOB XII, A&A. — em anãs, a intensidade das
  linhas de Balmer cresce monotonicamente com o tipo espectral e atinge o
  máximo perto de A2; a intensidade diminui com o aumento da luminosidade.
- Worthey, G. & Ottaviani, D. L. (1997). ApJS 111, 377. — índices de Balmer
  no sistema Lick, com bandas estreitas e largas.
- Gray & Corbally (2009) — as linhas de hidrogênio como melhores indicadores
  de temperatura efetiva em estrelas A.

**Ressalva.** A intensidade de Balmer é dupla-valorada em torno do máximo:
uma estrela B9 e uma A5 podem produzir valores semelhantes. É a razão
provável pela qual a fronteira B–A concentra a maior parte dos erros do
modelo, e deve ser mencionada na discussão. O mesmo vale, de forma ainda mais
acentuada, para Hα, cuja largura equivalente atinge o máximo em torno de A1.

---

### 1.4 `color_mag` — índice de cor

**O que mede.** −2,5·log₁₀ da razão entre o fluxo médio em 336–554 nm e o
fluxo médio em 802–1020 nm, ou seja, uma diferença de magnitudes sintéticas
entre duas bandas largas.

**Justificativa.** A distribuição espectral de energia de uma estrela é, em
primeira aproximação, um corpo negro cuja inclinação depende da temperatura
efetiva. Índices de cor são, historicamente, o indicador fotométrico primário
de temperatura, e o próprio Gaia usa BP−RP como entrada principal na
estimativa de parâmetros estelares.

**Fontes.**
- Andrae, R. et al. (2023). A&A 674, A27 (Gaia DR3 Apsis). — uso da
  espectrofotometria e da cor na estimativa de T_eff.
- Zhang, X., Green, G. M. & Rix, H.-W. (2023). MNRAS 524, 1855. — parâmetros
  estelares derivados dos espectros BP/RP para 220 milhões de estrelas.
- Casagrande, L. & VandenBerg, D. A. (2018). MNRAS 479, L102. — calibração
  de cores sintéticas do Gaia contra temperatura efetiva.

**Ressalva importante.** A cor é degenerada com o avermelhamento
interestelar. Estrelas O e B são jovens e concentram-se no plano galáctico,
onde a extinção é maior; uma estrela O avermelhada pode ter cor observada
semelhante à de uma estrela A próxima. Este atributo, portanto, carrega
informação de temperatura **e** de extinção, e o modelo não tem como
separá-las sem uma estimativa independente de E(B−V). Vale registrar na
discussão como limitação, e como possível explicação para parte da confusão
entre classes.

---

### 1.5 `slope_norm` — inclinação do contínuo

**O que mede.** A diferença de fluxo entre as extremidades da faixa espectral,
dividida pelo fluxo médio e multiplicada pela extensão em comprimento de onda.
É, portanto, uma inclinação relativa e adimensional.

**Justificativa.** Mesma física do índice de cor — a forma do contínuo reflete
a temperatura efetiva —, mas medida sobre toda a faixa amostrada em vez de
duas bandas. Nas estrelas quentes tratadas aqui, a faixa óptica situa-se no
lado Rayleigh-Jeans do pico de emissão, onde a inclinação varia de forma
suave e monotônica com T_eff.

**Fontes.** As mesmas de `color_mag`. A construção específica é própria deste
projeto e não replica um índice publicado; deve ser descrita como tal.

**Nota técnica obrigatória.** A versão anterior deste atributo, calculada
sobre fluxos absolutos do Gaia (~10⁻¹⁴ W m⁻² nm⁻¹), tinha amplitude abaixo do
limiar interno do divisor de árvores do scikit-learn e era silenciosamente
ignorada pelo modelo. A normalização pelo fluxo médio resolve o problema e é
o motivo de o atributo se chamar `slope_norm` e não `slope`.

---

### 1.6 e 1.7 `C2` e `C3` — inclinações do espectro normalizado

**O que medem.** São produzidos pelo próprio pacote de correção sistemática,
a partir do espectro **original**, e definidos como diferenças de fluxo
normalizado entre faixas adjacentes, divididas pela separação em micrômetros:

- `C2`: entre 390–434 nm e 434–478 nm;
- `C3`: entre 478–540 nm e 540–602 nm.

A normalização usa o fluxo médio em 460–490 nm.

**Justificativa.** São, portanto, inclinações locais do contínuo no azul e no
verde — fisicamente da mesma natureza que `color_mag` e `slope_norm`, mas
medidas em faixas mais estreitas e mais sensíveis à região onde a
distribuição espectral das estrelas quentes muda mais depressa. A região de
`C2` cobre ainda a vizinhança do salto de Balmer, cuja amplitude é um
indicador clássico de temperatura e gravidade superficial em estrelas OBA.

**Fonte.**
- Huang, Y. et al. (2024). ApJS 271, 13. — definição e implementação; o
  código-fonte publicado é a referência precisa para as faixas acima.

**Ressalva.** C2 e C3 não foram concebidos como atributos de classificação:
são quantidades intermediárias do procedimento de correção. Usá-los é
legítimo e produtivo, mas o texto deve deixar claro de onde vêm. Note ainda
que, por serem inclinações do contínuo, eles são parcialmente redundantes com
`color_mag` e `slope_norm` — o experimento P02 mostra que removê-los custa
apenas 0,016 em F1 macro.

---

## 2. Atributos de teste (pós-classificação principal)

### 2.1 `ew_h_alpha` — Hα

Balmer, mesma justificativa de Hβ. Fica fora do conjunto principal por duas
razões: cai na região de superposição entre os fotômetros azul e vermelho do
Gaia, onde os dois canais são combinados e a calibração é mais delicada; e
sua largura equivalente é fortemente dupla-valorada, com máximo em torno de
A1. Weiler et al. (2026, XP-TEAL I) observam que Hα é melhor amostrada no
canal vermelho, dada a resolução mais alta ali.

### 2.2 `ew_h_gamma` — Hγ

Balmer, mesma justificativa. É o atributo mais importante quando incluído
(0,252 no conjunto estendido), mas suas duas bandas laterais estão
contaminadas: a azul por Hδ, a vermelha por He I 4471. Fica como teste
justamente porque o ganho é grande e a fundamentação é fraca — a combinação
que mais merece investigação.

### 2.3 `ew_ca_ii_k` — Ca II K

A linha K do cálcio ionizado torna-se uma feição notável a partir de A0 e
cresce rapidamente para tipos mais frios, sendo um marcador clássico de
estrelas A (Gray & Corbally 2009). A ressalva é significativa: em estrelas B
a linha K é frequentemente de origem **interestelar**, não fotosférica.
Numa amostra concentrada no plano galáctico, esse atributo pode estar medindo
extinção em vez de tipo espectral. Além disso, nesta resolução Ca II H, K e
Hε formam um blend único.

### 2.4 `ew_he_i_5876` — He I 5876

Hélio neutro no vermelho. A razão He I 5876 / He II 5411 corresponde bem à
sequência de tipos espectrais e foi proposta justamente como alternativa para
estrelas fortemente avermelhadas, em que a região azul-violeta é inacessível
(Walborn 1980). Como a amostra deste projeto contém estrelas O muito
avermelhadas, é um candidato natural a teste. Fica fora do conjunto principal
apenas porque `ew_he_i_4471` já ocupa o papel do diagnóstico de He I.

### 2.5 `blue_red_norm`

Diferença normalizada entre as bandas azul e vermelha, (F_azul − F_verm) /
(F_azul + F_verm). Mesma informação de `color_mag` em outra parametrização,
limitada ao intervalo de −1 a 1. Mantido como teste de sensibilidade à forma
funcional do índice de cor, não como diagnóstico independente.

---

## 3. Resultado da separação

Validação cruzada estratificada de sete dobras, 8.686 estrelas.

| ID | Conjunto | Nº | Acurácia | F1 macro |
|---|---|---|---|---|
| P01 | **Principal** | 7 | 0,869 | **0,764 ± 0,010** |
| P02 | Principal sem C2/C3 | 5 | 0,863 | 0,748 ± 0,019 |
| P03 | Principal sem cor nem inclinação | 5 | 0,864 | 0,746 ± 0,008 |
| P04 | Principal, só as três linhas | 3 | 0,842 | 0,677 ± 0,016 |
| P05 | **Estendido** | 12 | 0,888 | **0,824 ± 0,011** |
| P06 | Só os atributos de teste | 5 | 0,873 | 0,798 ± 0,011 |

No conjunto de teste, o modelo principal atinge acurácia de 0,873 e F1 macro
de 0,777, contra 0,900 e 0,860 do conjunto completo.

**A diferença não é uniforme entre as classes.** A perda concentra-se
inteiramente na classe O:

| Classe | F1 (principal) | F1 (estendido) | Diferença |
|---|---|---|---|
| O | 0,676 | 0,859 | −0,183 |
| B | 0,719 | 0,777 | −0,058 |
| A | 0,937 | 0,942 | −0,005 |

A revocação da classe O cai de 0,83 para 0,63: com o conjunto principal, 37 %
das estrelas O são classificadas como B, contra 17 % no conjunto estendido.

A leitura é coerente com a física. Removidos Hα, Hγ, Ca II K e He I 5876, o
único diagnóstico de alta temperatura que resta é a região de He II 4686 — a
mesma que a resolução do instrumento não resolve. O modelo passa a depender
quase inteiramente da forma do contínuo para separar O de B, e a cor é
degenerada com o avermelhamento justamente nas estrelas O, que são as mais
avermelhadas da amostra.

Isso é um resultado a ser reportado, não um problema a ser escondido: mostra
concretamente qual é o custo de restringir-se aos diagnósticos MK
tradicionais quando o instrumento não os resolve, e sustenta o argumento de
que os índices adicionais não são conveniência estatística, mas compensação
por uma limitação instrumental identificável.

---

## 4. Referências completas

Andrae, R. et al. (2023). *Gaia Data Release 3: Analysis of the Gaia BP/RP
spectra using the General Stellar Parameterizer from Photometry*.
Astronomy & Astrophysics 674, A27.

Casagrande, L. & VandenBerg, D. A. (2018). *On the use of Gaia magnitudes and
new tables of bolometric corrections*. MNRAS 479, L102.

Conti, P. S. & Alschuler, W. R. (1971). *Spectroscopic Studies of O-Type
Stars. I. Classification and Absolute Magnitudes*. ApJ 170, 325.

De Angeli, F. et al. (2023). *Gaia Data Release 3: Processing and validation
of BP/RP low-resolution spectral data*. Astronomy & Astrophysics 674, A2.

Gray, R. O. & Corbally, C. J. (2009). *Stellar Spectral Classification*.
Princeton University Press.

Huang, Y. et al. (2024). *A Comprehensive Correction of the Gaia DR3 XP
Spectra*. ApJS 271, 13.

Martins, F. et al. (2018). *Quantitative spectral classification of Galactic
O stars*. Astronomy & Astrophysics 616, A135.

Mathys, G. (1988). *Spectral classification of O stars by equivalent widths*.
A&AS 76, 427. Continuação em A&AS 81, 237 (1989).

Montegriffo, P. et al. (2023). *Gaia Data Release 3: External calibration of
BP/RP low-resolution spectroscopic data*. Astronomy & Astrophysics 674, A3.

Morgan, W. W., Keenan, P. C. & Kellman, E. (1943). *An Atlas of Stellar
Spectra, with an Outline of Spectral Classification*. University of Chicago
Press.

Negueruela, I. et al. (2024). *The IACOB project XII. A new grid of northern
standards for the spectral classification of B-type stars*. Astronomy &
Astrophysics, arXiv:2407.04163.

Sota, A. et al. (2011). *The Galactic O-Star Spectroscopic Survey. I.*
ApJS 193, 24.

Trager, S. C. et al. (1998). *Old stellar populations. VI. Absorption-line
spectra of galaxy nuclei and globular clusters*. ApJS 116, 1.

Vincent, O. et al. (2024). *Classification and parameterization of a large
Gaia sample of white dwarfs using XP spectra*. Astronomy & Astrophysics.

Walborn, N. R. (1971). *Some Spectroscopic Characteristics of the OB Stars*.
ApJS 23, 257.

Walborn, N. R. (1980). *Spectral classification of OB stars in the
yellow-green*. ApJS 44, 535.

Weiler, M. et al. (2026). *XP-TEAL: a Gaia XP Tool for Emission and
Absorption Lines. I.* Astronomy & Astrophysics, arXiv:2606.16520.

Worthey, G. et al. (1994). *Old stellar populations. V. Absorption feature
indices for the complete Lick/IDS sample of stars*. ApJS 94, 687.

Worthey, G. & Ottaviani, D. L. (1997). *Hγ and Hδ absorption features in
stars and stellar populations*. ApJS 111, 377.

Zhang, X., Green, G. M. & Rix, H.-W. (2023). *Parameters of 220 million stars
from Gaia BP/RP spectra*. MNRAS 524, 1855.
