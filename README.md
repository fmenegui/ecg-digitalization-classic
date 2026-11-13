# ecgdig — digitalizador clássico de ECG (sem deep learning)

Converte a imagem de um ECG em papel (scan, PDF rasterizado ou foto) em sinais de 12 derivações em mV, a 500 Hz, usando apenas visão computacional clássica: geometria da grade, morfologia matemática, programação dinâmica e restrições fisiológicas. Não há nenhum modelo treinado.

Há também um [experimento opcional de U-Net com pesos programados](docs/programmed-unet.md)
para apoiar a segmentação, sem treino por gradiente. Use `--trace-support programmed-unet`
com o extra `wsp`. O padrão permanece clássico e o experimento exige avaliação separada.

![exemplo](docs/exemplo.png)

## Uso

```bash
pip install -e .
python -m ecgdig exame.jpg -o saida          # saida.csv, saida.json, saida_debug.jpg
python -m ecgdig exame.jpg --layout 6x2 --ppmm 11.81   # 300 dpi = 11.81 px/mm
```

```python
from ecgdig import digitize, Config
res = digitize("exame.jpg", Config(layout="auto"))
res["signals"]   # (12, 5000) em mV; NaN onde a derivação não aparece no papel
```

Layout, velocidade, duração e ganho são **descobertos na imagem** (`Config()` sem argumentos). Biblioteca de hipóteses: `3x4`, `3x4+1/+2/+3`, `6x2`, `6x2+1`, `12x1`, páginas de 6 tiras (`6x1-prec` V1–V6, `6x1-limb` I–aVF), todas também na ordem de **Cabrera** (aVL, I, −aVR, II, aVF, III), com tiras de ritmo de nome livre e ganho dividido (membros 10 / precordiais 5 mm/mV). Fora da biblioteca, uma hipótese "livre" monta o layout pelas fronteiras e pelo OCR. Também dá para forçar: `Config(layout="6x2")` ou `Config(layout=[["I","V1"],["II","V2"]])`.

## Pipeline

| Etapa | Técnica |
|---|---|
| Segmentação da folha | Teste de scan (borda tão clara quanto o centro); senão GrabCut com borda = fundo e envoltória convexa |
| Mapa da grade | Normalização de iluminação por filtro de máximo; tinta atenuada para não contaminar a geometria |
| Deskew e escala | Maximização da variância das projeções; período por autocorrelação em blocos, com detecção de linha de 5 mm vs 1 mm e de múltiplos do período |
| Retificação | Fase de Fourier local da grade em blocos + *unwrapping* 2D guiado por qualidade → correspondências → homografia (RANSAC) |
| Papel curvo | Resíduo de fase da linha de 1 mm → campo de deslocamento polinomial cúbico → `remap` |
| Vincos e dobras | Campo local numa malha de células de 5 mm (fase de 1 mm em blocos de 10 mm, desenrolada em 2D → passa de ±0,5 mm ao atravessar um vinco); maior região conexa de vértices confiáveis, vértices que destoam da mediana 3×3 descartados e faltantes preenchidos pelos vizinhos; encolhimento de Wiener pelo ruído do próprio campo (em papel plano vira zero); interpolação bilinear entre vértices = transformação de 4 cantos por célula, composta com o polinômio numa única reamostragem (`Config(nonrigid="cell")`, padrão) |
| Remoção da grade | Modelo de transmitância (tinta × grade): molde periódico mod 5 mm por mediana **local** (vizinhança de 25 mm) + resíduo por linha/coluna reta (percentil 80) — qualquer cor de grade, foco/iluminação variáveis |
| Binarização | Otsu calculado só sobre estruturas finas (tarjas, carimbos e blocos pretos > 2.5 mm excluídos, senão apagam traçado claro) + histerese com limiar baixo acima do vazamento medido da grade; remoção de componentes minúsculas |
| Linhas | Picos de cobertura horizontal, filtrados por "parecer traçado": longo depois de juntar quebras **e** longo sem juntar (rodapé de texto só é longo dilatado); réguas retas (borda da área quadriculada) removidas por abertura de 60 mm; orientação 0/90/180/270° por alongamento + posição dos pulsos |
| Layout (hipótese e verificação) | Cada layout candidato é pontuado por: OCR clássico dos rótulos (casamento de modelos com fontes TrueType, NCC, refinamento do último caractere, setas/pontos aparados), marcas de fronteira, fechamento de Einthoven/Goldberger nas derivações de membros, concordância de derivações repetidas (NRMSE) e prior de biblioteca; confiança = sigmoide da margem |
| Velocidade / ganho | Leitura do texto "NN mm/s" (com modelos negativos mm/mV, Hz); senão largura do QRS / RR. Ganho por linha pelo pulso (pulso em degrau = ganho dividido) |
| Calibração | Pulso de 1 mV (exige platô horizontal; aceita "meio pulso" cortado pela margem): ganho com ajuste a 5/10/20 mm/mV; 0 mV = platô − 1 mV (o pé do pulso é ambíguo, o platô não) |
| Origem temporal | Bordas sub-pixel esquerda/direita de cada linha votam no x₀; com os dois lados, a média cancela o viés da ponta do traço |
| Registro entre linhas | Todas as derivações são simultâneas: os QRS (centróide de energia) de todas as linhas definem uma referência comum; deslocamento e escala por linha, lineares na altura, corrigem cisalhamento residual da retificação em fotos |
| Rastreamento | Viterbi sobre os segmentos de tinta de cada coluna: continuidade, prior saturado da linha de base (R de 3–5 mV são fisiológicos), penalidade para trechos planos sobre a linha de base de outra derivação, estado de falha |
| Linha central | **Inversão morfológica**: o traço é a curva dilatada por uma pena elíptica. Borda de cima = erosão, de baixo = dilatação; abertura e fechamento recuperam a curva, exatos em picos e vales (amplitude do R preservada). Largura horizontal da pena medida nos QRS |
| Sobreposição | Bordas que caem dentro da tinta de um traço vizinho plano são tratadas como oclusão e interpoladas |
| Bordas de segmento | Descarte de ~1 raio de pena nas fronteiras internas (marcas separadoras) com extrapolação amortecida |
| Fusão | Média ponderada por confiança entre ocorrências repetidas (amostras em que o traço se funde com o de outra linha valem 5%) e projeção de mínimos quadrados em I − II + III = 0 e aVR + aVL + aVF = 0, em que as amostras ocluídas absorvem a correção; opcionalmente reconstrói derivações de membros por Goldberger |

## Resultados

Conjunto **de teste separado** (semente diferente da usada no desenvolvimento), 15 imagens por nível, 3x4+1, 150–300 dpi. SNR calculado só nas amostras visíveis de cada derivação, sem alinhamento posterior.

| Degradação | SNR médio | Mediana por imagem | Pior imagem | Correlação |
|---|---|---|---|---|
| Nenhuma (scan perfeito) | 22.4 dB | 22.7 dB | 19.3 dB | 0.995 |
| Leve | 21.2 dB | 21.9 dB | 16.6 dB | 0.992 |
| Média (foto: rotação, perspectiva, sombra, JPEG) | 21.6 dB | 22.3 dB | 17.2 dB | 0.993 |
| Pesada | 19.4 dB | 20.7 dB | 10.8 dB | 0.988 |

Outros layouts (degradação leve): 3x4 21.8 dB, 6x2 22.0 dB, 3x4+3 22.0 dB, 12x1 21.1 dB. Cerca de 8–12 s por imagem em CPU.

### Fotos reais (sem ground truth): autoconsistência

Sem sinal verdadeiro, a métrica honesta é a concordância entre as duas cópias independentes da mesma derivação (grade × tira de ritmo, mesmo instante) e o fechamento de Einthoven (I − II + III = 0) antes da fusão.

| | Foto 1 (685 px, ~2.6 px/mm) | Foto 2 (celular, ~6.7 px/mm) |
|---|---|---|
| II grade × ritmo (fora de oclusões) | r 0.90, 62 µV RMS | r 0.96, 48 µV RMS |
| V1 grade × ritmo | r 0.995, 19 µV RMS | r 0.98, 95 µV RMS |
| V5 grade × ritmo | r 0.71, 239 µV RMS | r 0.83, 228 µV RMS |
| Resíduo de Einthoven | 70 µV (RMS de II: 522 µV) | 90 µV (RMS de II: 257 µV) |

As discordâncias que sobram ficam concentradas em QRS: onde um R de 3 mV atravessa a linha vizinha exatamente no mesmo instante do R dela (foto 1) e num R que cai na fronteira entre colunas (V5 da foto 2).

**Checagem externa** com imagens do ECG-Image-Kit (o gerador usado no PhysioNet Challenge 2024, renderizador independente deste código) contra o registro PTB-XL 00001: a versão com dobras e vincos dá correlação média 0.92.

### Imagens reais da internet (layout, velocidade e duração automáticos)

Imagens públicas dos repositórios [Open-ECG-Digitizer](https://github.com/Ahus-AIM/open-ecg-digitizer) e [paper-ecg](https://github.com/Tereshchenkolab/paper-ecg); figuras em `docs/internet/`.

| Imagem | Na folha | Detectado |
|---|---|---|
| GE/Marquette, scan | 3x4 + ritmo V1, II · 25 mm/s | ✔ (V1 e II duplicados concordam a 8–12 µV, r 0.98–0.999; Einthoven 16 µV com II de 285 µV RMS) |
| Mesmo GE, scan girado | idem | ✔ |
| Schiller/Mortara, 3 scans | V1–V6 · 50 mm/s | ✔ (velocidade lida do texto "50mm/s") |
| Schiller, foto de celular em perspectiva | V1–V6 · 50 mm/s | ✔ |
| Foto com rótulos apagados | provavelmente V1–V6 | ✘ "6x1-limb-cabrera" (confiança 0.78) |

O caso que falha é uma ambiguidade física: precordiais em ordem varrem o tórax como o leque de Cabrera e quase fecham Goldberger; sem rótulos impressos, só a forma de onda não separa as duas hipóteses.

A foto sem rótulos agora sai com confiança 0,55 e o aviso "nenhum rótulo lido; decidido só pela fisiologia (alternativa: 6x1-prec)".

### Rodada de melhorias (12/11/2026)

| Conjunto | Antes | Depois | O que mudou |
|---|---|---|---|
| Misto de teste (24 imagens: Cabrera, ganho dividido, 50 mm/s, 3x4/+1/+2/+3, 6x2/+1, 12x1) | 18,4 dB | **19,2 dB** | registro entre linhas só acima de 0,65 mm; traço que atravessa rótulo |
| `bench/heavy` (15 imagens) | 16,7 dB | **19,3 dB** | velocidade lida no texto conferida pela fisiologia ("25mm/s" degradado lido como 12,5 ou 50) |
| `bench/medium` | 21,6 dB | 21,6 dB | — |
| Layouts corretos no misto | 24/24 | 24/24 | — |

- **Traço atravessando rótulo** (R passando por cima de "V4"): o Viterbi da extração final passou a pedir continuidade das *bordas* (topo ou base andam pouco entre colunas num traço feito por pena; uma letra colada fica no meio e quebra as duas). Imagem afetada: 20,0 → 22,3 dB.
- **Registro por batimentos**: o instante do QRS difere entre derivações por razões fisiológicas (p90 ≈ 12 ms nos sinais verdadeiros), o que imita cisalhamento de até ~0,5 mm. Agora o registro só corrige acima de 0,65 mm (escala com a velocidade), que é o caso das fotos com cisalhamento real (foto 2: até 0,9 mm).
- **Velocidade**: o texto vence, salvo quando a largura do QRS e o RR o contradizem com força (diferença de log-verossimilhança > 2) e a leitura é fraca.
- **Tiras de ritmo** não repetem a mesma derivação (penalidade); sem nenhum rótulo legível no início das linhas, a confiança é limitada e há aviso.
- Testado e descartado: filtro passa-baixa adaptativo à resolução (+0,3 dB só no SNR, achata QRS), suavização dependente da inclinação (piora), QT/RR para decidir 25 × 50 mm/s (não ganhou nada), teste de permutação para Goldberger (instável).

### Vincos e dobras: campo local por células de 5 mm (12/11/2026)

O sintético não tinha nenhuma deformação não rígida. O nível novo `fold` (`python -m ecgdig.synth data/x 15 fold 3x4+1 101`) adiciona curvatura cilíndrica suave e 1–2 vincos com sombra (quebra de inclinação de 0,8–2,5 %, cisalhamento ao longo do vinco). Conjuntos de teste novos, 15 imagens 3x4+1 cada (sementes 101–104):

| Conjunto | Polinômio | Por célula | Pior imagem (dif.) |
|---|---|---|---|
| `fold` | 9,66 dB | **17,34 dB** | +1,3 dB (todas melhoram; até +15 dB) |
| `medium` | 21,32 dB | 21,30 dB | −0,27 dB |
| `heavy` | 17,52 dB | 17,52 dB | −0,06 dB |
| misto (24, layouts/velocidade/ganho variados) | 21,79 dB | 21,77 dB | −0,10 dB |

Em scans reais o campo encolhe a zero e a saída é idêntica. Nas fotos reais da internet ele mede 0,25–0,32 mm de deformação local e o resíduo de Einthoven cai de ~234 para ~215 µV. Sem o encolhimento de Wiener o modo por célula perdia até 2,5 dB em papel plano: o polinômio já deixa resíduo < 0,01 mm, e 0,03 mm de ruído horizontal num QRS íngreme custa ~0,06 mV. É a ideia do Undistortion da PMcardio (4 cantos por célula), mas com os vértices tirados da fase da grade, sem Dotter nem Gridder.

Limite físico que continua: a 25 vs 50 mm/s com QRS estreito e FC ~110 bpm sem texto legível (uma imagem do misto) é ambíguo de verdade; derivações de baixa voltagem a 5 mm/mV ficam no piso de quantização.

### LITFL: 102 imagens reais sem gabarito (avaliação qualitativa)

Revisão visual de cada página retificada com o traçado sobreposto: **82 boas, 12 com um defeito pontual, 8 falhas**. Funciona em grade cinza ou rosa, ruído de scan e páginas com pulso à direita. Erra no ganho quando o pulso está deformado ou cortado, em traço muito claro ou quebrado, em página curva fotografada e em ondas gigantes que cruzam as linhas vizinhas. Correções que vieram dessa revisão:

- Linhas de traço fraco eram descartadas como texto: rodapé só é rejeitado na ponta do bloco e fora do passo regular das linhas; linhas "fracas" entram se seguirem esse passo.
- Orientação 0/180° desempatada pelos rótulos legíveis (aVR, V1... — I/II/III são simétricos), não só pela posição do pulso.
- Tira de ritmo: convenção dos fabricantes como prior (1 tira = II; 3 = V1, II, V5); tira que não é simultânea à grade sai da física; confusão I/II/III no OCR custa metade. Nomes DII: 46 → 68 de ~90 tiras; as duas fotos do usuário passaram a sair V1, II, V5.
- Cabrera exige alguma evidência (prior −1,5).

Regressão depois disso: misto 24/24 e 19,2 dB; fotos reais mantidas (Einthoven da foto 2: 165 → 90 µV, porque a tira V5 deixou de ser chamada de III).

### Avaliação independente do campo por células + calibração (12/11/2026)

Medido nos conjuntos desta máquina (diferentes dos da tabela acima), com o campo local só aplicado quando o resíduo local passa de 0,06 mm (papel plano: 0,01–0,04 mm; vincos e fotos: 0,09–0,74 mm). Sem essa trava, nos scans do LITFL o campo mexia em decisões marginais (nome de 4 tiras de ritmo, 2 páginas viravam Cabrera) sem ganho de coerência.

| Conjunto | Polinômio | Por célula (com trava) |
|---|---|---|
| `fold` (15, semente 777) | 7,1 dB | **18,1 dB** |
| misto de teste (24) | 19,24 dB, 24/24 | 19,24 dB, 24/24 |
| `bench/heavy` | 19,40 dB | 19,40 dB |
| `bench/medium` | 21,60 dB | 21,61 dB |
| Fotos reais: cópias duplicadas e Einthoven (média, µV) | 66,6 | **36,3** |

Na foto 2 a II da grade e a do ritmo passam a concordar a 8 µV (antes 48) e o resíduo de Einthoven cai de 90 para 21 µV.

**Calibração.** Pulso no fim da linha também é procurado (vários aparelhos imprimem a calibração à direita); ganho fora do padrão (3,7 / 6,5 / 7 mm/mV, pulso deformado ou cortado) não vale e a linha herda o ganho mais votado da página (ou 10); pulso da direita que discorda do consenso cede; ganho dividido só pelo pulso da esquerda; o 0 mV só vem de pulso confiável. Ganho certo no LITFL: 92 % → 97 %.

### Benchmark LITFL

`benchmarks/litfl/` transforma a revisão acima em benchmark: referência anotada à mão para cada imagem (layout, tiras de ritmo, velocidade, ganho, duração) e métricas de coerência sem gabarito (cobertura, Einthoven antes da fusão, ritmo × grade). Roda em todo PR que mexe em `ecgdig/` (`.github/workflows/litfl-benchmark.yml`) e compara com `benchmarks/litfl/baseline.json`.

| Métrica (102 imagens) | `main` anterior | Agora |
|---|---|---|
| Layout correto | 92,2 % | 92,2 % |
| Tiras de ritmo todas certas | 87,9 % | 87,9 % |
| Velocidade | 99 % | 99 % |
| Duração | 91,2 % | 91,2 % |
| Ganho | 92 % | **97 %** |
| Layout + velocidade + duração + ganho | 83,3 % | **87,3 %** |

### Correções guiadas pelos erros do benchmark LITFL (12/11/2026)

| LITFL (102 imagens) | Antes | Depois |
|---|---|---|
| Tiras de ritmo todas certas | 87,9 % | **92,9 %** |
| Duração | 91,2 % | **99,0 %** |
| Layout + velocidade + duração + ganho | 87,3 % | **91,2 %** |
| Einthoven antes da fusão (mediana, menor é melhor) | 0,62 | **0,52** |
| Ritmo II × grade II, NRMSE (mediana) | 0,58 | **0,54** |
| Layout / ganho / velocidade | 92,2 / 97 / 99 % | iguais |

- **Começo da linha:** o traço começa depois da descida do pulso; pulso que o detector não achou (platô plano de 0,5–2,5 mV seguido de queda) é cortado; a extrapolação nas pontas de segmento vai só até 2 raios de pena, além disso a amostra fica desconhecida (antes uma rampa inventada estragava o 1º batimento). Cobertura cai de 100 % para 98,9 % por isso, de propósito.
- **Nome da tira de ritmo:** decidido por rótulo, comparação direta com a mesma derivação na grade e convenção do fabricante, sem o termo de Goldberger (que puxava para III/aVF/V3); barra solta lida como "I" (fragmento de QRS, aresta de pulso ou "II" com uma barra colada ao traço) não desmente II/III e perde para leituras de vários caracteres; barras soltas de altura diferente dos rótulos da página são descartadas; vocabulário com parênteses ("Rhythm (II)").
- **Duração:** página de 3 ou mais linhas fora de qualquer duração padrão é quase sempre escala horizontal errada (imagem da web redimensionada sem manter a proporção, grade só com linhas grossas): assume 10 s e ajusta o eixo do tempo à largura medida, sem mexer na escala vertical (`duration_assumed`, `horizontal_scale` no resultado).

Sintético: misto 23/24 layouts (a tira V1 quase plana da imagem 018, a 5 mm/mV, passa a se chamar I: sem física nem rótulo legível, é o caso de piso de quantização; SNR igual), 19,19 dB; `fold` 18,1, `heavy` 19,4, `medium` 21,6 dB, iguais. Fotos reais: coerência 36,3 → 35,4 µV.

Os números sintéticos não são comparáveis diretamente com o placar do PhysioNet 2024: lá as imagens são mais variadas, e o gerador sintético daqui é mais fácil do que o mundo real.

## Limitações conhecidas

- Quando dois QRS altos de linhas vizinhas caem no mesmo instante, a tinta se funde numa única barra vertical e a amplitude de um deles fica genuinamente ambígua; o método marca essas amostras como pouco confiáveis e recorre à outra cópia da derivação ou a Einthoven, mas sem cópia redundante (ex.: V4) o pico pode sair subestimado.
- Nas fronteiras internas do 3x4, a marca separadora esconde ~10 ms de cada lado; um QRS exatamente ali é extrapolado.
- Folha cortada pela borda da imagem: se as duas pontas de todas as linhas estiverem fora, a origem temporal fica sem referência.
- Quando o traço começa colado à descida do pulso, o t = 0 depende da convenção do aparelho (pode haver ~±50 ms de deslocamento global, igual em todas as derivações).
- Sem grade nenhuma, a escala vem de `--ppmm` ou de uma suposição de largura de página.
- Vincos simples agora são corrigidos pelo campo local; papel muito amassado (muitas dobras em menos de 5 mm, sombra que apaga a grade) e fotos com desfoque de movimento continuam sendo o ponto fraco de qualquer método clássico.

## Validação

```bash
python -m ecgdig.synth data/teste 15 heavy 3x4+1   # gera imagens + ground truth
python -m ecgdig.evaluate data/teste               # SNR por imagem e por derivação
pytest -q tests
```

## Referências

- Reyna MA et al. Digitization and classification of ECG images: the George B. Moody PhysioNet Challenge 2024.
- Shivashankara KK et al. ECG-Image-Kit: a synthetic image generation toolbox to facilitate deep learning-based electrocardiogram digitization. Physiol Meas, 2024.
- Badilini F et al. ECGScan: a method for conversion of paper electrocardiographic printouts to digital electrocardiographic files. J Electrocardiol, 2005.
- Ghiglia DC, Pritt MD. Two-Dimensional Phase Unwrapping: Theory, Algorithms, and Software. Wiley, 1998.
- Hartley R, Zisserman A. Multiple View Geometry in Computer Vision. Cambridge, 2004.
- Fischler MA, Bolles RC. Random sample consensus. Commun ACM, 1981.
- Rother C, Kolmogorov V, Blake A. GrabCut: interactive foreground extraction using iterated graph cuts. SIGGRAPH, 2004.
- Serra J. Image Analysis and Mathematical Morphology. Academic Press, 1982.
- Otsu N. A threshold selection method from gray-level histograms. IEEE SMC, 1979.
- Viterbi AJ. Error bounds for convolutional codes and an asymptotically optimum decoding algorithm. IEEE Trans Inf Theory, 1967.

