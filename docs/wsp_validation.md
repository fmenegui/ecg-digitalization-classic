# Verificação da integração experimental

Data: 13/11/2026. Código-base: `49b836f0beb92a864ef35a858e6c46e63d334688`.
O ambiente está registrado em [`wsp_pilot_metrics.json`](wsp_pilot_metrics.json).

## Testes específicos da U-Net

Os 15 casos específicos passaram. Foram verificados:

- Reprodução dos pesos, ausência de parâmetros com gradientes e reconstrução do checkpoint.
- Resposta em campo constante e participação de encoder, contexto e skips.
- Inferência em blocos equivalente à imagem inteira, inclusive com dimensões ímpares.
- Suporte restrito à tinta medida e à vizinhança do suporte clássico.
- Importação do digitalizador clássico com PyTorch deliberadamente indisponível.
- Sinal completo idêntico quando o callback retorna a máscara clássica original.
- Rejeição de máscara em coordenadas incorretas.
- Penalização de amostras ausentes e inclusão de falhas na métrica do piloto.

As duas variantes da CLI também processaram o ECG real descrito em
[`wsp_litfl_smoke.json`](wsp_litfl_smoke.json), escrevendo CSV, JSON e imagem de
depuração. O checkpoint exportado foi carregado com `weights_only=True` e
reproduziu o mesmo SHA-256 da receita e do compilador.

## Testes anteriores do digitalizador

Os três testes de SNR sintético passaram. O teste existente
`tests/test_synthetic.py::test_fold_local_field` falhou. Repetimos esse caso em
uma cópia sem alterações do código-base, no mesmo ambiente, e obtivemos os
mesmos números:

| Código executado, sem ativar a U-Net | Correção polinomial, SNR | Correção por células, SNR |
|---|---:|---:|
| Com a integração opcional | 19,071901 dB | 18,974127 dB |
| Snapshot original | 19,071901 dB | 18,974127 dB |

O teste exige que a correção por células supere a polinomial em mais de 1,5 dB.
Essa condição já não é satisfeita pelo snapshot original neste ambiente. A
assertiva foi preservada. Portanto a suíte completa não está totalmente verde.

## Interpretação do piloto

O ganho médio de 0,0159 dB em três imagens sintéticas não estabelece vantagem
prática ou generalização. A intervenção não alterou o exemplo real. O código
deixa uma implementação executável, pesos inspecionáveis e um comparador para
próximas hipóteses, com o caminho clássico mantido como padrão.
