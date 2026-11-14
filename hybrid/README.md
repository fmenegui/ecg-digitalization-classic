# Experimento: clássico + pesos congelados do 5º colocado

Este experimento usa o CoaT-Lite-Medium + VGG19 de Đăng Nguyễn Hồng,
EMA 0,999, do [5º lugar do Kaggle](https://github.com/dangnh0611/kaggle_ecg_digitization).
Não é uma reprodução da submissão: substitui a retificação e os recortes
do template por geometria, layout e calibração encontrados pelo ecgdig.

## Google Colab

Abra [`colab_kaggle5.ipynb`](colab_kaggle5.ipynb) em uma sessão Colab com GPU.
Como este repositório é privado, o notebook lê `GITHUB_TOKEN` dos segredos
do Colab (ou pede um token em campo oculto). Basta permissão de leitura de
conteúdo para este repositório. O token não entra em URLs ou relatórios.
Não é necessário token Kaggle para os pesos públicos. Nenhuma célula escreve
no GitHub. O notebook resolve o commit do branch `hibrido`, executa o piloto,
exibe tabelas e figura e exporta um ZIP; salvar no Drive é opcional.

O caminho CUDA usa o decoder integral em FP16. Os testes locais validaram
contratos e equivalência das faixas; o notebook foi validado estruturalmente,
mas uma sessão Colab conectada é necessária para verificar a execução CUDA.

## Contratos de integração

1. O mapa clássico decide orientação, linhas, pulsos, velocidade, duração,
   ganho, rótulos e layout. A rede não altera essas decisões.
2. O ONNX opcional `ink_model` só orienta o rastreamento final. Medição de
   pena, envelopes e síntese continuam usando a tinta clássica. Sem modelo,
   o caminho numérico clássico permanece igual ao ponto de partida.
3. O backend `waveform_model` é uma alternativa ao ONNX: prevê coordenadas
   do sinal por janela local. Seu heatmap não é tratado como máscara de tinta.
4. Cada janela cobre 2,5 s e ±3,2 mV, usando as escalas medidas na imagem.
   Tiras longas são divididas, sem impor número ou ordem de derivações.
5. A entrada RGB recebe um quarto canal com grade anônima nominal, sem
   rótulos. Isso remove o template de página, mas introduz mudança de domínio:
   o checkpoint foi treinado com templates específicos. É justamente uma
   hipótese a avaliar, não uma garantia de transferência para outros formatos.
6. A saída neural usa o argmax de coluna do codec oficial. Pontas ausentes
   mantêm a cobertura do clássico. Einthoven/Goldberger e fusão são mantidos.
   Predições no limite vertical são registradas; sem fallback silencioso.

## Reproduzir

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,hybrid]'
git clone https://github.com/dangnh0611/kaggle_ecg_digitization.git ../kaggle-5th
git -C ../kaggle-5th checkout 8358d5b09a5b9519275e9b54c11716da5752a9d6
.venv/bin/python hybrid/download_kaggle5.py
.venv/bin/python -m hybrid.compare_kaggle5 --source ../kaggle-5th --device cpu
```

O downloader baixa apenas os tensores EMA usados na inferência (aproximadamente
236 MB, em vez do checkpoint Lightning de 941 MB), verifica CRCs do ZIP e
salva SHA-256, origem e configuração em `models/kaggle5/provenance.json`.
O modo padrão `--mode auto` tenta HTTP Range e, se indisponível, baixa o
checkpoint completo. No Colab, o notebook usa `--mode full` diretamente para
evitar depender de suporte a Range/redirecionamentos. O arquivo completo é
removido depois da extração; ficam apenas os pesos EMA e a configuração.
Os pesos ficam fora do Git. O código upstream é usado sob sua licença MIT;
não foi copiado para este repositório.

Em Macs de 8 GB, a inferência usa FP16 e executa a cabeça convolucional e os
blocos locais de alta resolução do decoder em faixas com halo de 2 pixels.
O encoder com atenção recebe a janela inteira; a interpolação usa índices
globais. Testes comparam as faixas contra os blocos integrais. Isso limita
temporários sem alterar pesos ou reduzir a resolução de 1024×1024. O consumo
e a velocidade dependem do dispositivo; `mps` e `cuda` podem ser selecionados.
Para limitar a alocação em Metal, execute com
`PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.6 PYTORCH_MPS_LOW_WATERMARK_RATIO=0.4`.

## Protocolo do piloto

12 imagens sintéticas, duas por formato: `3x4+1`, `6x2`, `3x4+1-cabrera`,
`12x1`, `6x1-prec`, `3x4+3`. Uma limpa e outra com degradação média por formato.
Inclui ganho dividido e 50 mm/s. Sementes fixadas no manifesto; sem fine-tuning,
sem escolha de pesos ou de variante usando o gabarito. São poucas imagens e
um gerador interno: não demonstram generalização clínica nem substituem uma
avaliação em fotografias reais ou no conjunto oculto do Kaggle.

`per_image.csv` traz SNR agrupado por imagem, RMSE em µV, RMSE nas 1% maiores
amplitudes verdadeiras, cobertura e tempo. Amostras visíveis sem previsão são
penalizadas como zero; não há alinhamento posterior. `per_format.csv` agrega
as comparações pareadas. O runner exige metadados estruturais idênticos.
As previsões completas são salvas em NPZ e os erros não são descartados.

O relatório HTML e JSON é atualizado a cada par e a execução pode ser retomada.
Use diretório de saída novo para mudar protocolo, checkpoint ou código.

## Limites

- Sem grid legível, rótulos ou pulsos, a rede local não resolve ambiguidades
  de calibração que já existem na imagem.
- ±3,2 mV é uma limitação deste checkpoint; os extremos são diagnosticados,
  sem redimensionar silenciosamente sinais grandes.
- Janelas menores que 2,5 s podem incluir contexto de derivações vizinhas.
- O canal de template, estatísticas de treino e a normalização de ganho/velocidade
  podem continuar carregando vieses do formato da competição.
- `ink_model` e `waveform_model` devem ser avaliados separadamente antes
  de combinar seus efeitos.

Próxima decisão depende do resultado por formato: manter congelado, fazer
fine-tuning multiformato ou descartar o backend. O modo padrão segue clássico.
