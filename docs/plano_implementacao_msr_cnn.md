# Adaptação da Arquitetura MSR-CNN para Séries Temporais Financeiras

Este projeto visa adaptar a arquitetura *Multi-Channel Subband Regularized CNN* (MSR-CNN) com *Adaptive Subband Decomposition* (ASD) para o domínio de séries temporais financeiras (1D). O objetivo principal é a classificação da direção do mercado (BUY, SELL, HOLD) de forma robusta e computacionalmente eficiente.

## Hipótese: Decomposição em Tendência, Sazonalidade e Ruído
Baseado na decomposição clássica de séries temporais, usaremos uma **decomposição assimétrica hierárquica em 3 subbandas**:
1. **Camada 1:** Separa a **Alta Frequência (Ruído / Volatilidade de Curto Prazo)**.
2. **Camada 2:** Decompõe o ramo de Baixa Frequência da Camada 1 em Alta Frequência (**Sazonalidade / Ciclos**) e Baixa Frequência (**Tendência de Longo Prazo**).

---

## Definição de Dados: Bolsa Brasileira (Ibovespa)
- **Ativos:** `^BVSP`, `PETR4.SA`, `VALE3.SA` e `ITUB4.SA`.
- **Granularidade:** Diária (últimos 10-15 anos).
- **Features de Entrada (Canais):** Retornos Logarítmicos, Volume Negociado, Volatilidade (ATR), Sazonalidade (RSI) e Tendência (MACD).

---

## User Review Required

> [!IMPORTANT]
> **Ambiente Local e Arquitetura Mac:** Como você utilizará sua própria máquina (macOS), vamos configurar o PyTorch para utilizar a aceleração de hardware nativa da Apple (MPS - Metal Performance Shaders) ao invés do CUDA (que é exclusivo para GPUs NVIDIA). Você possui o Python (versão 3.9+) já instalado na sua máquina para criarmos o ambiente virtual?

---

## Passo a Passo da Implementação (Execução Local)

### Fase 0: Configuração do Ambiente Local (macOS)
- **Ambiente Virtual:** Criação de um ambiente `venv` na pasta do projeto.
- **Dependências Chaves:**
  - `torch`, `torchvision`, `torchaudio` (Otimizados para uso de Metal `mps`).
  - `yfinance` (Coleta de dados).
  - `pandas`, `numpy`, `scikit-learn` (Manipulação e escalonamento).
  - `pandas-ta` (Cálculo dos indicadores técnicos).
  - `matplotlib`, `seaborn` (Visualização e análise de interpretabilidade).

### Fase 1: Aquisição e Pré-processamento de Dados
- Desenvolver um script (ex: `data_pipeline.py`) para baixar os dados do Yahoo Finance, calcular os indicadores via `pandas-ta`, gerar os *labels* de classificação e salvar localmente em arquivos `.csv` processados.

### Fase 2: Construção da Arquitetura (PyTorch)
- Desenvolver o arquivo `models.py` contendo:
  - O bloco de decomposição ASD 1D Asímetrico.
  - A Baseline CNN (Rede full-band para controle).
  - O modelo final MSR-CNN 1D conectando o ASD às redes independentes.
- Garantir que todos os tensores sejam dinamicamente mapeados para o *device* correto (CPU ou MPS).

### Fase 3: Pipeline de Treinamento
- Desenvolver o `train.py` para rodar o loop de treinamento localmente, acompanhando o log de métricas (Focal Loss, F1-Macro).
- Salvar os "checkpoints" do modelo (os pesos `.pt`) na máquina para que possam ser reaproveitados sem necessidade de treinar tudo novamente.

### Fase 4: Avaliação e Análise de Interpretabilidade (Espectral)
- Script de teste (`evaluate.py`) para medir os resultados no conjunto de validação/teste.
- Utilizar FFT (Transformada Rápida de Fourier) para plotar e salvar os gráficos das respostas em frequência dos filtros, demonstrando visualmente o aprendizado de Ruído, Sazonalidade e Tendência.


## Verification Plan

### Testes Manuais e Visuais
- Rodar o código do `data_pipeline.py` e verificar se a matriz gerada possui as colunas corretas e ausência de NaNs.
- Plotar os gráficos da resposta em frequência (FFT) para confirmar as subbandas.
- Acompanhar a convergência do *Loss* diretamente no terminal local.

### Testes Quantitativos
- Comparar tabelas numéricas (Salvas em arquivo) de F1-Macro, e tempo de treinamento local (segundos/época) entre o modelo Baseline e os MSR-CNNs.
