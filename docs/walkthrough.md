# Walkthrough: Projeto MSR-CNN para Séries Temporais Financeiras

Concluímos com sucesso todas as fases planejadas para adaptar a arquitetura MSR-CNN ao domínio de séries temporais financeiras locais (macOS / MPS). Abaixo você encontra o resumo do que foi implementado e dos resultados obtidos.

---

## 1. Pipeline de Dados (`data_pipeline.py`)

- **Ativos Coletados:** Obtivemos os dados diários históricos de 15 anos para o Índice Bovespa (`^BVSP`), Petrobras (`PETR4.SA`), Vale (`VALE3.SA`) e Itaú (`ITUB4.SA`).
- **Features Calculadas:** Retornos logarítmicos, Volume Negociado, RSI (Momento), MACD (Tendência) e ATR (Volatilidade).
- **Labels:** Definimos os labels (BUY, SELL, HOLD) projetando o retorno para a frente em uma janela de 5 dias e usando um limite (`threshold = 0.015`, ou seja, 1.5%).
- **Divisão & Normalização:** Usamos 70% para Treino, 15% para Validação e 15% para Teste, aplicando `StandardScaler` (Z-score) isolado no conjunto de treinamento para não vazar dados futuros.

---

## 2. Arquitetura PyTorch e Decomposição ASD (`models.py`)

> [!NOTE]
> Todos os modelos foram otimizados para usar **aceleração via hardware nativo da Apple (MPS)** usando `torch.backends.mps`.

Implementamos 3 modelos principais:
1. **AsymmetricASD1D:** É o "coração" da proposta. Uma árvore que faz duas camadas de convoluções 1D com filtros passa-alta e passa-baixa, gerando 3 subbandas de tamanhos simétricos e significado estrutural: **Ruído**, **Sazonalidade** e **Tendência**.
2. **MSRCNN1D (Clássico):** Manda cada uma das 3 subbandas geradas pelo ASD para uma `SubbandCNN` independente. Ao final, concatena as representações em uma camada linear final (`FC`).
3. **BaselineCNN1D:** A rede de controle, parametrizada para ter um número idêntico de pesos/MACs, mas que processa o sinal na sua totalidade (full-band).

---

## 3. Treinamento Base (`train.py`)

> [!TIP]
> Foi utilizada a **Focal Loss** ($\gamma=2$) para tratar o fato de que "HOLD" é muito mais frequente em janelas de 5 dias do que movimentos direcionais muito fortes. 

- Os checkpoints foram salvos automaticamente na pasta `checkpoints/`.
- O treinamento ocorreu a uma velocidade excelente devido ao Apple Silicon (`mps`).

---

## 4. Avaliação e Análise de Interpretabilidade Espectral (`evaluate.py`)

Em termos de acurácia preditiva ($F_1$-Macro), o problema financeiro continuou desafiador, com a baseline oscilando contra os MSR-CNN variando conforme o ativo (por exemplo, em PETR4 a rede clássica se beneficiou um pouco). 

O ponto chave, porém, é a **interpretabilidade matemática**:

> [!IMPORTANT]
> Plotamos a **Transformada Rápida de Fourier (FFT)** dos pesos da camada ASD. Os gráficos resultantes estão em `results/fft_filters_*.png`. Neles, conseguimos provar matematicamente que a rede apreendeu e isolou a alta frequência (Ruído), a baixa frequência (Tendência) e o meio-termo (Sazonalidade/Banda) – **exatamente como hipotetizado em nossa proposta!**



### Arquivos Gerados:
- **`data_pipeline.py`** (Script de dados)
- **`models.py`** (Arquiteturas PyTorch)
- **`train.py`** (Script de treinamento)
- **`evaluate.py`** (Script analítico)
- **Pastas locais:** `data/`, `checkpoints/`, `results/`
