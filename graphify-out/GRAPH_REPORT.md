# Graph Report - Projeto Redes Neurais  (2026-06-09)

## Corpus Check
- 12 files · ~17,061 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 96 nodes · 119 edges · 11 communities (9 shown, 2 thin omitted)
- Extraction: 88% EXTRACTED · 12% INFERRED · 0% AMBIGUOUS · INFERRED: 14 edges (avg confidence: 0.71)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `e769f8a8`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]

## God Nodes (most connected - your core abstractions)
1. `TimeSeriesDataset` - 11 edges
2. `MSRCNN1D` - 8 edges
3. `BaselineCNN1D` - 8 edges
4. `FocalLoss` - 7 edges
5. `MSR-CNN Finance: Adaptive Subband Decomposition para Séries Temporais 📈` - 7 edges
6. `Passo a Passo da Implementação (Execução Local)` - 7 edges
7. `main()` - 6 edges
8. `AsymmetricASD1D` - 6 edges
9. `SubbandCNN` - 6 edges
10. `MSRCNNAttention1D` - 6 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `MSRCNN1D`  [INFERRED]
  src/evaluate.py → src/models.py
- `main()` --calls--> `TimeSeriesDataset`  [INFERRED]
  src/evaluate.py → src/train.py
- `main()` --calls--> `MSRCNNAttention1D`  [INFERRED]
  src/evaluate_attention.py → src/models.py
- `main()` --calls--> `TimeSeriesDataset`  [INFERRED]
  src/evaluate_attention.py → src/train.py
- `FocalLoss` --uses--> `BaselineCNN1D`  [INFERRED]
  src/train.py → src/models.py

## Import Cycles
- None detected.

## Communities (11 total, 2 thin omitted)

### Community 0 - "Community 0"
Cohesion: 0.18
Nodes (9): Dataset, MSRCNN1D, MSR-CNN Clássico (sem atenção).     Usa o ASD 1D para gerar 3 subbandas e proces, main(), train_attention_model(), FocalLoss, main(), TimeSeriesDataset (+1 more)

### Community 1 - "Community 1"
Cohesion: 0.13
Nodes (14): Adaptação da Arquitetura MSR-CNN para Séries Temporais Financeiras, Definição de Dados: Bolsa Brasileira (Ibovespa), Fase 0: Configuração do Ambiente Local (macOS), Fase 1: Aquisição e Pré-processamento de Dados, Fase 2: Construção da Arquitetura (PyTorch), Fase 3: Pipeline de Treinamento, Fase 4: Avaliação e Análise de Interpretabilidade (Espectral), Fase 5: Extensão da Atenção entre Subbandas (+6 more)

### Community 2 - "Community 2"
Cohesion: 0.21
Nodes (6): AsymmetricASD1D, MSRCNNAttention1D, MSR-CNN com mecanismo de Atenção entre Subbandas.     Adiciona pesos dinâmicos p, CNN simples para processar cada subbanda de forma independente, Decomposição Adaptativa em Subbandas (ASD) 1D Asímetrica em 3 Subbandas.     Cam, SubbandCNN

### Community 3 - "Community 3"
Cohesion: 0.15
Nodes (12): 1. Clonar e Instalar Dependências, 2. Rodar o Pipeline de Dados, 3. Treinar os Modelos, 4. Avaliar e Analisar a Interpretabilidade, Arquiteturas Implementadas, 🚀 Como Replicar na sua Máquina, 📊 Dados Utilizados, 📂 Estrutura do Repositório (+4 more)

### Community 4 - "Community 4"
Cohesion: 0.28
Nodes (6): evaluate_model(), main(), plot_fft_filters(), Extrai os pesos dos filtros ASD e plota a resposta em frequência.     O objetivo, BaselineCNN1D, CNN Convencional Full-Band.     Mesmo número de parâmetros das subbands somadas

### Community 5 - "Community 5"
Cohesion: 0.29
Nodes (6): 1. Pipeline de Dados (`data_pipeline.py`), 2. Arquitetura PyTorch e Decomposição ASD (`models.py`), 3. Treinamento Base e com Atenção (`train.py` & `train_attention.py`), 4. Avaliação e Análise de Interpretabilidade Espectral (`evaluate.py` & `evaluate_attention.py`), Arquivos Gerados:, Walkthrough: Projeto MSR-CNN para Séries Temporais Financeiras

### Community 6 - "Community 6"
Cohesion: 0.48
Nodes (6): feature_engineering(), fetch_data(), generate_labels(), main(), Classifica o retorno futuro em H dias.     Se retorno_H > threshold -> BUY (1), split_and_normalize()

### Community 7 - "Community 7"
Cohesion: 0.33
Nodes (5): 1. Desempenho Preditivo (Acurácia e F1-Score), 2. Interpretabilidade Espectral (Diferencial da Arquitetura) 🌟, 3. Extensão do Modelo: Pesos de Atenção Dinâmica, Análise da Resposta em Frequência (FFT), Análise e Interpretação de Resultados (MSR-CNN)

### Community 8 - "Community 8"
Cohesion: 0.83
Nodes (3): evaluate_attention(), main(), plot_attention_weights()

## Knowledge Gaps
- **29 isolated node(s):** `graphify`, `Workflow: graphify`, `Arquiteturas Implementadas`, `📊 Dados Utilizados`, `📂 Estrutura do Repositório` (+24 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **2 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TimeSeriesDataset` connect `Community 0` to `Community 8`, `Community 4`?**
  _High betweenness centrality (0.067) - this node is a cross-community bridge._
- **Why does `MSRCNN1D` connect `Community 0` to `Community 2`, `Community 4`?**
  _High betweenness centrality (0.051) - this node is a cross-community bridge._
- **Why does `BaselineCNN1D` connect `Community 4` to `Community 0`, `Community 2`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `TimeSeriesDataset` (e.g. with `main()` and `main()`) actually correct?**
  _`TimeSeriesDataset` has 5 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `MSRCNN1D` (e.g. with `main()` and `FocalLoss`) actually correct?**
  _`MSRCNN1D` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `BaselineCNN1D` (e.g. with `main()` and `FocalLoss`) actually correct?**
  _`BaselineCNN1D` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `FocalLoss` (e.g. with `train_attention_model()` and `BaselineCNN1D`) actually correct?**
  _`FocalLoss` has 3 INFERRED edges - model-reasoned connections that need verification._