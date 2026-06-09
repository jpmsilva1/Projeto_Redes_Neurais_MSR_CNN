# MSR-CNN Finance: Adaptive Subband Decomposition para Séries Temporais 📈

Este repositório contém a implementação do projeto de mestrado que adapta a arquitetura **MSR-CNN** (Multi-Channel Subband Regularized Convolutional Neural Network) – originalmente proposta para processamento de imagens (Sinha et al., 2023) – para o domínio financeiro usando **séries temporais 1D**.

## 🧠 Sobre o Projeto

As séries financeiras são classicamente decompostas em três componentes: **Tendência** (Longo prazo), **Sazonalidade/Ciclos** (Médio prazo) e **Ruído** (Curto prazo). A maioria das redes neurais (CNNs) tradicionais processa esses sinais como um todo (full-band), o que dificulta o aprendizado e aumenta a sensibilidade ao ruído.

Neste projeto, utilizamos uma **Decomposição Adaptativa em Subbandas Assimétrica (ASD)**. O modelo aprende ativamente, de ponta a ponta (end-to-end), a separar o sinal de entrada nessas 3 frequências utilizando uma árvore de filtros `Conv1D`. Cada frequência é então processada por uma CNN independente.

### Arquiteturas Implementadas
1. **Baseline CNN 1D:** Rede tradicional para fins de controle e comparação de performance.
2. **MSR-CNN 1D (Clássico):** Rede que separa o sinal nas 3 subbandas, extrai as features isoladamente e as concatena para a classificação direcional do mercado (BUY, SELL, HOLD).
3. **MSR-CNN 1D (Atenção):** Extensão inédita do artigo base. Implementa um mecanismo de atenção visual (Softmax) entre as subbandas, permitindo que a rede dê pesos dinâmicos ao Ruído, Sazonalidade ou Tendência a depender do regime momentâneo do mercado.

## 📊 Dados Utilizados

O projeto utiliza a biblioteca `yfinance` para baixar dados diários da bolsa brasileira (Ibovespa):
- **Tickers:** `^BVSP`, `PETR4.SA`, `VALE3.SA`, `ITUB4.SA`.
- **Features extraídas:** Retornos Logarítmicos, Volume, RSI, MACD, MACD Signal, ATR (Average True Range).
- **Target (Label):** Retorno futuro em janela de 5 dias classificado como BUY (+1.5%), SELL (-1.5%) ou HOLD.

### Por que uma Decomposição Assimétrica?

A arquitetura MSR-CNN original utilizava uma árvore simétrica (dividindo os sinais em potências de 2, 4, 8...). Neste projeto de mestrado, optamos por uma **Decomposição Assimétrica** por três justificativas técnicas:
1. **Alinhamento com a Teoria Econômica (STL):** A análise estatística financeira foca em exatamente 3 componentes: Tendência, Sazonalidade e Ruído. A árvore assimétrica nos permite forçar a rede a gerar exatamente 3 canais de informação (Subbandas), mantendo o modelo interpretável.
2. **Foco na Frequência Correta:** O modelo descarta o Ruído na primeira etapa e aplica processamento profundo apenas nas baixas frequências e frequências médias, onde o "sinal real" da economia reside.
3. **Prevenção de Overfitting:** Evita o custo computacional inútil de fatiar o ruído repetidas vezes.

### Fluxo da Arquitetura

Abaixo detalhamos visualmente as duas versões da rede implementadas neste projeto.

#### 1. MSR-CNN Clássico (Original Adaptado)
Nesta versão, que replica a lógica do artigo original adaptada para séries temporais, as subbandas são concatenadas de forma bruta e enviadas diretamente para o classificador final (Camada Densa). A rede precisa descobrir sozinha na "força bruta" qual frequência importa mais.

```mermaid
flowchart TD
    classDef input fill:#e1f5fe,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef asd fill:#fff3e0,stroke:#0ea5e9,stroke-width:2px,color:#000
    classDef subband fill:#f3e8ff,stroke:#6366f1,stroke-width:2px,color:#000
    classDef cnn fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#000
    classDef concat fill:#f1f5f9,stroke:#64748b,stroke-width:2px,color:#000
    classDef output fill:#ecfdf5,stroke:#ec4899,stroke-width:2px,color:#000

    Input["Série Temporal (Janela de 32 dias, 6 Canais)"]:::input
    
    subgraph ASD["Módulo ASD (Decomposição Assimétrica 1D)"]
        L1_H["Filtro Passa-Alta L1\n(Conv1D + Pooling)"]:::asd
        L1_L["Filtro Passa-Baixa L1\n(Conv1D + Pooling)"]:::asd
        
        L2_H["Filtro Passa-Alta L2\n(Conv1D + Pooling)"]:::asd
        L2_L["Filtro Passa-Baixa L2\n(Conv1D + Pooling)"]:::asd
        
        Input --> L1_H
        Input --> L1_L
        
        L1_L --> L2_H
        L1_L --> L2_L
    end
    
    subgraph Subbands["Separação Estrutural"]
        SB1["Subbanda 1\n(Ruído / Alta Freq.)"]:::subband
        SB2["Subbanda 2\n(Sazonalidade / Freq. Média)"]:::subband
        SB3["Subbanda 3\n(Tendência / Baixa Freq.)"]:::subband
        
        L1_H --> SB1
        L2_H --> SB2
        L2_L --> SB3
    end
    
    subgraph CNN["Sub-Redes Independentes"]
        CNN1["CNN 1D\n(Especialista Ruído)\nAtivação: LeakyReLU"]:::cnn
        CNN2["CNN 1D\n(Especialista Sazonalidade)\nAtivação: LeakyReLU"]:::cnn
        CNN3["CNN 1D\n(Especialista Tendência)\nAtivação: LeakyReLU"]:::cnn
        
        SB1 --> CNN1
        SB2 --> CNN2
        SB3 --> CNN3
    end
    
    Concat{"Concatenação Bruta"}:::concat
    
    CNN1 --> Concat
    CNN2 --> Concat
    CNN3 --> Concat
    
    FC["Camadas Densas\n(Fully Connected)\nAtivação: ReLU + Dropout"]:::output
    Concat --> FC
    
    Output["Saída: Decisão de Mercado\n(BUY / SELL / HOLD)"]:::output
    FC --> Output
```

**Entendendo o Fluxo Clássico (Passo a Passo):**
1. **A Entrada:** A rede recebe uma matriz contendo os dados dos últimos 32 pregões divididos em 6 canais (Retorno Logarítmico, Volume e indicadores técnicos).
2. **A Separação (Módulo ASD):** A matriz passa por filtros de convolução 1D agrupados (*depthwise*). A rede "fatia" a matriz separando as oscilações caóticas diárias (Ruído), os movimentos cíclicos (Sazonalidade) e o direcional de longo prazo (Tendência).
3. **Sub-Redes Especializadas:** Cada um desses 3 sinais é enviado para uma pequena rede convolucional totalmente isolada das demais. Cada especialista procura padrões estruturais focando apenas na sua faixa de frequência, utilizando funções de ativação **LeakyReLU** (para preservar gradientes negativos sutis financeiros) seguidas de *Max Pooling*.
4. **Concatenação e Decisão Final:** As extrações dos três especialistas são unidas lado a lado (Concatenação Bruta) e entregues à Camada Densa final. A rede final (utilizando a ativação não-linear **ReLU** e **Dropout** para regularização) tenta deduzir a classe (`BUY/SELL/HOLD`) forçando sentido nessa mistura homogênea.

#### 2. MSR-CNN com Atenção Dinâmica (Nossa Extensão)
Para resolver a limitação da rede clássica, propusemos um mecanismo de Auto-Atenção (*Self-Attention*). O `Softmax` atua como um juiz que lê a concatenação bruta inteira e calcula três pesos dinâmicos. Esses pesos são multiplicados de volta contra as saídas dos especialistas, silenciando frequências irrelevantes para o dia específico antes da decisão final.

```mermaid
flowchart TD
    classDef input fill:#e1f5fe,stroke:#3b82f6,stroke-width:2px,color:#000
    classDef asd fill:#fff3e0,stroke:#0ea5e9,stroke-width:2px,color:#000
    classDef subband fill:#f3e8ff,stroke:#6366f1,stroke-width:2px,color:#000
    classDef cnn fill:#ede9fe,stroke:#8b5cf6,stroke-width:2px,color:#000
    classDef concat fill:#f1f5f9,stroke:#64748b,stroke-width:2px,color:#000
    classDef attn fill:#fce7f3,stroke:#d946ef,stroke-width:2px,color:#000
    classDef output fill:#ecfdf5,stroke:#ec4899,stroke-width:2px,color:#000

    Input["Série Temporal (Janela de 32 dias, 6 Canais)"]:::input
    
    subgraph ASD["Módulo ASD (Decomposição Assimétrica 1D)"]
        L1_H["Filtro Passa-Alta L1\n(Conv1D + Pooling)"]:::asd
        L1_L["Filtro Passa-Baixa L1\n(Conv1D + Pooling)"]:::asd
        
        L2_H["Filtro Passa-Alta L2\n(Conv1D + Pooling)"]:::asd
        L2_L["Filtro Passa-Baixa L2\n(Conv1D + Pooling)"]:::asd
        
        Input --> L1_H
        Input --> L1_L
        
        L1_L --> L2_H
        L1_L --> L2_L
    end
    
    subgraph Subbands["Separação Estrutural"]
        SB1["Subbanda 1\n(Ruído / Alta Freq.)"]:::subband
        SB2["Subbanda 2\n(Sazonalidade / Freq. Média)"]:::subband
        SB3["Subbanda 3\n(Tendência / Baixa Freq.)"]:::subband
        
        L1_H --> SB1
        L2_H --> SB2
        L2_L --> SB3
    end
    
    subgraph CNN["Sub-Redes Independentes"]
        CNN1["CNN 1D\n(Especialista Ruído)\nAtivação: LeakyReLU"]:::cnn
        CNN2["CNN 1D\n(Especialista Sazonalidade)\nAtivação: LeakyReLU"]:::cnn
        CNN3["CNN 1D\n(Especialista Tendência)\nAtivação: LeakyReLU"]:::cnn
        
        SB1 --> CNN1
        SB2 --> CNN2
        SB3 --> CNN3
    end
    
    Concat{"Concatenação Bruta"}:::concat
    
    CNN1 --> Concat
    CNN2 --> Concat
    CNN3 --> Concat
    
    subgraph Attention["Módulo de Atenção Dinâmica"]
        AttnCalc["Atenção Dinâmica\n(Tanh -> Softmax)"]:::attn
        Concat --> AttnCalc
        
        Mult1(("X")):::attn
        Mult2(("X")):::attn
        Mult3(("X")):::attn
        
        AttnCalc -.->|Peso Ruído| Mult1
        AttnCalc -.->|Peso Sazonalidade| Mult2
        AttnCalc -.->|Peso Tendência| Mult3
        
        CNN1 --> Mult1
        CNN2 --> Mult2
        CNN3 --> Mult3
    end
    
    Concat_Final{"Concatenação\nPonderada"}:::concat
    Mult1 --> Concat_Final
    Mult2 --> Concat_Final
    Mult3 --> Concat_Final
    
    FC["Camadas Densas\n(Fully Connected)\nAtivação: ReLU + Dropout"]:::output
    Concat_Final --> FC
    
    Output["Saída: Decisão de Mercado\n(BUY / SELL / HOLD)"]:::output
    FC --> Output
```

**Entendendo o Fluxo com Atenção (Passo a Passo):**
*(O trajeto inicial é idêntico ao modelo clássico. A diferença ocorre após a Concatenação Bruta).*
1. **A Visão Global do Juiz (Cálculo de Pesos):** Em vez de jogar a informação diretamente para o final, a `Concatenação Bruta` é enviada inteira para uma camada densa intermediária. Essa camada usa a ativação **Tanh** para comprimir a representação, seguida da função **Softmax**. Ela atua como um "Juiz" que lê o contexto total do dia e determina *qual frequência é a mais importante naquele cenário*, devolvendo 3 notas percentuais (pesos de 0 a 1).
2. **A Aplicação do Filtro Dinâmico:** Os sinais individuais de cada especialista (as setas paralelas no diagrama) passam por um nó multiplicador (`X`), onde recebem o peso definido pelo Juiz. Se a rede identificar um ambiente de pânico no mercado, ela pode gerar um peso de `0.80` para o Ruído e `0.10` para os outros, "silenciando" algoritmicamente as frequências inócuas para aquele momento.
3. **A Decisão Final:** É essa nova matriz filtrada (Concatenação Ponderada) que a Camada Densa final usa para calcular as probabilidades de `BUY`, `SELL` ou `HOLD`, tornando a predição altamente contextual e interpretável.

---
## 📂 Estrutura do Repositório

```text
├── src/                        # Códigos-fonte Python
│   ├── data_pipeline.py        # Coleta e pré-processamento de dados
│   ├── models.py               # Arquiteturas PyTorch (MSR-CNN, Baseline, Atenção)
│   ├── train.py                # Script de treinamento dos modelos base
│   ├── train_attention.py      # Script de treinamento do modelo com atenção
│   ├── evaluate.py             # Validação preditiva e extração de FFT
│   └── evaluate_attention.py   # Extração dos pesos de atenção
├── docs/                       # Documentação e relatórios do projeto
│   └── analise_resultados.md   # Relatório detalhado dos resultados
├── data/                       # Arquivos CSV gerados (Treino/Validação/Teste)
├── checkpoints/                # Pesos (.pt) dos modelos treinados
├── results/                    # Gráficos analíticos salvos (FFT, Matrizes)
└── README.md                   # Este arquivo
```

---

## 🚀 Como Replicar na sua Máquina

O código foi otimizado nativamente para uso de aceleração no **macOS (Apple Silicon - MPS)**, mas pode rodar em CUDA ou CPU facilmente (há fallback automático implementado nos scripts).

### 1. Clonar e Instalar Dependências

Crie um ambiente virtual usando o Python (versão 3.9+ recomendada):

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/msrcnn-finance.git
cd msrcnn-finance

# Crie e ative o ambiente virtual
python3 -m venv venv
source venv/bin/activate  # No Windows use: venv\\Scripts\\activate

# Instale os pacotes necessários
pip install torch torchvision torchaudio yfinance pandas ta scikit-learn matplotlib seaborn
```

### 2. Rodar o Pipeline de Dados

Este script fará o download da base e criará os arquivos CSV normalizados divididos em Treino, Validação e Teste na pasta `data/`.

```bash
python src/data_pipeline.py
```

### 3. Treinar os Modelos

Para treinar a Baseline e o MSR-CNN clássico:

```bash
python src/train.py
```

Para treinar a arquitetura com o mecanismo de Atenção proposto:

```bash
python src/train_attention.py
```
*(Os pesos dos modelos treinados ficarão salvos na pasta `checkpoints/`).*

### 4. Avaliar e Analisar a Interpretabilidade

A grande vantagem estrutural desse modelo é poder entender **o que** ele aprendeu. O script de avaliação não só diz a acurácia, mas também extrai a Transformada Rápida de Fourier (FFT) dos pesos convolucionais para provar que a rede se organizou isolando as Baixas Frequências, Frequências Médias e Altas Frequências (filtros Passa-Baixa, Passa-Banda e Passa-Alta).

```bash
python src/evaluate.py
python src/evaluate_attention.py
```
*(Os gráficos analíticos e espectrais serão salvos na pasta `results/`).*

---

## 🔬 Resultados de Interpretabilidade

Os gráficos gerados na pasta `results/` confirmam a eficácia analítica da rede:
1. **FFT dos Filtros:** O MSR-CNN aprende, sem supervisão humana imposta nas frequências, a estruturar o banco de filtros de tal forma que a "Subbanda 1" reage a picos altos de sinal (Ruído), enquanto a "Subbanda 3" age isolando o sinal em frequências lentas (Tendência).
2. **Pesos de Atenção:** Na versão com `Attention`, percebe-se a distribuição clara e assimétrica do "esforço" que a rede aplica em cada subbanda na hora de decidir uma compra (BUY) versus uma manutenção na carteira (HOLD).

---

## 📖 Referências
- *Sinha, P., Psaromiligkos, I., & Zilic, Z. (2023). A Structurally Regularized CNN Architecture via Adaptive Subband Decomposition. arXiv:2306.16604 [eess.IV].*
