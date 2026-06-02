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

A grande vantagem estrutural desse modelo é poder entender **o que** ele aprendeu. O script de avaliação não só diz a acurácia, mas também extrai a Transformada Rápida de Fourier (FFT) dos pesos convolucionais para provar que a rede se organizou em filtros Passa-Alta, Passa-Baixa e Passa-Banda.

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
