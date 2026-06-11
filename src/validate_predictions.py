"""
Validação Temporal das Previsões do MSR-CNN
============================================
Compara as classificações previstas (BUY/SELL/HOLD) dos 3 modelos
com o retorno real observado no mercado nos 5 dias seguintes.

Gera:
  - Matriz de Confusão (heatmap) para cada modelo/ticker
  - Gráfico temporal: previsões sobrepostas ao preço real da ação
  - Tabela comparativa de métricas (Acurácia, F1-Macro) dos 3 modelos
  - CSV detalhado com cada previsão individual

Uso:
    python src/validate_predictions.py
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import yfinance as yf
import os
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
from models import BaselineCNN1D, MSRCNN1D
from train import TimeSeriesDataset
from torch.utils.data import DataLoader

# ── Configurações ────────────────────────────────────────────────
TICKERS_MAP = {
    'BVSP': '^BVSP',
    'PETR4.SA': 'PETR4.SA',
    'VALE3.SA': 'VALE3.SA',
    'ITUB4.SA': 'ITUB4.SA',
}
WINDOW_SIZE = 32
IN_CHANNELS = 6
HORIZON = 5
THRESHOLD = 0.015
LABEL_NAMES = ['HOLD', 'BUY', 'SELL']
RESULTS_DIR = 'results/validation'

# ── Estilo visual ────────────────────────────────────────────────
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'axes.edgecolor': '#e0e0e0',
    'axes.labelcolor': '#e0e0e0',
    'text.color': '#e0e0e0',
    'xtick.color': '#e0e0e0',
    'ytick.color': '#e0e0e0',
    'grid.color': '#2a2a4a',
    'font.size': 11,
})


def fetch_raw_prices(yf_ticker, start='2010-01-01', end='2025-01-01'):
    """Baixa preços brutos de fechamento para calcular retornos reais."""
    df = yf.download(yf_ticker, start=start, end=end, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    return df[['Close']].dropna()


def compute_real_return(prices_series, date_idx, horizon=5):
    """Calcula o retorno real em % para os próximos `horizon` dias."""
    pos = prices_series.index.get_loc(date_idx)
    if pos + horizon >= len(prices_series):
        return np.nan
    p0 = prices_series.iloc[pos]
    p1 = prices_series.iloc[pos + horizon]
    return ((p1 - p0) / p0) * 100


def real_label_from_return(ret_pct, threshold_pct=1.5):
    """Converte retorno real em % para a classe correspondente."""
    if ret_pct > threshold_pct:
        return 1  # BUY
    elif ret_pct < -threshold_pct:
        return 2  # SELL
    else:
        return 0  # HOLD


def run_inference(model, test_loader, device, is_attention=False):
    """Roda inferência e retorna previsões e targets."""
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)

            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return np.array(all_preds), np.array(all_targets), np.array(all_probs)


def plot_confusion_matrix(targets, preds, model_name, ticker, save_path):
    """Gera heatmap da Matriz de Confusão."""
    cm = confusion_matrix(targets, preds, labels=[0, 1, 2])
    cm_pct = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        ax=ax,
        linewidths=0.5,
        linecolor='#2a2a4a',
        cbar_kws={'label': 'Contagem'},
    )

    # Adicionar percentuais em texto menor
    for i in range(3):
        for j in range(3):
            if cm.sum(axis=1)[i] > 0:
                ax.text(
                    j + 0.5, i + 0.75,
                    f'({cm_pct[i, j]:.1f}%)',
                    ha='center', va='center',
                    fontsize=8, color='#aaaaaa',
                )

    ax.set_xlabel('Previsão do Modelo')
    ax.set_ylabel('Classificação Real')
    ax.set_title(f'Matriz de Confusão — {model_name}\n{ticker}', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_temporal_predictions(dates, prices, preds, targets, model_name, ticker, save_path):
    """Gera gráfico temporal com previsões sobrepostas ao preço real."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[3, 1],
                                     sharex=True, gridspec_kw={'hspace': 0.05})

    # ── Painel superior: Preço + Marcadores de previsão ──
    ax1.plot(dates, prices, color='#00d2ff', linewidth=1.2, alpha=0.9, label='Preço de Fechamento')

    # Marcadores: acerto ou erro
    correct_mask = preds == targets
    wrong_mask = ~correct_mask

    # Acertos por classe
    for cls, color, marker, label in [
        (1, '#00ff88', '^', 'BUY correto'),
        (2, '#ff4466', 'v', 'SELL correto'),
        (0, '#888888', 's', 'HOLD correto'),
    ]:
        mask = correct_mask & (preds == cls)
        if mask.sum() > 0:
            ax1.scatter(dates[mask], prices[mask], c=color, marker=marker,
                       s=50, alpha=0.8, label=label, zorder=5, edgecolors='white', linewidths=0.3)

    # Erros (todos em amarelo com X)
    if wrong_mask.sum() > 0:
        ax1.scatter(dates[wrong_mask], prices[wrong_mask], c='#ffaa00', marker='X',
                   s=40, alpha=0.6, label='Erro de Classificação', zorder=4, edgecolors='white', linewidths=0.3)

    ax1.set_ylabel('Preço (R$)')
    ax1.set_title(f'Validação Temporal — {model_name} | {ticker}', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.7)
    ax1.grid(True, alpha=0.3)

    # ── Painel inferior: Barras de acerto/erro ──
    colors = ['#00ff88' if c else '#ff4466' for c in correct_mask]
    ax2.bar(dates, [1]*len(dates), color=colors, alpha=0.7, width=2)
    ax2.set_ylabel('Acerto')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Erro', 'Acerto'])
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b/%Y'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Usando device: {device}")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    # Tabela comparativa global
    summary_rows = []

    for ticker, yf_ticker in TICKERS_MAP.items():
        print(f"\n{'='*60}")
        print(f"  VALIDAÇÃO TEMPORAL — {ticker}")
        print(f"{'='*60}")

        # ── 1. Carregar dados de teste ──
        test_csv = f"data/{ticker}_test.csv"
        test_dataset = TimeSeriesDataset(test_csv, window_size=WINDOW_SIZE)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # ── 2. Buscar preços brutos ──
        print("Baixando preços brutos do Yahoo Finance...")
        raw_prices = fetch_raw_prices(yf_ticker)

        # ── 3. Mapear datas do conjunto de teste ──
        test_df = pd.read_csv(test_csv)
        test_df['Date'] = pd.to_datetime(test_df['Date'])

        # Cada amostra do dataset usa janela [idx : idx+32], label em idx+31
        sample_dates = []
        sample_prices = []
        sample_returns = []
        sample_real_labels = []

        for idx in range(len(test_dataset)):
            row_idx = idx + WINDOW_SIZE - 1  # Índice da linha do label
            date = test_df.iloc[row_idx]['Date']
            sample_dates.append(date)

            # Buscar preço bruto e retorno real
            if date in raw_prices.index:
                price = raw_prices.loc[date, 'Close']
                ret = compute_real_return(raw_prices['Close'], date, HORIZON)
            else:
                # Buscar a data mais próxima
                closest = raw_prices.index[raw_prices.index.get_indexer([date], method='nearest')[0]]
                price = raw_prices.loc[closest, 'Close']
                ret = compute_real_return(raw_prices['Close'], closest, HORIZON)

            sample_prices.append(float(price))
            sample_returns.append(ret if not np.isnan(ret) else 0.0)
            real_label = real_label_from_return(ret) if not np.isnan(ret) else int(test_df.iloc[row_idx]['Label'])
            sample_real_labels.append(real_label)

        sample_dates = np.array(sample_dates)
        sample_prices = np.array(sample_prices)
        sample_returns = np.array(sample_returns)

        # ── 4. Carregar e avaliar os 3 modelos ──
        models_config = [
            ('Baseline CNN', BaselineCNN1D(IN_CHANNELS, seq_len=WINDOW_SIZE),
             f'checkpoints/baseline_{ticker}.pt', False),
            ('MSR-CNN Clássico', MSRCNN1D(IN_CHANNELS, seq_len=WINDOW_SIZE),
             f'checkpoints/msrcnn_{ticker}.pt', False),
        ]

        ticker_results = []

        for model_name, model, ckpt_path, is_attn in models_config:
            print(f"\n  ▸ Avaliando {model_name}...")

            if not os.path.exists(ckpt_path):
                print(f"    ⚠ Checkpoint não encontrado: {ckpt_path}. Pulando.")
                continue

            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            model = model.to(device)

            preds, targets, probs = run_inference(model, test_loader, device, is_attn)

            acc = accuracy_score(targets, preds)
            f1 = f1_score(targets, preds, average='macro')

            print(f"    Acurácia: {acc:.4f} | F1-Macro: {f1:.4f}")
            print(f"\n{classification_report(targets, preds, target_names=LABEL_NAMES, zero_division=0)}")

            # ── Matriz de Confusão ──
            safe_name = model_name.replace(' ', '_').lower()
            cm_path = f"{RESULTS_DIR}/confusion_{safe_name}_{ticker}.png"
            plot_confusion_matrix(targets, preds, model_name, ticker, cm_path)
            print(f"    Matriz de Confusão salva: {cm_path}")

            # ── Gráfico Temporal (apenas para MSR-CNN Clássico para substituir) ──
            if model_name == 'MSR-CNN Clássico':
                temp_path = f"{RESULTS_DIR}/temporal_{safe_name}_{ticker}.png"
                plot_temporal_predictions(
                    sample_dates, sample_prices, preds, targets,
                    model_name, ticker, temp_path
                )
                print(f"    Gráfico temporal salvo: {temp_path}")

            # ── Salvar CSV detalhado ──
            detail_df = pd.DataFrame({
                'Data': sample_dates[:len(preds)],
                'Preco_Fechamento': sample_prices[:len(preds)],
                'Retorno_Real_5d_%': sample_returns[:len(preds)],
                'Label_Real': [LABEL_NAMES[t] for t in targets],
                'Previsao_Modelo': [LABEL_NAMES[p] for p in preds],
                'Acertou': preds == targets,
                'Prob_HOLD': probs[:, 0],
                'Prob_BUY': probs[:, 1],
                'Prob_SELL': probs[:, 2],
            })
            csv_path = f"{RESULTS_DIR}/detalhado_{safe_name}_{ticker}.csv"
            detail_df.to_csv(csv_path, index=False)
            print(f"    CSV detalhado salvo: {csv_path}")

            # Guardar para tabela comparativa
            ticker_results.append({
                'Ticker': ticker,
                'Modelo': model_name,
                'Acurácia': f"{acc:.4f}",
                'F1-Macro': f"{f1:.4f}",
                'Amostras': len(preds),
            })

        summary_rows.extend(ticker_results)

    # ── 5. Tabela comparativa final ──
    print(f"\n{'='*60}")
    print("  TABELA COMPARATIVA — TODOS OS MODELOS")
    print(f"{'='*60}\n")

    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    summary_path = f"{RESULTS_DIR}/tabela_comparativa.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\nTabela salva: {summary_path}")

    print(f"\n✅ Validação concluída! Resultados em: {RESULTS_DIR}/")


if __name__ == '__main__':
    main()
