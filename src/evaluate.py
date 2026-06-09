import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, f1_score, accuracy_score
import os

from models import BaselineCNN1D, MSRCNN1D
from train import TimeSeriesDataset
from torch.utils.data import DataLoader

def evaluate_model(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return acc, f1, all_targets, all_preds

def plot_fft_filters(model, save_path):
    """
    Extrai os pesos dos filtros ASD e plota a resposta em frequência.
    O objetivo é ver se eles se especializaram em:
    - Ruído (Camada 1, High-pass)
    - Sazonalidade (Camada 2, High-pass a partir da baixa frequência)
    - Tendência (Camada 2, Low-pass)
    """
    # Pegar pesos do primeiro canal apenas para visualização
    # L1_U (Alta Freq 1)
    w_l1_u = model.asd.L1_U.weight.data[0, 0, :].cpu().numpy()
    # L1_L (Baixa Freq 1)
    w_l1_l = model.asd.L1_L.weight.data[0, 0, :].cpu().numpy()
    
    # L2_U e L2_L
    w_l2_u = model.asd.L2_U.weight.data[0, 0, :].cpu().numpy()
    w_l2_l = model.asd.L2_L.weight.data[0, 0, :].cpu().numpy()
    
    # Calculando a Transformada de Fourier (FFT) para ver a Resposta em Frequência
    fft_noise = np.abs(np.fft.fft(w_l1_u, n=64))
    
    # Para as camadas 2, a resposta efetiva no sinal original é a convolução com a camada 1 (no domínio do tempo)
    # ou a multiplicação no domínio da frequência
    fft_l1_l = np.abs(np.fft.fft(w_l1_l, n=64))
    fft_l2_u = np.abs(np.fft.fft(w_l2_u, n=64))
    fft_l2_l = np.abs(np.fft.fft(w_l2_l, n=64))
    
    fft_seasonality = fft_l1_l * fft_l2_u
    fft_trend = fft_l1_l * fft_l2_l
    
    freqs = np.fft.fftfreq(64)
    # Pegar apenas a parte positiva
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    fft_noise = fft_noise[pos_mask]
    fft_seasonality = fft_seasonality[pos_mask]
    fft_trend = fft_trend[pos_mask]
    
    plt.figure(figsize=(10, 6))
    plt.plot(freqs, fft_noise, label='Ruído (Passa-Alta)', color='red')
    plt.plot(freqs, fft_seasonality, label='Sazonalidade (Freq. Média / Passa-Banda)', color='green')
    plt.plot(freqs, fft_trend, label='Tendência (Passa-Baixa)', color='blue')
    plt.title('Resposta em Frequência dos Filtros Aprendidos (ASD)')
    plt.xlabel('Frequência Normalizada')
    plt.ylabel('Magnitude')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    tickers = ['BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
    window_size = 32
    in_channels = 6
    
    os.makedirs('results', exist_ok=True)
    
    for ticker in tickers:
        print(f"\\n{'='*40}\\nAvaliando para {ticker}\\n{'='*40}")
        test_dataset = TimeSeriesDataset(f"data/{ticker}_test.csv", window_size=window_size)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        # Avaliar Baseline
        baseline = BaselineCNN1D(in_channels, seq_len=window_size).to(device)
        baseline.load_state_dict(torch.load(f"checkpoints/baseline_{ticker}.pt", map_location=device))
        
        acc_b, f1_b, _, _ = evaluate_model(baseline, test_loader, device)
        print(f"Baseline CNN -> Acc: {acc_b:.4f} | F1-Macro: {f1_b:.4f}")
        
        # Avaliar MSR-CNN
        msrcnn = MSRCNN1D(in_channels, seq_len=window_size).to(device)
        msrcnn.load_state_dict(torch.load(f"checkpoints/msrcnn_{ticker}.pt", map_location=device))
        
        acc_m, f1_m, _, _ = evaluate_model(msrcnn, test_loader, device)
        print(f"MSR-CNN Clássico -> Acc: {acc_m:.4f} | F1-Macro: {f1_m:.4f}")
        
        # Plotar FFT
        plot_fft_filters(msrcnn, f"results/fft_filters_{ticker}.png")
        print(f"Gráfico de resposta em frequência salvo em results/fft_filters_{ticker}.png")

if __name__ == '__main__':
    main()
