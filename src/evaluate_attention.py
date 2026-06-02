import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, f1_score
from models import MSRCNNAttention1D
from train import TimeSeriesDataset
from torch.utils.data import DataLoader

def evaluate_attention(model, test_loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    
    # Vamos guardar as médias dos pesos de atenção para cada classe predita
    attn_by_class = {0: [], 1: [], 2: []}
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs, attn_weights = model(X)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
            # attn_weights tem shape [batch_size, 3]
            for i in range(len(predicted)):
                c = predicted[i].item()
                w = attn_weights[i].cpu().numpy()
                attn_by_class[c].append(w)
                
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    
    # Calcular a média dos pesos de atenção para cada classe
    avg_attn = {}
    for c in [0, 1, 2]:
        if len(attn_by_class[c]) > 0:
            avg_attn[c] = np.mean(attn_by_class[c], axis=0)
        else:
            avg_attn[c] = np.array([0.0, 0.0, 0.0])
            
    return acc, f1, avg_attn

def plot_attention_weights(avg_attn, save_path):
    classes = ['HOLD', 'BUY', 'SELL']
    subbands = ['Ruído', 'Sazonalidade', 'Tendência']
    
    # Matrix 3x3 (Classes x Subbandas)
    attn_matrix = np.array([avg_attn[0], avg_attn[1], avg_attn[2]])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(attn_matrix, cmap='viridis')
    
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{attn_matrix[i, j]:.3f}", ha='center', va='center', color='white')
            
    ax.set_xticks(np.arange(len(subbands)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(subbands)
    ax.set_yticklabels(classes)
    
    plt.title('Pesos Médios de Atenção por Classe Predita', pad=20)
    plt.colorbar(cax)
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
        print(f"\\n{'='*40}\\nAvaliando Atenção para {ticker}\\n{'='*40}")
        test_dataset = TimeSeriesDataset(f"data/{ticker}_test.csv", window_size=window_size)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        msrcnn_attn = MSRCNNAttention1D(in_channels, seq_len=window_size).to(device)
        msrcnn_attn.load_state_dict(torch.load(f"checkpoints/msrcnn_attn_{ticker}.pt", map_location=device))
        
        acc, f1, avg_attn = evaluate_attention(msrcnn_attn, test_loader, device)
        print(f"MSR-CNN Attention -> Acc: {acc:.4f} | F1-Macro: {f1:.4f}")
        
        plot_attention_weights(avg_attn, f"results/attention_weights_{ticker}.png")
        print(f"Gráfico de pesos de atenção salvo em results/attention_weights_{ticker}.png")

if __name__ == '__main__':
    main()
