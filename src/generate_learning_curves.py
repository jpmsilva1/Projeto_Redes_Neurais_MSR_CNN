import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import seaborn as sns

from models import BaselineCNN1D, MSRCNN1D, MSRCNNAttention1D
from train import TimeSeriesDataset, FocalLoss

def train_and_plot(model, model_name, ticker, train_loader, val_loader, device, epochs=30, lr=1e-3):
    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(epochs):
        # Treino
        model.train()
        train_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            
            if model_name == 'MSRCNNAttention':
                outputs, _ = model(X)
            else:
                outputs = model(X)
                
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        # Validação
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                
                if model_name == 'MSRCNNAttention':
                    outputs, _ = model(X)
                else:
                    outputs = model(X)
                    
                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    epochs_range = range(1, epochs + 1)
    
    # Loss Plot
    ax1.plot(epochs_range, history['train_loss'], label='Train Loss', color='blue', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], label='Validation Loss', color='orange', linewidth=2)
    ax1.axvline(x=best_epoch + 1, color='red', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    ax1.set_title(f'Loss Evolution - {model_name} ({ticker})')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Focal Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy Plot
    ax2.plot(epochs_range, history['val_acc'], label='Validation Accuracy', color='green', linewidth=2)
    ax2.axvline(x=best_epoch + 1, color='red', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    ax2.set_title(f'Validation Accuracy - {model_name} ({ticker})')
    ax2.set_xlabel('Epochs')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs('results/learning_curves', exist_ok=True)
    save_path = f"results/learning_curves/curve_{model_name}_{ticker}.png"
    plt.savefig(save_path)
    plt.close()
    
    print(f"Curvas salvas para {model_name} ({ticker}) -> Sweet Spot na época {best_epoch+1}")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    # Para ser rápido e não gastar horas, vamos pegar só a PETR4.SA como exemplo
    # O usuário pode rodar para os outros mudando a lista
    tickers = ['PETR4.SA'] 
    window_size = 32
    in_channels = 6
    epochs = 30 # MSR-CNN Attention treinava 15, base treinava 30. Vamos rodar 30 para todos para ver o overfitting.
    
    for ticker in tickers:
        print(f"\nGerando curvas de aprendizado para {ticker}...")
        
        train_dataset = TimeSeriesDataset(f"data/{ticker}_train.csv", window_size=window_size)
        val_dataset = TimeSeriesDataset(f"data/{ticker}_val.csv", window_size=window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Baseline
        print("Treinando Baseline...")
        model_base = BaselineCNN1D(in_channels, seq_len=window_size).to(device)
        train_and_plot(model_base, 'BaselineCNN', ticker, train_loader, val_loader, device, epochs=epochs)
        
        # MSR-CNN Clássico
        print("Treinando MSR-CNN Clássico...")
        model_msrcnn = MSRCNN1D(in_channels, seq_len=window_size).to(device)
        train_and_plot(model_msrcnn, 'MSRCNN_Classico', ticker, train_loader, val_loader, device, epochs=epochs)
        
        # MSR-CNN Attention
        print("Treinando MSR-CNN Attention...")
        model_attn = MSRCNNAttention1D(in_channels, seq_len=window_size).to(device)
        train_and_plot(model_attn, 'MSRCNNAttention', ticker, train_loader, val_loader, device, epochs=epochs)

if __name__ == '__main__':
    sns.set_theme(style='darkgrid')
    main()
