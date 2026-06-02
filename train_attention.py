import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from models import MSRCNNAttention1D
from train import TimeSeriesDataset, FocalLoss

def train_attention_model(model, train_loader, val_loader, device, epochs=15, lr=1e-3, save_path='model_attn.pt'):
    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs, _ = model(X) # Ignorar attn_weights no treino
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * X.size(0)
            
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs, _ = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)
                
                _, predicted = torch.max(outputs, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
                
        val_loss /= len(val_loader.dataset)
        val_acc = correct / total
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"--> Modelo salvo em {save_path}!")

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    tickers = ['BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
    window_size = 32
    in_channels = 6
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for ticker in tickers:
        print(f"\\n{'='*40}\\nTreinando Atenção para {ticker}\\n{'='*40}")
        
        train_dataset = TimeSeriesDataset(f"data/{ticker}_train.csv", window_size=window_size)
        val_dataset = TimeSeriesDataset(f"data/{ticker}_val.csv", window_size=window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        model_attn = MSRCNNAttention1D(in_channels, seq_len=window_size).to(device)
        train_attention_model(model_attn, train_loader, val_loader, device, epochs=15, 
                              save_path=f"checkpoints/msrcnn_attn_{ticker}.pt")

if __name__ == '__main__':
    main()
