import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
from models import MSRCNN1D, BaselineCNN1D

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.alpha is not None:
            alpha_t = self.alpha[targets]
            focal_loss = alpha_t * focal_loss
            
        return focal_loss.mean()

class TimeSeriesDataset(Dataset):
    def __init__(self, csv_file, window_size=20):
        self.data = pd.read_csv(csv_file)
        self.features = ['Log_Return', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
        self.window_size = window_size
        
        # O label 0=HOLD, 1=BUY, 2=SELL
        self.X = self.data[self.features].values
        self.y = self.data['Label'].values
        
    def __len__(self):
        return len(self.X) - self.window_size
        
    def __getitem__(self, idx):
        # Janela deslizante: X shape -> (in_channels, seq_len)
        x_window = self.X[idx : idx + self.window_size]
        x_tensor = torch.tensor(x_window, dtype=torch.float32).transpose(0, 1)
        y_tensor = torch.tensor(self.y[idx + self.window_size - 1], dtype=torch.long)
        return x_tensor, y_tensor

def train_model(model, train_loader, val_loader, device, epochs=30, lr=1e-3, save_path='model.pt'):
    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            optimizer.zero_grad()
            outputs = model(X)
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
                outputs = model(X)
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
    # Usar MPS se disponível (Mac M1/M2/M3), caso contrário CPU
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Usando device: {device}")
    
    tickers = ['BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
    window_size = 32 # Importante ser múltiplo de 16 por causa das 4 decimações (poolings) da rede
    in_channels = 6
    
    os.makedirs('checkpoints', exist_ok=True)
    
    for ticker in tickers:
        print(f"\\n{'='*40}\\nTreinando para {ticker}\\n{'='*40}")
        
        train_dataset = TimeSeriesDataset(f"data/{ticker}_train.csv", window_size=window_size)
        val_dataset = TimeSeriesDataset(f"data/{ticker}_val.csv", window_size=window_size)
        
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
        
        # Treinar Baseline CNN
        print("Treinando Baseline CNN...")
        model_baseline = BaselineCNN1D(in_channels, seq_len=window_size).to(device)
        train_model(model_baseline, train_loader, val_loader, device, epochs=15, 
                    save_path=f"checkpoints/baseline_{ticker}.pt")
        
        # Treinar MSR-CNN
        print("\\nTreinando MSR-CNN Clássico...")
        model_msr = MSRCNN1D(in_channels, seq_len=window_size).to(device)
        train_model(model_msr, train_loader, val_loader, device, epochs=15, 
                    save_path=f"checkpoints/msrcnn_{ticker}.pt")

if __name__ == '__main__':
    main()
