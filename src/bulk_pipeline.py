import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score

# Importar nossas arquiteturas e funções existentes
from models import BaselineCNN1D, MSRCNN1D
from train import FocalLoss

# ==========================================
# 1. DEFINIÇÃO DOS SEGMENTOS
# ==========================================
SEGMENTS = {
    'BlueChips': ['VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 
                  'ABEV3.SA', 'WEGE3.SA', 'RENT3.SA', 'SUZB3.SA', 'BPAC11.SA', 
                  'EQTL3.SA', 'RADL3.SA', 'B3SA3.SA', 'VIVT3.SA', 'JBSS3.SA'],
                  
    'SmallCaps': ['MGLU3.SA', 'COGN3.SA', 'YDUQ3.SA', 'CVCB3.SA', 'ALPA4.SA', 
                  'TASA4.SA', 'GOLL4.SA', 'AZUL4.SA', 'LWSA3.SA', 'MRVE3.SA', 
                  'EZTC3.SA', 'POMO4.SA', 'RAPT4.SA', 'TEND3.SA', 'DIRR3.SA'],
                  
    'Commodities': ['GC=F', 'SI=F', 'CL=F', 'BZ=F', 'ZS=F', 
                    'ZC=F', 'KC=F', 'LE=F', 'SB=F', 'HG=F'],
                    
    'FIIs': ['KNRI11.SA', 'HGLG11.SA', 'MXRF11.SA', 'XPLG11.SA', 'VISC11.SA', 
             'BTLG11.SA', 'CPTS11.SA', 'IRDM11.SA', 'VILG11.SA', 'RECR11.SA', 
             'TGAR11.SA', 'HGRE11.SA', 'BRCR11.SA', 'HGBS11.SA', 'ALZR11.SA'],
             
    'BDRs': ['AAPL34.SA', 'MSFT34.SA', 'AMZO34.SA', 'GOGL34.SA', 'META34.SA', 
             'TSLA34.SA', 'NVDC34.SA', 'MELI34.SA', 'BABA34.SA', 'DISB34.SA']
}

DATA_DIR = './data_bulk'
CHECKPOINT_DIR = './checkpoints_bulk'
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs('./results', exist_ok=True)

# ==========================================
# 2. PIPELINE DE DADOS (Criação de CSVs)
# ==========================================
def process_ticker_data(ticker):
    try:
        # Pega de 2020 em diante para ter dados o suficiente, validando até 2025
        df = yf.download(ticker, start='2020-01-01', end='2025-01-01', progress=False)
        if df.empty or len(df) < 200:
            return False
            
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.droplevel(1)
            
        df.dropna(inplace=True)
        
        # Feature Engineering
        df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
        df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
        macd = ta.trend.MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
        df['Volume'] = df['Volume'].astype(float)
        
        # Labels
        horizon = 5
        threshold = 0.015
        df[f'Future_Return_{horizon}d'] = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
        conditions = [
            (df[f'Future_Return_{horizon}d'] > threshold),
            (df[f'Future_Return_{horizon}d'] < -threshold)
        ]
        df['Label'] = np.select(conditions, [1, 2], default=0)
        df.dropna(inplace=True)
        
        # Split e Normalização Temporal
        features = ['Log_Return', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
        target = 'Label'
        
        n = len(df)
        train_end = int(n * 0.7)
        val_end = int(n * 0.85)
        
        train_df = df.iloc[:train_end].copy()
        val_df = df.iloc[train_end:val_end].copy()
        test_df = df.iloc[val_end:].copy()
        
        scaler = StandardScaler()
        train_df[features] = scaler.fit_transform(train_df[features])
        val_df[features] = scaler.transform(val_df[features])
        test_df[features] = scaler.transform(test_df[features])
        
        safe_ticker = ticker.replace('^', '').replace('=', '_')
        train_df[features + [target]].to_csv(f"{DATA_DIR}/{safe_ticker}_train.csv")
        val_df[features + [target]].to_csv(f"{DATA_DIR}/{safe_ticker}_val.csv")
        test_df[features + [target]].to_csv(f"{DATA_DIR}/{safe_ticker}_test.csv")
        return True
    except Exception as e:
        print(f"Erro ao processar {ticker}: {e}")
        return False

# ==========================================
# 3. DATASET PARA SEGMENTOS (Agrupador)
# ==========================================
class SegmentTimeSeriesDataset(Dataset):
    """Junta múltiplos arquivos CSV de ativos diferentes em um único Dataset sem cruzar janelas."""
    def __init__(self, file_paths, window_size=32):
        self.X = []
        self.y = []
        
        for file_path in file_paths:
            if not os.path.exists(file_path): continue
            df = pd.read_csv(file_path, index_col=0)
            
            features = [c for c in df.columns if c != 'Label']
            data_X = df[features].values
            data_y = df['Label'].values
            
            for i in range(len(df) - window_size):
                self.X.append(data_X[i:(i + window_size)])
                self.y.append(data_y[i + window_size])
                
        # Se algum segmento falhar em todos, evita crash
        if len(self.X) > 0:
            self.X = torch.tensor(np.array(self.X), dtype=torch.float32)
            self.y = torch.tensor(np.array(self.y), dtype=torch.long)
        else:
            self.X, self.y = None, None
            
    def __len__(self):
        return len(self.X) if self.X is not None else 0
        
    def __getitem__(self, idx):
        return self.X[idx].transpose(0, 1), self.y[idx]

# ==========================================
# 4. LOOP DE TREINAMENTO E AVALIAÇÃO
# ==========================================
def train_and_eval_segment_model(model_class, model_name, segment, train_loader, val_loader, test_loader, device):
    print(f"  Treinando {model_name} para {segment}...")
    model = model_class(in_channels=6, seq_len=32).to(device)
    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    epochs = 10 # Early stopping rápido provado pelos gráficos de learning curves
    save_path = f"{CHECKPOINT_DIR}/{model_name}_{segment}.pt"
    
    for epoch in range(epochs):
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
            
        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, y)
                val_loss += loss.item() * X.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            
    # Avaliação final no Teste
    model.load_state_dict(torch.load(save_path))
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return acc, f1

# ==========================================
# 5. ORQUESTRAÇÃO PRINCIPAL
# ==========================================
def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Iniciando Pipeline de Larga Escala (Device: {device})")
    
    # Baixar todos os dados
    print("\n[1/3] Processando dados dos 65 ativos...")
    for segment, tickers in SEGMENTS.items():
        for ticker in tickers:
            safe_ticker = ticker.replace('^', '').replace('=', '_')
            if not os.path.exists(f"{DATA_DIR}/{safe_ticker}_train.csv"):
                process_ticker_data(ticker)
                
    results = []
    
    # Loop de Treino por Segmento
    print("\n[2/3] Iniciando Treinamento e Avaliação por Segmento...")
    for segment, tickers in SEGMENTS.items():
        print(f"\n--- Segmento: {segment} ---")
        train_files = [f"{DATA_DIR}/{t.replace('^', '').replace('=', '_')}_train.csv" for t in tickers]
        val_files = [f"{DATA_DIR}/{t.replace('^', '').replace('=', '_')}_val.csv" for t in tickers]
        test_files = [f"{DATA_DIR}/{t.replace('^', '').replace('=', '_')}_test.csv" for t in tickers]
        
        train_dataset = SegmentTimeSeriesDataset(train_files)
        val_dataset = SegmentTimeSeriesDataset(val_files)
        test_dataset = SegmentTimeSeriesDataset(test_files)
        
        if len(train_dataset) == 0:
            print(f"Erro: Sem dados para {segment}")
            continue
            
        print(f"Amostras: Treino({len(train_dataset)}) | Val({len(val_dataset)}) | Teste({len(test_dataset)})")
        
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
        
        # Baseline
        acc_base, f1_base = train_and_eval_segment_model(BaselineCNN1D, 'Baseline', segment, train_loader, val_loader, test_loader, device)
        # MSRCNN
        acc_msr, f1_msr = train_and_eval_segment_model(MSRCNN1D, 'MSRCNN', segment, train_loader, val_loader, test_loader, device)
        
        results.append({
            'Segmento': segment,
            'Baseline_Acc': acc_base, 'Baseline_F1': f1_base,
            'MSRCNN_Acc': acc_msr, 'MSRCNN_F1': f1_msr
        })
        
    print("\n[3/3] Consolidando Resultados...")
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/segmentos_comparativo.csv', index=False)
    print(df_results)
    print("\nTeste em larga escala concluído com sucesso!")

if __name__ == '__main__':
    main()
