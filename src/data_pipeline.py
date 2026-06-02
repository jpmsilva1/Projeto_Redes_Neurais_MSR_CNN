import yfinance as yf
import pandas as pd
import numpy as np
import ta
import os
from sklearn.preprocessing import StandardScaler

# Configurações globais
TICKERS = ['^BVSP', 'PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
START_DATE = '2010-01-01'
END_DATE = '2025-01-01'
HORIZON = 5 # Janela de 5 dias para o label
DATA_DIR = './data'

# Cria a pasta de dados se não existir
os.makedirs(DATA_DIR, exist_ok=True)

def fetch_data(ticker):
    print(f"Baixando dados para {ticker}...")
    df = yf.download(ticker, start=START_DATE, end=END_DATE)
    # Se baixar múltiplos níveis por causa do yfinance novo, tratar colunas
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    df.dropna(inplace=True)
    return df

def feature_engineering(df):
    print("Calculando indicadores técnicos...")
    # Log Returns
    df['Log_Return'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # RSI (Sazonalidade/Momento)
    df['RSI'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
    
    # MACD (Tendência)
    macd = ta.trend.MACD(close=df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    
    # ATR (Volatilidade/Ruído)
    df['ATR'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()
    
    # Volume
    df['Volume'] = df['Volume'].astype(float)
    
    df.dropna(inplace=True)
    return df

def generate_labels(df, horizon=5, threshold=0.015):
    """
    Classifica o retorno futuro em H dias.
    Se retorno_H > threshold -> BUY (1)
    Se retorno_H < -threshold -> SELL (2)
    Caso contrário -> HOLD (0)
    """
    # Retorno futuro de H dias
    df[f'Future_Return_{horizon}d'] = (df['Close'].shift(-horizon) - df['Close']) / df['Close']
    
    conditions = [
        (df[f'Future_Return_{horizon}d'] > threshold),
        (df[f'Future_Return_{horizon}d'] < -threshold)
    ]
    choices = [1, 2] # 1=BUY, 2=SELL
    
    df['Label'] = np.select(conditions, choices, default=0) # 0=HOLD
    df.dropna(inplace=True)
    return df

def split_and_normalize(df, ticker):
    # Features que usaremos na rede
    features = ['Log_Return', 'Volume', 'RSI', 'MACD', 'MACD_Signal', 'ATR']
    target = 'Label'
    
    # Split temporal (70% treino, 15% validação, 15% teste)
    n = len(df)
    train_end = int(n * 0.7)
    val_end = int(n * 0.85)
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    
    # Normalização fitada apenas no treino
    scaler = StandardScaler()
    train_df[features] = scaler.fit_transform(train_df[features])
    val_df[features] = scaler.transform(val_df[features])
    test_df[features] = scaler.transform(test_df[features])
    
    # Salvar
    train_df[features + [target]].to_csv(f"{DATA_DIR}/{ticker.replace('^', '')}_train.csv")
    val_df[features + [target]].to_csv(f"{DATA_DIR}/{ticker.replace('^', '')}_val.csv")
    test_df[features + [target]].to_csv(f"{DATA_DIR}/{ticker.replace('^', '')}_test.csv")
    
    print(f"Dados salvos para {ticker}.")

def main():
    for ticker in TICKERS:
        df = fetch_data(ticker)
        df = feature_engineering(df)
        df = generate_labels(df, HORIZON)
        split_and_normalize(df, ticker)
        
if __name__ == '__main__':
    main()
