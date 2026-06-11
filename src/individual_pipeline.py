import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score

# Importar nossas arquiteturas e funções existentes
from models import BaselineCNN1D, MSRCNN1D, MSRCNNAttention1D
from train import FocalLoss
from bulk_pipeline import SEGMENTS, DATA_DIR, CHECKPOINT_DIR, SegmentTimeSeriesDataset

def train_and_eval_individual_model(model_class, model_name, ticker, train_loader, val_loader, test_loader, device):
    model = model_class(in_channels=6, seq_len=32).to(device)
    criterion = FocalLoss(gamma=2)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    best_val_loss = float('inf')
    epochs = 10 
    save_path = f"{CHECKPOINT_DIR}/{model_name}_{ticker}.pt"
    
    for epoch in range(epochs):
        model.train()
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
            
        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                if model_name == 'MSRCNNAttention':
                    outputs, _ = model(X)
                else:
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
            if model_name == 'MSRCNNAttention':
                outputs, _ = model(X)
            else:
                outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro')
    return acc, f1

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Iniciando Pipeline Individual (Device: {device})")
    
    results = []
    
    print("\nIniciando Treinamento e Avaliação por Ativo...")
    for segment, tickers in SEGMENTS.items():
        for ticker in tickers:
            safe_ticker = ticker.replace('^', '').replace('=', '_')
            
            train_file = f"{DATA_DIR}/{safe_ticker}_train.csv"
            val_file = f"{DATA_DIR}/{safe_ticker}_val.csv"
            test_file = f"{DATA_DIR}/{safe_ticker}_test.csv"
            
            # Pula se os dados não existem (ex: ações delistadas como JBSS3, GOLL4)
            if not os.path.exists(train_file):
                print(f"  Pulo: {ticker} (dados não encontrados)")
                continue
                
            train_dataset = SegmentTimeSeriesDataset([train_file])
            val_dataset = SegmentTimeSeriesDataset([val_file])
            test_dataset = SegmentTimeSeriesDataset([test_file])
            
            if len(train_dataset) == 0:
                print(f"  Pulo: {ticker} (dataset vazio)")
                continue
                
            print(f"[{segment}] Treinando ativo {ticker}...")
            
            train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
            
            acc_base, f1_base = train_and_eval_individual_model(BaselineCNN1D, 'Baseline', safe_ticker, train_loader, val_loader, test_loader, device)
            acc_msr, f1_msr = train_and_eval_individual_model(MSRCNN1D, 'MSRCNN', safe_ticker, train_loader, val_loader, test_loader, device)
            acc_att, f1_att = train_and_eval_individual_model(MSRCNNAttention1D, 'MSRCNNAttention', safe_ticker, train_loader, val_loader, test_loader, device)
            
            results.append({
                'Segmento': segment,
                'Ativo': ticker,
                'Baseline_Acc': acc_base, 'Baseline_F1': f1_base,
                'MSRCNN_Acc': acc_msr, 'MSRCNN_F1': f1_msr,
                'Attention_Acc': acc_att, 'Attention_F1': f1_att
            })
            
    print("\nConsolidando Resultados Individuais...")
    df_results = pd.DataFrame(results)
    df_results.to_csv('results/individual_comparativo.csv', index=False)
    
    # Gerando um resumo de vitórias do MSRCNN_Attention
    df_results['Winner_Acc'] = df_results[['Baseline_Acc', 'MSRCNN_Acc', 'Attention_Acc']].idxmax(axis=1)
    df_results['Winner_F1'] = df_results[['Baseline_F1', 'MSRCNN_F1', 'Attention_F1']].idxmax(axis=1)
    
    print("\nEstatísticas de Vitória por Acurácia:")
    print(df_results['Winner_Acc'].value_counts())
    
    print("\nPipeline individual concluído com sucesso!")

if __name__ == '__main__':
    main()
