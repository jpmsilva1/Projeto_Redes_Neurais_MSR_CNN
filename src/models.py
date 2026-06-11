import torch
import torch.nn as nn

class AsymmetricASD1D(nn.Module):
    """
    Decomposição Adaptativa em Subbandas (ASD) 1D Asímetrica em 3 Subbandas.
    Camada 1 gera Alta Frequência (Ruído) e Baixa Frequência.
    Camada 2 decompõe a Baixa Frequência em Média Freq (Sazonalidade) e Baixa Freq (Tendência).
    """
    def __init__(self, in_channels, filter_size=5):
        super(AsymmetricASD1D, self).__init__()
        
        padding = filter_size // 2
        
        # Camada 1
        self.L1_U = nn.Conv1d(in_channels, in_channels, kernel_size=filter_size, padding=padding, groups=in_channels)
        self.L1_L = nn.Conv1d(in_channels, in_channels, kernel_size=filter_size, padding=padding, groups=in_channels)
        
        # Camada 2 (Atua apenas na saída L1_L)
        self.L2_U = nn.Conv1d(in_channels, in_channels, kernel_size=filter_size, padding=padding, groups=in_channels)
        self.L2_L = nn.Conv1d(in_channels, in_channels, kernel_size=filter_size, padding=padding, groups=in_channels)
        
        # O artigo usa decimação por 2 após cada filtro
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # x shape: (batch, in_channels, sequence_length)
        
        # Camada 1
        y1_u = self.pool(self.L1_U(x)) # Alta frequência 1 (Ruído)
        y1_l = self.pool(self.L1_L(x)) # Baixa frequência 1
        
        # Camada 2
        y2_u = self.pool(self.L2_U(y1_l)) # Alta frequência 2 (Sazonalidade)
        y2_l = self.pool(self.L2_L(y1_l)) # Baixa frequência 2 (Tendência)
        
        # Precisamos igualar as dimensões temporais para as 3 subbandas
        # y1_u sofreu 1 decimação, y2_u e y2_l sofreram 2 decimações.
        # Faremos max_pool em y1_u para igualar
        y1_u_sync = self.pool(y1_u)
        
        # Retorna: Ruído, Sazonalidade, Tendência
        return y1_u_sync, y2_u, y2_l


class SubbandCNN(nn.Module):
    """CNN simples para processar cada subbanda de forma independente"""
    def __init__(self, in_channels, out_channels=16):
        super(SubbandCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels*2, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool1d(2)
        
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        return self.flatten(x)


class MSRCNN1D(nn.Module):
    """
    MSR-CNN Clássico (sem atenção).
    Usa o ASD 1D para gerar 3 subbandas e processa cada uma com uma CNN isolada.
    """
    def __init__(self, in_channels, seq_len, num_classes=3):
        super(MSRCNN1D, self).__init__()
        self.asd = AsymmetricASD1D(in_channels=in_channels, filter_size=5)
        
        self.cnn_noise = SubbandCNN(in_channels, out_channels=16)
        self.cnn_seasonality = SubbandCNN(in_channels, out_channels=16)
        self.cnn_trend = SubbandCNN(in_channels, out_channels=16)
        
        # Para calcular a dimensão final do Flatten (depende do seq_len original)
        # O ASD faz 2 decimações (seq_len // 4)
        # Cada SubbandCNN faz mais 2 decimações ((seq_len // 4) // 4) = seq_len // 16
        final_seq = seq_len // 16
        if final_seq == 0:
            final_seq = 1 # Fallback caso a janela seja pequena
            
        cnn_out_features = 32 * final_seq # 32 é out_channels*2 da SubbandCNN
        
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_features * 3, 128), # *3 por causa das 3 subbandas
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        noise, seasonality, trend = self.asd(x)
        
        feat_noise = self.cnn_noise(noise)
        feat_seasonality = self.cnn_seasonality(seasonality)
        feat_trend = self.cnn_trend(trend)
        
        # Concatena as features extraídas
        combined = torch.cat([feat_noise, feat_seasonality, feat_trend], dim=1)
        
        out = self.fc(combined)
        return out


class BaselineCNN1D(nn.Module):
    """
    CNN Convencional Full-Band.
    Mesmo número de parâmetros das subbands somadas para comparação justa.
    """
    def __init__(self, in_channels, seq_len, num_classes=3):
        super(BaselineCNN1D, self).__init__()
        
        # O MSR-CNN tem 3 subbands com 16 canais iniciais = 48
        self.conv1 = nn.Conv1d(in_channels, 48, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.1)
        self.pool1 = nn.MaxPool1d(2)
        
        self.conv2 = nn.Conv1d(48, 96, kernel_size=3, padding=1)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool2 = nn.MaxPool1d(2)
        
        # Simular a quantidade de decimações totais do MSR-CNN para comparar o tempo final
        self.pool_extra = nn.MaxPool1d(4)
        
        self.flatten = nn.Flatten()
        
        final_seq = seq_len // 16
        if final_seq == 0:
            final_seq = 1
            
        self.fc = nn.Sequential(
            nn.Linear(96 * final_seq, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.pool_extra(x)
        x = self.flatten(x)
        out = self.fc(x)
        return out

