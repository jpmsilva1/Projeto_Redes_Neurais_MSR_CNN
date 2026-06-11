import pandas as pd
import glob

files = glob.glob('data/*.csv')
total_counts = {0: 0, 1: 0, 2: 0}

for f in files:
    df = pd.read_csv(f)
    if 'Label' in df.columns:
        counts = df['Label'].value_counts().to_dict()
        for k, v in counts.items():
            total_counts[k] += v

total_samples = sum(total_counts.values())
print("Distribuição Total de Classes (Todos os Datasets):")
for label, name in [(0, 'HOLD'), (1, 'BUY'), (2, 'SELL')]:
    count = total_counts.get(label, 0)
    pct = (count / total_samples) * 100 if total_samples > 0 else 0
    print(f"{name}: {count} amostras ({pct:.2f}%)")
