import pandas as pd
import glob
import os

files = glob.glob('data/*.csv')
dist = {}

for f in files:
    df = pd.read_csv(f)
    if 'Label' in df.columns:
        counts = df['Label'].value_counts().to_dict()
        total = sum(counts.values())
        
        # split filename to get ticker and split type
        basename = os.path.basename(f)
        parts = basename.split('_')
        split_type = parts[-1].split('.')[0] # train, test, val
        ticker = "_".join(parts[:-1])
        
        if split_type not in dist:
            dist[split_type] = {0:0, 1:0, 2:0}
            
        for k, v in counts.items():
            dist[split_type][k] += v

for split_type, counts in dist.items():
    total = sum(counts.values())
    print(f"\nDistribuição no conjunto de {split_type.upper()}:")
    for label, name in [(0, 'HOLD'), (1, 'BUY'), (2, 'SELL')]:
        count = counts.get(label, 0)
        pct = (count / total) * 100 if total > 0 else 0
        print(f"  {name}: {count} amostras ({pct:.2f}%)")
