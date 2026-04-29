# src/data/ingest.py

import pandas as pd
import os

def load_data(path: str) -> pd.DataFrame:
    """Raw CSV load karta hai"""
    
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ File nahi mili: {path}")
    
    df = pd.read_csv(path, index_col=0)
    
    print("=" * 50)
    print("📊 DATASET LOADED SUCCESSFULLY")
    print("=" * 50)
    print(f"Rows    : {df.shape[0]:,}")
    print(f"Columns : {df.shape[1]}")
    print(f"Size    : {df.memory_usage().sum() / 1024**2:.2f} MB")
    print(f"\nColumns : {df.columns.tolist()}")
    
    return df

if __name__ == "__main__":
    df = load_data("data/raw/cs-training.csv")
    print("\nFirst 3 rows:")
    print(df.head(3))