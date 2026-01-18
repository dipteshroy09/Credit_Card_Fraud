"""Script to create a sample CSV file for testing the fraud detection app."""

import pandas as pd
import numpy as np
from pathlib import Path

# Read a subset of the original data
print("Reading data...")
df = pd.read_csv("data/creditcard.csv", nrows=1000)

# Create a balanced sample with both normal and fraud cases
normal_df = df[df["Class"] == 0].sample(n=90, random_state=42)
fraud_df = df[df["Class"] == 1].sample(n=min(10, len(df[df["Class"] == 1])), random_state=42)

# Combine and shuffle
sample_df = pd.concat([normal_df, fraud_df]).sample(frac=1, random_state=42).reset_index(drop=True)

# Save to sample file (in data directory but with a specific name we can track)
output_path = Path("data/creditcard_sample.csv")
sample_df.to_csv(output_path, index=False)

print(f"\n[OK] Created sample CSV: {output_path}")
print(f"   Total rows: {len(sample_df)}")
print(f"   Normal transactions: {(sample_df['Class'] == 0).sum()}")
print(f"   Fraud transactions: {sample_df['Class'].sum()}")
print(f"   File size: {output_path.stat().st_size / 1024:.2f} KB")
print(f"\nColumns: {list(sample_df.columns)}")
