import os, sys, gc, time
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Test search root
print("Testing vectorized cell 7 logic...")
t0 = time.time()

# Check if work/output/matching_results.tsv exists
baseline_tsv = Path("work/output/matching_results.tsv")
if not baseline_tsv.exists():
    print(f"File {baseline_tsv} not found, searching...")
    # search
    for p in Path(".").rglob("*matching*.tsv"):
        print(f"Found: {p}")
        baseline_tsv = p
        break

if baseline_tsv.exists():
    print(f"Reading {baseline_tsv}...")
    df_prev = pd.read_csv(baseline_tsv, sep='\t', dtype=str)
    print(f"df_prev shape: {df_prev.shape}")
    s1_col = df_prev.columns[0]
    m_col = df_prev.columns[1]
    
    # 1. Explode to pairs
    df_valid = df_prev[df_prev[m_col].notna() & (df_prev[m_col] != '') & (df_prev[m_col] != 'nan')].copy()
    print(f"Non-empty rows: {len(df_valid)}")
    
    # Split matches
    df_valid['candidate_id'] = df_valid[m_col].str.split(',')
    df_pairs = df_valid.explode('candidate_id')[[s1_col, 'candidate_id']]
    df_pairs.rename(columns={s1_col: 's1_id'}, inplace=True)
    df_pairs['candidate_id'] = df_pairs['candidate_id'].str.strip()
    df_pairs['score'] = 0.95
    print(f"Total baseline pairs: {len(df_pairs):,}")
    
    # Check duplicate targets
    dup_targets = df_pairs['candidate_id'].duplicated().sum()
    print(f"Duplicate target assignments in baseline: {dup_targets:,}")
    
    # Enforce 1-to-1 target exclusivity
    df_pairs = df_pairs.drop_duplicates(subset=['candidate_id'], keep='first')
    print(f"Pairs after exclusivity: {len(df_pairs):,}")
    print(f"Finished in {time.time()-t0:.2f}s")
