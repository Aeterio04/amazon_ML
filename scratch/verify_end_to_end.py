import os, sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Test end-to-end
work_dir = Path("work/test_sim")
work_dir.mkdir(parents=True, exist_ok=True)

# 1. Create a dummy test_source1.tsv with 10 entities
df_s1 = pd.DataFrame({
    'entity_id': [f"s1_{i}" for i in range(10)],
    'business_name': [f"Name {i}" for i in range(10)],
    'business_address': [f"Address {i}" for i in range(10)],
    'country': ['US']*10
})
df_s1.to_csv(work_dir / "test_source1.tsv", sep='\t', index=False)

# 2. Create a dummy baseline matching TSV
df_base = pd.DataFrame({
    'source1_entity_id': [f"s1_{i}" for i in range(10)],
    'matched_entity_ids': [
        "t1,t2",    # s1_0
        "t3",       # s1_1
        "",         # s1_2 singleton
        "t4,t5",    # s1_3
        "",         # s1_4 singleton
        "t2,t6",    # s1_5 (conflict with s1_0 on t2!)
        "t7",       # s1_6
        "t8,t9",    # s1_7
        "",         # s1_8 singleton
        "t10"       # s1_9
    ]
})
df_base.to_csv(work_dir / "matching_results.tsv", sep='\t', index=False)

# 3. Create a dummy candidates parquet
df_cands = pd.DataFrame({
    's1_id': ['s1_0', 's1_2', 's1_3', 's1_3'],
    'candidate_id': ['t99', 't100', 't5', 't101'],
    'sim_name': [0.91, 0.95, 0.88, 0.86],
    'sim_addr': [0.89, 0.92, 0.80, 0.85]
})
df_cands.to_parquet(work_dir / "candidates_test.parquet", index=False)

print("Setup test files complete. Now running self-contained logic...")

# RUN SELF-CONTAINED LOGIC
t0 = time.time()
test_s1_tsv = work_dir / "test_source1.tsv"
baseline_tsv = work_dir / "matching_results.tsv"
cand_parquet = work_dir / "candidates_test.parquet"
output_dir = work_dir / "output"
output_dir.mkdir(parents=True, exist_ok=True)

df_s1_raw = pd.read_csv(test_s1_tsv, sep='\t', usecols=[0], dtype=str)
s1_ids_all = df_s1_raw.iloc[:, 0].str.strip().tolist()
n_s1 = len(s1_ids_all)

df_prev = pd.read_csv(baseline_tsv, sep='\t', dtype=str)
s1_col = df_prev.columns[0]
match_col = df_prev.columns[1]

mask_has_match = df_prev[match_col].notna() & (df_prev[match_col].str.strip() != '') & (df_prev[match_col] != 'nan')
df_matched = df_prev[mask_has_match].copy()
confirmed_s1 = set(df_matched[s1_col].str.strip())

df_matched['candidate_id'] = df_matched[match_col].str.split(',')
df_base_pairs = df_matched.explode('candidate_id')[[s1_col, 'candidate_id']].copy()
df_base_pairs.rename(columns={s1_col: 's1_id'}, inplace=True)
df_base_pairs['s1_id'] = df_base_pairs['s1_id'].str.strip()
df_base_pairs['candidate_id'] = df_base_pairs['candidate_id'].str.strip()
df_base_pairs = df_base_pairs[df_base_pairs['candidate_id'] != '']
df_base_pairs['score'] = 0.95

cand_dfs = [df_base_pairs]

if cand_parquet and cand_parquet.exists():
    pf = pq.ParquetFile(str(cand_parquet))
    for batch in pf.iter_batches(batch_size=1000, columns=['s1_id', 'candidate_id', 'sim_name', 'sim_addr']):
        df_b = batch.to_pandas()
        df_b['s1_id'] = df_b['s1_id'].astype(str)
        df_b['candidate_id'] = df_b['candidate_id'].astype(str)
        # Protect singletons strictly
        df_b = df_b[df_b['s1_id'].isin(confirmed_s1)]
        if len(df_b) == 0:
            continue
        mask_sim = (df_b['sim_name'] >= 0.82) | ((df_b['sim_addr'] >= 0.85) & (df_b['sim_name'] >= 0.55))
        df_filt = df_b[mask_sim].copy()
        if len(df_filt) > 0:
            df_filt['score'] = np.maximum(df_filt['sim_name'].astype(float), df_filt['sim_addr'].astype(float))
            cand_dfs.append(df_filt[['s1_id', 'candidate_id', 'score']])

df_all = pd.concat(cand_dfs, ignore_index=True)
df_all['s1_id'] = df_all['s1_id'].astype(str)
df_all['candidate_id'] = df_all['candidate_id'].astype(str)

df_all.sort_values(by='score', ascending=False, inplace=True)
df_all.drop_duplicates(subset=['s1_id', 'candidate_id'], keep='first', inplace=True)
# Target exclusivity
df_all.drop_duplicates(subset=['candidate_id'], keep='first', inplace=True)

grouped = df_all.groupby('s1_id')['candidate_id'].agg(','.join)
match_map = grouped.to_dict()

final_matches = {sid: match_map.get(sid, '') for sid in s1_ids_all}

# Export
out_path = output_dir / "matching_results.tsv"
with open(out_path, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in s1_ids_all:
        f.write(f"{sid}\t{final_matches.get(sid, '')}\n")

with open(out_path, encoding='utf-8') as f:
    n_lines = sum(1 for _ in f) - 1

print(f"Verified rows: {n_lines} == {n_s1}")
assert n_lines == n_s1

# Check results
print("\nExported file content:")
print(out_path.read_text(encoding='utf-8'))
