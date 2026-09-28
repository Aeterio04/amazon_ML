import os, sys, gc, time
from pathlib import Path
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

print("Benchmarking vectorized exclusivity & aggregation on 1.73M entities...")
t0 = time.time()

# 1. Generate 1.73M simulated S1 IDs
n_s1 = 1_732_544
s1_ids = [f"s1_{i}" for i in range(n_s1)]

# 2. Simulate 6M candidate pairs (avg 3.5 per entity, with some duplicates)
n_pairs = 6_000_000
rand_s1 = np.random.choice(s1_ids, size=n_pairs)
cand_ids = [f"target_{i}" for i in np.random.randint(0, 5_000_000, size=n_pairs)]
scores = np.random.uniform(0.6, 1.0, size=n_pairs)

df_all = pd.DataFrame({
    's1_id': rand_s1,
    'candidate_id': cand_ids,
    'score': scores
})
print(f"Generated {len(df_all):,} pairs in {time.time()-t0:.2f}s")

# 3. Vectorized Exclusivity: sort and drop duplicates
t1 = time.time()
df_all.sort_values(by='score', ascending=False, inplace=True)
df_all.drop_duplicates(subset=['candidate_id'], keep='first', inplace=True)
df_all.drop_duplicates(subset=['s1_id', 'candidate_id'], inplace=True)
print(f"Exclusivity filter finished in {time.time()-t1:.2f}s | Remaining pairs: {len(df_all):,}")

# 4. Vectorized groupby
t2 = time.time()
grouped = df_all.groupby('s1_id')['candidate_id'].agg(','.join)
match_map = grouped.to_dict()
print(f"Groupby finished in {time.time()-t2:.2f}s | Entities with matches: {len(match_map):,}")

# 5. Write to file
t3 = time.time()
out_file = Path("work/benchmark_matching.tsv")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in s1_ids:
        f.write(f"{sid}\t{match_map.get(sid, '')}\n")
print(f"Writing TSV finished in {time.time()-t3:.2f}s | Total elapsed: {time.time()-t0:.2f}s")

# Verify lines
with open(out_file, encoding='utf-8') as f:
    n_lines = sum(1 for _ in f) - 1
print(f"Verified line count: {n_lines:,} == {n_s1:,} -> {n_lines == n_s1}")
