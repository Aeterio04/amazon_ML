import pandas as pd
import numpy as np
from rapidfuzz.distance import JaroWinkler
import re

print("Loading 20,000 rows of train_ground_truth...")
gt = pd.read_csv("data/student_resource/dataset/train/train_ground_truth.tsv", sep="\t", nrows=25000)
s1 = pd.read_csv("data/student_resource/dataset/train/train_source1.tsv", sep="\t", nrows=25000)

# Filter gt to those in s1
s1_set = set(s1['entity_id'])
gt = gt[gt['source1_entity_id'].isin(s1_set)]

# Collect all matched S2 and S3 IDs needed
needed_s2 = set()
needed_s3 = set()
for m in gt['matched_entity_ids'].dropna():
    for x in m.split(','):
        x = x.strip()
        if x.startswith('S2'):
            needed_s2.add(x)
        elif x.startswith('S3'):
            needed_s3.add(x)

print(f"Entities in sample: {len(gt):,} | Needed S2: {len(needed_s2):,} | Needed S3: {len(needed_s3):,}")

# Read only matching rows from S2 and S3
s2_rows = []
for chunk in pd.read_csv("data/student_resource/dataset/train/train_source2.tsv", sep="\t", chunksize=100000):
    c = chunk[chunk['entity_id'].isin(needed_s2)]
    if len(c) > 0:
        s2_rows.append(c)
    if sum(len(x) for x in s2_rows) >= len(needed_s2):
        break
df_s2 = pd.concat(s2_rows, ignore_index=True) if s2_rows else pd.DataFrame()

s3_rows = []
for chunk in pd.read_csv("data/student_resource/dataset/train/train_source3.tsv", sep="\t", chunksize=100000):
    c = chunk[chunk['entity_id'].isin(needed_s3)]
    if len(c) > 0:
        s3_rows.append(c)
    if sum(len(x) for x in s3_rows) >= len(needed_s3):
        break
df_s3 = pd.concat(s3_rows, ignore_index=True) if s3_rows else pd.DataFrame()

print(f"Loaded matched targets: S2={len(df_s2):,}, S3={len(df_s3):,}")

s1_dict = s1.set_index('entity_id').to_dict('index')
tgt_dict = {}
for _, r in df_s2.iterrows():
    tgt_dict[r['entity_id']] = r.to_dict()
for _, r in df_s3.iterrows():
    tgt_dict[r['entity_id']] = r.to_dict()

# Analyze true pairs
exact_name_cnt = 0
exact_addr_cnt = 0
jw_name_high = 0 # >= 0.90
jw_addr_high = 0 # >= 0.90
total_pairs = 0

def clean(s):
    if not s or pd.isna(s):
        return ""
    s = str(s).lower()
    return re.sub(r'[^a-z0-9\s]', ' ', s).strip()

jw_names = []
jw_addrs = []

for _, row in gt.iterrows():
    sid = row['source1_entity_id']
    m = row['matched_entity_ids']
    if pd.isna(m) or not m:
        continue
    r1 = s1_dict.get(sid, {})
    n1 = clean(r1.get('business_name', ''))
    a1 = clean(r1.get('business_address', ''))
    
    for tid in m.split(','):
        tid = tid.strip()
        if tid not in tgt_dict:
            continue
        r2 = tgt_dict[tid]
        n2 = clean(r2.get('business_name', ''))
        a2 = clean(r2.get('business_address', ''))
        
        total_pairs += 1
        if n1 == n2:
            exact_name_cnt += 1
        if a1 == a2:
            exact_addr_cnt += 1
            
        jwn = JaroWinkler.similarity(n1, n2) if n1 and n2 else 0.0
        jwa = JaroWinkler.similarity(a1, a2) if a1 and a2 else 0.0
        jw_names.append(jwn)
        jw_addrs.append(jwa)
        if jwn >= 0.90:
            jw_name_high += 1
        if jwa >= 0.90:
            jw_addr_high += 1

print("\n" + "=" * 60)
print(f"ANALYSIS OF {total_pairs:,} TRUE MATCHING PAIRS")
print("=" * 60)
print(f"Exact Clean Name:     {exact_name_cnt:,} ({exact_name_cnt/total_pairs*100:.2f}%)")
print(f"Exact Clean Address:  {exact_addr_cnt:,} ({exact_addr_cnt/total_pairs*100:.2f}%)")
print(f"Name JW >= 0.90:      {jw_name_high:,} ({jw_name_high/total_pairs*100:.2f}%)")
print(f"Addr JW >= 0.90:      {jw_addr_high:,} ({jw_addr_high/total_pairs*100:.2f}%)")
print(f"Mean Name JW:         {np.mean(jw_names):.4f}")
print(f"Mean Addr JW:         {np.mean(jw_addrs):.4f}")
print(f"Median Name JW:       {np.median(jw_names):.4f}")
print(f"Median Addr JW:       {np.median(jw_addrs):.4f}")

# Check correlations and sample pairs
print("\n--- SAMPLE 5 TRUE PAIRS ---")
sample_idx = 0
for _, row in gt.iterrows():
    sid = row['source1_entity_id']
    m = row['matched_entity_ids']
    if pd.isna(m) or not m:
        continue
    r1 = s1_dict.get(sid, {})
    for tid in m.split(','):
        tid = tid.strip()
        if tid not in tgt_dict:
            continue
        r2 = tgt_dict[tid]
        print(f"\nPair {sample_idx+1}:")
        print(f"  S1 [{sid}]: {r1.get('business_name')} || {r1.get('business_address')}")
        print(f"  Tgt [{tid}]: {r2.get('business_name')} || {r2.get('business_address')}")
        sample_idx += 1
        if sample_idx >= 5:
            break
    if sample_idx >= 5:
        break
