import pandas as pd
import numpy as np
import time, re, unicodedata

t0 = time.time()
print("Testing Multi-Key Lexical Blocking Recall on Train Ground Truth...")

# 1. Load ground truth sample (first 100,000 S1 queries)
SAMPLE_SIZE = 100000
df_gt = pd.read_csv('data/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=SAMPLE_SIZE)

gt_map = {}
total_true = 0
for sid, raw_m in zip(df_gt['source1_entity_id'].values, df_gt['matched_entity_ids'].fillna('').values):
    if raw_m and raw_m != 'nan':
        m_set = {m.strip() for m in raw_m.split(',') if m.strip()}
        gt_map[str(sid)] = m_set
        total_true += len(m_set)
    else:
        gt_map[str(sid)] = set()

print(f"Sample: {len(gt_map):,} queries, {total_true:,} true matches")

# 2. Load matching S1 queries
df_s1 = pd.read_csv('data/student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=SAMPLE_SIZE)
s1_set = set(df_s1['entity_id'].values)

# Clean text helper
def clean_toks(s):
    if not s or pd.isna(s):
        return []
    s = unicodedata.normalize('NFKD', str(s).lower())
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return [w for w in s.split() if len(w) >= 2]

s1_data = []
for eid, name, addr in zip(df_s1['entity_id'].values, df_s1['business_name'].values, df_s1['business_address'].values):
    n_toks = clean_toks(name)
    a_toks = clean_toks(addr)
    # Extract postal code (5 digits or 6 digits)
    m_zip = re.search(r'\b(\d{5,6})\b', str(addr))
    zip_code = m_zip.group(1) if m_zip else ""
    s1_data.append((eid, n_toks, a_toks, zip_code))

print(f"Loaded {len(s1_data):,} S1 queries in {time.time()-t0:.1f}s")
