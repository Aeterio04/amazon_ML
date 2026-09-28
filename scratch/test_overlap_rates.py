import pandas as pd
import numpy as np
import time, re, unicodedata

t0 = time.time()
print("Analyzing True Match Overlaps in Ground Truth...")

# 1. Load ground truth
df_gt = pd.read_csv('data/student_resource/dataset/train/train_ground_truth.tsv', sep='\t', nrows=50000)

gt_pairs = []
for sid, raw_m in zip(df_gt['source1_entity_id'].values, df_gt['matched_entity_ids'].fillna('').values):
    if raw_m and raw_m != 'nan':
        for m in raw_m.split(','):
            m = m.strip()
            if m:
                gt_pairs.append((str(sid), m))

print(f"Loaded {len(gt_pairs):,} true matching pairs across {len(df_gt):,} S1 queries.")

# 2. Load S1 records for these pairs
df_s1 = pd.read_csv('data/student_resource/dataset/train/train_source1.tsv', sep='\t', nrows=50000)
s1_map = {}
for eid, name, addr in zip(df_s1['entity_id'].values, df_s1['business_name'].values, df_s1['business_address'].values):
    s1_map[str(eid)] = (str(name), str(addr))

# 3. Find which S2/S3 IDs are needed
needed_s2 = {m for s, m in gt_pairs if m.startswith('S2')}
needed_s3 = {m for s, m in gt_pairs if m.startswith('S3')}
print(f"Target matches needed: {len(needed_s2):,} S2, {len(needed_s3):,} S3")

# Load matching target records from S2 and S3 in chunks
target_map = {}
for chunk in pd.read_csv('data/student_resource/dataset/train/train_source2.tsv', sep='\t', chunksize=500000):
    sub = chunk[chunk['entity_id'].isin(needed_s2)]
    for eid, name, addr in zip(sub['entity_id'].values, sub['business_name'].values, sub['business_address'].values):
        target_map[str(eid)] = (str(name), str(addr))
    if len(target_map) >= len(needed_s2):
        break

for chunk in pd.read_csv('data/student_resource/dataset/train/train_source3.tsv', sep='\t', chunksize=500000):
    sub = chunk[chunk['entity_id'].isin(needed_s3)]
    for eid, name, addr in zip(sub['entity_id'].values, sub['business_name'].values, sub['business_address'].values):
        target_map[str(eid)] = (str(name), str(addr))
    if len(target_map) >= len(needed_s2) + len(needed_s3):
        break

print(f"Loaded {len(target_map):,} target records in {time.time()-t0:.1f}s")

# 4. Now evaluate blocking rules on all true pairs!
def clean_toks(s):
    if not s or pd.isna(s):
        return []
    s = unicodedata.normalize('NFKD', str(s).lower())
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    return [w for w in s.split() if len(w) >= 2]

exact_name_hits = 0
first2_tok_hits = 0
first1_tok_hits = 0
any_name_tok_hits = 0
addr_tok_hits = 0
combined_hits = 0

eval_pairs = [p for p in gt_pairs if p[0] in s1_map and p[1] in target_map]
print(f"Evaluating {len(eval_pairs):,} true pairs with loaded data...")

for sid, tid in eval_pairs:
    s_name, s_addr = s1_map[sid]
    t_name, t_addr = target_map[tid]
    
    s_ntok = clean_toks(s_name)
    t_ntok = clean_toks(t_name)
    s_atok = clean_toks(s_addr)
    t_atok = clean_toks(t_addr)
    
    # Rule 1: Exact stripped name
    is_exact = (' '.join(s_ntok) == ' '.join(t_ntok)) if s_ntok and t_ntok else False
    if is_exact:
        exact_name_hits += 1
        
    # Rule 2: First 2 tokens match
    is_first2 = (s_ntok[:2] == t_ntok[:2]) if len(s_ntok) >= 2 and len(t_ntok) >= 2 else False
    if is_first2:
        first2_tok_hits += 1
        
    # Rule 3: First token matches
    is_first1 = (s_ntok[0] == t_ntok[0]) if s_ntok and t_ntok else False
    if is_first1:
        first1_tok_hits += 1
        
    # Rule 4: ANY name token overlap
    has_name_tok = bool(set(s_ntok) & set(t_ntok))
    if has_name_tok:
        any_name_tok_hits += 1
        
    # Rule 5: Address token overlap >= 2
    has_addr_tok = (len(set(s_atok) & set(t_atok)) >= 2)
    if has_addr_tok:
        addr_tok_hits += 1
        
    # Combined Multi-Key Rule: (First token matches) OR (Any name token + address token)
    is_captured = is_first1 or (has_name_tok and has_addr_tok)
    if is_captured:
        combined_hits += 1

n = len(eval_pairs)
print("=" * 60)
print(f"RESULTS ON {n:,} TRUE MATCH PAIRS:")
print("=" * 60)
print(f"  Exact Full Name:              {exact_name_hits/n*100:.2f}% ({exact_name_hits:,})")
print(f"  First 2 Tokens Match:         {first2_tok_hits/n*100:.2f}% ({first2_tok_hits:,})")
print(f"  First 1 Token Match:          {first1_tok_hits/n*100:.2f}% ({first1_tok_hits:,})")
print(f"  ANY Name Token Overlap:       {any_name_tok_hits/n*100:.2f}% ({any_name_tok_hits:,})")
print(f"  Combined (1st Tok OR Name+Addr): {combined_hits/n*100:.2f}% ({combined_hits:,})")
print("=" * 60)
