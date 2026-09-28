import json
from pathlib import Path

p = Path("notebooks/06_stage6_test_submission_v2.ipynb")
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

# -------------------------------------------------------------
# 1. UPDATE CELL 2: Robust Auto-Discovery for User's Uploaded Files
# -------------------------------------------------------------
cell2_src = """# ============================================================
# ROBUST ARTIFACT & PATH DISCOVERY (Auto-Detects All User Uploads)
# ============================================================

import os, sys, gc, time, re, unicodedata, subprocess, shutil
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

IS_KAGGLE = os.path.exists('/kaggle')

def find_file(pattern):
    search_root = Path('/kaggle/input') if IS_KAGGLE else Path('.')
    matches = list(search_root.rglob(pattern))
    if not matches:
        return None
    matches.sort(key=lambda p: (p.stat().st_size, p.stat().st_mtime), reverse=True)
    return matches[0]

WORK_DIR = Path('/kaggle/working') if IS_KAGGLE else Path('work')
OUTPUT_DIR = WORK_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 1. Test Raw TSVs
TEST_S1_TSV = find_file('test_source1.tsv')
TEST_S2_TSV = find_file('test_source2.tsv')
TEST_S3_TSV = find_file('test_source3.tsv')

# 2. Normalized parquets
TEST_S1_NORM = find_file('test_s1_norm.parquet')
TEST_S2_NORM = find_file('test_s2_norm.parquet')
TEST_S3_NORM = find_file('test_s3_norm.parquet')

# 3. Model Weights (Optional if using precomputed matches)
MODEL_FOLD0 = find_file('lgbm_model_fold0.txt')
MODEL_FOLD1 = find_file('lgbm_model_fold1.txt')

# 4. Dense Embeddings
TEST_S1_NAME_EMB = find_file('test_s1_name_emb.npy')
TEST_S1_ADDR_EMB = find_file('test_s1_addr_emb.npy')
TEST_S2_NAME_EMB = find_file('test_s2_name_emb.npy')
TEST_S2_ADDR_EMB = find_file('test_s2_addr_emb.npy')
TEST_S3_NAME_EMB = find_file('test_s3_name_emb.npy')
TEST_S3_ADDR_EMB = find_file('test_s3_addr_emb.npy')

# 5. User Uploaded Dataset Artifacts (Fast-Track Discovery)
# Finds matchingresult.tsv, candidatepairs.tsv, and candidates parquet from attached dataset
PREV_MATCHING_TSV = None
for pat in ['*matching*result*.tsv', '*matchingresult*.tsv', '*matching*.tsv', '*result*.tsv']:
    cand = find_file(pat)
    if cand and 'train' not in str(cand).lower():
        PREV_MATCHING_TSV = cand
        break

PREV_CAND_TSV = None
for pat in ['*candidate*pair*.tsv', '*candidatepair*.tsv', '*candidate*.tsv']:
    cand = find_file(pat)
    if cand and 'train' not in str(cand).lower():
        PREV_CAND_TSV = cand
        break

CAND_PARQUET_FILE = None
for pat in ['candidates_test.parquet', '*candidate*.parquet', '*paraquet*', '*.parquet']:
    cand = find_file(pat)
    if cand and 'norm' not in str(cand).lower() and 'train' not in str(cand).lower() and 's1' not in str(cand).lower():
        CAND_PARQUET_FILE = cand
        break

VALIDATOR_SCRIPT = find_file('validate_submission.py')

print("\\n" + "=" * 60)
print(" DISCOVERED INPUT ARTIFACTS")
print("=" * 60)
print(f"  Test S1 TSV:         {TEST_S1_TSV}")
print(f"  Test S1 Norm:        {TEST_S1_NORM}")
print(f"  Precomputed Matches: {PREV_MATCHING_TSV}")
print(f"  Precomputed Cands:   {PREV_CAND_TSV}")
print(f"  Candidate Parquet:   {CAND_PARQUET_FILE}")
print(f"  Output Dir:          {OUTPUT_DIR}")
print("=" * 60)

assert TEST_S1_TSV is not None and TEST_S1_TSV.exists(), "ERROR: test_source1.tsv not found!"
"""

nb['cells'][2]['source'] = [line + '\n' for line in cell2_src.strip().split('\n')]

# -------------------------------------------------------------
# 2. UPDATE CELL 5: Skip FAISS if candidates/matches exist
# -------------------------------------------------------------
cell5_code = """# ============================================================
# CANDIDATE GENERATION: FAISS GPU RETRIEVAL (AUTO-BYPASS IF FOUND)
# ============================================================

cand_parquet_path = CAND_PARQUET_FILE or (WORK_DIR / 'candidates_test.parquet')
cand_tsv_path = OUTPUT_DIR / 'candidate_pairs.tsv'

TOP_K = 50

# Fast-track check: If user uploaded candidates parquet or matching results, SKIP FAISS!
if PREV_MATCHING_TSV is not None and PREV_MATCHING_TSV.exists():
    print(f"\\n[FAST TRACK] Found precomputed matches: {PREV_MATCHING_TSV.name}")
    print("  Skipping FAISS candidate retrieval completely!")
elif cand_parquet_path is not None and cand_parquet_path.exists() and cand_parquet_path.stat().st_size > 10_000_000:
    print(f"\\n[FAST TRACK] Using existing candidates parquet: {cand_parquet_path} ({cand_parquet_path.stat().st_size / 1e6:.1f} MB)")
    print("  Skipping FAISS candidate retrieval completely!")
elif TEST_S1_NAME_EMB is not None and TEST_S2_NAME_EMB is not None:
    print("\\n" + "=" * 60)
    print(f" EXECUTING DUAL-CHANNEL FAISS GPU RETRIEVAL (TOP-K={TOP_K})")
    print("=" * 60)
    import faiss
    
    dim = 384
    nlist = 4096
    nprobe = 64
    
    def load_any_emb(filepath, dim=384):
        filepath = Path(filepath)
        try:
            if filepath.stat().st_size > 1000 and not str(filepath).endswith('.tmp'):
                return np.load(filepath, mmap_mode='r')
        except Exception:
            pass
        size = filepath.stat().st_size
        n_rows = size // (dim * 2)
        return np.memmap(filepath, dtype=np.float16, mode='r', shape=(n_rows, dim))

    def run_gpu_faiss(channel_name, s1_path, s2_path, s3_path):
        print(f"  Building GPU SQ8 index for {channel_name} channel...")
        t0 = time.time()
        a2 = load_any_emb(s2_path)
        a3 = load_any_emb(s3_path)
        n_a2 = len(a2)
        n_a3 = len(a3)
        n_target = n_a2 + n_a3
        print(f"    Targets: S2={n_a2:,}, S3={n_a3:,} | Total={n_target:,}")
        
        quantizer = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
        )
        
        s2_step = max(1, n_a2 // 100000)
        s3_step = max(1, n_a3 // 100000)
        t_s2 = np.asarray(a2[0 : 100000 * s2_step : s2_step], dtype=np.float32)
        t_s3 = np.asarray(a3[0 : 100000 * s3_step : s3_step], dtype=np.float32)
        train_data = np.vstack([t_s2, t_s3])
        del t_s2, t_s3
        
        print(f"    Training SQ8 IVF quantizer on {len(train_data):,} samples...")
        cpu_index.train(train_data)
        del train_data; gc.collect()
        
        res = faiss.StandardGpuResources()
        res.setTempMemory(512 * 1024 * 1024)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.nprobe = nprobe
        del cpu_index; gc.collect()
        
        for s in range(0, n_a2, 500000):
            e = min(s + 500000, n_a2)
            gpu_index.add(np.asarray(a2[s:e], dtype=np.float32))
            
        for s in range(0, n_a3, 500000):
            e = min(s + 500000, n_a3)
            gpu_index.add(np.asarray(a3[s:e], dtype=np.float32))
            
        del a2, a3; gc.collect()
        print(f"    Index ready ({gpu_index.ntotal:,} vectors in GPU VRAM). Searching...")
        
        s1_embs = load_any_emb(s1_path)
        all_sims, all_indices = [], []
        query_chunk = 10000
        for q in range(0, len(s1_embs), query_chunk):
            qe = min(q + query_chunk, len(s1_embs))
            q_batch = np.asarray(s1_embs[q:qe], dtype=np.float32)
            sb, ib = gpu_index.search(q_batch, TOP_K)
            all_sims.append(sb)
            all_indices.append(ib)
            
        sims = np.vstack(all_sims)
        indices = np.vstack(all_indices)
        print(f"    {channel_name} search finished in {time.time()-t0:.1f}s")
        del gpu_index, res, s1_embs, all_sims, all_indices; gc.collect()
        return indices, sims

    name_idx, name_sim = run_gpu_faiss("Name", TEST_S1_NAME_EMB, TEST_S2_NAME_EMB, TEST_S3_NAME_EMB)
    addr_idx, addr_sim = run_gpu_faiss("Addr", TEST_S1_ADDR_EMB, TEST_S2_ADDR_EMB, TEST_S3_ADDR_EMB)
    
    schema = pa.schema([
        ('s1_id', pa.string()),
        ('candidate_id', pa.string()),
        ('sim_name', pa.float32()),
        ('sim_addr', pa.float32()),
        ('route_name', pa.int8()),
        ('route_addr', pa.int8()),
    ])
    writer = pq.ParquetWriter(cand_parquet_path, schema, compression='snappy')
    tsv_out = open(cand_tsv_path, 'w', encoding='utf-8')
    tsv_out.write("source1_entity_id\\tcandidate_entity_ids\\n")
    
    print("  Merging candidates with country filter...")
    batch_size = 50000
    total_pairs = 0
    
    for b_start in range(0, n_s1, batch_size):
        b_end = min(b_start + batch_size, n_s1)
        b_s1, b_cand, b_sn, b_sa, b_rn, b_ra = [], [], [], [], [], []
        
        for i in range(b_start, b_end):
            sid = s1_ids_all[i]
            s_country = s1_countries_all[i]
            cand_dict = {}
            
            for k in range(TOP_K):
                t_idx = name_idx[i, k]
                if 0 <= t_idx < n_tgt:
                    if tgt_countries[t_idx] == s_country or s_country == 'UNKNOWN':
                        cand_dict[t_idx] = [float(name_sim[i, k]), 0.0, 1, 0]
                        
            if addr_idx is not None:
                for k in range(TOP_K):
                    t_idx = addr_idx[i, k]
                    if 0 <= t_idx < n_tgt:
                        if tgt_countries[t_idx] == s_country or s_country == 'UNKNOWN':
                            if t_idx in cand_dict:
                                cand_dict[t_idx][1] = float(addr_sim[i, k])
                                cand_dict[t_idx][3] = 1
                            else:
                                cand_dict[t_idx] = [0.0, float(addr_sim[i, k]), 0, 1]
                                
            c_ids = [tgt_ids[ti] for ti in cand_dict.keys()]
            tsv_out.write(f"{sid}\\t{','.join(c_ids)}\\n")
            for ti, vals in cand_dict.items():
                b_s1.append(sid)
                b_cand.append(tgt_ids[ti])
                b_sn.append(vals[0])
                b_sa.append(vals[1])
                b_rn.append(vals[2])
                b_ra.append(vals[3])
                
        tbl = pa.Table.from_arrays([
            pa.array(b_s1, type=pa.string()),
            pa.array(b_cand, type=pa.string()),
            pa.array(b_sn, type=pa.float32()),
            pa.array(b_sa, type=pa.float32()),
            pa.array(b_rn, type=pa.int8()),
            pa.array(b_ra, type=pa.int8()),
        ], schema=schema)
        writer.write_table(tbl)
        total_pairs += len(b_s1)
        del tbl, b_s1, b_cand, b_sn, b_sa, b_rn, b_ra; gc.collect()
        
    writer.close()
    tsv_out.close()
    print(f"\\nFAISS candidate generation finished: {total_pairs:,} candidate pairs exported!")
    del name_idx, name_sim, addr_idx, addr_sim; gc.collect()
else:
    print("Candidates ready.")
"""

nb['cells'][5]['source'] = [line + '\n' for line in cell5_code.strip().split('\n')]

# -------------------------------------------------------------
# 3. UPDATE CELL 6: Fast Bypass if PREV_MATCHING_TSV exists
# -------------------------------------------------------------
cell6_code = """# ============================================================
# LIGHTGBM SCORING (BYPASS IF PRECOMPUTED MATCHES PROVIDED)
# ============================================================

scored_s1 = []
scored_cand = []
scored_probs = []

if PREV_MATCHING_TSV is not None and PREV_MATCHING_TSV.exists():
    print("\\n" + "=" * 60)
    print(f" [FAST TRACK] FOUND PRECOMPUTED MATCHES: {PREV_MATCHING_TSV.name}")
    print("=" * 60)
    print("  Skipping LightGBM scoring. Jumping directly to Deterministic Rule-Based Matching!")
else:
    print("\\n" + "=" * 60)
    print(" EVALUATING CANDIDATES WITH LIGHTGBM ENSEMBLE")
    print("=" * 60)
    
    models = []
    for p in [MODEL_FOLD0, MODEL_FOLD1]:
        if p and p.exists():
            models.append(lgb.Booster(model_file=str(p)))
            
    active_cand_parquet = CAND_PARQUET_FILE or (WORK_DIR / 'candidates_test.parquet')
    pf = pq.ParquetFile(str(active_cand_parquet))
    chunk_size = 2_000_000
    
    t0_score = time.time()
    for batch_idx, batch in enumerate(pf.iter_batches(batch_size=chunk_size)):
        df_chunk = batch.to_pandas()
        c_s1_ids = df_chunk['s1_id'].values
        c_cand_ids = df_chunk['candidate_id'].values
        c_sim_name = df_chunk['sim_name'].values.astype(np.float32)
        c_sim_addr = df_chunk['sim_addr'].values.astype(np.float32)
        c_r_name = df_chunk['route_name'].values.astype(np.int8)
        c_r_addr = df_chunk['route_addr'].values.astype(np.int8)
        del df_chunk; gc.collect()
        
        s1_idx = np.array([s1_id_map.get(sid, -1) for sid in c_s1_ids], dtype=np.int32)
        tgt_idx = np.array([tgt_id_map.get(cid, -1) for cid in c_cand_ids], dtype=np.int32)
        valid_mask = (s1_idx >= 0) & (tgt_idx >= 0)
        
        s1_idx = s1_idx[valid_mask]
        tgt_idx = tgt_idx[valid_mask]
        c_s1_ids = c_s1_ids[valid_mask]
        c_cand_ids = c_cand_ids[valid_mask]
        c_sim_name = c_sim_name[valid_mask]
        c_sim_addr = c_sim_addr[valid_mask]
        c_r_name = c_r_name[valid_mask]
        c_r_addr = c_r_addr[valid_mask]
        n_valid = len(s1_idx)
        
        r_both = (c_r_name & c_r_addr).astype(np.int8)
        sim_prod = (c_sim_name * c_sim_addr).astype(np.float32)
        sim_max = np.maximum(c_sim_name, c_sim_addr).astype(np.float32)
        
        f_jw = np.zeros(n_valid, dtype=np.float32)
        f_name_jacc = np.zeros(n_valid, dtype=np.float32)
        f_name_exact = np.zeros(n_valid, dtype=np.int8)
        f_len_diff = np.zeros(n_valid, dtype=np.int16)
        f_first_tok = np.zeros(n_valid, dtype=np.int8)
        f_house = np.zeros(n_valid, dtype=np.int8)
        f_state = np.zeros(n_valid, dtype=np.int8)
        f_addr_jacc = np.zeros(n_valid, dtype=np.float32)
        
        for i in range(n_valid):
            si = s1_idx[i]
            ti = tgt_idx[i]
            sn = s1_names_all[si]
            tn = tgt_names[ti]
            if sn and tn:
                if sn == tn:
                    f_jw[i] = 1.0; f_name_exact[i] = 1; f_name_jacc[i] = 1.0
                else:
                    f_jw[i] = JaroWinkler.similarity(sn, tn)
                    s_toks, t_toks = sn.split(), tn.split()
                    s_set, t_set = set(s_toks), set(t_toks)
                    u_len = len(s_set | t_set)
                    f_name_jacc[i] = len(s_set & t_set) / u_len if u_len > 0 else 0.0
                f_len_diff[i] = abs(len(sn) - len(tn))
                s_first = sn.split()[0] if sn else ''
                t_first = tn.split()[0] if tn else ''
                f_first_tok[i] = 1 if (s_first and t_first and s_first == t_first) else 0
                
            sh = s1_houses_all[si]
            th = tgt_houses[ti]
            if sh and th:
                f_house[i] = 1 if sh == th else -1
            ss = s1_states_all[si]
            ts = tgt_states[ti]
            if ss and ts:
                f_state[i] = 1 if ss == ts else -1
            sa = s1_addrs_all[si]
            ta = tgt_addrs[ti]
            if sa and ta:
                sa_set, ta_set = set(sa.split()), set(ta.split())
                u_len = len(sa_set | ta_set)
                f_addr_jacc[i] = len(sa_set & ta_set) / u_len if u_len > 0 else 0.0
                
        s1_cat = pd.Categorical(c_s1_ids)
        s_series = pd.Series(sim_max)
        grouped_max = s_series.groupby(s1_cat, observed=True).transform('max').values.astype(np.float32)
        f_gap_to_top = grouped_max - sim_max
        f_cand_rank = s_series.groupby(s1_cat, observed=True).rank(ascending=False, method='min').values.astype(np.int16)
        f_n_cands = s_series.groupby(s1_cat, observed=True).transform('count').values.astype(np.int16)
        del s1_cat, s_series, grouped_max; gc.collect()
        
        X_chunk = np.column_stack([
            c_sim_name, c_sim_addr, c_r_name, c_r_addr, r_both,
            f_jw, f_name_jacc, f_name_exact, f_len_diff, f_first_tok,
            f_house, f_state, f_addr_jacc, tgt_is_s2[tgt_idx],
            sim_prod, sim_max, f_gap_to_top, f_cand_rank, f_n_cands
        ]).astype(np.float32)
        
        preds = np.zeros(n_valid, dtype=np.float32)
        for m in models:
            preds += m.predict(X_chunk)
        preds /= len(models)
        
        keep_mask = (preds >= 0.35)
        scored_s1.extend(c_s1_ids[keep_mask])
        scored_cand.extend(c_cand_ids[keep_mask])
        scored_probs.extend(preds[keep_mask])
        del X_chunk, preds, keep_mask; gc.collect()
        
    print(f"Scoring completed in {time.time()-t0_score:.1f}s.")
"""

nb['cells'][6]['source'] = [line + '\n' for line in cell6_code.strip().split('\n')]

# -------------------------------------------------------------
# 4. UPDATE CELL 7: Fast Deterministic Rule-Based Engine
# -------------------------------------------------------------
cell7_code = """# ============================================================
# DETERMINISTIC RULE-BASED MATCHING & EXCLUSIVITY ENGINE
# ============================================================

t0_rule = time.time()
print("\\n" + "=" * 60)
print(" EXECUTING DETERMINISTIC RULE-BASED MATCHING & EXCLUSIVITY")
print("=" * 60)

final_matches = {sid: [] for sid in s1_ids_all}

if PREV_MATCHING_TSV is not None and PREV_MATCHING_TSV.exists():
    print(f"  Loading baseline matches from {PREV_MATCHING_TSV.name}...")
    df_prev = pd.read_csv(PREV_MATCHING_TSV, sep='\\t', dtype=str)
    s1_col = df_prev.columns[0]
    match_col = df_prev.columns[1]
    
    raw_matches = {}
    for sid, m_str in zip(df_prev[s1_col], df_prev[match_col].fillna('')):
        sid = str(sid).strip()
        if m_str and str(m_str).strip() and str(m_str) != 'nan':
            raw_matches[sid] = [x.strip() for x in str(m_str).split(',') if x.strip()]
        else:
            raw_matches[sid] = []
    del df_prev; gc.collect()
    
    # 1. Load candidate similarities for scoring claims & sibling expansion
    active_cand_parquet = CAND_PARQUET_FILE or (WORK_DIR / 'candidates_test.parquet')
    sim_dict = {} # (s1_id, cand_id) -> max(sim_name, sim_addr)
    s1_cand_map = {} # s1_id -> list of (cand_id, sn, sa)
    
    if active_cand_parquet and active_cand_parquet.exists():
        print(f"  Indexing candidate similarities from {active_cand_parquet.name}...")
        pf = pq.ParquetFile(str(active_cand_parquet))
        for batch in pf.iter_batches(batch_size=5_000_000, columns=['s1_id', 'candidate_id', 'sim_name', 'sim_addr']):
            df_b = batch.to_pandas()
            for sid, cid, sn, sa in zip(df_b['s1_id'], df_b['candidate_id'], df_b['sim_name'], df_b['sim_addr']):
                sid = str(sid); cid = str(cid)
                score = max(float(sn), float(sa))
                sim_dict[(sid, cid)] = score
                if sid not in s1_cand_map:
                    s1_cand_map[sid] = []
                s1_cand_map[sid].append((cid, float(sn), float(sa)))
            del df_b; gc.collect()
            
    print(f"  Indexed candidate pairs in {time.time()-t0_rule:.1f}s.")
    
    # 2. RULE 1: 1-to-1 TARGET EXCLUSIVITY (Ground truth: 0.000% targets match >1 S1)
    print("  Applying Rule 1: 1-to-1 Target Exclusivity Deduplication...")
    best_claim = {} # cid -> (sid, score)
    for sid, cids in raw_matches.items():
        for cid in cids:
            score = sim_dict.get((sid, cid), 0.75)
            if cid not in best_claim or score > best_claim[cid][1]:
                best_claim[cid] = (sid, score)
                
    excl_matches = {sid: [] for sid in s1_ids_all}
    for cid, (sid, score) in best_claim.items():
        excl_matches[sid].append(cid)
        
    # 3. RULE 2: HIGH-CONFIDENCE SIBLING EXPANSION (Ground truth: 72% have >=3 matches)
    print("  Applying Rule 2: High-Confidence Sibling Expansion for confirmed entities...")
    expanded_matches = {}
    for sid in s1_ids_all:
        cur_matched = excl_matches.get(sid, [])
        if not cur_matched:
            # Protect singletons: strictly empty
            expanded_matches[sid] = []
            continue
            
        accepted = list(cur_matched)
        existing_cids = set(cur_matched)
        
        # Look through all candidates for this confirmed non-singleton
        for cid, sn, sa in s1_cand_map.get(sid, []):
            if cid in existing_cids:
                continue
            # Sibling match criteria: high name similarity OR high address similarity
            if sn >= 0.82 or (sa >= 0.85 and sn >= 0.55):
                # Ensure no conflict with another higher-scoring S1 entity
                claim = best_claim.get(cid)
                if claim is None or claim[0] == sid or max(sn, sa) > claim[1]:
                    accepted.append(cid)
                    best_claim[cid] = (sid, max(sn, sa))
                    existing_cids.add(cid)
                    
        expanded_matches[sid] = accepted
        
    final_matches = expanded_matches

else:
    # Use LightGBM scored probabilities with calibrated thresholds
    print("  Applying Calibrated Decision Rule on LightGBM predictions...")
    T_FIRST = 0.580
    T_REST  = 0.420
    
    cands_by_s1 = {}
    for sid, cid, prob in zip(scored_s1, scored_cand, scored_probs):
        sid = str(sid)
        if sid not in cands_by_s1:
            cands_by_s1[sid] = []
        cands_by_s1[sid].append((str(cid), float(prob)))
        
    for sid in cands_by_s1:
        cands_by_s1[sid].sort(key=lambda x: -x[1])
        
    initial_matches = {}
    for sid in s1_ids_all:
        cands = cands_by_s1.get(sid, [])
        if not cands or cands[0][1] < T_FIRST:
            initial_matches[sid] = []
        else:
            top_prob = cands[0][1]
            accepted = [(cands[0][0], top_prob)]
            for cid, p in cands[1:]:
                if p >= T_REST and p >= top_prob * 0.55:
                    accepted.append((cid, p))
            initial_matches[sid] = accepted
            
    best_claim = {}
    for sid, matches in initial_matches.items():
        for cid, prob in matches:
            if cid not in best_claim or prob > best_claim[cid][1]:
                best_claim[cid] = (sid, prob)
                
    for cid, (sid, prob) in best_claim.items():
        final_matches[sid].append(cid)

n_singletons = sum(1 for v in final_matches.values() if len(v) == 0)
n_matched = len(final_matches) - n_singletons
total_matched_ids = sum(len(v) for v in final_matches.values())
singleton_pct = (n_singletons / n_s1) * 100

print(f"\\nDecision Summary (Finished in {time.time()-t0_rule:.1f}s):")
print(f"  Total S1 Entities:     {n_s1:,}")
print(f"  Singletons (Empty):    {n_singletons:,} ({singleton_pct:.2f}% | Expected ~5.6%)")
print(f"  Resolved Entities:     {n_matched:,} ({100-singleton_pct:.2f}%)")
print(f"  Total Matched IDs:     {total_matched_ids:,}")
print(f"  Average Matches/Query: {total_matched_ids / max(n_matched, 1):.2f} (Target ~3.5 - 4.0)")
"""

nb['cells'][7]['source'] = [line + '\n' for line in cell7_code.strip().split('\n')]

# -------------------------------------------------------------
# 5. UPDATE CELL 8: Fast Export of matching_results.tsv Only
# -------------------------------------------------------------
cell8_code = """# ============================================================
# EXPORT ONLY MATCHING RESULTS (FAST 5-SECOND EXPORT)
# ============================================================

matching_path = OUTPUT_DIR / 'matching_results.tsv'

print(f"\\nWriting {matching_path}...")
with open(matching_path, 'w', encoding='utf-8') as f:
    f.write("source1_entity_id\\tmatched_entity_ids\\n")
    for sid in s1_ids_all:
        matches = final_matches.get(sid, [])
        f.write(f"{sid}\\t{','.join(matches)}\\n")

print(f"  Exported {matching_path.name} ({matching_path.stat().st_size / 1e6:.1f} MB)")

with open(matching_path, encoding='utf-8') as f:
    n_m_lines = sum(1 for _ in f) - 1

print(f"\\nDeliverable Verification:")
print(f"  matching_results.tsv rows: {n_m_lines:,} (Expected: {n_s1:,})")
assert n_m_lines == n_s1, f"Row mismatch in matching_results! Expected {n_s1}, got {n_m_lines}"

print("\\n[SUCCESS] matching_results.tsv is 100% READY TO DOWNLOAD AND SUBMIT!")
"""

nb['cells'][8]['source'] = [line + '\n' for line in cell8_code.strip().split('\n')]

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Successfully updated notebooks/06_stage6_test_submission_v2.ipynb with complete deterministic rule-based matching wiring!")
