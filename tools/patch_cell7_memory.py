import json
from pathlib import Path

p = Path("notebooks/06_stage6_test_submission_v2.ipynb")
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

cell7_code = """# ============================================================
# MEMORY-EFFICIENT DETERMINISTIC RULE-BASED MATCHING & EXCLUSIVITY
# ============================================================

# 1. Clean up unused large arrays from previous cells to free ~8 GB RAM immediately
try:
    del df_tgt, tgt_names, tgt_addrs, tgt_houses, tgt_states
except Exception:
    pass
gc.collect()

t0_rule = time.time()
print("\\n" + "=" * 60)
print(" EXECUTING STREAMING DETERMINISTIC RULE-BASED MATCHING & EXCLUSIVITY")
print("=" * 60)

final_matches = {sid: [] for sid in s1_ids_all}

if PREV_MATCHING_TSV is not None and PREV_MATCHING_TSV.exists():
    print(f"  Loading baseline matches from {PREV_MATCHING_TSV.name}...")
    df_prev = pd.read_csv(PREV_MATCHING_TSV, sep='\\t', dtype=str)
    s1_col = df_prev.columns[0]
    match_col = df_prev.columns[1]
    
    raw_matches = {}
    confirmed_s1 = set()
    best_claim = {}  # cid -> (sid, score)
    
    for sid, m_str in zip(df_prev[s1_col], df_prev[match_col].fillna('')):
        sid = str(sid).strip()
        if m_str and str(m_str).strip() and str(m_str) != 'nan':
            c_list = [x.strip() for x in str(m_str).split(',') if x.strip()]
            raw_matches[sid] = c_list
            confirmed_s1.add(sid)
            for cid in c_list:
                best_claim[cid] = (sid, 0.90)  # High baseline score for already accepted matches
        else:
            raw_matches[sid] = []
    del df_prev; gc.collect()
    
    print(f"  Loaded baseline: {len(confirmed_s1):,} confirmed non-singletons, {len(best_claim):,} matched targets.")
    
    # 2. STREAMING SIBLING EXPANSION (Processes in small chunks to keep RAM < 1 GB)
    active_cand_parquet = CAND_PARQUET_FILE or (WORK_DIR / 'candidates_test.parquet')
    
    if active_cand_parquet and active_cand_parquet.exists():
        print(f"  Streaming sibling expansion from {active_cand_parquet.name}...")
        pf = pq.ParquetFile(str(active_cand_parquet))
        chunk_size = 2_000_000
        n_added = 0
        
        for batch in pf.iter_batches(batch_size=chunk_size, columns=['s1_id', 'candidate_id', 'sim_name', 'sim_addr']):
            df_b = batch.to_pandas()
            # Fast vectorized filter: keep only high-similarity candidates
            mask_sim = (df_b['sim_name'] >= 0.82) | ((df_b['sim_addr'] >= 0.85) & (df_b['sim_name'] >= 0.55))
            df_filt = df_b[mask_sim]
            del df_b; gc.collect()
            
            for sid, cid, sn, sa in zip(df_filt['s1_id'], df_filt['candidate_id'], df_filt['sim_name'], df_filt['sim_addr']):
                sid = str(sid)
                if sid not in confirmed_s1:
                    continue  # Protect singletons strictly
                cid = str(cid)
                score = max(float(sn), float(sa))
                claim = best_claim.get(cid)
                # Assign to highest-confidence S1 entity (1-to-1 Exclusivity)
                if claim is None or score > claim[1]:
                    best_claim[cid] = (sid, score)
                    n_added += 1
            del df_filt; gc.collect()
            
        print(f"  Sibling expansion complete: {n_added:,} candidates evaluated.")
        
    # 3. RULE: 1-to-1 TARGET EXCLUSIVITY AGGREGATION
    print("  Applying 1-to-1 Target Exclusivity aggregation...")
    for cid, (sid, score) in best_claim.items():
        final_matches[sid].append(cid)
        
else:
    # Fallback to LightGBM scored probabilities with calibrated thresholds
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
print(f"  Average Matches/Query: {total_matched_ids / max(n_matched, 1):.2f} (Target ~3.5 - 4.0)")"""

nb['cells'][7]['source'] = [line + '\n' for line in cell7_code.split('\n')]
if nb['cells'][7]['source'] and nb['cells'][7]['source'][-1] == '\n':
    nb['cells'][7]['source'].pop()

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Successfully patched Cell 7 with memory-efficient streaming logic!")
