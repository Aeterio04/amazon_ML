import json
from pathlib import Path

p = Path("notebooks/06_stage6_test_submission_v2.ipynb")
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

# -------------------------------------------------------------
# 1. UPDATE CELL 2: Add Discovery for SCORED_CAND_PARQUET
# -------------------------------------------------------------
cell2_src = "".join(nb['cells'][2]['source'])

old_discovery = """# 5. Submission Validator
VALIDATOR_SCRIPT = find_file('validate_submission.py')"""

new_discovery = """# 5. Submission Validator
VALIDATOR_SCRIPT = find_file('validate_submission.py')

# 6. Pre-scored Candidates Dataset (Fast Track Auto-Discovery)
SCORED_CAND_PARQUET = find_file('scored_candidates_test.parquet')
if SCORED_CAND_PARQUET is None:
    for pat in ['*scored*.parquet', 'test_scores*.parquet', 'score_test*.parquet']:
        cand = find_file(pat)
        if cand and 'train' not in str(cand).lower():
            SCORED_CAND_PARQUET = cand
            break"""

cell2_src = cell2_src.replace(old_discovery, new_discovery)

old_print = """print(f"  Tgt Addr Emb:      {TEST_TGT_ADDR_EMB or (TEST_S2_ADDR_EMB and TEST_S3_ADDR_EMB)}")
print(f"  Output Dir:        {OUTPUT_DIR}")"""

new_print = """print(f"  Tgt Addr Emb:      {TEST_TGT_ADDR_EMB or (TEST_S2_ADDR_EMB and TEST_S3_ADDR_EMB)}")
print(f"  Scored Dataset:    {SCORED_CAND_PARQUET}")
print(f"  Output Dir:        {OUTPUT_DIR}")"""

cell2_src = cell2_src.replace(old_print, new_print)

nb['cells'][2]['source'] = [line + '\n' for line in cell2_src.split('\n')]
if nb['cells'][2]['source'] and nb['cells'][2]['source'][-1] == '\n':
    nb['cells'][2]['source'].pop()

# -------------------------------------------------------------
# 2. UPDATE CELL 5: Skip FAISS if SCORED_CAND_PARQUET exists
# -------------------------------------------------------------
cell5_src = "".join(nb['cells'][5]['source'])

old_c5_top = """cand_parquet_path = WORK_DIR / 'candidates_test.parquet'
cand_tsv_path = OUTPUT_DIR / 'candidate_pairs.tsv'

TOP_K = 50

if cand_parquet_path.exists() and cand_parquet_path.stat().st_size > 10_000_000:"""

new_c5_top = """cand_parquet_path = WORK_DIR / 'candidates_test.parquet'
cand_tsv_path = OUTPUT_DIR / 'candidate_pairs.tsv'

TOP_K = 50

if SCORED_CAND_PARQUET and SCORED_CAND_PARQUET.exists() and SCORED_CAND_PARQUET.stat().st_size > 100_000:
    print(f"\\n[FAST TRACK] Found pre-scored dataset: {SCORED_CAND_PARQUET} ({SCORED_CAND_PARQUET.stat().st_size / 1e6:.1f} MB)")
    print("  Skipping FAISS candidate retrieval completely!")
elif cand_parquet_path.exists() and cand_parquet_path.stat().st_size > 10_000_000:"""

cell5_src = cell5_src.replace(old_c5_top, new_c5_top)

nb['cells'][5]['source'] = [line + '\n' for line in cell5_src.split('\n')]
if nb['cells'][5]['source'] and nb['cells'][5]['source'][-1] == '\n':
    nb['cells'][5]['source'].pop()

# -------------------------------------------------------------
# 3. UPDATE CELL 6: Fast Load from SCORED_CAND_PARQUET or Export
# -------------------------------------------------------------
cell6_src = "".join(nb['cells'][6]['source'])

old_c6_start = """print("\\n" + "=" * 60)
print(" EVALUATING CANDIDATES WITH LIGHTGBM ENSEMBLE")
print("=" * 60)"""

new_c6_start = """if SCORED_CAND_PARQUET and SCORED_CAND_PARQUET.exists() and SCORED_CAND_PARQUET.stat().st_size > 100_000:
    print("\\n" + "=" * 60)
    print(f" [FAST TRACK] LOADING PRE-SCORED CANDIDATES: {SCORED_CAND_PARQUET.name}")
    print("=" * 60)
    t0_load = time.time()
    df_scored = pd.read_parquet(SCORED_CAND_PARQUET)
    
    p_col = next((c for c in ['prob', 'score', 'score_C', 'preds', 'probability'] if c in df_scored.columns), None)
    s1_col = next((c for c in ['s1_id', 'source1_entity_id', 's1'] if c in df_scored.columns), None)
    cand_col = next((c for c in ['candidate_id', 'candidate_entity_id', 'cand_id', 'cand'] if c in df_scored.columns), None)
    
    assert p_col and s1_col and cand_col, f"Unrecognized columns in {SCORED_CAND_PARQUET.name}: {list(df_scored.columns)}"
    
    scored_s1 = df_scored[s1_col].values.astype(str)
    scored_cand = df_scored[cand_col].values.astype(str)
    scored_probs = df_scored[p_col].values.astype(np.float32)
    del df_scored; gc.collect()
    print(f"  Loaded {len(scored_s1):,} pre-scored pairs in {time.time()-t0_load:.1f}s! Skipped feature extraction & LightGBM scoring.")
else:
    print("\\n" + "=" * 60)
    print(" EVALUATING CANDIDATES WITH LIGHTGBM ENSEMBLE")
    print("=" * 60)"""

cell6_src = cell6_src.replace(old_c6_start, new_c6_start)

# Add automatic export of scored candidates at the end of Cell 6
old_c6_end = """print(f"Scoring completed in {time.time()-t0_score:.1f}s. Total candidates to evaluate: {len(scored_s1):,}")"""

new_c6_end = """    print(f"Scoring completed in {time.time()-t0_score:.1f}s. Total candidates to evaluate: {len(scored_s1):,}")
    
    # Export scored candidates checkpoint for 1-second future fast-tracking
    df_out_scored = pd.DataFrame({
        's1_id': scored_s1,
        'candidate_id': scored_cand,
        'prob': scored_probs
    })
    save_path = OUTPUT_DIR / 'scored_candidates_test.parquet'
    df_out_scored.to_parquet(save_path, index=False)
    try:
        df_out_scored.to_parquet(WORK_DIR / 'scored_candidates_test.parquet', index=False)
    except Exception:
        pass
    print(f"  [Checkpoint] Exported scored predictions to {save_path.name} ({save_path.stat().st_size / 1e6:.1f} MB)")
    del df_out_scored; gc.collect()"""

cell6_src = cell6_src.replace(old_c6_end, new_c6_end)

nb['cells'][6]['source'] = [line + '\n' for line in cell6_src.split('\n')]
if nb['cells'][6]['source'] and nb['cells'][6]['source'][-1] == '\n':
    nb['cells'][6]['source'].pop()

# Save updated notebook
with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Successfully updated 06_stage6_test_submission_v2.ipynb with Fast-Track Dataset Auto-Discovery!")
