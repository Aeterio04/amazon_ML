import json
from pathlib import Path

p = Path("notebooks/06_stage6_test_submission_v2.ipynb")
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ---------------------------------------------------------
# UPDATE CELL 7: HIGH-PRECISION CALIBRATED DECISION ENGINE
# ---------------------------------------------------------
cell7_code = """# ============================================================
# HIGH-PRECISION CALIBRATED DECISION ENGINE & 1-TO-1 EXCLUSIVITY
# ============================================================

T_FIRST = 0.580   # Strong bar for top-1 to protect singletons (~5.6% target)
T_REST  = 0.420   # Recovers multi-match siblings (mean 3.67 matches in GT)

print("\\n" + "=" * 60)
print(f" APPLYING CALIBRATED DECISION RULE: t_first={T_FIRST:.3f}, t_rest={T_REST:.3f} + EXCLUSIVITY")
print("=" * 60)

# Group scored candidates by S1 entity
cands_by_s1 = {}
for sid, cid, prob in zip(scored_s1, scored_cand, scored_probs):
    sid = str(sid)
    if sid not in cands_by_s1:
        cands_by_s1[sid] = []
    cands_by_s1[sid].append((str(cid), float(prob)))

for sid in cands_by_s1:
    cands_by_s1[sid].sort(key=lambda x: -x[1])

# Pass 1: S1-level initial match acceptance
initial_matches = {}
for sid in s1_ids_all:
    cands = cands_by_s1.get(sid, [])
    if not cands or cands[0][1] < T_FIRST:
        initial_matches[sid] = []
    else:
        top_prob = cands[0][1]
        accepted = [(cands[0][0], top_prob)]
        for cid, p in cands[1:]:
            # Accept if passes T_REST and within 55% relative margin of top candidate
            if p >= T_REST and p >= top_prob * 0.55:
                accepted.append((cid, p))
        initial_matches[sid] = accepted

# Pass 2: 1-to-1 Target Exclusivity (Ground truth: 0.000% targets match >1 S1)
# If multiple S1 entities claim the same target record, award it strictly to the highest probability claim
best_claim = {} # cid -> (sid, prob)
for sid, matches in initial_matches.items():
    for cid, prob in matches:
        if cid not in best_claim or prob > best_claim[cid][1]:
            best_claim[cid] = (sid, prob)

final_matches = {sid: [] for sid in s1_ids_all}
for cid, (sid, prob) in best_claim.items():
    final_matches[sid].append(cid)

n_singletons = sum(1 for v in final_matches.values() if len(v) == 0)
n_matched = len(final_matches) - n_singletons
total_matched_ids = sum(len(v) for v in final_matches.values())
singleton_pct = (n_singletons / n_s1) * 100

print(f"\\nDecision Summary:")
print(f"  Total S1 Entities:     {n_s1:,}")
print(f"  Singletons (Empty):    {n_singletons:,} ({singleton_pct:.2f}% | Expected ~5.6%)")
print(f"  Resolved Entities:     {n_matched:,} ({100-singleton_pct:.2f}%)")
print(f"  Total Matched IDs:     {total_matched_ids:,}")
print(f"  Average Matches/Query: {total_matched_ids / max(n_matched, 1):.2f} (Target ~3.5 - 4.0)")"""

nb['cells'][7]['source'] = [line + '\n' for line in cell7_code.split('\n')]
if nb['cells'][7]['source'] and nb['cells'][7]['source'][-1] == '\n':
    nb['cells'][7]['source'].pop()

# ---------------------------------------------------------
# UPDATE CELL 8: EXPORT ONLY MATCHING RESULTS (FAST 5-SECOND EXPORT)
# ---------------------------------------------------------
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

# Verify row count
with open(matching_path, encoding='utf-8') as f:
    n_m_lines = sum(1 for _ in f) - 1

print(f"\\nDeliverable Verification:")
print(f"  matching_results.tsv rows: {n_m_lines:,} (Expected: {n_s1:,})")
assert n_m_lines == n_s1, f"Row mismatch in matching_results! Expected {n_s1}, got {n_m_lines}"

print("\\n[SUCCESS] matching_results.tsv is 100% READY TO DOWNLOAD AND SUBMIT!")"""

nb['cells'][8]['source'] = [line + '\n' for line in cell8_code.split('\n')]
if nb['cells'][8]['source'] and nb['cells'][8]['source'][-1] == '\n':
    nb['cells'][8]['source'].pop()

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Successfully updated notebooks/06_stage6_test_submission_v2.ipynb")
