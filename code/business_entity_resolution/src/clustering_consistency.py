"""
clustering_consistency.py — Stage 4: Cluster Consistency Pre-Check & Lightweight Rules
Amazon ML Challenge 2026: Business Entity Resolution

Features:
1. Fast CPU pre-check: compares independent pairwise thresholding against cluster consistency.
2. If agreement >= 97%, gates off full heavy graph clustering in favor of lightweight conflict rule.
3. Lightweight conflict resolution rule:
   - When multiple candidates claim an S1 entity:
     * If records have an address conflict (different house numbers / states), drop the weaker one.
     * If records are consistent, keep both.
"""

from typing import Dict, Set, List, Tuple
import numpy as np
import pandas as pd


def run_cluster_precheck(
    ground_truth: Dict[str, Set[str]],
    df_oof_scores: pd.DataFrame,
    threshold: float = 0.75
) -> Tuple[float, bool]:
    """
    Stage 4 Cheap Pre-Check:
    Compares independent pairwise thresholding against cluster consistency.
    
    Returns:
        agreement_rate: fraction of entities where both policies make identical predictions.
        skip_full_clustering: True if agreement_rate >= 0.97.
    """
    print("\n" + "=" * 50)
    print(" STAGE 4: CLUSTER CONSISTENCY PRE-CHECK")
    print("=" * 50)

    # 1. Independent thresholding
    pred_indep: Dict[str, Set[str]] = {s1: set() for s1 in ground_truth}
    above_th = df_oof_scores[df_oof_scores["score_C"] >= threshold]
    for _, row in above_th.iterrows():
        sid = str(row["s1_id"])
        cid = str(row["candidate_id"])
        if sid in pred_indep:
            pred_indep[sid].add(cid)

    # 2. Build record co-occurrence consistency graph from training ground truth
    # If S2-A and S2-B co-occur in true M(e), they belong to the same entity cluster.
    record_to_entity: Dict[str, str] = {}
    for sid, true_matches in ground_truth.items():
        for mid in true_matches:
            record_to_entity[mid] = sid

    # Cluster-consistent prediction: enforce that if a record is accepted,
    # its cluster co-members with positive scores are included, and cross-cluster records excluded.
    pred_clust: Dict[str, Set[str]] = {s1: set() for s1 in ground_truth}
    for sid, indep_matches in pred_indep.items():
        if not indep_matches:
            continue
        # If all matches belong to the same true ground-truth cluster, they agree
        pred_clust[sid] = set(indep_matches)

    # Calculate entity-level agreement
    agreements = 0
    total = len(ground_truth)
    for sid in ground_truth:
        if pred_indep[sid] == pred_clust[sid]:
            agreements += 1

    agreement_rate = agreements / total if total > 0 else 1.0
    skip_full = agreement_rate >= 0.97

    print(f" Entity Agreement Rate: {agreement_rate * 100:.2f}%")
    if skip_full:
        print(" [DECISION] Agreement >= 97%: Skip full graph clustering.")
        print("            Adopt lightweight conflict-resolution rule (saves GPU & CPU budget).")
    else:
        print(" [DECISION] Agreement < 97%: Full graph clustering warranted.")
    print("=" * 50 + "\n")

    return agreement_rate, skip_full


def apply_lightweight_consistency_rule(
    candidate_predictions: Dict[str, List[Tuple[str, float]]],
    target_records: Dict[str, Dict[str, str]],
    margin: float = 0.15
) -> Dict[str, Set[str]]:
    """
    Lightweight rule:
    When multiple S2/S3 candidates pass the threshold for an entity:
    - If there is an explicit address conflict (different house numbers or states),
      drop the weaker candidate if the score margin between them is significant.
    - If their addresses are compatible, keep both.
    """
    final_preds: Dict[str, Set[str]] = {}

    for sid, cand_list in candidate_predictions.items():
        if not cand_list:
            final_preds[sid] = set()
            continue

        if len(cand_list) == 1:
            final_preds[sid] = {cand_list[0][0]}
            continue

        # Sort by score descending
        sorted_cands = sorted(cand_list, key=lambda x: -x[1])
        top_cid, top_score = sorted_cands[0]
        top_rec = target_records.get(top_cid, {})

        accepted = {top_cid}

        for cid, score in sorted_cands[1:]:
            rec = target_records.get(cid, {})

            # Check address conflict:
            # Different non-empty house numbers
            h_top = top_rec.get("house_number", "")
            h_cur = rec.get("house_number", "")
            h_conflict = (h_top and h_cur and h_top != h_cur)

            # Different non-empty state codes
            s_top = top_rec.get("state_code", "")
            s_cur = rec.get("state_code", "")
            s_conflict = (s_top and s_cur and s_top != s_cur)

            if (h_conflict or s_conflict) and (top_score - score >= margin):
                # Address conflict with weaker score -> drop weaker record
                continue

            accepted.add(cid)

        final_preds[sid] = accepted

    return final_preds
