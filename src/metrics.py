"""
metrics.py — Official Evaluation Metrics & Blocking Diagnostics
Amazon ML Challenge 2026: Business Entity Resolution

Implements:
1. Macro F0.5 score:
   - Evaluated per Source 1 entity, then macro-averaged across all S1 entities.
   - Singleton rule:
     * If true M(e) is empty: score = 1.0 if predicted P(e) is empty, else 0.0.
   - Non-singleton rule:
     * If true M(e) is non-empty:
       - if P(e) is empty: score = 0.0
       - else: F0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
2. Blocking quality:
   - Pair Completeness (PC): |true pairs in candidates| / |all true pairs|
   - Reduction Ratio (RR): 1 - |candidates| / (|S1| * (|S2| + |S3|))
   - Candidate set distribution (mean, median, p95, max)
"""

from typing import Dict, Set, Tuple, List, Optional
import numpy as np


def compute_entity_f05(pred_ids: Set[str], true_ids: Set[str]) -> float:
    """
    Compute the official F0.5 score for a single Source 1 entity.
    """
    if len(true_ids) == 0:
        # Singleton entity: no true matches
        return 1.0 if len(pred_ids) == 0 else 0.0

    if len(pred_ids) == 0:
        # Non-singleton, but predicted empty -> recall is 0.0
        return 0.0

    tp = len(pred_ids & true_ids)
    if tp == 0:
        return 0.0

    precision = tp / len(pred_ids)
    recall = tp / len(true_ids)
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    return (1.25 * precision * recall) / denom


def compute_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    return_details: bool = False
) -> Tuple[float, Optional[Dict[str, float]]]:
    """
    Compute macro-average F0.5 across all S1 entities in ground truth.
    
    Args:
        ground_truth: dict mapping s1_id -> set of true matching IDs (empty set for singletons)
        predictions: dict mapping s1_id -> set of predicted matching IDs
        return_details: whether to return breakdown for singletons vs non-singletons
        
    Returns:
        macro_f05: float
        details: dict with 'singleton_f05', 'non_singleton_f05', 'singleton_count', 'total_count'
    """
    total_entities = len(ground_truth)
    if total_entities == 0:
        return 0.0, None

    singleton_scores = []
    non_singleton_scores = []
    all_scores = []

    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(pred_set, true_set)
        all_scores.append(score)
        if len(true_set) == 0:
            singleton_scores.append(score)
        else:
            non_singleton_scores.append(score)

    macro_f05 = float(np.mean(all_scores))

    if return_details:
        details = {
            "macro_f05": macro_f05,
            "singleton_f05": float(np.mean(singleton_scores)) if singleton_scores else 0.0,
            "non_singleton_f05": float(np.mean(non_singleton_scores)) if non_singleton_scores else 0.0,
            "singleton_count": len(singleton_scores),
            "non_singleton_count": len(non_singleton_scores),
            "total_entities": total_entities,
        }
        return macro_f05, details

    return macro_f05, None


def compute_blocking_metrics(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    total_s23_count: Optional[int] = None
) -> Dict[str, float]:
    """
    Compute Pair Completeness (PC), Reduction Ratio (RR), and candidate stats.
    
    Args:
        ground_truth: s1_id -> set of true matched IDs
        candidates: s1_id -> set of candidate IDs retrieved during blocking
        total_s23_count: |S2| + |S3| for exact RR computation
    """
    total_true_pairs = 0
    covered_true_pairs = 0
    candidate_counts = []

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        candidate_counts.append(len(cand_set))
        if len(true_set) > 0:
            total_true_pairs += len(true_set)
            covered_true_pairs += len(true_set & cand_set)

    pc = covered_true_pairs / total_true_pairs if total_true_pairs > 0 else 1.0

    counts_arr = np.array(candidate_counts, dtype=np.int32)
    stats = {
        "pair_completeness": float(pc),
        "total_true_pairs": int(total_true_pairs),
        "covered_true_pairs": int(covered_true_pairs),
        "cand_count_mean": float(np.mean(counts_arr)),
        "cand_count_median": float(np.median(counts_arr)),
        "cand_count_p95": float(np.percentile(counts_arr, 95)),
        "cand_count_max": int(np.max(counts_arr)) if len(counts_arr) > 0 else 0,
        "cand_count_total": int(np.sum(counts_arr)),
    }

    if total_s23_count is not None and len(ground_truth) > 0:
        total_possible = len(ground_truth) * total_s23_count
        rr = 1.0 - (stats["cand_count_total"] / total_possible)
        stats["reduction_ratio"] = float(rr)

    return stats
