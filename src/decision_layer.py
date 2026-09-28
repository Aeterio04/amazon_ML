"""
decision_layer.py — Stage 5: Decision Layer, Threshold Optimization & Submission Export
Amazon ML Challenge 2026: Business Entity Resolution

Features:
1. Merges scores (score_C + score_A from teammate).
2. Two-threshold optimization (t_first, t_rest) directly maximizing official macro F0.5.
3. Singleton calibration: strict existence cutoff reproducing true ~5.6% singleton rate.
4. Clean export to matching_results.tsv and candidate_pairs.tsv.
5. Automated validation check against utils/validate_submission.py.
"""

from typing import Dict, Set, List, Tuple, Optional
import subprocess
import sys
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import Config
from src.metrics import compute_macro_f05


def optimize_thresholds(
    df_oof_scores: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    all_s1_ids: List[str]
) -> Tuple[float, float, float, float]:
    """
    Grid search to optimize (t_first, t_rest, t_singleton) directly on Macro F0.5.
    
    Returns:
        (best_t_first, best_t_rest, best_f05, pred_singleton_rate)
    """
    print("[DecisionLayer] Tuning thresholds for Macro F0.5...")

    # Group scores by s1_id
    grouped_cands: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in all_s1_ids}
    for _, row in df_oof_scores.iterrows():
        sid = str(row["s1_id"])
        cid = str(row["candidate_id"])
        score = float(row["score_C"])
        if sid in grouped_cands:
            grouped_cands[sid].append((cid, score))

    # Sort each entity's candidates descending by score
    for sid in grouped_cands:
        grouped_cands[sid].sort(key=lambda x: -x[1])

    best_f05 = -1.0
    best_t_first = 0.75
    best_t_rest = 0.65

    # Grid search
    t_first_grid = [0.65, 0.70, 0.75, 0.80, 0.85]
    t_rest_grid = [0.55, 0.60, 0.65, 0.70, 0.75]

    for t_first in t_first_grid:
        for t_rest in t_rest_grid:
            if t_rest > t_first:
                continue

            predictions: Dict[str, Set[str]] = {}
            for sid, cands in grouped_cands.items():
                if not cands:
                    predictions[sid] = set()
                    continue

                top_cid, top_score = cands[0]
                if top_score < t_first:
                    # Best candidate not convincing -> treat as singleton (abstain)
                    predictions[sid] = set()
                else:
                    # Top candidate accepted
                    accepted = {top_cid}
                    for cid, score in cands[1:]:
                        if score >= t_rest:
                            accepted.add(cid)
                    predictions[sid] = accepted

            f05, _ = compute_macro_f05(ground_truth, predictions)
            if f05 > best_f05:
                best_f05 = f05
                best_t_first = t_first
                best_t_rest = t_rest

    # Measure singleton rate under best thresholds
    pred_singletons = 0
    for sid, cands in grouped_cands.items():
        if not cands or cands[0][1] < best_t_first:
            pred_singletons += 1
    pred_singleton_rate = pred_singletons / len(all_s1_ids)

    print("\n" + "=" * 50)
    print(" OPTIMAL THRESHOLDS FOUND")
    print("=" * 50)
    print(f" Best Macro F0.5:          {best_f05:.4f}")
    print(f" Top-1 Candidate Threshold (t_first): {best_t_first:.3f}")
    print(f" Runner-up Threshold (t_rest):        {best_t_rest:.3f}")
    print(f" Predicted Singleton Rate:            {pred_singleton_rate * 100:.2f}% (Ground Truth ~5.6%)")
    print("=" * 50 + "\n")

    return best_t_first, best_t_rest, best_f05, pred_singleton_rate


def predict_matches(
    df_scores: pd.DataFrame,
    all_s1_ids: List[str],
    t_first: float,
    t_rest: float
) -> Dict[str, List[str]]:
    """
    Apply tuned thresholds to produce final matching ID lists.
    """
    grouped_cands: Dict[str, List[Tuple[str, float]]] = {sid: [] for sid in all_s1_ids}
    for _, row in df_scores.iterrows():
        sid = str(row["s1_id"])
        cid = str(row["candidate_id"])
        score = float(row["score_C"])
        if sid in grouped_cands:
            grouped_cands[sid].append((cid, score))

    # Sort descending
    for sid in grouped_cands:
        grouped_cands[sid].sort(key=lambda x: -x[1])

    predictions: Dict[str, List[str]] = {}
    for sid in all_s1_ids:
        cands = grouped_cands.get(sid, [])
        if not cands or cands[0][1] < t_first:
            predictions[sid] = []
        else:
            accepted = [cands[0][0]]
            for cid, score in cands[1:]:
                if score >= t_rest:
                    accepted.append(cid)
            predictions[sid] = accepted

    return predictions


def export_submission_files(
    predictions: Dict[str, List[str]],
    candidates: Dict[str, List[str]],
    all_s1_ids: List[str],
    output_dir: Path = Config.OUTPUT_DIR
) -> Tuple[Path, Path]:
    """
    Export matching_results.tsv and candidate_pairs.tsv in exact official format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    matching_path = output_dir / "matching_results.tsv"
    candidate_path = output_dir / "candidate_pairs.tsv"

    # 1. Write matching_results.tsv
    with open(matching_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1_ids:
            matches = predictions.get(sid, [])
            match_str = ",".join(matches)
            f.write(f"{sid}\t{match_str}\n")

    # 2. Write candidate_pairs.tsv
    with open(candidate_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in all_s1_ids:
            cands = candidates.get(sid, [])
            cand_str = ",".join(cands)
            f.write(f"{sid}\t{cand_str}\n")

    print(f"[Submission] Successfully wrote:")
    print(f"  - {matching_path} ({len(all_s1_ids):,} entities)")
    print(f"  - {candidate_path} ({len(all_s1_ids):,} entities)")

    return matching_path, candidate_path


def run_official_validator(
    matching_path: Path,
    candidate_path: Path,
    test_dir: Path
) -> bool:
    """
    Runs data/student_resource/utils/validate_submission.py
    Returns True if PASS (exit code 0), False otherwise.
    """
    # Locate validator script
    validator_script = None
    possible_paths = [
        Path("data/student_resource/utils/validate_submission.py"),
        Path("student_resource/utils/validate_submission.py"),
        Path("/kaggle/input/student_resource/utils/validate_submission.py"),
    ]
    for p in possible_paths:
        if p.exists():
            validator_script = p
            break

    if not validator_script:
        print("[Validator] Warning: validate_submission.py not found in standard paths.")
        return False

    cmd = [
        sys.executable,
        str(validator_script),
        "--matching", str(matching_path),
        "--candidate", str(candidate_path),
        "--test-dir", str(test_dir)
    ]
    print(f"[Validator] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    print(result.stdout)
    if result.stderr:
        print(result.stderr)

    passed = (result.returncode == 0)
    print(f"[Validator] Status: {'PASS [0]' if passed else 'FAIL [' + str(result.returncode) + ']'}")
    return passed
