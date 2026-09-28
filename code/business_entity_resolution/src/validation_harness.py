"""
validation_harness.py — Grouped Splitting, India Holdout & CV Harness
Amazon ML Challenge 2026: Business Entity Resolution

Features:
1. Leakage-free Grouped K-Fold splitting by S1 entity.
2. Stratified by singleton status and match cardinality.
3. Country-holdout split generator (holding out India to simulate unseen France behavior).
4. Efficient ground truth parsing & caching.
"""

from typing import Dict, Set, List, Tuple, Optional
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import KFold


def load_ground_truth(gt_path: Path) -> Dict[str, Set[str]]:
    """
    Load train_ground_truth.tsv into {source1_id: set_of_matched_ids}.
    Singletons map to empty sets.
    """
    gt_map: Dict[str, Set[str]] = {}
    # Use chunked read for memory efficiency
    for chunk in pd.read_csv(gt_path, sep="\t", chunksize=200000):
        for _, row in chunk.iterrows():
            s1_id = str(row["source1_entity_id"]).strip()
            matched_raw = str(row["matched_entity_ids"])
            if not matched_raw or matched_raw == "nan":
                gt_map[s1_id] = set()
            else:
                m_ids = {m.strip() for m in matched_raw.split(",") if m.strip()}
                gt_map[s1_id] = m_ids
    return gt_map


def create_grouped_kfold_splits(
    s1_df: pd.DataFrame,
    gt_map: Dict[str, Set[str]],
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create leakage-free K-Fold splits grouped by S1 entity.
    Stratified by singleton status and match cardinality.
    
    Returns list of (train_idx, val_idx) arrays.
    """
    s1_ids = s1_df["entity_id"].values
    n = len(s1_ids)

    # Compute stratification bins: 0 for singletons, 1 for 1-2 matches, 2 for 3-5 matches, 3 for 6+
    strata = np.zeros(n, dtype=np.int32)
    for i, sid in enumerate(s1_ids):
        m_count = len(gt_map.get(sid, set()))
        if m_count == 0:
            strata[i] = 0
        elif m_count <= 2:
            strata[i] = 1
        elif m_count <= 5:
            strata[i] = 2
        else:
            strata[i] = 3

    # Stratified split into folds
    fold_assignments = np.zeros(n, dtype=np.int32)
    rng = np.random.default_rng(seed)

    for s_val in np.unique(strata):
        indices = np.where(strata == s_val)[0]
        rng.shuffle(indices)
        for fold, fold_idx in enumerate(np.array_split(indices, n_splits)):
            fold_assignments[fold_idx] = fold

    splits = []
    for fold in range(n_splits):
        val_idx = np.where(fold_assignments == fold)[0]
        train_idx = np.where(fold_assignments != fold)[0]
        splits.append((train_idx, val_idx))

    return splits


def create_india_holdout_split(
    s1_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates an India-holdout split:
    - Train: US entities only
    - Val: India entities only
    Used specifically to validate that normalization and model do not collapse
    on an unseen country, simulating how France will behave at test time.
    """
    countries = s1_df["country"].fillna("UNKNOWN").values
    train_idx = np.where(countries == "US")[0]
    val_idx = np.where(countries == "India")[0]
    return train_idx, val_idx
