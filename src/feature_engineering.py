"""
feature_engineering.py — Pairwise Feature Extraction for Stage 2 Matcher
Amazon ML Challenge 2026: Business Entity Resolution

Extracts lightweight, high-signal CPU features for each candidate pair:
1. Dense embedding similarities: cosine(name_emb), cosine(addr_emb).
2. Route indicators: route_name, route_addr, route_both.
3. String similarities: Jaro-Winkler, token Jaccard, length ratios.
4. Structural matching flags: house number match (1 / 0 / -1), state code match (1 / 0 / -1).
5. Source indicator: S2 vs S3 indicator flag.
6. Context / neighborhood features: candidate rank, score gap to top candidate.
"""

from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from rapidfuzz.distance import JaroWinkler


def compute_token_jaccard(tokens1: set, tokens2: set) -> float:
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def extract_pairwise_features(
    df_candidates: pd.DataFrame,
    s1_records: Dict[str, Dict[str, str]],
    target_records: Dict[str, Dict[str, str]],
    is_training: bool = False,
    ground_truth: Optional[Dict[str, set]] = None
) -> pd.DataFrame:
    """
    Extracts vectorized/optimized features for all candidate pairs in df_candidates.
    
    Args:
        df_candidates: DataFrame with ['s1_id', 'candidate_id', 'sim_name', 'sim_addr', 'route_name', 'route_addr']
        s1_records: dict of normalized S1 records
        target_records: dict of normalized S2/S3 target records
        is_training: if True, appends binary 'label' column
        ground_truth: ground truth dict required if is_training=True
        
    Returns:
        DataFrame containing all engineered feature columns (and 'label' if training).
    """
    n_pairs = len(df_candidates)
    print(f"[FeatureEngineering] Generating features for {n_pairs:,} candidate pairs...")

    # Pre-tokenize names and addresses for speed
    s1_name_tokens = {sid: set(rec.get("name_core", "").split()) for sid, rec in s1_records.items()}
    s1_addr_tokens = {sid: set(rec.get("addr_core", "").split()) for sid, rec in s1_records.items()}

    target_name_tokens = {cid: set(rec.get("name_core", "").split()) for cid, rec in target_records.items()}
    target_addr_tokens = {cid: set(rec.get("addr_core", "").split()) for cid, rec in target_records.items()}

    # Initialize feature arrays
    name_sim = df_candidates["sim_name"].values
    addr_sim = df_candidates["sim_addr"].values
    route_name = df_candidates["route_name"].values
    route_addr = df_candidates["route_addr"].values
    route_both = (route_name & route_addr).astype(np.int32)

    jw_sim = np.zeros(n_pairs, dtype=np.float32)
    name_jaccard = np.zeros(n_pairs, dtype=np.float32)
    name_exact = np.zeros(n_pairs, dtype=np.int32)
    name_len_diff = np.zeros(n_pairs, dtype=np.float32)

    house_match = np.zeros(n_pairs, dtype=np.int32)
    state_match = np.zeros(n_pairs, dtype=np.int32)
    addr_jaccard = np.zeros(n_pairs, dtype=np.float32)
    is_source2 = np.zeros(n_pairs, dtype=np.int32)

    labels = np.zeros(n_pairs, dtype=np.int32) if is_training else None

    s1_ids = df_candidates["s1_id"].values
    cand_ids = df_candidates["candidate_id"].values

    for i in range(n_pairs):
        sid = s1_ids[i]
        cid = cand_ids[i]

        s1_rec = s1_records.get(sid, {})
        t_rec = target_records.get(cid, {})

        # Source indicator
        is_source2[i] = 1 if cid.startswith("S2-") else 0

        # Name string features
        s1_name = s1_rec.get("name_core", "")
        t_name = t_rec.get("name_core", "")

        if s1_name and t_name:
            if s1_name == t_name:
                jw_sim[i] = 1.0
                name_exact[i] = 1
                name_jaccard[i] = 1.0
            else:
                jw_sim[i] = JaroWinkler.similarity(s1_name, t_name)
                tok1 = s1_name_tokens.get(sid, set())
                tok2 = target_name_tokens.get(cid, set())
                name_jaccard[i] = compute_token_jaccard(tok1, tok2)
            name_len_diff[i] = abs(len(s1_name) - len(t_name))

        # House number matching (1 = match, -1 = mismatch, 0 = missing/redacted)
        h1 = s1_rec.get("house_number", "")
        h2 = t_rec.get("house_number", "")
        if h1 and h2:
            house_match[i] = 1 if h1 == h2 else -1
        else:
            house_match[i] = 0

        # State matching (1 = match, -1 = mismatch, 0 = missing)
        st1 = s1_rec.get("state_code", "")
        st2 = t_rec.get("state_code", "")
        if st1 and st2:
            state_match[i] = 1 if st1 == st2 else -1
        else:
            state_match[i] = 0

        # Address token jaccard
        addr_tok1 = s1_addr_tokens.get(sid, set())
        addr_tok2 = target_addr_tokens.get(cid, set())
        addr_jaccard[i] = compute_token_jaccard(addr_tok1, addr_tok2)

        # Label if training
        if is_training and ground_truth is not None:
            true_set = ground_truth.get(sid, set())
            labels[i] = 1 if cid in true_set else 0

    # Build feature DataFrame
    feat_df = pd.DataFrame({
        "s1_id": s1_ids,
        "candidate_id": cand_ids,
        "name_sim": name_sim,
        "addr_sim": addr_sim,
        "route_name": route_name,
        "route_addr": route_addr,
        "route_both": route_both,
        "jw_sim": jw_sim,
        "name_jaccard": name_jaccard,
        "name_exact": name_exact,
        "name_len_diff": name_len_diff,
        "house_match": house_match,
        "state_match": state_match,
        "addr_jaccard": addr_jaccard,
        "is_source2": is_source2,
    })

    # Interaction & context features:
    # 1. Combined similarity product
    feat_df["sim_product"] = feat_df["name_sim"] * feat_df["addr_sim"]
    # 2. Maximum similarity
    feat_df["sim_max"] = np.maximum(feat_df["name_sim"], feat_df["addr_sim"])

    # 3. Neighborhood rank & gap to top per s1_id
    grouped_max = feat_df.groupby("s1_id")["sim_max"].transform("max")
    feat_df["gap_to_top"] = grouped_max - feat_df["sim_max"]
    feat_df["cand_rank"] = feat_df.groupby("s1_id")["sim_max"].rank(ascending=False, method="min")

    if is_training:
        feat_df["label"] = labels

    print("[FeatureEngineering] Feature extraction completed.")
    return feat_df


FEATURE_COLUMNS = [
    "name_sim", "addr_sim", "route_name", "route_addr", "route_both",
    "jw_sim", "name_jaccard", "name_exact", "name_len_diff",
    "house_match", "state_match", "addr_jaccard", "is_source2",
    "sim_product", "sim_max", "gap_to_top", "cand_rank"
]
