"""
pipeline.py — Master Orchestration Pipeline for Track C + Add-ons
Amazon ML Challenge 2026: Business Entity Resolution

Orchestrates all 5 stages with full checkpointing:
Stage 0: Normalization & Grouped Splitting (CPU)
Stage 1: Dual-Channel Multilingual Embedding Retrieval & Gate 1 (GPU/CPU)
Stage 2: Pairwise Feature Engineering & LightGBM Scorer (score_C) (CPU)
Stage 3: Gated Selective Cross-Encoder Reranking (GPU)
Stage 4: Cluster Consistency Pre-Check & Conflict Rules (CPU)
Stage 5: Decision Layer, Threshold Optimization & Submission Export (CPU)
"""

import argparse
import sys
import gc
from pathlib import Path
import numpy as np
import pandas as pd

from src.config import Config
from src.normalization import normalize_record
from src.validation_harness import load_ground_truth, create_grouped_kfold_splits, create_india_holdout_split
from src.embedding_retrieval import DualChannelEncoder, DualFaissRetriever, run_gate1_audit
from src.feature_engineering import extract_pairwise_features
from src.pairwise_scorer import PairwiseLightGBMScorer
from src.selective_cross_encoder import measure_ambiguous_band_size, SelectiveCrossEncoder
from src.clustering_consistency import run_cluster_precheck, apply_lightweight_consistency_rule
from src.decision_layer import optimize_thresholds, predict_matches, export_submission_files, run_official_validator


def run_stage0(sample_size: Optional[int] = None):
    """
    Stage 0: Normalize raw data and save to Parquet.
    """
    print("\n" + "#" * 60)
    print(" STAGE 0: DATA NORMALIZATION & CLEANING")
    print("#" * 60)

    Config.setup_directories()

    # Process Train S1
    print(f"[Stage 0] Normalizing train_source1.tsv...")
    df_s1 = pd.read_csv(Config.TRAIN_SOURCE1, sep="\t", nrows=sample_size)
    s1_norm = [
        normalize_record(r["entity_id"], r["business_name"], r["business_address"], r["country"])
        for _, r in df_s1.iterrows()
    ]
    pd.DataFrame(s1_norm).to_parquet(Config.TRAIN_S1_NORM, index=False)
    print(f"  Saved {len(s1_norm):,} records to {Config.TRAIN_S1_NORM}")

    # Process Train S2
    print(f"[Stage 0] Normalizing train_source2.tsv...")
    df_s2 = pd.read_csv(Config.TRAIN_SOURCE2, sep="\t", nrows=sample_size * 2 if sample_size else None)
    s2_norm = [
        normalize_record(r["entity_id"], r["business_name"], r["business_address"], r["country"])
        for _, r in df_s2.iterrows()
    ]
    pd.DataFrame(s2_norm).to_parquet(Config.TRAIN_S2_NORM, index=False)
    print(f"  Saved {len(s2_norm):,} records to {Config.TRAIN_S2_NORM}")

    # Process Train S3
    print(f"[Stage 0] Normalizing train_source3.tsv...")
    df_s3 = pd.read_csv(Config.TRAIN_SOURCE3, sep="\t", nrows=sample_size * 2 if sample_size else None)
    s3_norm = [
        normalize_record(r["entity_id"], r["business_name"], r["business_address"], r["country"])
        for _, r in df_s3.iterrows()
    ]
    pd.DataFrame(s3_norm).to_parquet(Config.TRAIN_S3_NORM, index=False)
    print(f"  Saved {len(s3_norm):,} records to {Config.TRAIN_S3_NORM}")

    print("[Stage 0] Completed successfully.\n")


def run_stage1_and_gate1(sample_size: Optional[int] = None) -> pd.DataFrame:
    """
    Stage 1: Encode dual embeddings, build FAISS indices, retrieve candidates, audit Gate 1.
    """
    print("\n" + "#" * 60)
    print(" STAGE 1: DUAL-CHANNEL EMBEDDING RETRIEVAL & GATE 1")
    print("#" * 60)

    # Load normalized data
    df_s1 = pd.read_parquet(Config.TRAIN_S1_NORM)
    df_s2 = pd.read_parquet(Config.TRAIN_S2_NORM)
    df_s3 = pd.read_parquet(Config.TRAIN_S3_NORM)

    df_target = pd.concat([df_s2, df_s3], ignore_index=True)
    print(f"[Stage 1] Query records (S1): {len(df_s1):,} | Target records (S2+S3): {len(df_target):,}")

    encoder = DualChannelEncoder()

    # Encode S1
    print("[Stage 1] Encoding S1 names...")
    s1_name_emb = encoder.encode_texts(df_s1["name_core"].fillna("").tolist(), prefix="query: ")
    print("[Stage 1] Encoding S1 addresses...")
    s1_addr_emb = encoder.encode_texts(df_s1["addr_core"].fillna("").tolist(), prefix="query: ")

    # Encode Target (S2 + S3)
    print("[Stage 1] Encoding Target names...")
    tgt_name_emb = encoder.encode_texts(df_target["name_core"].fillna("").tolist(), prefix="passage: ")
    print("[Stage 1] Encoding Target addresses...")
    tgt_addr_emb = encoder.encode_texts(df_target["addr_core"].fillna("").tolist(), prefix="passage: ")

    # Build FAISS indices
    retriever = DualFaissRetriever(use_ivf=False)
    retriever.build_indices(df_target["entity_id"].tolist(), tgt_name_emb, tgt_addr_emb)

    # Query dual routes
    df_cands = retriever.query(
        df_s1["entity_id"].tolist(),
        s1_name_emb,
        s1_addr_emb,
        top_k=Config.RETRIEVAL_TOP_K
    )

    # Gate 1 Audit
    gt_map = load_ground_truth(Config.TRAIN_GROUND_TRUTH)
    audit = run_gate1_audit(df_cands, gt_map, total_s23_count=len(df_target))

    cand_parquet = Config.WORK_DIR / "candidates_train.parquet"
    df_cands.to_parquet(cand_parquet, index=False)
    print(f"[Stage 1] Saved candidates to {cand_parquet}")
    return df_cands


def run_stage2(df_cands: pd.DataFrame) -> Tuple[pd.DataFrame, PairwiseLightGBMScorer]:
    """
    Stage 2: Pairwise feature extraction and LightGBM training (score_C).
    """
    print("\n" + "#" * 60)
    print(" STAGE 2: PAIRWISE FEATURE ENGINEERING & LIGHTGBM SCORER")
    print("#" * 60)

    df_s1 = pd.read_parquet(Config.TRAIN_S1_NORM)
    df_s2 = pd.read_parquet(Config.TRAIN_S2_NORM)
    df_s3 = pd.read_parquet(Config.TRAIN_S3_NORM)
    df_target = pd.concat([df_s2, df_s3], ignore_index=True)

    s1_dict = {r["entity_id"]: r for _, r in df_s1.iterrows()}
    tgt_dict = {r["entity_id"]: r for _, r in df_target.iterrows()}
    gt_map = load_ground_truth(Config.TRAIN_GROUND_TRUTH)

    # Extract features
    df_features = extract_pairwise_features(
        df_cands,
        s1_dict,
        tgt_dict,
        is_training=True,
        ground_truth=gt_map
    )

    # Create Grouped K-Fold splits
    splits = create_grouped_kfold_splits(df_s1, gt_map, n_splits=5)

    # Train LightGBM
    scorer = PairwiseLightGBMScorer()
    oof_preds, df_oof = scorer.train_cv(df_features, splits)

    score_path = Config.WORK_DIR / "score_C.parquet"
    scorer.save_scores(df_oof, score_path)

    return df_oof, scorer


def run_stage4_and_stage5(df_oof: pd.DataFrame):
    """
    Stage 4 pre-check + Stage 5 decision layer.
    """
    print("\n" + "#" * 60)
    print(" STAGE 4 & 5: CLUSTER PRE-CHECK & DECISION LAYER")
    print("#" * 60)

    gt_map = load_ground_truth(Config.TRAIN_GROUND_TRUTH)
    df_s1 = pd.read_parquet(Config.TRAIN_S1_NORM)
    all_s1_ids = df_s1["entity_id"].tolist()

    # Stage 4 Pre-Check
    agree_rate, skip_clustering = run_cluster_precheck(gt_map, df_oof, threshold=0.75)

    # Stage 3 Ambiguous Band Audit
    band_stats = measure_ambiguous_band_size(df_oof, threshold=0.75, margin=0.10)

    # Stage 5 Threshold Optimization
    t_first, t_rest, best_f05, pred_singleton_rate = optimize_thresholds(df_oof, gt_map, all_s1_ids)

    # Generate predictions
    predictions = predict_matches(df_oof, all_s1_ids, t_first, t_rest)

    # Format candidates for candidate_pairs.tsv audit
    cands_grouped: Dict[str, List[str]] = {sid: [] for sid in all_s1_ids}
    for _, row in df_oof.iterrows():
        sid = str(row["s1_id"])
        cid = str(row["candidate_id"])
        if sid in cands_grouped:
            cands_grouped[sid].append(cid)

    # Export
    match_path, cand_path = export_submission_files(predictions, cands_grouped, all_s1_ids)

    print("\n[Pipeline] Complete run finished successfully!")


def main():
    parser = argparse.ArgumentParser(description="Amazon ML Challenge 2026 - Pipeline")
    parser.add_argument("--stage", type=str, default="all", help="Stage to run: 0, 1, 2, 4, 5, or all")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for fast validation run")
    args = parser.parse_args()

    if args.stage in ["0", "all"]:
        run_stage0(sample_size=args.sample)

    if args.stage in ["1", "all"]:
        df_cands = run_stage1_and_gate1(sample_size=args.sample)
    else:
        cand_parquet = Config.WORK_DIR / "candidates_train.parquet"
        if cand_parquet.exists():
            df_cands = pd.read_parquet(cand_parquet)
        else:
            df_cands = None

    if args.stage in ["2", "all"] and df_cands is not None:
        df_oof, scorer = run_stage2(df_cands)
    else:
        score_path = Config.WORK_DIR / "score_C.parquet"
        if score_path.exists():
            df_oof = pd.read_parquet(score_path)
        else:
            df_oof = None

    if args.stage in ["4", "5", "all"] and df_oof is not None:
        run_stage4_and_stage5(df_oof)


if __name__ == "__main__":
    main()
