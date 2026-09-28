"""
test_pipeline_slice.py — End-to-end integration test on a real dataset slice
Verifies that all 5 stages execute seamlessly from raw TSV to submission export.
"""

import unittest
import os
import shutil
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import Config
from src.normalization import normalize_record
from src.embedding_retrieval import DualChannelEncoder, DualFaissRetriever, run_gate1_audit
from src.feature_engineering import extract_pairwise_features
from src.pairwise_scorer import PairwiseLightGBMScorer
from src.clustering_consistency import run_cluster_precheck
from src.decision_layer import optimize_thresholds, predict_matches, export_submission_files


class TestPipelineSlice(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.test_work_dir = Path("work/test_slice")
        cls.test_work_dir.mkdir(parents=True, exist_ok=True)

        # 1. Load small slice of real data
        df_gt_full = pd.read_csv(Config.TRAIN_GROUND_TRUTH, sep="\t", nrows=100)
        cls.gt_map = {}
        target_ids_needed = set()

        for _, r in df_gt_full.iterrows():
            sid = str(r["source1_entity_id"]).strip()
            raw_m = str(r["matched_entity_ids"])
            if raw_m and raw_m != "nan":
                m_list = {m.strip() for m in raw_m.split(",") if m.strip()}
                cls.gt_map[sid] = m_list
                target_ids_needed.update(m_list)
            else:
                cls.gt_map[sid] = set()

        s1_ids = list(cls.gt_map.keys())
        df_s1_full = pd.read_csv(Config.TRAIN_SOURCE1, sep="\t")
        cls.df_s1_slice = df_s1_full[df_s1_full["entity_id"].isin(s1_ids)].copy()

        # Load S2 and S3 rows for targets plus distractors
        df_s2_full = pd.read_csv(Config.TRAIN_SOURCE2, sep="\t", nrows=500)
        df_s3_full = pd.read_csv(Config.TRAIN_SOURCE3, sep="\t", nrows=500)
        cls.df_target_slice = pd.concat([df_s2_full, df_s3_full], ignore_index=True).drop_duplicates(subset=["entity_id"])

    def test_end_to_end_flow(self):
        # Stage 0: Normalization
        s1_norm = [
            normalize_record(r["entity_id"], r["business_name"], r["business_address"], r["country"])
            for _, r in self.df_s1_slice.iterrows()
        ]
        tgt_norm = [
            normalize_record(r["entity_id"], r["business_name"], r["business_address"], r["country"])
            for _, r in self.df_target_slice.iterrows()
        ]

        df_s1_n = pd.DataFrame(s1_norm)
        df_tgt_n = pd.DataFrame(tgt_norm)
        self.assertGreater(len(df_s1_n), 0)
        self.assertGreater(len(df_tgt_n), 0)

        # Stage 1: Dual Embedding & Retrieval
        encoder = DualChannelEncoder(batch_size=32)

        s1_name_emb = encoder.encode_texts(df_s1_n["name_core"].fillna("").tolist(), prefix="query: ")
        s1_addr_emb = encoder.encode_texts(df_s1_n["addr_core"].fillna("").tolist(), prefix="query: ")
        tgt_name_emb = encoder.encode_texts(df_tgt_n["name_core"].fillna("").tolist(), prefix="passage: ")
        tgt_addr_emb = encoder.encode_texts(df_tgt_n["addr_core"].fillna("").tolist(), prefix="passage: ")

        self.assertEqual(s1_name_emb.shape[1], 384)
        self.assertEqual(tgt_name_emb.shape[1], 384)

        retriever = DualFaissRetriever(embedding_dim=384, use_ivf=False)
        retriever.build_indices(df_tgt_n["entity_id"].tolist(), tgt_name_emb, tgt_addr_emb)

        df_cands = retriever.query(
            df_s1_n["entity_id"].tolist(),
            s1_name_emb,
            s1_addr_emb,
            top_k=10
        )
        self.assertGreater(len(df_cands), 0)

        # Gate 1 Check
        metrics = run_gate1_audit(df_cands, self.gt_map, total_s23_count=len(df_tgt_n))
        self.assertIn("pair_completeness", metrics)

        # Stage 2: Feature Engineering
        s1_dict = {r["entity_id"]: r for _, r in df_s1_n.iterrows()}
        tgt_dict = {r["entity_id"]: r for _, r in df_tgt_n.iterrows()}

        df_features = extract_pairwise_features(
            df_cands,
            s1_dict,
            tgt_dict,
            is_training=True,
            ground_truth=self.gt_map
        )
        self.assertIn("jw_sim", df_features.columns)
        self.assertIn("label", df_features.columns)

        # Stage 2: Train LightGBM on 2-fold CV
        scorer = PairwiseLightGBMScorer()
        splits = [
            (np.arange(0, len(s1_norm) // 2), np.arange(len(s1_norm) // 2, len(s1_norm))),
            (np.arange(len(s1_norm) // 2, len(s1_norm)), np.arange(0, len(s1_norm) // 2))
        ]
        oof_preds, df_oof = scorer.train_cv(df_features, splits)
        self.assertEqual(len(oof_preds), len(df_features))

        # Stage 4: Cluster Pre-Check
        agree_rate, skip_full = run_cluster_precheck(self.gt_map, df_oof, threshold=0.75)
        self.assertGreaterEqual(agree_rate, 0.0)

        # Stage 5: Threshold Optimization & Export
        all_s1_ids = df_s1_n["entity_id"].tolist()
        t_first, t_rest, best_f05, pred_sing_rate = optimize_thresholds(df_oof, self.gt_map, all_s1_ids)

        preds = predict_matches(df_oof, all_s1_ids, t_first, t_rest)
        cands_grouped = {sid: list(df_cands[df_cands["s1_id"] == sid]["candidate_id"]) for sid in all_s1_ids}

        out_match, out_cand = export_submission_files(
            preds, cands_grouped, all_s1_ids, output_dir=self.test_work_dir
        )
        self.assertTrue(out_match.exists())
        self.assertTrue(out_cand.exists())

        # Verify output formats
        df_match_check = pd.read_csv(out_match, sep="\t")
        self.assertEqual(list(df_match_check.columns), ["source1_entity_id", "matched_entity_ids"])
        self.assertEqual(len(df_match_check), len(all_s1_ids))


if __name__ == "__main__":
    unittest.main()
