"""
pairwise_scorer.py — Stage 2: LightGBM Pairwise Scorer (score_C)
Amazon ML Challenge 2026: Business Entity Resolution

Features:
1. Trains LightGBM GBDT classifier with grouped CV by S1 entity.
2. Binary logloss objective with early stopping.
3. Produces out-of-fold probability predictions on validation set.
4. Predicts test candidate pairs.
5. Saves versioned score_C.parquet keyed by (s1_id, candidate_id).
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV

from src.config import Config
from src.feature_engineering import FEATURE_COLUMNS


class PairwiseLightGBMScorer:
    """
    LightGBM model for scoring candidate pairs (s1_id, candidate_id).
    """

    def __init__(self, params: Optional[Dict] = None):
        self.params = params or Config.LGBM_PARAMS.copy()
        self.models: List[lgb.Booster] = []
        self.feature_names = FEATURE_COLUMNS

    def train_cv(
        self,
        df_train_pairs: pd.DataFrame,
        splits: List[Tuple[np.ndarray, np.ndarray]]
    ) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Train LightGBM across grouped CV folds.
        
        Args:
            df_train_pairs: DataFrame containing FEATURE_COLUMNS + ['s1_id', 'candidate_id', 'label']
            splits: list of (train_idx, val_idx) grouped by S1 entity
            
        Returns:
            oof_preds: out-of-fold predicted probabilities
            df_oof: DataFrame with ['s1_id', 'candidate_id', 'score_C', 'label']
        """
        print(f"[LightGBM] Training {len(splits)}-fold CV over {len(df_train_pairs):,} pairs...")
        X = df_train_pairs[self.feature_names].values
        y = df_train_pairs["label"].values
        s1_ids = df_train_pairs["s1_id"].values
        cand_ids = df_train_pairs["candidate_id"].values

        oof_preds = np.zeros(len(df_train_pairs), dtype=np.float32)
        self.models = []

        # Map S1 entities to row indices
        s1_to_rows: Dict[str, List[int]] = {}
        for r_idx, sid in enumerate(s1_ids):
            s1_to_rows.setdefault(sid, []).append(r_idx)

        unique_s1 = np.unique(s1_ids)

        for fold, (train_s1_idx, val_s1_idx) in enumerate(splits):
            val_entities = set(unique_s1[val_s1_idx])

            val_row_idx = []
            train_row_idx = []
            for sid, r_indices in s1_to_rows.items():
                if sid in val_entities:
                    val_row_idx.extend(r_indices)
                else:
                    train_row_idx.extend(r_indices)

            val_row_idx = np.array(val_row_idx, dtype=np.int32)
            train_row_idx = np.array(train_row_idx, dtype=np.int32)

            X_tr, y_tr = X[train_row_idx], y[train_row_idx]
            X_va, y_va = X[val_row_idx], y[val_row_idx]

            dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=self.feature_names)
            dval = lgb.Dataset(X_va, label=y_va, reference=dtrain, feature_name=self.feature_names)

            callbacks = [
                lgb.early_stopping(stopping_rounds=50, verbose=False),
                lgb.log_evaluation(period=0)
            ]

            booster = lgb.train(
                self.params,
                dtrain,
                valid_sets=[dtrain, dval],
                callbacks=callbacks
            )
            self.models.append(booster)

            # Predict on val
            val_preds = booster.predict(X_va, num_iteration=booster.best_iteration)
            oof_preds[val_row_idx] = val_preds
            print(f"  Fold {fold + 1}/{len(splits)} finished (best_iter={booster.best_iteration})")

        df_oof = pd.DataFrame({
            "s1_id": s1_ids,
            "candidate_id": cand_ids,
            "score_C": oof_preds,
            "label": y
        })

        return oof_preds, df_oof

    def predict(self, df_test_pairs: pd.DataFrame) -> pd.DataFrame:
        """
        Ensemble prediction across all trained fold boosters.
        Returns DataFrame with ['s1_id', 'candidate_id', 'score_C'].
        """
        if not self.models:
            raise ValueError("No trained models found. Call train_cv first.")

        X_test = df_test_pairs[self.feature_names].values
        test_preds = np.zeros(len(df_test_pairs), dtype=np.float32)

        for booster in self.models:
            test_preds += booster.predict(X_test, num_iteration=booster.best_iteration)

        test_preds /= len(self.models)

        df_scores = pd.DataFrame({
            "s1_id": df_test_pairs["s1_id"].values,
            "candidate_id": df_test_pairs["candidate_id"].values,
            "score_C": test_preds
        })
        return df_scores

    def save_scores(self, df_scores: pd.DataFrame, output_path: Path):
        """
        Save versioned score_C.parquet
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_scores.to_parquet(output_path, index=False)
        print(f"[LightGBM] Saved scores to {output_path} ({len(df_scores):,} pairs).")
