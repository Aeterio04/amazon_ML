"""
selective_cross_encoder.py — Stage 3: Margin-Gated Selective Cross-Encoder (D)
Amazon ML Challenge 2026: Business Entity Resolution

Features:
1. Measures ambiguous band size: |{pairs with score_C in [threshold - margin, threshold + margin]}|.
2. Gates cross-encoder inference based on measured band size and GPU budget.
3. Reranks only ambiguous candidate pairs using lightweight cross-encoder (MIT/Apache 2.0).
4. Blends cross-encoder score with pairwise score_C.
"""

from typing import Dict, List, Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from src.config import Config


def measure_ambiguous_band_size(
    df_scores: pd.DataFrame,
    threshold: float = 0.75,
    margin: float = 0.10
) -> Dict[str, float]:
    """
    Measures how many candidate pairs fall into the ambiguous margin band.
    """
    scores = df_scores["score_C"].values
    total_pairs = len(scores)

    low_bound = max(0.0, threshold - margin)
    high_bound = min(1.0, threshold + margin)

    in_band = (scores >= low_bound) & (scores <= high_bound)
    band_size = int(np.sum(in_band))
    band_pct = (band_size / total_pairs * 100) if total_pairs > 0 else 0.0

    # Rough estimate on Kaggle T4 GPU: ~1,500 pairs/sec
    est_gpu_hours = (band_size / 1500) / 3600

    stats = {
        "total_pairs": total_pairs,
        "band_low": low_bound,
        "band_high": high_bound,
        "band_size": band_size,
        "band_percentage": band_pct,
        "est_t4_gpu_hours": est_gpu_hours,
        "feasible": float(est_gpu_hours <= 3.0)
    }

    print("\n" + "=" * 50)
    print(" STAGE 3: AMBIGUOUS BAND AUDIT")
    print("=" * 50)
    print(f" Threshold: {threshold:.3f} | Margin: ±{margin:.3f} ([{low_bound:.3f}, {high_bound:.3f}])")
    print(f" Total Candidate Pairs:    {total_pairs:,}")
    print(f" Ambiguous Band Pairs:     {band_size:,} ({band_pct:.2f}%)")
    print(f" Est. T4 GPU Inference Time: {est_gpu_hours:.2f} hours")
    print(f" Gated Decision:           {'PROCEED WITH RERANKING' if stats['feasible'] else 'SKIP RERANKING (TOO EXPENSIVE)'}")
    print("=" * 50 + "\n")

    return stats


class SelectiveCrossEncoder:
    """
    Cross-Encoder for reranking ambiguous pairs.
    """

    def __init__(
        self,
        model_name: str = Config.CROSS_ENCODER_MODEL,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[CrossEncoder] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def rerank_ambiguous_pairs(
        self,
        df_scores: pd.DataFrame,
        s1_records: Dict[str, Dict[str, str]],
        target_records: Dict[str, Dict[str, str]],
        threshold: float = 0.75,
        margin: float = 0.10,
        batch_size: int = 128
    ) -> pd.DataFrame:
        """
        Reranks pairs inside [threshold - margin, threshold + margin] and returns updated DataFrame.
        """
        scores = df_scores["score_C"].values
        low = threshold - margin
        high = threshold + margin
        ambiguous_mask = (scores >= low) & (scores <= high)
        ambiguous_indices = np.where(ambiguous_mask)[0]

        if len(ambiguous_indices) == 0:
            print("[CrossEncoder] No pairs in ambiguous band. Returning original scores.")
            return df_scores.copy()

        print(f"[CrossEncoder] Reranking {len(ambiguous_indices):,} ambiguous pairs...")

        s1_ids = df_scores["s1_id"].values
        cand_ids = df_scores["candidate_id"].values

        pairs_text_a = []
        pairs_text_b = []

        for idx in ambiguous_indices:
            sid = s1_ids[idx]
            cid = cand_ids[idx]
            s1_rec = s1_records.get(sid, {})
            t_rec = target_records.get(cid, {})

            text_a = f"{s1_rec.get('name_core', '')} | {s1_rec.get('addr_core', '')}"
            text_b = f"{t_rec.get('name_core', '')} | {t_rec.get('addr_core', '')}"
            pairs_text_a.append(text_a)
            pairs_text_b.append(text_b)

        ce_scores = []
        for i in range(0, len(pairs_text_a), batch_size):
            batch_a = pairs_text_a[i:i + batch_size]
            batch_b = pairs_text_b[i:i + batch_size]

            inputs = self.tokenizer(
                batch_a, batch_b,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            logits = self.model(**inputs).logits
            if logits.shape[1] == 1:
                probs = torch.sigmoid(logits.squeeze(-1)).cpu().numpy()
            else:
                probs = torch.softmax(logits, dim=-1)[:, 1].cpu().numpy()
            ce_scores.extend(probs)

        # Update scores with weighted average for ambiguous pairs
        updated_scores = scores.copy()
        updated_scores[ambiguous_indices] = 0.5 * scores[ambiguous_indices] + 0.5 * np.array(ce_scores)

        df_out = df_scores.copy()
        df_out["score_C"] = updated_scores
        df_out["reranked_by_D"] = 0
        df_out.loc[ambiguous_indices, "reranked_by_D"] = 1

        print("[CrossEncoder] Reranking complete.")
        return df_out
