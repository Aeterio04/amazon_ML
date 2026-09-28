"""
embedding_retrieval.py — Stage 1: Dual-Channel Multilingual Embedding Retrieval
Amazon ML Challenge 2026: Business Entity Resolution

Features:
1. Dual independent embedding spaces: name_core and addr_core.
2. Pretrained multilingual sentence encoder (intfloat/multilingual-e5-small, MIT license, 118M params).
3. Zero-shot bi-encoder with PyTorch FP16 and chunked disk checkpointing (resumable).
4. FAISS IVF / Flat inner product index for high-speed top-k candidate retrieval.
5. Dual route union and deduplication.
6. Gate 1 Pair Completeness (PC) audit against training labels (target >= 0.95).
"""

import os
import gc
from typing import Dict, List, Tuple, Set, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel
import faiss

from src.config import Config
from src.metrics import compute_blocking_metrics


class TextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


class DualChannelEncoder:
    """
    Encodes text into dense normalized embeddings using multilingual transformer bi-encoder.
    """

    def __init__(
        self,
        model_name: str = Config.ENCODER_MODEL_NAME,
        batch_size: int = Config.EMB_BATCH_SIZE,
        max_length: int = Config.EMB_MAX_SEQ_LENGTH,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        print(f"[Encoder] Loading {model_name} on {self.device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        if self.device == "cuda" and torch.cuda.device_count() > 1:
            print(f"[Encoder] Using {torch.cuda.device_count()} GPUs with DataParallel")
            self.model = torch.nn.DataParallel(self.model)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    @torch.no_grad()
    def encode_texts(
        self,
        texts: List[str],
        prefix: str = "passage: ",
        show_progress: bool = True
    ) -> np.ndarray:
        """
        Encode a list of text strings into L2-normalized float32 numpy embeddings.
        """
        dataset = TextDataset([f"{prefix}{t}" if prefix else t for t in texts])
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

        all_embeddings = []
        use_amp = (self.device == "cuda")

        for batch_texts in loader:
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)

            with torch.amp.autocast(device_type=self.device, enabled=use_amp):
                outputs = self.model(**encoded)
                embeddings = self._mean_pooling(outputs, encoded["attention_mask"])
                # L2 normalize
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings.cpu().to(torch.float32).numpy())

        return np.vstack(all_embeddings)


class DualFaissRetriever:
    """
    Manages independent FAISS indices for Name and Address channels.
    """

    def __init__(
        self,
        embedding_dim: int = Config.EMBEDDING_DIM,
        use_ivf: bool = False,
        nlist: int = 4096,
        nprobe: int = 32
    ):
        self.dim = embedding_dim
        self.use_ivf = use_ivf
        self.nlist = nlist
        self.nprobe = nprobe
        self.index_name = None
        self.index_addr = None
        self.target_ids: List[str] = []

    def build_indices(
        self,
        target_ids: List[str],
        name_embeddings: np.ndarray,
        addr_embeddings: np.ndarray
    ):
        """
        Build dual indices for candidate database (S2 + S3).
        Vectors must be L2 normalized so Inner Product equals Cosine Similarity.
        """
        self.target_ids = list(target_ids)
        n = len(self.target_ids)
        print(f"[FAISS] Building dual indices over {n:,} target records (dim={self.dim})...")

        if self.use_ivf and n >= self.nlist * 4:
            quantizer_name = faiss.IndexFlatIP(self.dim)
            self.index_name = faiss.IndexIVFFlat(quantizer_name, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index_name.train(name_embeddings)
            self.index_name.add(name_embeddings)
            self.index_name.nprobe = self.nprobe

            quantizer_addr = faiss.IndexFlatIP(self.dim)
            self.index_addr = faiss.IndexIVFFlat(quantizer_addr, self.dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index_addr.train(addr_embeddings)
            self.index_addr.add(addr_embeddings)
            self.index_addr.nprobe = self.nprobe
        else:
            # Exact inner product (flat)
            self.index_name = faiss.IndexFlatIP(self.dim)
            self.index_name.add(name_embeddings)

            self.index_addr = faiss.IndexFlatIP(self.dim)
            self.index_addr.add(addr_embeddings)

        print("[FAISS] Indices successfully built.")

    def query(
        self,
        s1_ids: List[str],
        s1_name_embeddings: np.ndarray,
        s1_addr_embeddings: np.ndarray,
        top_k: int = Config.RETRIEVAL_TOP_K
    ) -> pd.DataFrame:
        """
        Query both name and address routes, take union, deduplicate, and record similarities.
        Returns DataFrame with columns:
        ['s1_id', 'candidate_id', 'sim_name', 'sim_addr', 'route_name', 'route_addr']
        """
        print(f"[FAISS] Querying top-{top_k} per route for {len(s1_ids):,} entities...")
        # 1. Search Name Index
        sims_name, idxs_name = self.index_name.search(s1_name_embeddings, top_k)
        # 2. Search Address Index
        sims_addr, idxs_addr = self.index_addr.search(s1_addr_embeddings, top_k)

        rows = []
        for i, sid in enumerate(s1_ids):
            cand_map: Dict[str, Dict] = {}

            # Process Name route
            for k in range(top_k):
                t_idx = idxs_name[i, k]
                if t_idx < 0 or t_idx >= len(self.target_ids):
                    continue
                cid = self.target_ids[t_idx]
                score = float(sims_name[i, k])
                cand_map[cid] = {
                    "sim_name": score,
                    "sim_addr": 0.0,
                    "route_name": 1,
                    "route_addr": 0
                }

            # Process Address route
            for k in range(top_k):
                t_idx = idxs_addr[i, k]
                if t_idx < 0 or t_idx >= len(self.target_ids):
                    continue
                cid = self.target_ids[t_idx]
                score = float(sims_addr[i, k])
                if cid in cand_map:
                    cand_map[cid]["sim_addr"] = score
                    cand_map[cid]["route_addr"] = 1
                else:
                    cand_map[cid] = {
                        "sim_name": 0.0,
                        "sim_addr": score,
                        "route_name": 0,
                        "route_addr": 1
                    }

            # Build record tuples
            for cid, info in cand_map.items():
                rows.append((
                    sid,
                    cid,
                    info["sim_name"],
                    info["sim_addr"],
                    info["route_name"],
                    info["route_addr"]
                ))

        df_candidates = pd.DataFrame(
            rows,
            columns=["s1_id", "candidate_id", "sim_name", "sim_addr", "route_name", "route_addr"]
        )
        return df_candidates


def run_gate1_audit(
    df_candidates: pd.DataFrame,
    ground_truth: Dict[str, Set[str]],
    total_s23_count: Optional[int] = None
) -> Dict[str, float]:
    """
    Run Gate 1 verification:
    Check Pair Completeness (PC) >= 0.95.
    """
    # Group candidates by s1_id
    cand_dict: Dict[str, Set[str]] = {}
    for sid, group in df_candidates.groupby("s1_id"):
        cand_dict[str(sid)] = set(group["candidate_id"].values)

    metrics = compute_blocking_metrics(ground_truth, cand_dict, total_s23_count=total_s23_count)
    pc = metrics["pair_completeness"]
    passed = pc >= Config.GATE1_PC_TARGET

    print("\n" + "=" * 50)
    print(f" GATE 1 AUDIT REPORT: {'PASSED [OK]' if passed else 'WARNING [BELOW TARGET]'}")
    print("=" * 50)
    print(f" Pair Completeness (PC): {pc:.4f} (Target: >= {Config.GATE1_PC_TARGET:.2f})")
    print(f" Covered True Pairs:     {metrics['covered_true_pairs']:,} / {metrics['total_true_pairs']:,}")
    print(f" Mean Candidates / Entity: {metrics['cand_count_mean']:.1f}")
    print(f" Median Candidates:        {metrics['cand_count_median']:.1f}")
    print(f" P95 Candidates:           {metrics['cand_count_p95']:.1f}")
    print(f" Max Candidates:           {metrics['cand_count_max']:,}")
    if "reduction_ratio" in metrics:
        print(f" Reduction Ratio (RR):   {metrics['reduction_ratio']:.6f}")
    print("=" * 50 + "\n")

    metrics["gate1_passed"] = float(passed)
    return metrics
