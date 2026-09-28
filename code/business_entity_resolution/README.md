# Amazon ML Challenge 2026: Business Entity Resolution
## Reproduction Guide (Track C + Shared Pipeline)

This package contains the complete, reproducible pipeline for Business Entity Resolution under the official rules and constraints:
- License-compliant models: `intfloat/multilingual-e5-small` (118M parameters, MIT License, <=8B parameters).
- Zero external data lookups, zero internet lookups at inference.
- Evaluated on official Macro F0.5 with explicit singleton handling.

---

### 1. Environment Setup

Install pinned dependencies:
```bash
pip install -r requirements.txt
```

---

### 2. Execution Pipeline

The pipeline can be executed in two modes:

#### Option A: Modular Kaggle Notebooks (Recommended for GPU Scale)
Under `notebooks/`:
1. `00_stage0_normalization.ipynb` (CPU): Normalizes text, transliterates Devanagari, parses addresses, outputs parquets.
2. `01_stage1_dual_channel_embeddings.ipynb` (GPU 2x T4): Encodes dual independent embeddings (`name_core` and `addr_core`) with FP16 autocast.
3. `02_stage1_faiss_retrieval_gate1.ipynb` (CPU): Builds FAISS dual indexes, retrieves top-20 per route, audits Gate 1 (PC >= 0.95).
4. `03_stage2_pairwise_scorer_lightgbm.ipynb` (CPU): Extracts pairwise features, trains 5-fold grouped LightGBM, generates `score_C.parquet`.
5. `04_gated_cross_encoder_and_clustering.ipynb` (GPU/CPU): Stage 4 cluster pre-check and Stage 3 selective ambiguous band reranking.
6. `05_stage5_decision_and_submission.ipynb` (CPU): Optimizes two thresholds for Macro F0.5, generates `matching_results.tsv` and `candidate_pairs.tsv`, runs validator.

#### Option B: Unified Command-Line Runner
Run end-to-end:
```bash
python -m src.pipeline --stage all
```
Or execute stage by stage:
```bash
python -m src.pipeline --stage 0  # Normalization
python -m src.pipeline --stage 1  # Dual Embedding Retrieval & Gate 1
python -m src.pipeline --stage 2  # Feature Engineering & LightGBM Scorer
python -m src.pipeline --stage 4  # Cluster Pre-Check
python -m src.pipeline --stage 5  # Threshold Tuning & TSV Export
```

For quick verification on a slice:
```bash
python -m src.pipeline --stage all --sample 1000
```

---

### 3. Submission Validation

Before submitting to the portal, run the official validation script:
```bash
python utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```
Exit code 0 (`PASS`) guarantees zero formatting rejections.
