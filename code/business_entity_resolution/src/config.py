"""
config.py — Central Configuration & Path Resolution
Amazon ML Challenge 2026: Business Entity Resolution

Automatically detects whether execution is on Kaggle or a local workstation,
and manages paths, constants, and hyperparameters across all stages.
"""

import os
from pathlib import Path


class Config:
    # --- ENVIRONMENT DETECTION ---
    IS_KAGGLE = os.path.exists("/kaggle") or "KAGGLE_KERNEL_RUN_TYPE" in os.environ

    # --- BASE PATHS ---
    if IS_KAGGLE:
        # In Kaggle notebooks:
        BASE_INPUT_DIR = Path("/kaggle/input")
        # Find competition / dataset input path
        # Typically /kaggle/input/amazon-ml-challenge-2026 or similar
        input_subdirs = list(BASE_INPUT_DIR.glob("*"))
        if input_subdirs:
            DATA_DIR = input_subdirs[0]
            # Check if student_resource or dataset is inside
            if (DATA_DIR / "dataset").exists():
                DATA_DIR = DATA_DIR / "dataset"
            elif (DATA_DIR / "student_resource" / "dataset").exists():
                DATA_DIR = DATA_DIR / "student_resource" / "dataset"
        else:
            DATA_DIR = Path("/kaggle/input/dataset")

        WORK_DIR = Path("/kaggle/working")
        ARTIFACTS_DIR = WORK_DIR / "artifacts"
        OUTPUT_DIR = WORK_DIR / "output"
    else:
        # Local workspace
        ROOT_DIR = Path("c:/projects/MLchallenge")
        DATA_DIR = ROOT_DIR / "data" / "student_resource" / "dataset"
        WORK_DIR = ROOT_DIR / "work"
        ARTIFACTS_DIR = WORK_DIR / "artifacts"
        OUTPUT_DIR = ROOT_DIR / "output"

    # Raw dataset paths
    TRAIN_SOURCE1 = DATA_DIR / "train" / "train_source1.tsv"
    TRAIN_SOURCE2 = DATA_DIR / "train" / "train_source2.tsv"
    TRAIN_SOURCE3 = DATA_DIR / "train" / "train_source3.tsv"
    TRAIN_GROUND_TRUTH = DATA_DIR / "train" / "train_ground_truth.tsv"

    TEST_SOURCE1 = DATA_DIR / "test" / "test_source1.tsv"
    TEST_SOURCE2 = DATA_DIR / "test" / "test_source2.tsv"
    TEST_SOURCE3 = DATA_DIR / "test" / "test_source3.tsv"

    # Preprocessed / Normalized Parquet Paths
    PARQUET_DIR = WORK_DIR / "normalized"
    TRAIN_S1_NORM = PARQUET_DIR / "train_s1_norm.parquet"
    TRAIN_S2_NORM = PARQUET_DIR / "train_s2_norm.parquet"
    TRAIN_S3_NORM = PARQUET_DIR / "train_s3_norm.parquet"

    TEST_S1_NORM = PARQUET_DIR / "test_s1_norm.parquet"
    TEST_S2_NORM = PARQUET_DIR / "test_s2_norm.parquet"
    TEST_S3_NORM = PARQUET_DIR / "test_s3_norm.parquet"

    # Embedding Paths & Parameters
    EMB_DIR = WORK_DIR / "embeddings"
    # Selected encoder: intfloat/multilingual-e5-small (118M params, MIT License, <=8B params, 384 dims)
    ENCODER_MODEL_NAME = "intfloat/multilingual-e5-small"
    EMBEDDING_DIM = 384
    EMB_BATCH_SIZE = 512
    EMB_MAX_SEQ_LENGTH = 64

    # Retrieval / FAISS Parameters (Stage 1)
    RETRIEVAL_TOP_K = 20  # top-20 nearest neighbors per route
    GATE1_PC_TARGET = 0.95  # Pair completeness gate

    # Pairwise Scorer / LightGBM (Stage 2)
    LGBM_PARAMS = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 30,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "n_estimators": 1000,
        "random_state": 42,
        "n_jobs": -1,
        "verbose": -1,
    }

    # Selective Cross-Encoder (Stage 3)
    CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    AMBIGUOUS_MARGIN = 0.10  # [t_opt - margin, t_opt + margin]

    # Decision Layer (Stage 5)
    DEFAULT_DECISION_THRESHOLD = 0.75
    DEFAULT_FIRST_THRESHOLD = 0.80
    DEFAULT_REST_THRESHOLD = 0.65
    SINGLETON_CUTOFF = 0.70

    # Ensure required working directories exist
    @classmethod
    def setup_directories(cls):
        for d in [cls.WORK_DIR, cls.ARTIFACTS_DIR, cls.OUTPUT_DIR, cls.PARQUET_DIR, cls.EMB_DIR]:
            d.mkdir(parents=True, exist_ok=True)


# Initialize directories
Config.setup_directories()
