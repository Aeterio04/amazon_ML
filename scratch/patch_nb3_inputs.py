import json
from pathlib import Path

nb_path = Path("notebooks/03_stage2_pairwise_scorer_lightgbm.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# ── Update Cell 1 ──
cell1_source = """!pip install -q rapidfuzz lightgbm

import os, sys, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from rapidfuzz import fuzz, distance

print("Libraries imported.")

# ============================================================
# HARDWARE CONFIGURATION
# ============================================================

USE_GPU = False
device_param = 'cpu'

try:
    import torch
    if torch.cuda.is_available():
        USE_GPU = True
        device_param = 'gpu'
        gpu_name = torch.cuda.get_device_name(0)
        n_gpus = torch.cuda.device_count()
except Exception:
    pass

if USE_GPU:
    print(f"🚀 GPU Acceleration: ENABLED ({n_gpus} GPU(s) available: {gpu_name})")
else:
    print("ℹ️ GPU Acceleration: OFF (Running on multithreaded CPU)")

# ============================================================
# PATH RESOLUTION (robust: handles Kaggle directory flattening)
# ============================================================

IS_KAGGLE = os.path.exists('/kaggle')

def find_dir_by_file(pattern):
    for p in Path('/kaggle/input').rglob(pattern):
        return p.parent
    return None

def find_norm_dir():
    # Method 1: rglob search for train_s1_norm.parquet
    found = find_dir_by_file('train_s1_norm.parquet')
    if found:
        return found, 'parquet'
    
    # Method 2: check known Kaggle notebook output paths (including ml1nbv2 & ml1nbv1)
    known_paths = [
        Path('/kaggle/input/ml1nbv2/normalized'),
        Path('/kaggle/input/ml1nbv2'),
        Path('/kaggle/input/notebooks/ojassangwai/ml1nbv2/normalized'),
        Path('/kaggle/input/notebooks/ojassangwai/ml1nbv2'),
        Path('/kaggle/input/ml1nbv1/normalized'),
        Path('/kaggle/input/ml1nbv1'),
        Path('/kaggle/input/notebooks/ojassangwai/ml1nbv1/normalized'),
        Path('/kaggle/input/notebooks/ojassangwai/ml1nbv1'),
    ]
    for p in known_paths:
        if (p / 'train_s1_norm.parquet').exists():
            return p, 'parquet'
    
    # Method 3: fallback to raw TSV files from the competition dataset
    data_dir = find_dir_by_file('train_ground_truth.tsv')
    if data_dir and (data_dir / 'train' / 'train_source1.tsv').exists():
        print("  ⚠️ Normalized parquets not found. Will normalize from raw TSVs (slower but works).")
        return data_dir, 'raw_tsv'
    
    # Diagnostic
    print("\\n❌ ERROR: Could not find normalized parquets (train_s1_norm.parquet).")
    print("  Listing /kaggle/input data files for debugging:")
    for p in sorted(Path('/kaggle/input').rglob('*')):
        if p.is_file() and p.suffix in ('.parquet', '.tsv', '.csv', '.npy'):
            print(f"    {p}")
    return None, None

def find_candidates_file():
    # If multiple candidates_train.parquet exist (e.g. from ml2nbv1 and ml2nbv2),
    # prioritize ml2nbv2 over ml2nbv1 and ensure non-empty.
    cands = [p for p in Path('/kaggle/input').rglob('candidates_train.parquet') if p.is_file()]
    if not cands:
        return None
    
    def sort_key(p):
        path_str = str(p).lower()
        # Prefer v2 over v1
        v_score = 2 if 'v2' in path_str else (1 if 'v1' in path_str else 0)
        try:
            sz = p.stat().st_size
            mt = p.stat().st_mtime
        except Exception:
            sz, mt = 0, 0
        return (v_score, sz, mt)
    
    cands_sorted = sorted(cands, key=sort_key, reverse=True)
    best = cands_sorted[0]
    print(f"  Found {len(cands)} candidates_train.parquet candidate(s):")
    for cp in cands_sorted:
        tag = "[SELECTED]" if cp == best else "[IGNORED] "
        sz_mb = cp.stat().st_size / (1024 * 1024)
        print(f"    {tag} {cp} ({sz_mb:.1f} MB)")
    return best

if IS_KAGGLE:
    WORK_DIR = Path('/kaggle/working')
    
    # 1. candidates_train.parquet from Stage 1 (ml2nbv2 / ml2nbv1)
    CAND_FILE = find_candidates_file()
    if CAND_FILE is not None:
        CAND_DIR = CAND_FILE.parent
    else:
        CAND_DIR = WORK_DIR
        CAND_FILE = CAND_DIR / 'candidates_train.parquet'
    
    # 2. Normalized parquets from Stage 0 (ml1nbv2 / ml1nbv1)
    NORM_DIR, NORM_FORMAT = find_norm_dir()
    assert NORM_DIR is not None, (
        "Could not find normalized parquets!\\n"
        "FIX: In the Kaggle notebook sidebar, click '+ Add Input' and attach\\n"
        "     the output of your Stage 0 normalization notebook (ml1nbv2 or ml1nbv1).\\n"
        "     It should contain: train_s1_norm.parquet, train_s2_norm.parquet, train_s3_norm.parquet"
    )
    
    # 3. Ground truth — exhaustive search
    GT_FILE = None
    # Method 1: rglob under all of /kaggle/input
    for p in Path('/kaggle/input').rglob('train_ground_truth.tsv'):
        GT_FILE = p
        break
    # Method 2: check known nested structures
    if GT_FILE is None:
        gt_candidates = [
            Path('/kaggle/input/mlchallengedata/student_resource/dataset/train/train_ground_truth.tsv'),
            Path('/kaggle/input/mlchallengedata/dataset/train/train_ground_truth.tsv'),
            Path('/kaggle/input/mlchallengedata/train/train_ground_truth.tsv'),
            Path('/kaggle/input/mlchallengedata/train_ground_truth.tsv'),
            Path('/kaggle/input/datasets/ojassangwai/mlchallengedata/student_resource/dataset/train/train_ground_truth.tsv'),
            Path('/kaggle/input/datasets/ojassangwai/mlchallengedata/dataset/train/train_ground_truth.tsv'),
            Path('/kaggle/input/datasets/ojassangwai/mlchallengedata/train/train_ground_truth.tsv'),
            Path('/kaggle/input/datasets/ojassangwai/mlchallengedata/train_ground_truth.tsv'),
        ]
        for gp in gt_candidates:
            if gp.exists():
                GT_FILE = gp
                break
    # Diagnostics if still not found
    if GT_FILE is None:
        print('\\n❌ Ground truth NOT FOUND! Listing all .tsv files under /kaggle/input:')
        for p in sorted(Path('/kaggle/input').rglob('*.tsv')):
            print(f'    {p}')
        raise FileNotFoundError(
            'train_ground_truth.tsv not found anywhere under /kaggle/input.\\n'
            'FIX: Attach the competition dataset (mlchallengedata) via + Add Input.'
        )
    DATA_DIR = GT_FILE.parent
else:
    WORK_DIR = Path('work')
    CAND_DIR = WORK_DIR
    CAND_FILE = CAND_DIR / 'candidates_train.parquet'
    NORM_DIR = WORK_DIR / 'normalized'
    NORM_FORMAT = 'parquet'
    DATA_DIR = Path('data/student_resource/dataset')
    GT_FILE = DATA_DIR / 'train' / 'train_ground_truth.tsv'

OUTPUT_DIR = WORK_DIR / 'output'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Candidates File: {CAND_FILE}")
print(f"Norm Dir:        {NORM_DIR} (format: {NORM_FORMAT})")
print(f"Data Dir:        {DATA_DIR}")
print(f"GT File:         {GT_FILE}")
print(f"Work Dir:        {WORK_DIR}")
"""

# Replace cell 1 source
nb['cells'][1]['source'] = [line + '\n' for line in cell1_source.split('\n')]
# Ensure last line doesn't have double newline
if nb['cells'][1]['source'] and nb['cells'][1]['source'][-1] == '\n':
    nb['cells'][1]['source'] = nb['cells'][1]['source'][:-1]

# ── Update Cell 2: use CAND_FILE directly ──
cell2_src = nb['cells'][2]['source']
for i, line in enumerate(cell2_src):
    if 'cand_file =' in line:
        cell2_src[i] = "cand_file = CAND_FILE\n"
    elif 'assert cand_file.exists()' in line:
        cell2_src[i] = "assert cand_file is not None and cand_file.exists(), f\"ERROR: {cand_file} not found! Check that ml2nbv2 is attached and finished successfully.\"\n"
    elif 'train_ground_truth.tsv' in line and 'read_csv' in line:
        cell2_src[i] = "df_gt = pd.read_csv(GT_FILE, sep='\\t', dtype=str)\n"

nb['cells'][2]['source'] = cell2_src

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("[SUCCESS] Updated notebook 03 successfully!")
