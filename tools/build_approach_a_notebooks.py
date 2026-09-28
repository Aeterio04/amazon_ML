"""
build_approach_a_notebooks.py
Generates hyper-optimized, production-ready Jupyter notebooks for Approach A:
1. notebooks/approach_a_01_prep_target_index.ipynb (Prepares S23 target index, vectorizers & sparse matrices)
2. notebooks/approach_a_02a_sharded_test_inference.ipynb (Worker Part 1: S1 rows 0 to 866,272)
3. notebooks/approach_a_02b_sharded_test_inference.ipynb (Worker Part 2: S1 rows 866,272 to 1,732,544)
4. notebooks/approach_a_03_assemble_and_validate.ipynb (Merges 02a + 02b predictions & validates deliverables)
"""

import json
from pathlib import Path


def create_ipynb(cells, output_path: Path):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2, ensure_ascii=False)
    print(f"Created notebook: {output_path}")


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")]
    }


def code_cell(code):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in code.strip().split("\n")]
    }


def build_nb1_prep_target():
    cells = [
        md_cell("""# Approach A — Notebook 1: Target Prep & Pre-Computed Index
**Amazon ML Challenge 2026: Business Entity Resolution**

### Purpose:
1. Ingest 10.3M S2 + S3 target records (either raw TSVs or precomputed parquets).
2. Fit the 3 character $n$-gram TF-IDF vectorizers (`name_core`, `addr_norm`, `combined`).
3. Pre-transform and partition S23 sparse CSR matrices by country bucket (`US`, `India`, `France`, `__unknown__`).
4. Save the target bundle into `/kaggle/working/target_index/`.

**Output Artifacts:**
- `target_metadata.pkl`: S23 entity IDs, countries, raw text columns, fitted vectorizers.
- `target_sparse_matrices.pkl`: Pre-transformed sparse CSR matrices per country bucket.

*Publish `/kaggle/working/target_index/` as a Kaggle Dataset named `approach-a-target-index` for Notebooks 2A and 2B.*
"""),
        code_cell("""# 0. Dependencies Setup
!pip install sparse_dot_topn rapidfuzz lightgbm pyarrow --quiet
"""),
        code_cell("""# 1. Environment & Thread Budget Setup
import os, sys, time, glob, re, gc, pickle, unicodedata, warnings
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize

warnings.filterwarnings("ignore")

N_CORES = os.cpu_count() or 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(N_CORES))
os.environ.pop("GOMP_CPU_AFFINITY", None)
os.environ.pop("KMP_AFFINITY", None)

print(f"[Config] Detected CPU cores: {N_CORES}")
"""),
        code_cell("""# 2. Path Resolution (Auto-detects Kaggle vs Local)
def find_file(filename, search_root="/kaggle/input"):
    hits = glob.glob(os.path.join(search_root, "**", filename), recursive=True)
    return hits[0] if hits else None

if os.path.isdir("/kaggle/input"):
    TEST_DIR = os.path.dirname(find_file("test_source1.tsv") or "/kaggle/input/dataset/test/test_source1.tsv")
    WORK_DIR = "/kaggle/working"
else:
    TEST_DIR = os.path.abspath("data/student_resource/dataset/test")
    WORK_DIR = os.path.abspath("output/approach_a")

INDEX_DIR = os.path.join(WORK_DIR, "target_index")
os.makedirs(INDEX_DIR, exist_ok=True)
print("TEST_DIR:", TEST_DIR)
print("INDEX_DIR:", INDEX_DIR)
"""),
        code_cell("""# 3. Fast Text Normalization
LEGAL_FORMS = frozenset({
    "inc","incorporated","corp","corporation","co","company","ltd","limited",
    "llc","llp","plc","pvt","private","gmbh","ag","sa","sarl","srl","bv","nv",
    "dba","pte","pty","group","holdings","enterprises","ventures",
})

STREET_ABBR = {
    "rd":"road","st":"street","ave":"avenue","blvd":"boulevard","dr":"drive",
    "ln":"lane","ct":"court","pl":"place","cir":"circle","pkwy":"parkway",
    "hwy":"highway","trl":"trail","sq":"square","ter":"terrace","xing":"crossing",
    "nr":"near","opp":"opposite","flr":"floor","apt":"apartment",
}

US_STATES = {
    "al":"alabama","ak":"alaska","az":"arizona","ar":"arkansas","ca":"california",
    "co":"colorado","ct":"connecticut","de":"delaware","fl":"florida","ga":"georgia",
    "hi":"hawaii","id":"idaho","il":"illinois","in":"indiana","ia":"iowa",
    "ks":"kansas","ky":"kentucky","la":"louisiana","me":"maine","md":"maryland",
    "ma":"massachusetts","mi":"michigan","mn":"minnesota","ms":"mississippi",
    "mo":"missouri","mt":"montana","ne":"nebraska","nv":"nevada","nh":"new hampshire",
    "nj":"new jersey","nm":"new mexico","ny":"new york","nc":"north carolina",
    "nd":"north dakota","oh":"ohio","ok":"oklahoma","or":"oregon","pa":"pennsylvania",
    "ri":"rhode island","sc":"south carolina","sd":"south dakota","tn":"tennessee",
    "tx":"texas","ut":"utah","vt":"vermont","va":"virginia","wa":"washington",
    "wv":"west virginia","wi":"wisconsin","wy":"wyoming","dc":"district of columbia",
}

DEV_MAP = {
    'अ':'a','आ':'aa','इ':'i','ई':'ii','उ':'u','ऊ':'uu','ए':'e','ऐ':'ai','ओ':'o','औ':'au','ऋ':'ri',
    'क':'k','ख':'kh','ग':'g','घ':'gh','ङ':'ng','च':'ch','छ':'chh','ज':'j','झ':'jh','ञ':'ny',
    'ट':'t','ठ':'th','ड':'d','ढ':'dh','ण':'n','त':'t','थ':'th','द':'d','ध':'dh','न':'n',
    'प':'p','फ':'ph','ब':'b','भ':'bh','म':'m','य':'y','र':'r','ल':'l','व':'v','श':'sh','ष':'sh',
    'स':'s','ह':'h','ा':'aa','ि':'i','ी':'ii','ु':'u','ू':'uu','े':'e','ै':'ai','ो':'o','ौ':'au',
    'ं':'n','ः':'h','्':'','ृ':'ri','़':'','ँ':'n','ॉ':'o',
}

_punct_re = re.compile(r"[^a-z0-9\\s]")
_ws_re    = re.compile(r"\\s+")

def _translit(text):
    return "".join(DEV_MAP.get(ch, ch) for ch in text)

def _has_nonlatin(text):
    return any(ord(ch) > 127 and unicodedata.category(ch).startswith("L") for ch in text)

def _base(text):
    if not text or pd.isna(text): return ""
    text = str(text)
    if _has_nonlatin(text): text = _translit(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()

def norm_name(text):
    t = _base(text)
    if not t: return "", "", ""
    t = t.replace("&", " and ")
    t = _punct_re.sub(" ", t)
    t = _ws_re.sub(" ", t).strip()
    name_full = t
    parts = re.split(r"\\bdba\\b", t)
    primary = parts[0].strip() if parts else t
    trade = parts[1].strip() if len(parts) > 1 else ""
    core = [tok for tok in primary.split() if tok not in LEGAL_FORMS]
    name_core = " ".join(core) if core else primary
    trade_core = " ".join(tok for tok in trade.split() if tok not in LEGAL_FORMS)
    return name_full, name_core, trade_core

def norm_addr(text):
    t = _base(text)
    if not t: return ""
    t = t.replace("&", " and ")
    t = re.sub(r"#+(\\d)", r"\\1", t)
    t = _punct_re.sub(" ", t)
    t = _ws_re.sub(" ", t).strip()
    out = [STREET_ABBR.get(tok, US_STATES.get(tok, tok)) for tok in t.split()]
    return " ".join(out)
"""),
        code_cell("""# 4. Load & Ingest Target S2 + S3
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

parquet_hit = find_file("s23_norm.parquet") or find_file("s23_target.parquet")

if parquet_hit:
    print(f"Loading precomputed S23 target parquet: {parquet_hit}")
    s23 = pd.read_parquet(parquet_hit)
else:
    def _norm_chunk(texts, is_name=True):
        return [norm_name(t) for t in texts] if is_name else [norm_addr(t) for t in texts]

    def parallel_norm(series, is_name=True, n_workers=N_CORES):
        vals = series.tolist()
        n = len(vals)
        chunk_size = max(1, -(-n // n_workers))
        chunks = [vals[i:i + chunk_size] for i in range(0, n, chunk_size)]
        results = [None] * len(chunks)
        ctx = mp.get_context("fork") if "fork" in mp.get_all_start_methods() else mp.get_context()
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
            futs = {ex.submit(_norm_chunk, c, is_name): i for i, c in enumerate(chunks)}
            for fut in as_completed(futs):
                results[futs[fut]] = fut.result()
        out = []
        for r in results: out.extend(r)
        return out

    def load_source_tsv(path):
        print(f"Reading {path} with pyarrow...")
        t0 = time.time()
        cols = ["entity_id", "business_name", "business_address", "country"]
        try:
            df = pd.read_csv(path, sep="\\t", dtype=str, usecols=cols, engine="pyarrow").fillna("")
        except Exception:
            df = pd.read_csv(path, sep="\\t", dtype=str, usecols=cols).fillna("")
        print(f"  Read {len(df):,} rows in {time.time()-t0:.1f}s. Normalizing...")
        t_n = time.time()
        name_res = parallel_norm(df["business_name"], is_name=True)
        df["name_full"]  = [r[0] for r in name_res]
        df["name_core"]  = [r[1] for r in name_res]
        df["name_trade"] = [r[2] for r in name_res]
        del name_res
        df["addr_norm"] = parallel_norm(df["business_address"], is_name=False)
        df["combined"] = df["name_core"] + " " + df["addr_norm"]
        df["country_norm"] = df["country"].map(lambda x: x.strip().lower() if x.strip() else "__unknown__")
        print(f"  Normalized {len(df):,} rows in {time.time()-t_n:.1f}s")
        return df[["entity_id","name_full","name_core","name_trade","addr_norm","combined","country","country_norm"]]

    s2 = load_source_tsv(os.path.join(TEST_DIR, "test_source2.tsv"))
    s3 = load_source_tsv(os.path.join(TEST_DIR, "test_source3.tsv"))
    s23 = pd.concat([s2, s3], ignore_index=True)
    del s2, s3; gc.collect()

print(f"Total S23 target pool: {len(s23):,} entities")
"""),
        code_cell("""# 5. Fit 3 Vectorizers & Pre-Transform Country Sparse Matrices
print("Fitting TF-IDF Vectorizers...")
t0 = time.time()

vec_name = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=200_000, sublinear_tf=True, dtype=np.float32)
vec_name.fit(s23["name_core"])

vec_addr = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=150_000, sublinear_tf=True, dtype=np.float32)
vec_addr.fit(s23["addr_norm"])

vec_comb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=200_000, sublinear_tf=True, dtype=np.float32)
vec_comb.fit(s23["combined"])
print(f"Fitted all 3 vectorizers in {time.time()-t0:.1f}s")

# Pre-transform sparse matrices per country bucket
print("Pre-transforming sparse country matrices for instant worker loading...")
t_sparse = time.time()

countries = sorted(set(s23["country_norm"].unique()))
print(f"Unique countries in S23: {countries}")

unknown_idx = np.where(s23["country_norm"].values == "__unknown__")[0].astype(np.int32)
sparse_country_dict = {}

for route_name, col, vec in [("name", "name_core", vec_name),
                             ("addr", "addr_norm", vec_addr),
                             ("comb", "combined", vec_comb)]:
    col_vals = s23[col].values
    unknown_mat = sk_normalize(vec.transform(col_vals[unknown_idx])) if len(unknown_idx) else None

    sparse_country_dict[route_name] = {"unknown_idx": unknown_idx, "buckets": {}}
    for c in countries:
        if c == "__unknown__":
            sparse_country_dict[route_name]["buckets"][c] = {
                "idx": np.arange(len(s23), dtype=np.int32),
                "mat_T": sk_normalize(vec.transform(col_vals)).T.tocsr()
            }
        else:
            exact_idx = np.where(s23["country_norm"].values == c)[0].astype(np.int32)
            if len(exact_idx) == 0 and len(unknown_idx) == 0: continue
            exact_mat = sk_normalize(vec.transform(col_vals[exact_idx])) if len(exact_idx) else None
            if exact_mat is not None and unknown_mat is not None:
                full_mat = sparse.vstack([exact_mat, unknown_mat], format="csr")
                full_idx = np.concatenate([exact_idx, unknown_idx])
                order = np.argsort(full_idx, kind="stable")
                full_idx = full_idx[order]
                full_mat = full_mat[order]
            elif exact_mat is not None:
                full_mat, full_idx = exact_mat, exact_idx
            else:
                full_mat, full_idx = unknown_mat, unknown_idx

            sparse_country_dict[route_name]["buckets"][c] = {
                "idx": full_idx,
                "mat_T": full_mat.T.tocsr()
            }

print(f"Pre-transformed country matrices in {time.time()-t_sparse:.1f}s")
"""),
        code_cell("""# 6. Save Artifacts for Worker Notebooks
meta = {
    "entity_ids": s23["entity_id"].values,
    "country_norm": s23["country_norm"].values,
    "country_raw": s23["country"].values,
    "name_core": s23["name_core"].values,
    "name_full": s23["name_full"].values,
    "name_trade": s23["name_trade"].values,
    "addr_norm": s23["addr_norm"].values,
    "combined": s23["combined"].values,
    "vec_name": vec_name,
    "vec_addr": vec_addr,
    "vec_comb": vec_comb,
}

meta_path = os.path.join(INDEX_DIR, "target_metadata.pkl")
matrices_path = os.path.join(INDEX_DIR, "target_sparse_matrices.pkl")

print(f"Saving {meta_path}...")
with open(meta_path, "wb") as f:
    pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saving {matrices_path}...")
with open(matrices_path, "wb") as f:
    pickle.dump(sparse_country_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

print("="*60)
print(f"STAGE 1 COMPLETE: Target index artifacts successfully saved in {INDEX_DIR}")
print("Ready for Notebooks 2A and 2B!")
print("="*60)
""")
    ]
    return cells


def build_nb2_worker(part_label: str, start_ratio: float, end_ratio: float):
    part_upper = part_label.upper()
    start_pct = int(start_ratio * 100)
    end_pct = int(end_ratio * 100)
    start_approx = int(1_732_544 * start_ratio)
    end_approx = int(1_732_544 * end_ratio)

    cells = [
        md_cell(f"""# Approach A — Notebook {part_upper}: Test Inference Worker ({part_upper})
**Amazon ML Challenge 2026: Business Entity Resolution**

### Partition Scope:
- **Partition:** {part_upper} ({start_pct}% to {end_pct}% of Test $S_1$)
- **Query Range:** Rows {start_approx:,} to {end_approx:,} (~{end_approx - start_approx:,} entities)
- **Output File:** `part_{part_label}_results.pkl`

**Inputs Required:**
1. Competition Test Dataset (`test_source1.tsv`)
2. Model parameters (`iso_thr.pkl` and `lgbm_fold0/1/2.txt`)
3. Pre-computed Target Index from Notebook 1 (`target_metadata.pkl` & `target_sparse_matrices.pkl`)
"""),
        code_cell("""# 0. Dependencies Setup
!pip install sparse_dot_topn rapidfuzz lightgbm pyarrow --quiet
"""),
        code_cell(f"""# 1. Partition Configuration
PART_LABEL      = "{part_label}"
START_RATIO     = {start_ratio}
END_RATIO       = {end_ratio}
SUB_BATCH_SIZE  = 220000   # Internal batching to keep peak RAM strictly under 6 GB

BLOCK_TOP_K        = 50
BLOCK_MIN_SIM       = 0.12
SPARSE_CHUNK_SIZE   = 12000

import os, sys, time, glob, re, gc, pickle, unicodedata, warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

N_CORES = os.cpu_count() or 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(N_CORES))
os.environ.pop("GOMP_CPU_AFFINITY", None)
os.environ.pop("KMP_AFFINITY", None)

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sparse_dot_topn import sp_matmul_topn
import lightgbm as lgb
from rapidfuzz import fuzz, process as rf_process
from rapidfuzz.distance import Levenshtein as Lev

warnings.filterwarnings("ignore")
print(f"[Worker {part_upper}] Processing partition {{START_RATIO:.1%}} to {{END_RATIO:.1%}} on {{N_CORES}} CPU cores")
"""),
        code_cell("""# 2. Path & Model Artifact Discovery
def find_file(filename, search_root="/kaggle/input"):
    hits = glob.glob(os.path.join(search_root, "**", filename), recursive=True)
    return hits[0] if hits else None

if os.path.isdir("/kaggle/input"):
    TEST_DIR   = os.path.dirname(find_file("test_source1.tsv") or "/kaggle/input/dataset/test/test_source1.tsv")
    MODEL_DIR  = os.path.dirname(find_file("iso_thr.pkl") or "/kaggle/input/approach_a_models/iso_thr.pkl")
    INDEX_DIR  = os.path.dirname(find_file("target_metadata.pkl") or "/kaggle/working/target_index/target_metadata.pkl")
    WORK_DIR   = "/kaggle/working"
else:
    TEST_DIR   = os.path.abspath("data/student_resource/dataset/test")
    MODEL_DIR  = os.path.abspath("models")
    INDEX_DIR  = os.path.abspath("output/approach_a/target_index")
    WORK_DIR   = os.path.abspath("output/approach_a")

# Load trained models & calibration
with open(os.path.join(MODEL_DIR, "iso_thr.pkl"), "rb") as f:
    saved = pickle.load(f)
iso, t_first, t_rest = saved["iso"], saved["t_first"], saved["t_rest"]
feature_names = saved["feature_names"]

fold_files = sorted(glob.glob(os.path.join(MODEL_DIR, "lgbm_fold*.txt")))
models = [lgb.Booster(model_file=f) for f in fold_files]
print(f"Loaded {len(models)} LightGBM folds. Thresholds: t_first={t_first:.4f}, t_rest={t_rest:.4f}")

# Load target index
print(f"Loading target metadata from {INDEX_DIR}...")
with open(os.path.join(INDEX_DIR, "target_metadata.pkl"), "rb") as f:
    target_meta = pickle.load(f)

# Safe attribute extraction (compatible with any version of Notebook 1)
target_countries = target_meta.get("country_raw", target_meta.get("country_norm", target_meta.get("countries")))
target_combined  = target_meta.get("combined", target_meta["name_core"] + " " + target_meta["addr_norm"])
target_trade     = target_meta.get("name_trade", np.array([""] * len(target_meta["entity_ids"])))

matrices_file = find_file("target_sparse_matrices.pkl")
if matrices_file and os.path.exists(matrices_file):
    print(f"Loading pre-transformed sparse country matrices from {matrices_file}...")
    with open(matrices_file, "rb") as f:
        target_sparse_dict = pickle.load(f)
else:
    print("target_sparse_matrices.pkl not found in dataset — building country sparse matrices on the fly from target_meta (~90s)...")
    t_sp = time.time()
    c_norm = target_meta.get("country_norm", target_meta.get("countries"))
    countries = sorted(set(c_norm))
    unknown_idx = np.where(c_norm == "__unknown__")[0].astype(np.int32)
    n_s23 = len(target_meta["entity_ids"])

    target_sparse_dict = {}
    for r_key, text_arr, vec in [("name", target_meta["name_core"], target_meta["vec_name"]),
                                 ("addr", target_meta["addr_norm"], target_meta["vec_addr"]),
                                 ("comb", target_combined, target_meta["vec_comb"])]:
        print(f"  Transforming {r_key} target corpus ({len(text_arr):,} rows)...", flush=True)
        t_r = time.time()
        full_all_mat = sk_normalize(vec.transform(text_arr))
        print(f"    Transformed {r_key} in {time.time()-t_r:.1f}s. Slicing country buckets...", flush=True)

        unknown_mat = full_all_mat[unknown_idx] if len(unknown_idx) else None
        target_sparse_dict[r_key] = {"unknown_idx": unknown_idx, "buckets": {}}
        for c in countries:
            if c == "__unknown__":
                target_sparse_dict[r_key]["buckets"][c] = {
                    "idx": np.arange(n_s23, dtype=np.int32),
                    "mat_T": full_all_mat.T.tocsr()
                }
            else:
                exact_idx = np.where(c_norm == c)[0].astype(np.int32)
                if len(exact_idx) == 0 and len(unknown_idx) == 0: continue
                exact_mat = full_all_mat[exact_idx] if len(exact_idx) else None
                if exact_mat is not None and unknown_mat is not None:
                    full_mat = sparse.vstack([exact_mat, unknown_mat], format="csr")
                    full_idx = np.concatenate([exact_idx, unknown_idx])
                    order = np.argsort(full_idx, kind="stable")
                    full_mat = full_mat[order]
                    full_idx = full_idx[order]
                elif exact_mat is not None:
                    full_mat, full_idx = exact_mat, exact_idx
                else:
                    full_mat, full_idx = unknown_mat, unknown_idx

                target_sparse_dict[r_key]["buckets"][c] = {
                    "idx": full_idx,
                    "mat_T": full_mat.T.tocsr()
                }
    print(f"Built sparse country matrices on the fly in {time.time()-t_sp:.1f}s", flush=True)

print(f"Loaded target index with {len(target_meta['entity_ids']):,} records", flush=True)
"""),
        code_cell("""# 3. Load & Slice S1 for This Partition
def norm_s1_row(text):
    if not text or pd.isna(text): return "", "", ""
    t = str(text).lower().strip().replace("&", " and ")
    t = re.sub(r"[^a-z0-9\\s]", " ", t)
    t = re.sub(r"\\s+", " ", t).strip()
    parts = re.split(r"\\bdba\\b", t)
    p = parts[0].strip() if parts else t
    trade = parts[1].strip() if len(parts) > 1 else ""
    return t, p, trade

s1_full = pd.read_csv(os.path.join(TEST_DIR, "test_source1.tsv"), sep="\\t", dtype=str).fillna("")
total_s1 = len(s1_full)

part_start = int(round(total_s1 * START_RATIO))
part_end   = int(round(total_s1 * END_RATIO))
s1_part    = s1_full.iloc[part_start:part_end].reset_index(drop=True)

print(f"Partition {PART_LABEL}: rows {part_start:,} to {part_end:,} ({len(s1_part):,} entities out of {total_s1:,})")
"""),
        code_cell("""# 4. Core Pipeline Functions: Blocking, Features, Inference
def run_fast_country_blocking(s1_sub, route_key, vec, top_k, min_sim):
    min_sim_strict = float(np.nextafter(np.float32(min_sim), np.float32(-np.inf)))
    text_col = "name_core" if route_key == "name" else ("addr_norm" if route_key == "addr" else "combined")
    s1_texts = s1_sub[text_col].values
    s1_countries = s1_sub["country_norm"].values

    buckets_data = target_sparse_dict[route_key]["buckets"]
    out_s1, out_s23, out_sim = [], [], []

    for c, b_info in buckets_data.items():
        if c == "__unknown__":
            s1_idx = np.where(s1_countries == "__unknown__")[0].astype(np.int32)
        else:
            s1_idx = np.where(s1_countries == c)[0].astype(np.int32)
        if len(s1_idx) == 0: continue

        s23_idx_global = b_info["idx"]
        s23_mat_T = b_info["mat_T"]

        for ci in range(0, len(s1_idx), SPARSE_CHUNK_SIZE):
            chunk_g = s1_idx[ci:ci+SPARSE_CHUNK_SIZE]
            c_mat = sk_normalize(vec.transform(s1_texts[chunk_g]))
            C = sp_matmul_topn(c_mat, s23_mat_T, top_n=top_k, threshold=min_sim_strict, sort=True, n_threads=N_CORES).tocoo()
            if len(C.row):
                out_s1.append(chunk_g[C.row.astype(np.int32)])
                out_s23.append(s23_idx_global[C.col.astype(np.int32)])
                out_sim.append(C.data.astype(np.float32))

    if out_s1:
        return np.concatenate(out_s1), np.concatenate(out_s23), np.concatenate(out_sim)
    return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

def _binary_intersection(text1, text23, idx1, idx23, analyzer, token_pattern=None, ngram_range=(1, 1)):
    n = len(idx1)
    nonempty = pd.concat([pd.Series(text1)[pd.Series(text1) != ""], pd.Series(text23)[pd.Series(text23) != ""]], ignore_index=True)
    if nonempty.empty:
        z = np.zeros(n, dtype=np.float32)
        return z, z.copy(), z.copy()
    kwargs = dict(analyzer=analyzer, ngram_range=ngram_range, binary=True, dtype=np.float32)
    if token_pattern is not None: kwargs["token_pattern"] = token_pattern
    cv = CountVectorizer(**kwargs)
    cv.fit(nonempty)
    X1 = cv.transform(text1); X23 = cv.transform(text23)
    A, B = X1[idx1], X23[idx23]
    inter = np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)
    size1 = np.asarray(X1.sum(axis=1)).ravel()[idx1].astype(np.float32)
    size2 = np.asarray(X23.sum(axis=1)).ravel()[idx23].astype(np.float32)
    return inter, size1, size2

def _jac_ov(inter, size1, size2):
    union = size1 + size2 - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    minsz = np.minimum(size1, size2)
    ov = np.divide(inter, minsz, out=np.zeros_like(inter), where=minsz > 0)
    return jac.astype(np.float32), ov.astype(np.float32)
"""),
        code_cell(f"""# 5. Process S1 in Bounded Sub-Batches
total_part_rows = len(s1_part)
n_sub_batches = -(-total_part_rows // SUB_BATCH_SIZE)
print(f"Partition {{PART_LABEL}}: processing {{total_part_rows:,}} rows across {{n_sub_batches}} sub-batches...")

all_matches = {{}}
all_cands   = {{}}
t_part_start = time.time()

for b_idx in range(n_sub_batches):
    blo = b_idx * SUB_BATCH_SIZE
    bhi = min(total_part_rows, (b_idx + 1) * SUB_BATCH_SIZE)
    s1_sub = s1_part.iloc[blo:bhi].reset_index(drop=True)
    print(f"\\n--- [Sub-Batch {{b_idx+1}}/{{n_sub_batches}}] rows {{blo:,}} to {{bhi:,}} ({{len(s1_sub):,}} queries) ---")

    t_sub = time.time()
    names = [norm_s1_row(t) for t in s1_sub["business_name"]]
    s1_sub["name_full"]  = [x[0] for x in names]
    s1_sub["name_core"]  = [x[1] for x in names]
    s1_sub["name_trade"] = [x[2] for x in names]
    s1_sub["addr_norm"]  = s1_sub["business_address"].str.lower().str.replace(r"[^a-z0-9\\s]", " ", regex=True).str.strip()
    s1_sub["combined"]   = s1_sub["name_core"] + " " + s1_sub["addr_norm"]
    s1_sub["country_norm"] = s1_sub["country"].map(lambda x: x.strip().lower() if x.strip() else "__unknown__")

    # 1. Fast Blocking
    t_b = time.time()
    n_i, n_c, n_v = run_fast_country_blocking(s1_sub, "name", target_meta["vec_name"], BLOCK_TOP_K, BLOCK_MIN_SIM)
    a_i, a_c, a_v = run_fast_country_blocking(s1_sub, "addr", target_meta["vec_addr"], BLOCK_TOP_K // 2, BLOCK_MIN_SIM)
    c_i, c_c, c_v = run_fast_country_blocking(s1_sub, "comb", target_meta["vec_comb"], BLOCK_TOP_K // 2, BLOCK_MIN_SIM)

    n23 = len(target_meta["entity_ids"])
    def make_key(i1, i23): return i1.astype(np.int64) * (n23 + 1) + i23.astype(np.int64)

    df_n = pd.DataFrame({{"key": make_key(n_i, n_c), "nsim": n_v}})
    df_a = pd.DataFrame({{"key": make_key(a_i, a_c), "asim": a_v}})
    df_c = pd.DataFrame({{"key": make_key(c_i, c_c), "csim": c_v}})

    cand = df_n.merge(df_a, on="key", how="outer").merge(df_c, on="key", how="outer")
    for col in ["nsim", "asim", "csim"]: cand[col] = cand[col].fillna(0.0).astype(np.float32)

    cand["s1i"]  = (cand["key"] // (n23 + 1)).astype(np.int32)
    cand["s23i"] = (cand["key"] %  (n23 + 1)).astype(np.int32)
    cand = cand.drop(columns="key")

    cand["s1_id"]  = s1_sub["entity_id"].values[cand["s1i"].values]
    cand["s23_id"] = target_meta["entity_ids"][cand["s23i"].values]
    print(f"  Blocking done: {{len(cand):,}} candidate pairs in {{time.time()-t_b:.1f}}s")

    # 2. Features
    t_f = time.time()
    s1i, s23i = cand["s1i"].values, cand["s23i"].values
    nc1, nc2 = s1_sub["name_core"].values[s1i], target_meta["name_core"][s23i]
    nf1, nf2 = s1_sub["name_full"].values[s1i], target_meta["name_full"][s23i]
    a1,  a2  = s1_sub["addr_norm"].values[s1i],  target_meta["addr_norm"][s23i]
    c1,  c2  = s1_sub["country"].values[s1i],    target_countries[s23i]
    s23_ids = cand["s23_id"].values
    n = len(cand)

    feat = {{}}
    feat["nsim"] = cand["nsim"].values.astype(np.float32)
    feat["asim"] = cand["asim"].values.astype(np.float32)
    feat["csim"] = cand["csim"].values.astype(np.float32)
    feat["max_sim"] = np.maximum(np.maximum(feat["nsim"], feat["asim"]), feat["csim"])

    n_inter, n_sz1, n_sz2 = _binary_intersection(s1_sub["name_core"].values, target_meta["name_core"], s1i, s23i, analyzer="word", token_pattern=r"\\S+")
    feat["name_jaccard_tok"], feat["name_overlap"] = _jac_ov(n_inter, n_sz1, n_sz2)
    n3i, n3s1, n3s2 = _binary_intersection(s1_sub["name_core"].values, target_meta["name_core"], s1i, s23i, analyzer="char", ngram_range=(3, 3))
    feat["name_jac3"], _ = _jac_ov(n3i, n3s1, n3s2)
    n4i, n4s1, n4s2 = _binary_intersection(s1_sub["name_core"].values, target_meta["name_core"], s1i, s23i, analyzer="char", ngram_range=(4, 4))
    feat["name_jac4"], _ = _jac_ov(n4i, n4s1, n4s2)

    a_inter, a_sz1, a_sz2 = _binary_intersection(s1_sub["addr_norm"].values, target_meta["addr_norm"], s1i, s23i, analyzer="word", token_pattern=r"\\S+")
    feat["addr_jaccard_tok"], feat["addr_overlap"] = _jac_ov(a_inter, a_sz1, a_sz2)
    a3i, a3s1, a3s2 = _binary_intersection(s1_sub["addr_norm"].values, target_meta["addr_norm"], s1i, s23i, analyzer="char", ngram_range=(3, 3))
    feat["addr_jac3"], _ = _jac_ov(a3i, a3s1, a3s2)
    num_inter, _, _ = _binary_intersection(s1_sub["addr_norm"].values, target_meta["addr_norm"], s1i, s23i, analyzer="word", token_pattern=r"\\b\\d+\\b")
    feat["addr_num_overlap"] = num_inter

    d1i, d1s1, d1s2 = _binary_intersection(s1_sub["name_trade"].values, target_meta["name_core"], s1i, s23i, analyzer="word", token_pattern=r"\\S+")
    d1_j, _ = _jac_ov(d1i, d1s1, d1s2)
    d2i, d2s1, d2s2 = _binary_intersection(s1_sub["name_core"].values, target_trade, s1i, s23i, analyzer="word", token_pattern=r"\\S+")
    d2_j, _ = _jac_ov(d2i, d2s1, d2s2)
    feat["dba_trade_jaccard"] = np.maximum(feat["name_jaccard_tok"], np.maximum(d1_j, d2_j)).astype(np.float32)

    feat["name_exact"] = ((nf1 == nf2) & (nf1 != "")).astype(np.float32)
    nlen1 = np.array([len(x) for x in nc1], dtype=np.float32)
    nlen2 = np.array([len(x) for x in nc2], dtype=np.float32)
    feat["name_len_diff"] = np.abs(nlen1 - nlen2)
    maxn = np.maximum(nlen1, nlen2)
    feat["name_len_ratio"] = np.divide(np.minimum(nlen1, nlen2), maxn, out=np.zeros_like(nlen1), where=maxn > 0)
    feat["name_tok_diff"] = np.abs(n_sz1 - n_sz2)
    ft1 = np.array([x.split()[0] if x.split() else "" for x in nc1])
    ft2 = np.array([x.split()[0] if x.split() else "" for x in nc2])
    feat["name_first_tok"] = ((ft1 == ft2) & (ft1 != "")).astype(np.float32)

    alen1 = np.array([len(x) for x in a1], dtype=np.float32)
    alen2 = np.array([len(x) for x in a2], dtype=np.float32)
    feat["addr_missing"] = ((alen1 == 0) | (alen2 == 0)).astype(np.float32)
    maxa = np.maximum(alen1, alen2)
    feat["addr_len_ratio"] = np.divide(np.minimum(alen1, alen2), maxa, out=np.zeros_like(alen1), where=maxa > 0)

    feat["country_match"] = ((c1 == c2) & (c1 != "")).astype(np.float32)
    feat["country_mismatch"] = ((c1 != c2) & (c1 != "") & (c2 != "")).astype(np.float32)
    feat["src_is_s2"] = np.array([sid.startswith("S2-") for sid in s23_ids], dtype=np.float32)
    feat["name_x_addr"] = feat["name_jaccard_tok"] * feat["addr_jaccard_tok"]

    # In-Process Grouped RapidFuzz (C++ cdist)
    t_rf = time.time()
    order = np.argsort(s1i, kind="stable")
    sorted_s1i = s1i[order]
    _, g_start = np.unique(sorted_s1i, return_index=True)
    bounds = np.append(g_start, n)
    group_bounds = list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))
    n_groups = len(group_bounds)

    wratio = np.zeros(n, np.float32); tsort = np.zeros(n, np.float32)
    tset = np.zeros(n, np.float32); partial = np.zeros(n, np.float32); levn = np.zeros(n, np.float32)
    addr_ts = np.zeros(n, np.float32); addr_te = np.zeros(n, np.float32)

    print(f"  Computing RapidFuzz on {{n:,}} pairs ({{n_groups:,}} S1 groups)...", flush=True)
    t_last_log = time.time()
    for g_idx, (start, end) in enumerate(group_bounds):
        rows = order[start:end]
        a_name = nc1[rows[0]]; b_names = nc2[rows].tolist()
        a_addr = a1[rows[0]];  b_addrs = a2[rows].tolist()
        wratio[rows]  = np.array(rf_process.cdist([a_name], b_names, scorer=fuzz.WRatio)[0], np.float32) / 100.0
        tsort[rows]   = np.array(rf_process.cdist([a_name], b_names, scorer=fuzz.token_sort_ratio)[0], np.float32) / 100.0
        tset[rows]    = np.array(rf_process.cdist([a_name], b_names, scorer=fuzz.token_set_ratio)[0], np.float32) / 100.0
        partial[rows] = np.array(rf_process.cdist([a_name], b_names, scorer=fuzz.partial_ratio)[0], np.float32) / 100.0
        levn[rows]    = np.array(rf_process.cdist([a_name], b_names, scorer=Lev.normalized_similarity)[0], np.float32)
        addr_ts[rows] = np.array(rf_process.cdist([a_addr], b_addrs, scorer=fuzz.token_sort_ratio)[0], np.float32) / 100.0
        addr_te[rows] = np.array(rf_process.cdist([a_addr], b_addrs, scorer=fuzz.token_set_ratio)[0], np.float32) / 100.0
        if time.time() - t_last_log > 15:
            print(f"    RapidFuzz progress: {{g_idx+1:,}}/{{n_groups:,}} groups ({{time.time()-t_rf:.1f}}s elapsed)", flush=True)
            t_last_log = time.time()

    feat.update(dict(name_wratio=wratio, name_tsort=tsort, name_tset=tset, name_partial=partial, name_lev_norm=levn, addr_tsort=addr_ts, addr_tset=addr_te))
    print(f"  RapidFuzz completed in {{time.time()-t_rf:.1f}}s", flush=True)

    tmp = pd.DataFrame({{"s1_id": cand["s1_id"].values, "s23_id": cand["s23_id"].values, "max_sim": feat["max_sim"]}})
    tmp["rank_in_cand"] = tmp.groupby("s1_id")["max_sim"].rank(ascending=False, method="first")
    tmp["top_sim"] = tmp.groupby("s1_id")["max_sim"].transform("max")
    tmp["n_cand_for_entity"] = tmp.groupby("s1_id")["max_sim"].transform("size")
    tmp["reverse_rank"] = tmp.groupby("s23_id")["max_sim"].rank(ascending=False, method="first")

    feat["rank_in_cand"] = tmp["rank_in_cand"].values.astype(np.float32)
    feat["gap_to_top"] = (tmp["top_sim"] - tmp["max_sim"]).values.astype(np.float32)
    feat["n_cand_for_entity"] = tmp["n_cand_for_entity"].values.astype(np.float32)
    feat["reverse_rank"] = tmp["reverse_rank"].values.astype(np.float32)
    feat["mutual_top1"] = ((tmp["rank_in_cand"] == 1) & (tmp["reverse_rank"] == 1)).values.astype(np.float32)

    s1_nf = s1_sub["name_core"].value_counts()
    s1_af = s1_sub["addr_norm"].value_counts()
    feat["name_freq_s1"] = pd.Series(nc1).map(s1_nf).fillna(0).values.astype(np.float32)
    feat["addr_freq_s1"] = pd.Series(a1).map(s1_af).fillna(0).values.astype(np.float32)

    feat_df = pd.DataFrame(feat)[feature_names]
    print(f"  Features engineered in {{time.time()-t_f:.1f}}s")

    # 3. LightGBM Scoring + Calibration
    X = feat_df.values.astype(np.float32)
    raw_scores = np.mean([m.predict(X) for m in models], axis=0)
    cal_scores = iso.transform(raw_scores)
    cand["score"] = cal_scores

    # 4. Two-Threshold Decision Rule
    sub_matches, sub_cands = {{}}, {{}}
    grouped = cand.groupby("s1_id")
    for sid, grp in grouped:
        grp_s = grp.sort_values("score", ascending=False)
        ids = grp_s["s23_id"].tolist()
        scs = grp_s["score"].tolist()
        sub_cands[sid] = list(dict.fromkeys(ids))
        if scs and scs[0] >= t_first:
            keep = [i for i, s in zip(ids, scs) if s >= t_rest]
            sub_matches[sid] = list(dict.fromkeys(keep))
        else:
            sub_matches[sid] = []

    for sid in s1_sub["entity_id"]:
        if sid not in sub_matches:
            sub_matches[sid] = []
            sub_cands[sid] = []

    all_matches.update(sub_matches)
    all_cands.update(sub_cands)
    print(f"  Sub-batch {{b_idx+1}} done in {{time.time()-t_sub:.1f}}s. Accumulated {{len(all_matches):,}} queries.")
    del cand, feat_df, X, raw_scores, cal_scores, s1_sub; gc.collect()

# Save final partition output
out_file = os.path.join(WORK_DIR, f"part_{{PART_LABEL}}_results.pkl")
print(f"\\nSaving partition results to {{out_file}}...")
with open(out_file, "wb") as f:
    pickle.dump({{"part": PART_LABEL, "matches": all_matches, "cands": all_cands}}, f, protocol=pickle.HIGHEST_PROTOCOL)

print("="*60)
print(f"PARTITION {{PART_LABEL.upper()}} FINISHED in {{time.time()-t_part_start:.1f}}s")
print(f"Covered {{len(all_matches):,}} entities.")
print(f"Output saved to: {{out_file}}")
print("="*60)
""")
    ]
    return cells


def build_nb3_assembler():
    cells = [
        md_cell("""# Approach A — Notebook 3: Assemble & Official Validation
**Amazon ML Challenge 2026: Business Entity Resolution**

### Purpose:
1. Gathers the output pickle files from Notebook 2A (`part_02a_results.pkl`) and Notebook 2B (`part_02b_results.pkl`).
2. Orders rows to precisely match official `test_source1.tsv`.
3. Exports official submission deliverables:
   - `matching_results.tsv` (1,732,544 rows)
   - `candidate_pairs.tsv` (1,732,544 rows)
4. Executes comprehensive verification checks.
"""),
        code_cell("""# 1. Environment & Path Setup
import os, sys, glob, pickle, warnings
import pandas as pd

def find_file(filename, search_root="/kaggle/input"):
    hits = glob.glob(os.path.join(search_root, "**", filename), recursive=True)
    return hits[0] if hits else None

if os.path.isdir("/kaggle/input"):
    TEST_DIR = os.path.dirname(find_file("test_source1.tsv") or "/kaggle/input/dataset/test/test_source1.tsv")
    WORK_DIR = "/kaggle/working"
else:
    TEST_DIR = os.path.abspath("data/student_resource/dataset/test")
    WORK_DIR = os.path.abspath("output/approach_a")

FINAL_DIR = os.path.join(WORK_DIR, "final_submission")
os.makedirs(FINAL_DIR, exist_ok=True)
"""),
        code_cell("""# 2. Gather Partition 02a & 02b Results
result_files = sorted(glob.glob("/kaggle/input/**/part_02*_results.pkl", recursive=True) or
                      glob.glob(os.path.join(WORK_DIR, "**", "part_02*_results.pkl"), recursive=True) or
                      glob.glob("/kaggle/input/**/shard_*_results.pkl", recursive=True) or
                      glob.glob(os.path.join(WORK_DIR, "**", "shard_*_results.pkl"), recursive=True))

print(f"Discovered {len(result_files)} result files: {result_files}")
assert len(result_files) > 0, "No partition files found! Attach datasets containing part_02a_results.pkl and part_02b_results.pkl"

all_matches = {}
all_cands = {}

for rf in result_files:
    print(f"Loading {rf}...")
    with open(rf, "rb") as f:
        data = pickle.load(f)
    all_matches.update(data["matches"])
    all_cands.update(data["cands"])

print(f"Aggregated predictions for {len(all_matches):,} unique S1 entities")
"""),
        code_cell("""# 3. Export Deliverables to Official Competition Format
s1_test_path = os.path.join(TEST_DIR, "test_source1.tsv")
s1_ids_order = pd.read_csv(s1_test_path, sep="\t", dtype=str, usecols=["entity_id"])["entity_id"].tolist()
total_expected = len(s1_ids_order)
print(f"Official test S1 expects exactly {total_expected:,} entities")

assert len(all_matches) == total_expected, f"Entity count mismatch! Aggregated {len(all_matches):,}, expected {total_expected:,}"

mp = os.path.join(FINAL_DIR, "matching_results.tsv")
cp = os.path.join(FINAL_DIR, "candidate_pairs.tsv")

print(f"Writing {mp}...")
with open(mp, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tmatched_entity_ids\n")
    for sid in s1_ids_order:
        f.write(f"{sid}\t{','.join(all_matches.get(sid, []))}\n")

print(f"Writing {cp}...")
with open(cp, "w", encoding="utf-8") as f:
    f.write("source1_entity_id\tcandidate_entity_ids\n")
    for sid in s1_ids_order:
        f.write(f"{sid}\t{','.join(all_cands.get(sid, []))}\n")

n_matched = sum(1 for sid in s1_ids_order if all_matches.get(sid, []))
n_preds = sum(len(all_matches.get(sid, [])) for sid in s1_ids_order)
print(f"Summary: {n_matched:,}/{total_expected:,} entities predicted with matches ({n_preds:,} total pairs)")
"""),
        code_cell("""# 4. Comprehensive Validation Harness
s2_ids = set(pd.read_csv(os.path.join(TEST_DIR, "test_source2.tsv"), sep="\t", dtype=str, usecols=["entity_id"])["entity_id"])
s3_ids = set(pd.read_csv(os.path.join(TEST_DIR, "test_source3.tsv"), sep="\t", dtype=str, usecols=["entity_id"])["entity_id"])
valid_targets = s2_ids | s3_ids

mdf = pd.read_csv(mp, sep="\t", dtype=str).fillna("")
errors = []

if len(mdf) != total_expected:
    errors.append(f"Row count mismatch: got {len(mdf):,}, expected {total_expected:,}")

if mdf["source1_entity_id"].duplicated().any():
    errors.append("Duplicate source1_entity_id entries detected in matching_results.tsv!")

for idx, row in mdf.iterrows():
    mids = [x for x in row["matched_entity_ids"].split(",") if x]
    if len(mids) != len(set(mids)):
        errors.append(f"Duplicate IDs inside matched list for {row['source1_entity_id']}")
        break
    bad = [x for x in mids if x not in valid_targets]
    if bad:
        errors.append(f"Invalid target IDs for {row['source1_entity_id']}: {bad[:3]}")
        break

if not errors:
    print("="*60)
    print("ALL VALIDATION CHECKS PASSED!")
    print(f"Primary output ready: {mp}")
    print(f"Candidate output ready: {cp}")
    print("="*60)
else:
    print(f"VALIDATION FAILED with errors: {errors}")
""")
    ]
    return cells


def main():
    root = Path(__file__).resolve().parent.parent
    nb_dir = root / "notebooks"
    nb_dir.mkdir(exist_ok=True)

    create_ipynb(build_nb1_prep_target(), nb_dir / "approach_a_01_prep_target_index.ipynb")
    create_ipynb(build_nb2_worker("02a", 0.0, 0.5), nb_dir / "approach_a_02a_sharded_test_inference.ipynb")
    create_ipynb(build_nb2_worker("02b", 0.5, 1.0), nb_dir / "approach_a_02b_sharded_test_inference.ipynb")
    create_ipynb(build_nb3_assembler(), nb_dir / "approach_a_03_assemble_and_validate.ipynb")


if __name__ == "__main__":
    main()
