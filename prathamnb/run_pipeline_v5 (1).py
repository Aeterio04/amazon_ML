#!/usr/bin/env python3
"""
Auto-generated from entity_resolution_pipeline_v5.ipynb -- run directly with:
    python3 -u run_pipeline_v5.py

One-time setup (run manually once, NOT part of this script -- the notebook's
setup cell used Jupyter "!shell" magics, which only work under ipython, so
that cell is intentionally left out of this script):

    pip3 install lightgbm rapidfuzz scikit-learn pandas scipy pyarrow \
                 sparse_dot_topn --break-system-packages
    aws s3 sync s3://ml-challenge-dataset-team-baymax/dataset/ ./dataset/ --region eu-north-1
    # (skip the s3 sync if the data is already present / mounted)
"""


# ==============================================================================
# Amazon ML Challenge — Business Entity Resolution
# ==============================================================================

# ==============================================================================
# 0. Setup: install deps, pull data from S3
# ==============================================================================

# --- notebook cell 3 ---
import os, re, gc, sys, time, pickle, warnings, unicodedata, glob
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# ─── (patch) thread/process budget — set BEFORE numpy/sklearn/sparse_dot_topn
# are imported, since OpenBLAS/MKL/libgomp read these env vars once at import
# / first-use time. If something upstream (shell profile, conda env activate
# script, etc.) already exported a smaller OMP_NUM_THREADS, that's almost
# certainly why only a handful of cores were lighting up in htop regardless
# of what gets passed as n_threads= at the call site below — an inherited env
# var can silently cap the OpenMP thread pool before any in-script argument
# has a chance to matter. Run `env | grep -iE "omp|mkl|openblas|numexpr"` on
# the box BEFORE running this to see what, if anything, was already set.
# os.environ.setdefault(...) means: respect an explicit override if the user
# really did export one on purpose, otherwise default to using every core.
N_CORES = os.cpu_count() or 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(N_CORES))
os.environ.pop("GOMP_CPU_AFFINITY", None)   # don't let an inherited affinity mask pin OpenMP to a subset of cores
os.environ.pop("KMP_AFFINITY", None)
print(f"[thread-config] N_CORES detected={N_CORES}  "
      f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}", flush=True)

import numpy as np
import pandas as pd
from scipy import sparse

warnings.filterwarnings("ignore")

# ─── paths (Kaggle- and SageMaker/local-aware) ────────────────────────────
def find_dir(filename, search_root="/kaggle/input"):
    hits = glob.glob(os.path.join(search_root, "**", filename), recursive=True)
    return os.path.dirname(hits[0]) if hits else None

if os.path.isdir("/kaggle/input"):
    # Kaggle: locate the real mount path by searching rather than assuming a
    # fixed /kaggle/input/<slug>/... layout — it can be nested behind an extra
    # owner/dataset folder, e.g. /kaggle/input/datasets/<user>/<slug>/...
    TRAIN_DIR = find_dir("train_source1.tsv")
    TEST_DIR  = find_dir("test_source1.tsv")
    if TRAIN_DIR is None or TEST_DIR is None:
        raise FileNotFoundError(
            "Running on Kaggle but train_source1.tsv / test_source1.tsv weren't "
            "found anywhere under /kaggle/input. Check the Input panel: is the "
            "dataset actually attached, and has the session been restarted "
            "since adding it? Run `!find /kaggle/input -iname '*.tsv'` to see "
            "what's actually mounted."
        )
    BASE_DIR = "/kaggle/working"
else:
    # SageMaker / local: expects ./dataset/{train,test} already populated
    # (e.g. by the `aws s3 sync` in the setup cell above).
    BASE_DIR  = os.getcwd()
    TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
    TEST_DIR  = os.path.join(BASE_DIR, "dataset", "test")

OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODEL_DIR  = os.path.join(BASE_DIR, "models")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
print("TRAIN_DIR:", TRAIN_DIR)
print("TEST_DIR:", TEST_DIR)

# ─── debug / scale knobs ─────────────────────────────────────────────────
# Set SAMPLE_S1 to an int (e.g. 20_000) for a fast end-to-end smoke test.
# Set to None for the full run. ALWAYS smoke-test before a full run —
# there is no way to estimate real throughput without running on real data.
SAMPLE_S1 = 100_000

# NOTE on instance size: solution.py's docstring targeted ml.t3.medium
# (2 vCPU / 4GB). Your own methodology doc's audit puts the full training
# set at ~2.2M S1 / ~5.0M S2 / ~5.3M S3 records. Even with the vectorized
# fixes below, a run at that scale needs materially more RAM than 4GB
# (the TF-IDF matrices and candidate/feature frames for tens of millions
# of pairs are the dominant cost). Start this notebook on something in the
# ml.m5.4xlarge / ml.r5.4xlarge range (16 vCPU, 64GB+) for the full run;
# t3.medium is realistically only enough for SAMPLE_S1 debugging. On Kaggle,
# use "Save Version -> Save & Run All (Commit)" for the full run instead of
# an interactive session, so a browser/kernel disconnect can't lose progress.

# ─── blocking / feature hyperparameters ─────────────────────────────────
BLOCK_TOP_K        = 50
BLOCK_MIN_SIM       = 0.12
CHUNK_SIZE          = 12000   # (patch) was 2000 -- more work per sp_matmul_topn call, same
                              # total candidate set (chunking only affects call granularity,
                              # not which S1/S23 rows get compared), should push the measured
                              # 2.8x blocking speedup closer to the 3-5x expected from n_threads alone
NGRAM_RANGE_NAME    = (3, 4)
NGRAM_RANGE_ADDR    = (3, 4)
MAX_FEAT_NAME       = 200_000
MAX_FEAT_ADDR       = 150_000
USE_COUNTRY_BLOCKING = True   # data-driven bucketing, safe on unseen countries (see md doc Part 13)

# ─── v4: I/O engine flag ──────────────────────────────────────────────────
# pyarrow is typically several times faster than the default C engine for
# parsing the full-scale TSVs. This is a speed-only change for well-formed
# data; if you suspect your TSVs rely on quote-character parsing quirks that
# differ between engines, set this to False for a byte-identical read with
# the default engine.
USE_PYARROW_ENGINE = True

N_FOLDS             = 3       # grouped K-fold by S1 entity
THRESH_TUNE_SAMPLE  = 100_000 # cap on #S1 entities used for threshold grid search

# ─── v5: progress-logging cadence ─────────────────────────────────────────
# Purely cosmetic — how often the long-running loops below print a
# heartbeat. Does not affect anything that's computed.
LOG_EVERY_SEC  = 20    # seconds between heartbeat prints in blocking/feature loops
LGB_LOG_PERIOD = 100   # print LightGBM's validation loss every N boosting rounds

# ─── (patch) multiprocessing knobs ────────────────────────────────────────
# Used by load_source() (name/address normalization) and compute_features()
# (rapidfuzz scoring), both of which were pure single-core Python loops.
# N_WORKERS uses every core rather than cores-1: both stages are almost
# entirely CPU-bound with only cheap periodic prints, so there's no real
# responsiveness cost to using all N_CORES here. Drop this by 1 if SSH/
# htop feels laggy while a stage is running.
N_WORKERS = N_CORES
# Below this many rows/pairs, ProcessPoolExecutor's fork + IPC overhead
# costs more than it saves, so both stages fall back to the plain serial
# loop (this matters for the SAMPLE_S1 debug run, and for the small addr
# sanity-check matrix above).
PARALLEL_MIN_ROWS = 50_000
N_THREADS_BLOCKING = N_CORES  # passed explicitly to sp_matmul_topn (see below) —
                              # explicit beats relying on n_threads=-1 or env-var
                              # inheritance, which is what silently capped things before.

T_START = time.time()
def elapsed():
    return f"[{time.time()-T_START:.0f}s]"

# ==============================================================================
# 1. Text normalization (unchanged logic from solution.py, plus DBA trade-name kept as its own field)
# ==============================================================================

# --- notebook cell 5 ---
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

# Minimal, hand-written Devanagari -> Latin table (deterministic mapping table,
# not a third-party transliteration library / external lookup — see md doc Part 12).
DEV_MAP = {}
for pair in [
    ('अ','a'),('आ','aa'),('इ','i'),('ई','ii'),('उ','u'),('ऊ','uu'),
    ('ए','e'),('ऐ','ai'),('ओ','o'),('औ','au'),('ऋ','ri'),
    ('क','k'),('ख','kh'),('ग','g'),('घ','gh'),('ङ','ng'),
    ('च','ch'),('छ','chh'),('ज','j'),('झ','jh'),('ञ','ny'),
    ('ट','t'),('ठ','th'),('ड','d'),('ढ','dh'),('ण','n'),
    ('त','t'),('थ','th'),('द','d'),('ध','dh'),('न','n'),
    ('प','p'),('फ','ph'),('ब','b'),('भ','bh'),('म','m'),
    ('य','y'),('र','r'),('ल','l'),('व','v'),('श','sh'),('ष','sh'),('स','s'),('ह','h'),
    ('ा','aa'),('ि','i'),('ी','ii'),('ु','u'),('ू','uu'),
    ('े','e'),('ै','ai'),('ो','o'),('ौ','au'),('ं','n'),('ः','h'),('्',''),('ृ','ri'),
    ('़',''),('ँ','n'),('ॉ','o'),
]:
    DEV_MAP[pair[0]] = pair[1]

_punct_re = re.compile(r"[^a-z0-9\s]")
_ws_re    = re.compile(r"\s+")

def _translit(text):
    return "".join(DEV_MAP.get(ch, ch) for ch in text)

def _has_nonlatin(text):
    return any(ord(ch) > 127 and unicodedata.category(ch).startswith("L") for ch in text)

def _base(text):
    if not text or pd.isna(text):
        return ""
    text = str(text)
    if _has_nonlatin(text):
        text = _translit(text)
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().strip()

def norm_name(text):
    # Returns (name_full, name_core, name_trade) -- name_trade is the part
    # after a "dba" marker, legal-form-stripped, or "" if no dba present.
    t = _base(text)
    if not t:
        return "", "", ""
    t = t.replace("&", " and ")
    t = _punct_re.sub(" ", t)
    t = _ws_re.sub(" ", t).strip()
    name_full = t
    parts = re.split(r"\bdba\b", t)
    primary = parts[0].strip() if parts else t
    trade = parts[1].strip() if len(parts) > 1 else ""
    core = [tok for tok in primary.split() if tok not in LEGAL_FORMS]
    name_core = " ".join(core) if core else primary
    trade_core = " ".join(tok for tok in trade.split() if tok not in LEGAL_FORMS)
    return name_full, name_core, trade_core

def norm_addr(text):
    t = _base(text)
    if not t:
        return ""
    t = t.replace("&", " and ")
    t = re.sub(r"#+(\d)", r"\1", t)
    t = _punct_re.sub(" ", t)
    t = _ws_re.sub(" ", t).strip()
    out = []
    for tok in t.split():
        if tok in STREET_ABBR:
            out.append(STREET_ABBR[tok])
        elif tok in US_STATES:
            out.append(US_STATES[tok])
        else:
            out.append(tok)
    return " ".join(out)

# ==============================================================================
# 2. Data loading
# ==============================================================================

# --- notebook cell 7 ---
# (patch) module-level, picklable chunk-workers for parallel normalization.
# Must be top-level functions (not closures/lambdas) so ProcessPoolExecutor
# can send them to worker processes. norm_name/norm_addr and the lookup
# tables they use (LEGAL_FORMS, STREET_ABBR, DEV_MAP, ...) are module
# globals defined above, so under the default Linux "fork" start method
# each worker inherits them via copy-on-write at fork time — no need to
# re-pickle those tables per task.
def _norm_name_chunk(texts):
    return [norm_name(t) for t in texts]

def _norm_addr_chunk(texts):
    return [norm_addr(t) for t in texts]

def _parallel_map(values, chunk_func, label, n_workers=None):
    """Runs chunk_func over `values` split into n_workers contiguous chunks,
    one process per chunk (order-preserving). Falls back to a single
    in-process call for small inputs, where fork/IPC overhead isn't worth
    it. This is real multiprocessing, not threading — necessary here
    because norm_name/norm_addr are pure-Python/regex, so the GIL means
    plain threads would not run concurrently."""
    n_workers = n_workers or N_WORKERS
    values = values.tolist() if hasattr(values, "tolist") else list(values)
    n = len(values)
    if n < PARALLEL_MIN_ROWS or n_workers <= 1:
        return chunk_func(values)

    chunk_size = max(1, -(-n // n_workers))  # ceil div
    chunks = [values[i:i + chunk_size] for i in range(0, n, chunk_size)]
    results = [None] * len(chunks)
    t0 = time.time()
    done = 0
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {ex.submit(chunk_func, c): idx for idx, c in enumerate(chunks)}
        for fut in as_completed(futures):
            idx = futures[fut]
            results[idx] = fut.result()
            done += 1
            print(f"    {elapsed()} [load_source] {label}: {done}/{len(chunks)} chunks done "
                  f"({time.time()-t0:.1f}s elapsed, {n_workers} workers)", flush=True)
    out = []
    for r in results:
        out.extend(r)
    return out


def load_source(path, nrows=None):
    print(f"  {elapsed()} [load_source] reading {path} ...", flush=True)
    t_read = time.time()
    cols = ["entity_id", "business_name", "business_address", "country"]
    df = None
    if USE_PYARROW_ENGINE:
        try:
            df = pd.read_csv(path, sep="\t", dtype=str, usecols=cols, nrows=nrows, engine="pyarrow")
        except Exception as e:
            print(f"  [load_source] pyarrow engine failed ({e!r}), falling back to the default engine")
    if df is None:
        df = pd.read_csv(path, sep="\t", dtype=str, usecols=cols, nrows=nrows)
    df = df.fillna("")
    print(f"  {elapsed()} [load_source] read {len(df):,} rows in {time.time()-t_read:.1f}s, "
          f"normalizing (parallel across {N_WORKERS} workers when >= {PARALLEL_MIN_ROWS:,} rows) ...",
          flush=True)

    t_norm = time.time()
    name_results = _parallel_map(df["business_name"], _norm_name_chunk, "name")
    df["name_full"]  = [r[0] for r in name_results]
    df["name_core"]  = [r[1] for r in name_results]
    df["name_trade"] = [r[2] for r in name_results]
    del name_results

    df["addr_norm"] = _parallel_map(df["business_address"], _norm_addr_chunk, "address")

    df["combined"]   = df["name_core"] + " " + df["addr_norm"]
    df["country_norm"] = df["country"].map(lambda x: x.strip().lower() if x.strip() else "__unknown__")
    df = df[["entity_id","name_full","name_core","name_trade","addr_norm","combined","country","country_norm"]]
    print(f"  {elapsed()} [load_source] normalization done in {time.time()-t_norm:.1f}s", flush=True)
    return df.reset_index(drop=True)

# ==============================================================================
# 3. Blocking — TF-IDF char n-gram, fully vectorized aggregation
# ==============================================================================

# --- notebook cell 9 ---
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sparse_dot_topn import sp_matmul_topn

# (patch) confirm the installed sparse_dot_topn build actually accepts
# n_threads before relying on it below — versions have differed on the
# exact kwarg name/semantics, and a silently-ignored kwarg is exactly the
# kind of thing that would produce "I passed n_threads and nothing changed".
import inspect as _inspect
try:
    _sig = _inspect.signature(sp_matmul_topn)
    print(f"[thread-config] sp_matmul_topn signature: {_sig}", flush=True)
    if "n_threads" not in _sig.parameters:
        print("  WARNING: this sparse_dot_topn build has no n_threads parameter — "
              "check `pip3 show sparse_dot_topn` and the library's release notes; "
              "the n_threads= passed below will raise TypeError until this is resolved.", flush=True)
except (TypeError, ValueError):
    print("[thread-config] could not introspect sp_matmul_topn signature (may be a compiled built-in)", flush=True)


def route_topk_vectorized(sims_csr, min_sim, top_k):
    # v5: no longer called from the main pipeline — tfidf_block_one_route
    # below now uses sparse_dot_topn's fused top-n multiply instead, so the
    # full product this function filters is never computed. Kept only as
    # the reference implementation for the equivalence check in cell 3a.
    indptr = sims_csr.indptr
    if sims_csr.nnz == 0:
        empty = np.array([], dtype=np.int32)
        return empty, empty, np.array([], dtype=np.float32)
    row_ids = np.repeat(np.arange(sims_csr.shape[0], dtype=np.int32), np.diff(indptr))
    col_ids = sims_csr.indices.astype(np.int32)
    vals    = sims_csr.data

    mask = vals >= min_sim
    row_ids, col_ids, vals = row_ids[mask], col_ids[mask], vals[mask]
    if len(vals) == 0:
        empty = np.array([], dtype=np.int32)
        return empty, empty, np.array([], dtype=np.float32)

    order = np.lexsort((-vals, row_ids))          # sort by row asc, val desc
    row_ids, col_ids, vals = row_ids[order], col_ids[order], vals[order]

    _, group_start = np.unique(row_ids, return_index=True)
    group_sizes = np.diff(np.append(group_start, len(row_ids)))
    rank_within = np.arange(len(row_ids)) - np.repeat(group_start, group_sizes)
    keep = rank_within < top_k
    return row_ids[keep], col_ids[keep], vals[keep]


def build_country_buckets(s1, s23, use_country_blocking):
    """(patch) Same candidate-pair coverage as v5, restructured so the caller
    can transform the shared "unknown-country S23" pool ONCE instead of once
    per known country.

    v5 built each known-country bucket's S23 side as
    np.union1d(exact_country_match, unknown_s23) -- correct for recall (an
    S23 row with a blank country field is never safely excludable, so it's
    included as a safety net in every known-country bucket) but wasteful: the
    same multi-million-row unknown_s23 pool got vec.transform()'d from
    scratch inside every country's loop iteration. With N known countries
    that's N redundant transforms of an identical pool.

    Returned structure: {"unknown_s23_idx": ..., "buckets": [...]}. Each
    bucket dict is {"s1_idx", "exact_s23_idx", "shared_unknown"}; when
    shared_unknown is True, the bucket's actual S23 side is
    exact_s23_idx UNION unknown_s23_idx -- identical set to v5's
    np.union1d, just assembled by the caller from a cached transform of
    unknown_s23_idx plus a fresh transform of exact_s23_idx, instead of a
    fresh transform of the whole union. Candidate pairs, similarities and
    recall are unchanged from v5 -- this only removes duplicate work."""
    n1 = len(s1)
    if not use_country_blocking:
        return {"unknown_s23_idx": np.array([], dtype=np.int32),
                "buckets": [{"s1_idx": np.arange(n1, dtype=np.int32),
                             "exact_s23_idx": np.arange(len(s23), dtype=np.int32),
                             "shared_unknown": False}]}
    s1_c = s1["country_norm"].values
    s23_c = s23["country_norm"].values
    countries = sorted(set(s1_c.tolist()) | set(s23_c.tolist()))
    unknown_s23_idx = np.where(s23_c == "__unknown__")[0].astype(np.int32)
    has_unknown_pool = len(unknown_s23_idx) > 0
    buckets = []
    for c in countries:
        s1_idx = np.where(s1_c == c)[0].astype(np.int32)
        if len(s1_idx) == 0:
            continue
        if c == "__unknown__":
            # unknown S1 country -> compare vs everything, unchanged from v5
            buckets.append({"s1_idx": s1_idx,
                             "exact_s23_idx": np.arange(len(s23), dtype=np.int32),
                             "shared_unknown": False})
        else:
            exact_idx = np.where(s23_c == c)[0].astype(np.int32)
            if len(exact_idx) == 0 and not has_unknown_pool:
                continue   # nothing on the S23 side for this country at all
            buckets.append({"s1_idx": s1_idx,
                             "exact_s23_idx": exact_idx,
                             "shared_unknown": has_unknown_pool})
    return {"unknown_s23_idx": unknown_s23_idx, "buckets": buckets}


def tfidf_block_one_route(s1, s23, text_col, label, ngram_range, max_features, top_k,
                          use_country_blocking=USE_COUNTRY_BLOCKING):
    print(f"  {elapsed()} Blocking [{label}] ngram={ngram_range} topk={top_k} country_blocking={use_country_blocking}", flush=True)
    t_fit = time.time()
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=ngram_range,
                          max_features=max_features, sublinear_tf=True, dtype=np.float32)
    vec.fit(pd.concat([s1[text_col], s23[text_col]], ignore_index=True))
    print(f"    {elapsed()} [{label}] vectorizer fit done in {time.time()-t_fit:.1f}s, "
          f"vocab size={len(vec.vocabulary_):,}", flush=True)

    s1_text_all  = s1[text_col].values
    s23_text_all = s23[text_col].values
    bucket_info = build_country_buckets(s1, s23, use_country_blocking)
    buckets = bucket_info["buckets"]
    unknown_s23_idx = bucket_info["unknown_s23_idx"]
    n_chunks_total = sum(-(-len(b["s1_idx"]) // CHUNK_SIZE) for b in buckets)  # ceil division
    print(f"    {elapsed()} [{label}] {len(buckets)} country buckets, {n_chunks_total} chunks "
          f"total (chunk_size={CHUNK_SIZE})", flush=True)

    # v5: sp_matmul_topn's threshold is a strict '>' comparison (verified
    # empirically), while the old route_topk_vectorized used '>=' — nudge
    # down by one float32 ULP so the surviving-candidate set at the exact
    # boundary is identical to before (see cell 3a).
    min_sim_strict = float(np.nextafter(np.float32(BLOCK_MIN_SIM), np.float32(-np.inf)))

    # (patch) transform the shared "unknown-country S23" pool ONCE and reuse
    # it (via sparse.vstack) across every known-country bucket that needs it,
    # instead of re-transforming the same pool from scratch inside each
    # bucket's loop iteration as v5 did. This changes nothing about which
    # rows end up being compared or what their TF-IDF vectors are --
    # TfidfVectorizer.transform() weights each row independently using the
    # already-fit idf_, so transforming unknown_s23_idx once and stacking it
    # with a separately-transformed exact_s23_idx is numerically identical to
    # transforming their union in one call, just without doing it N times for
    # N known countries. Where before an M-row unknown pool got transformed
    # once per known country (2x for train's US+India, 3x for test's
    # US+India+France), it's now transformed exactly once, period.
    unknown_mat = None
    if len(unknown_s23_idx):
        t_shared = time.time()
        unknown_mat = sk_normalize(vec.transform(s23_text_all[unknown_s23_idx]))
        print(f"    {elapsed()} [{label}] shared unknown-country S23 pool "
              f"({len(unknown_s23_idx):,} rows) transformed once in "
              f"{time.time()-t_shared:.1f}s, reused across every known-country bucket", flush=True)

    out_s1, out_s23, out_sim = [], [], []
    chunks_done = 0
    t_last_log = time.time()
    for bi, b in enumerate(buckets):
        s1_idx, exact_idx, shared_unknown = b["s1_idx"], b["exact_s23_idx"], b["shared_unknown"]
        exact_mat = sk_normalize(vec.transform(s23_text_all[exact_idx])) if len(exact_idx) else None
        if shared_unknown and unknown_mat is not None:
            if exact_mat is not None:
                s23_mat = sparse.vstack([exact_mat, unknown_mat], format="csr")
                s23_idx = np.concatenate([exact_idx, unknown_s23_idx])
                # (patch) exact_idx and unknown_s23_idx are each individually
                # ascending (np.where preserves order), but concatenating
                # them interleaves the two ranges out of global order. v5's
                # np.union1d always returned a single ascending array, and
                # when two candidates tie exactly on similarity at the
                # top-k cutoff, sp_matmul_topn's tie-break depends on that
                # physical row order -- so re-sort back to ascending here to
                # keep every tie-break identical to v5, not just the score.
                order = np.argsort(s23_idx, kind="stable")
                s23_idx = s23_idx[order]
                s23_mat = s23_mat[order]
            else:
                s23_mat, s23_idx = unknown_mat, unknown_s23_idx
        else:
            s23_mat, s23_idx = exact_mat, exact_idx
        if s23_mat is None or s23_mat.shape[0] == 0:
            continue
        s23_mat_T = s23_mat.T.tocsr()
        for ci in range(0, len(s1_idx), CHUNK_SIZE):
            chunk_global = s1_idx[ci:ci+CHUNK_SIZE]
            chunk_mat = sk_normalize(vec.transform(s1_text_all[chunk_global]))
            # v5: fused top-n multiply — never materializes the full
            # (chunk_size x len(s23_idx)) product. This is the actual fix:
            # that full product is what stalled for hours on large buckets.
            C = sp_matmul_topn(chunk_mat, s23_mat_T, top_n=top_k,
                               threshold=min_sim_strict, sort=True,
                               n_threads=N_THREADS_BLOCKING).tocoo()
            r_local = C.row.astype(np.int32); c_local = C.col.astype(np.int32); v = C.data.astype(np.float32)
            if len(r_local):
                out_s1.append(chunk_global[r_local])
                out_s23.append(s23_idx[c_local])
                out_sim.append(v)
            del chunk_mat, C
            chunks_done += 1
            if time.time() - t_last_log > LOG_EVERY_SEC:
                done_pairs = sum(len(x) for x in out_sim)
                print(f"    {elapsed()} [{label}] chunk {chunks_done}/{n_chunks_total} "
                      f"(bucket {bi+1}/{len(buckets)}, {len(s23_idx):,} s23 rows)  "
                      f"candidates so far: {done_pairs:,}", flush=True)
                t_last_log = time.time()
        del s23_mat, s23_mat_T
        gc.collect()

    total_pairs = sum(len(x) for x in out_sim)
    print(f"    {elapsed()} [{label}] done: {chunks_done} chunks, {total_pairs:,} candidates", flush=True)

    if out_s1:
        return (np.concatenate(out_s1).astype(np.int32),
                np.concatenate(out_s23).astype(np.int32),
                np.concatenate(out_sim))
    return (np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32))


def run_blocking(s1, s23, top_k=BLOCK_TOP_K, use_country_blocking=USE_COUNTRY_BLOCKING):
    print(f"\n{elapsed()} ─── BLOCKING ───", flush=True)
    s1_ids  = s1["entity_id"].values
    s23_ids = s23["entity_id"].values
    n23 = len(s23)

    n_i, n_c, n_v = tfidf_block_one_route(s1, s23, "name_core", "name",
                                          NGRAM_RANGE_NAME, MAX_FEAT_NAME, top_k, use_country_blocking)
    a_i, a_c, a_v = tfidf_block_one_route(s1, s23, "addr_norm", "addr",
                                          NGRAM_RANGE_ADDR, MAX_FEAT_ADDR, max(5, top_k//2), use_country_blocking)
    c_i, c_c, c_v = tfidf_block_one_route(s1, s23, "combined", "combined",
                                          NGRAM_RANGE_NAME, MAX_FEAT_NAME, max(5, top_k//2), use_country_blocking)

    # v4: merge on a single combined int64 key instead of two int columns —
    # a scalar-key hash-merge is faster than a composite-key merge at this
    # row count. s1i*(n23+1)+s23i is a bijection since 0 <= s23i < n23
    # always, so it's losslessly reversible below (same candidate set and
    # similarity values as the two-column merge it replaces).
    def make_key(idx1, idx23):
        return idx1.astype(np.int64) * (n23 + 1) + idx23.astype(np.int64)

    df_n = pd.DataFrame({"key": make_key(n_i, n_c), "nsim": n_v})
    df_a = pd.DataFrame({"key": make_key(a_i, a_c), "asim": a_v})
    df_c = pd.DataFrame({"key": make_key(c_i, c_c), "csim": c_v})

    cand = df_n.merge(df_a, on="key", how="outer").merge(df_c, on="key", how="outer")
    for col in ["nsim","asim","csim"]:
        cand[col] = cand[col].fillna(0.0).astype(np.float32)

    cand["s1i"]  = (cand["key"] // (n23 + 1)).astype(np.int32)
    cand["s23i"] = (cand["key"] %  (n23 + 1)).astype(np.int32)
    cand = cand.drop(columns="key")
    cand["s1_id"]  = s1_ids[cand["s1i"].values]
    cand["s23_id"] = s23_ids[cand["s23i"].values]
    cand = cand.reset_index(drop=True)

    n = len(cand); nu = cand["s1_id"].nunique()
    print(f"  {elapsed()} Candidates: {n:,} pairs for {nu:,}/{len(s1):,} S1 entities "
          f"(avg {n/max(nu,1):.1f}/entity)", flush=True)
    return cand[["s1_id","s23_id","s1i","s23i","nsim","asim","csim"]]

# ==============================================================================
# 3a. Sanity check — `sparse_dot_topn` fused top-n vs. the old full-product + filter
# ==============================================================================

# --- notebook cell 11 ---
from scipy import sparse as _sp_test

print(f"\n{elapsed()} ─── SANITY CHECK (sparse_dot_topn fused top-n vs. old full-product+filter) ───")
_rng_bc = np.random.RandomState(0)
_A = sk_normalize(_sp_test.random(80, 3000, density=0.02, format="csr", dtype=np.float32, random_state=_rng_bc))
_B = sk_normalize(_sp_test.random(4000, 3000, density=0.02, format="csr", dtype=np.float32, random_state=_rng_bc))
_B_T = _B.T.tocsr()

_old_full = _A.dot(_B_T).tocsr()
_old_r, _old_c, _old_v = route_topk_vectorized(_old_full, BLOCK_MIN_SIM, BLOCK_TOP_K)

_thr = float(np.nextafter(np.float32(BLOCK_MIN_SIM), np.float32(-np.inf)))
_new_coo = sp_matmul_topn(_A, _B_T, top_n=BLOCK_TOP_K, threshold=_thr, sort=True,
                          n_threads=N_THREADS_BLOCKING).tocoo()
_new_r = _new_coo.row.astype(np.int32); _new_c = _new_coo.col.astype(np.int32); _new_v = _new_coo.data.astype(np.float32)

def _as_pair_set(r, c, v):
    return set(zip(r.tolist(), c.tolist(), np.round(v, 5).tolist()))

_old_set, _new_set = _as_pair_set(_old_r, _old_c, _old_v), _as_pair_set(_new_r, _new_c, _new_v)
_ok = (_old_set == _new_set)
print(f"  old candidates: {len(_old_set):,}   new candidates: {len(_new_set):,}")
print(f"  identical candidate set incl. similarity values: {_ok}")
if not _ok:
    print(f"  MISMATCH -- only in old: {len(_old_set - _new_set)}   only in new: {len(_new_set - _old_set)}")
    print("  Do not trust the blocking swap until this is resolved.")
del _A, _B, _B_T, _old_full

# ==============================================================================
# 4. Feature engineering — vectorized gather, cached token sets, batched rapidfuzz
# ==============================================================================

# --- notebook cell 13 ---
try:
    from rapidfuzz import fuzz as _fuzz
    from rapidfuzz import process as _rf_process
    from rapidfuzz.distance import Levenshtein as _Lev
    HAS_RF = True
except ImportError:
    HAS_RF = False

from sklearn.feature_extraction.text import CountVectorizer

# (patch) rapidfuzz is parallelized by splitting S1 GROUPS across processes,
# not by passing workers=-1 into each cdist() call. Each cdist call here is
# tiny (1 name vs. <=~150 candidates), and this loop runs once per unique S1
# entity — millions of times at full scale. Asking rapidfuzz's own C++
# thread pool to spin up on every one of those millions of small calls
# would spend more time on thread-pool setup/teardown than on the actual
# scoring, and would likely be SLOWER than the plain serial loop, not
# faster (this is why we did not just do the earlier sed-based
# `workers=-1` patch — it optimizes at the wrong granularity). Instead,
# _rf_group_worker keeps workers=1 per cdist call and a contiguous run of
# S1 groups (plus only the slice of nc1/nc2/a1/a2 those groups need — not
# the full arrays) is handed to each of N_WORKERS processes, so
# parallelism comes from process count. Passing each worker exactly the
# slice it needs (rather than the whole array via a fork-shared global)
# keeps memory bounded and doesn't depend on the fork start method.
def _rf_group_worker(task):
    rows_chunk, nc1_c, nc2_c, a1_c, a2_c, local_bounds = task
    wr, ts, tt, pa, lv, ats, ate = [], [], [], [], [], [], []
    for start, end in local_bounds:
        a_name = nc1_c[start]; b_names = nc2_c[start:end].tolist()
        a_addr = a1_c[start];  b_addrs = a2_c[start:end].tolist()
        wr.append(np.array(_rf_process.cdist([a_name], b_names, scorer=_fuzz.WRatio, workers=1)[0], np.float32) / 100.0)
        ts.append(np.array(_rf_process.cdist([a_name], b_names, scorer=_fuzz.token_sort_ratio, workers=1)[0], np.float32) / 100.0)
        tt.append(np.array(_rf_process.cdist([a_name], b_names, scorer=_fuzz.token_set_ratio, workers=1)[0], np.float32) / 100.0)
        pa.append(np.array(_rf_process.cdist([a_name], b_names, scorer=_fuzz.partial_ratio, workers=1)[0], np.float32) / 100.0)
        lv.append(np.array(_rf_process.cdist([a_name], b_names, scorer=_Lev.normalized_similarity, workers=1)[0], np.float32))
        ats.append(np.array(_rf_process.cdist([a_addr], b_addrs, scorer=_fuzz.token_sort_ratio, workers=1)[0], np.float32) / 100.0)
        ate.append(np.array(_rf_process.cdist([a_addr], b_addrs, scorer=_fuzz.token_set_ratio, workers=1)[0], np.float32) / 100.0)
    empty = np.array([], dtype=np.float32)
    return (rows_chunk,
            np.concatenate(wr) if wr else empty, np.concatenate(ts) if ts else empty,
            np.concatenate(tt) if tt else empty, np.concatenate(pa) if pa else empty,
            np.concatenate(lv) if lv else empty, np.concatenate(ats) if ats else empty,
            np.concatenate(ate) if ate else empty)


def _binary_intersection(text1, text23, idx1, idx23, analyzer, token_pattern=None, ngram_range=(1, 1)):
    r"""Exact set-intersection / set-size vectors for a given tokenization,
    computed with sparse binary matrices instead of a per-pair Python loop.
    Reproduces len(set_a & set_b), len(set_a), len(set_b) exactly for
    whatever tokenization analyzer/token_pattern/ngram_range encode:
      - word tokens matching str.split():            analyzer="word", token_pattern=r"\S+"
      - char k-grams matching s[i:i+k] (no word-boundary padding, i.e. NOT
        sklearn's char_wb): analyzer="char", ngram_range=(k, k)
      - digit tokens matching \b\d+\b:                analyzer="word", token_pattern=r"\b\d+\b"
    text1/text23 are the RAW per-record columns (length n1/n23), not
    gathered per pair — the sparse row-gather by idx1/idx23 happens inside
    this function, which is cheaper than materializing per-pair string
    arrays first. Returns (inter, size1, size2) as float32 arrays aligned
    to idx1/idx23 (length = number of candidate pairs).
    """
    n = len(idx1)
    nonempty = pd.concat([text1[text1 != ""], text23[text23 != ""]], ignore_index=True)
    if nonempty.empty:
        z = np.zeros(n, dtype=np.float32)
        return z, z.copy(), z.copy()
    kwargs = dict(analyzer=analyzer, ngram_range=ngram_range, binary=True, dtype=np.float32)
    if token_pattern is not None:
        kwargs["token_pattern"] = token_pattern
    cv = CountVectorizer(**kwargs)
    cv.fit(nonempty)
    X1  = cv.transform(text1)
    X23 = cv.transform(text23)
    A, B = X1[idx1], X23[idx23]
    inter = np.asarray(A.multiply(B).sum(axis=1)).ravel().astype(np.float32)
    size1 = np.asarray(X1.sum(axis=1)).ravel()[idx1].astype(np.float32)
    size2 = np.asarray(X23.sum(axis=1)).ravel()[idx23].astype(np.float32)
    del X1, X23, A, B
    return inter, size1, size2


def _jac_ov(inter, size1, size2):
    union = size1 + size2 - inter
    jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    minsz = np.minimum(size1, size2)
    ov = np.divide(inter, minsz, out=np.zeros_like(inter), where=minsz > 0)
    return jac.astype(np.float32), ov.astype(np.float32)


def compute_features(cand, s1, s23):
    print(f"\n{elapsed()} ─── FEATURES ({len(cand):,} pairs) ───")
    s1i  = cand["s1i"].values
    s23i = cand["s23i"].values

    # gathered per-pair arrays (same as v3) — used for elementwise features
    # that don't benefit from record-level dedup, and for the rapidfuzz
    # batching below (unchanged from v3).
    nc1 = s1["name_core"].values[s1i];  nc2 = s23["name_core"].values[s23i]
    nf1 = s1["name_full"].values[s1i];  nf2 = s23["name_full"].values[s23i]
    a1  = s1["addr_norm"].values[s1i];  a2  = s23["addr_norm"].values[s23i]
    c1  = s1["country"].values[s1i];    c2  = s23["country"].values[s23i]
    nsims = cand["nsim"].values.astype(np.float32)
    asims = cand["asim"].values.astype(np.float32)
    csims = cand["csim"].values.astype(np.float32)
    s23_ids = cand["s23_id"].values
    n = len(cand)

    feat = {}
    feat["nsim"] = nsims; feat["asim"] = asims; feat["csim"] = csims
    feat["max_sim"] = np.maximum(np.maximum(nsims, asims), csims)

    # ── v4: token / n-gram Jaccard & overlap, vectorized with sparse binary
    #    matrices instead of the v3 per-pair Python loop (see md note above
    #    and the equivalence-check in cell 6a). These work off the RAW
    #    (un-gathered) per-record columns — the row-gather by s1i/s23i
    #    happens inside _binary_intersection. ──
    name_core_1, name_core_2 = s1["name_core"], s23["name_core"]
    addr_1, addr_2           = s1["addr_norm"], s23["addr_norm"]
    trade_1, trade_2         = s1["name_trade"], s23["name_trade"]

    n_inter, n_sz1, n_sz2 = _binary_intersection(name_core_1, name_core_2, s1i, s23i,
                                                 analyzer="word", token_pattern=r"\S+")
    name_jt, name_ov = _jac_ov(n_inter, n_sz1, n_sz2)

    n3i, n3s1, n3s2 = _binary_intersection(name_core_1, name_core_2, s1i, s23i,
                                           analyzer="char", ngram_range=(3, 3))
    name_j3, _ = _jac_ov(n3i, n3s1, n3s2)
    n4i, n4s1, n4s2 = _binary_intersection(name_core_1, name_core_2, s1i, s23i,
                                           analyzer="char", ngram_range=(4, 4))
    name_j4, _ = _jac_ov(n4i, n4s1, n4s2)
    print(f"    {elapsed()} name token/char3/char4 jaccard done", flush=True)

    a_inter, a_sz1, a_sz2 = _binary_intersection(addr_1, addr_2, s1i, s23i,
                                                 analyzer="word", token_pattern=r"\S+")
    addr_jt, addr_ov = _jac_ov(a_inter, a_sz1, a_sz2)
    a3i, a3s1, a3s2 = _binary_intersection(addr_1, addr_2, s1i, s23i,
                                           analyzer="char", ngram_range=(3, 3))
    addr_j3, _ = _jac_ov(a3i, a3s1, a3s2)

    addr_num_ov, _, _ = _binary_intersection(addr_1, addr_2, s1i, s23i,
                                             analyzer="word", token_pattern=r"\b\d+\b")
    print(f"    {elapsed()} address token/char3 jaccard + numeric overlap done", flush=True)

    # dba/trade-name best match: max(name_jt, jac(trade1,name2), jac(name1,trade2)).
    # jac(...) is already 0 whenever the trade side is empty (an empty set
    # intersected with anything is empty), so computing it unconditionally
    # and taking the max reproduces the v3 loop's `if nt1[i]:`/`if nt2[i]:`
    # guards exactly — max(x, 0) is a no-op against an already >=0 score.
    d1i, d1s1, d1s2 = _binary_intersection(trade_1, name_core_2, s1i, s23i,
                                           analyzer="word", token_pattern=r"\S+")
    dba_j1, _ = _jac_ov(d1i, d1s1, d1s2)
    d2i, d2s1, d2s2 = _binary_intersection(name_core_1, trade_2, s1i, s23i,
                                           analyzer="word", token_pattern=r"\S+")
    dba_j2, _ = _jac_ov(d2i, d2s1, d2s2)
    dba_match = np.maximum(name_jt, np.maximum(dba_j1, dba_j2)).astype(np.float32)
    print(f"    {elapsed()} dba/trade-name match done", flush=True)

    # ── elementwise features — vectorized with numpy/pandas string ops.
    #    Lengths/first-tokens are computed once over the raw per-record
    #    columns (n1+n23 rows) then gathered by s1i/s23i, instead of
    #    recomputing len()/split() once per candidate pair (the same S1 or
    #    S23 record can appear in dozens to hundreds of pairs). ──
    name_exact = ((nf1 == nf2) & (nf1 != "")).astype(np.float32)

    nlen1_full = name_core_1.str.len().fillna(0).values.astype(np.float32)
    nlen2_full = name_core_2.str.len().fillna(0).values.astype(np.float32)
    nlen1, nlen2 = nlen1_full[s1i], nlen2_full[s23i]
    name_len_diff = np.abs(nlen1 - nlen2)
    maxnlen = np.maximum(nlen1, nlen2)
    name_len_ratio = np.divide(np.minimum(nlen1, nlen2), maxnlen,
                               out=np.zeros_like(nlen1), where=maxnlen > 0)
    name_tok_diff = np.abs(n_sz1 - n_sz2)   # word-token counts already computed above, already pair-aligned

    ft1_full = name_core_1.str.split().str[0].fillna("").values
    ft2_full = name_core_2.str.split().str[0].fillna("").values
    ft1, ft2 = ft1_full[s1i], ft2_full[s23i]
    name_first = ((ft1 == ft2) & (ft1 != "")).astype(np.float32)

    alen1_full = addr_1.str.len().fillna(0).values.astype(np.float32)
    alen2_full = addr_2.str.len().fillna(0).values.astype(np.float32)
    alen1, alen2 = alen1_full[s1i], alen2_full[s23i]
    addr_missing = ((alen1 == 0) | (alen2 == 0)).astype(np.float32)
    maxalen = np.maximum(alen1, alen2)
    addr_len_ratio = np.divide(np.minimum(alen1, alen2), maxalen,
                               out=np.zeros_like(alen1), where=maxalen > 0)

    feat.update(dict(
        name_exact=name_exact, name_jaccard_tok=name_jt, name_overlap=name_ov,
        name_jac3=name_j3, name_jac4=name_j4, name_first_tok=name_first,
        name_len_diff=name_len_diff, name_len_ratio=name_len_ratio, name_tok_diff=name_tok_diff,
        addr_jaccard_tok=addr_jt, addr_overlap=addr_ov, addr_jac3=addr_j3,
        addr_num_overlap=addr_num_ov, addr_missing=addr_missing, addr_len_ratio=addr_len_ratio,
        dba_trade_jaccard=dba_match,
        country_match=((c1==c2) & (c1!="")).astype(np.float32),
        country_mismatch=((c1!=c2) & (c1!="") & (c2!="")).astype(np.float32),
        src_is_s2=np.array([str(sid).startswith("S2-") for sid in s23_ids], dtype=np.float32),
        name_x_addr=name_jt*addr_jt,
    ))
    print(f"    {elapsed()} elementwise features done", flush=True)

    # ── batched rapidfuzz, grouped per S1 entity, parallelized across
    #    N_WORKERS processes (each still bounded by #unique s1i, not #pairs;
    #    see the note above _rf_group_worker for why this is process-level
    #    parallelism rather than workers=-1 inside each cdist call) ──
    if HAS_RF:
        wratio = np.zeros(n, np.float32); tsort = np.zeros(n, np.float32)
        tset = np.zeros(n, np.float32); partial = np.zeros(n, np.float32); levn = np.zeros(n, np.float32)
        addr_ts = np.zeros(n, np.float32); addr_te = np.zeros(n, np.float32)

        order = np.argsort(s1i, kind="stable")
        sorted_idx = s1i[order]
        _, group_start = np.unique(sorted_idx, return_index=True)
        bounds = np.append(group_start, n)
        group_bounds = list(zip(bounds[:-1].tolist(), bounds[1:].tolist()))
        n_groups_total = len(group_bounds)
        t_rf = time.time()

        if n >= PARALLEL_MIN_ROWS and N_WORKERS > 1:
            print(f"    {elapsed()} rapidfuzz: {n_groups_total:,} S1 groups over {n:,} pairs, "
                  f"parallel across {N_WORKERS} workers", flush=True)
            # split by GROUP index (not raw row index) into N_WORKERS
            # contiguous group-ranges, so each worker's rows_chunk is still
            # a whole run of complete groups.
            edges = np.linspace(0, n_groups_total, N_WORKERS + 1).round().astype(np.int64)
            tasks = []
            for gi in range(N_WORKERS):
                g_lo, g_hi = int(edges[gi]), int(edges[gi + 1])
                if g_lo >= g_hi:
                    continue
                pos_lo, pos_hi = int(bounds[g_lo]), int(bounds[g_hi])   # contiguous span within `order`
                rows_chunk = order[pos_lo:pos_hi]                       # original pair-indices, this worker's job
                lb = (bounds[g_lo:g_hi + 1] - pos_lo)
                local_bounds = list(zip(lb[:-1].tolist(), lb[1:].tolist()))
                # pre-gather only the slice this worker needs (aligned with rows_chunk's order),
                # instead of shipping/sharing the full nc1/nc2/a1/a2 arrays.
                tasks.append((rows_chunk, nc1[rows_chunk], nc2[rows_chunk],
                              a1[rows_chunk], a2[rows_chunk], local_bounds))
            done = 0
            ctx = mp.get_context("fork")
            with ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=ctx) as ex:
                futures = {ex.submit(_rf_group_worker, t): i for i, t in enumerate(tasks)}
                for fut in as_completed(futures):
                    rows_chunk, wr, ts_, tt, pa, lv, ats, ate = fut.result()
                    wratio[rows_chunk] = wr; tsort[rows_chunk] = ts_; tset[rows_chunk] = tt
                    partial[rows_chunk] = pa; levn[rows_chunk] = lv
                    addr_ts[rows_chunk] = ats; addr_te[rows_chunk] = ate
                    done += 1
                    print(f"    {elapsed()} rapidfuzz: {done}/{len(tasks)} worker chunks done "
                          f"({time.time()-t_rf:.1f}s elapsed)", flush=True)
        else:
            print(f"    {elapsed()} rapidfuzz: batching {n_groups_total:,} S1 groups over {n:,} pairs "
                  f"(serial — below PARALLEL_MIN_ROWS)", flush=True)
            t_last_log = time.time(); groups_done = 0
            for start, end in group_bounds:
                rows = order[start:end]
                a_name = nc1[rows[0]]; b_names = nc2[rows].tolist()
                a_addr = a1[rows[0]];  b_addrs = a2[rows].tolist()
                wratio[rows] = np.array(_rf_process.cdist([a_name], b_names, scorer=_fuzz.WRatio)[0], np.float32) / 100.0
                tsort[rows]  = np.array(_rf_process.cdist([a_name], b_names, scorer=_fuzz.token_sort_ratio)[0], np.float32) / 100.0
                tset[rows]   = np.array(_rf_process.cdist([a_name], b_names, scorer=_fuzz.token_set_ratio)[0], np.float32) / 100.0
                partial[rows]= np.array(_rf_process.cdist([a_name], b_names, scorer=_fuzz.partial_ratio)[0], np.float32) / 100.0
                levn[rows]   = np.array(_rf_process.cdist([a_name], b_names, scorer=_Lev.normalized_similarity)[0], np.float32)
                addr_ts[rows]= np.array(_rf_process.cdist([a_addr], b_addrs, scorer=_fuzz.token_sort_ratio)[0], np.float32) / 100.0
                addr_te[rows]= np.array(_rf_process.cdist([a_addr], b_addrs, scorer=_fuzz.token_set_ratio)[0], np.float32) / 100.0
                groups_done += 1
                if time.time() - t_last_log > LOG_EVERY_SEC:
                    print(f"    {elapsed()} rapidfuzz: {groups_done:,}/{n_groups_total:,} S1 groups", flush=True)
                    t_last_log = time.time()

        feat.update(dict(name_wratio=wratio, name_tsort=tsort, name_tset=tset,
                          name_partial=partial, name_lev_norm=levn,
                          addr_tsort=addr_ts, addr_tset=addr_te))
        print(f"    {elapsed()} rapidfuzz done in {time.time()-t_rf:.1f}s for {n_groups_total:,} groups", flush=True)

    # ── rank / gap / mutual-best-match / frequency features (unchanged, already vectorized) ──
    tmp = pd.DataFrame({"s1_id": cand["s1_id"].values, "s23_id": cand["s23_id"].values,
                        "max_sim": feat["max_sim"]})
    tmp["rank_in_cand"] = tmp.groupby("s1_id")["max_sim"].rank(ascending=False, method="first")
    tmp["top_sim"]      = tmp.groupby("s1_id")["max_sim"].transform("max")
    tmp["n_cand_for_entity"] = tmp.groupby("s1_id")["max_sim"].transform("size")
    tmp["reverse_rank"] = tmp.groupby("s23_id")["max_sim"].rank(ascending=False, method="first")
    feat["rank_in_cand"] = tmp["rank_in_cand"].values.astype(np.float32)
    feat["gap_to_top"] = (tmp["top_sim"] - tmp["max_sim"]).values.astype(np.float32)
    feat["n_cand_for_entity"] = tmp["n_cand_for_entity"].values.astype(np.float32)
    feat["reverse_rank"] = tmp["reverse_rank"].values.astype(np.float32)
    feat["mutual_top1"] = ((tmp["rank_in_cand"]==1) & (tmp["reverse_rank"]==1)).values.astype(np.float32)

    s1_name_freq = s1["name_core"].value_counts()
    s1_addr_freq = s1["addr_norm"].value_counts()
    feat["name_freq_s1"] = pd.Series(nc1).map(s1_name_freq).fillna(0).values.astype(np.float32)
    feat["addr_freq_s1"] = pd.Series(a1).map(s1_addr_freq).fillna(0).values.astype(np.float32)
    print(f"    {elapsed()} rank/gap/frequency features done", flush=True)

    out = pd.DataFrame(feat)
    print(f"  {elapsed()} FEATURES done: {len(out):,} rows x {len(out.columns)} columns", flush=True)
    return out

# ==============================================================================
# 5. Training — grouped K-fold LightGBM, isotonic calibration, two-threshold decision rule
# ==============================================================================

# --- notebook cell 15 ---
import lightgbm as lgb
from sklearn.model_selection import GroupKFold
from sklearn.isotonic import IsotonicRegression

LGB_PARAMS = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.05, num_leaves=63, min_data_in_leaf=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=5,
    is_unbalance=True, verbose=-1, n_jobs=-1, seed=42,
)

def load_gt_dict(gt_path):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str).fillna("")
    d = {}
    for sid, mids in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        d[sid] = set(mids.split(",")) if mids.strip() else set()
    return d

def build_labels(cand, gt_dict):
    rows = [(sid, m) for sid, mids in gt_dict.items() for m in mids]
    gt_pairs = pd.DataFrame(rows, columns=["s1_id","s23_id"])
    gt_pairs["label"] = np.int8(1)
    merged = cand[["s1_id","s23_id"]].merge(gt_pairs, on=["s1_id","s23_id"], how="left")
    return merged["label"].fillna(0).values.astype(np.int8)

def f05(p, r):
    return 1.25*p*r/(0.25*p+r) if (p+r) > 0 else 0.0

def build_pred_dict(cand, scores):
    # v4: single global sort + boundary slicing instead of a pandas groupby
    # loop over every S1 entity — identical (s23_id, score) lists per entity
    # (now pre-sorted by score descending, which downstream code re-sorts
    # anyway, so this changes no computed value, only how fast it's built).
    s1i = cand["s1i"].values
    s1_id_arr = cand["s1_id"].values
    s23_id_arr = cand["s23_id"].values
    order = np.lexsort((-scores, s1i))
    s1i_s, s1id_s, s23id_s, sc_s = s1i[order], s1_id_arr[order], s23_id_arr[order], scores[order]
    _, start = np.unique(s1i_s, return_index=True)
    bounds = np.append(start, len(s1i_s))
    d = {}
    for k in range(len(start)):
        lo, hi = bounds[k], bounds[k+1]
        d[s1id_s[lo]] = list(zip(s23id_s[lo:hi], sc_s[lo:hi]))
    return d

def macro_f05_single(gt_dict, pred_dict, t, all_s1_ids):
    scores = []
    for s1id in all_s1_ids:
        M = gt_dict.get(s1id, set())
        P = {sid for sid, sc in pred_dict.get(s1id, []) if sc >= t}
        if not M:
            scores.append(1.0 if not P else 0.0)
        elif not P:
            scores.append(0.0)
        else:
            tp = len(P & M); scores.append(f05(tp/len(P), tp/len(M)))
    return float(np.mean(scores))

def macro_f05_two(gt_dict, pred_dict, t_first, t_rest, all_s1_ids):
    scores = []
    for s1id in all_s1_ids:
        M = gt_dict.get(s1id, set())
        plist = sorted(pred_dict.get(s1id, []), key=lambda x: -x[1])
        if not plist or plist[0][1] < t_first:
            P = set()
        else:
            P = {sid for sid, sc in plist if sc >= t_rest}
        if not M:
            scores.append(1.0 if not P else 0.0)
        elif not P:
            scores.append(0.0)
        else:
            tp = len(P & M); scores.append(f05(tp/len(P), tp/len(M)))
    return float(np.mean(scores))

def best_single_threshold(gt_dict, pred_dict, all_s1_ids):
    best_t, best_s = 0.5, 0.0
    for t in np.arange(0.30, 0.96, 0.02):
        s = macro_f05_single(gt_dict, pred_dict, t, all_s1_ids)
        if s > best_s:
            best_s, best_t = s, t
    for t in np.arange(max(0.10, best_t-0.05), min(0.99, best_t+0.06), 0.005):
        s = macro_f05_single(gt_dict, pred_dict, t, all_s1_ids)
        if s > best_s:
            best_s, best_t = s, t
    return best_t, best_s


def train_cv(feat_df, cand, gt_dict, n_folds=N_FOLDS):
    print(f"\n{elapsed()} ─── TRAINING ({n_folds}-fold grouped CV) ───")
    labels = build_labels(cand, gt_dict)
    print(f"  Pos: {labels.sum():,}  Neg: {(len(labels)-labels.sum()):,}  Rate: {labels.mean():.4f}")

    X = feat_df.values.astype(np.float32)
    groups = cand["s1_id"].values
    gkf = GroupKFold(n_splits=n_folds)

    oof = np.zeros(len(cand), dtype=np.float32)
    models = []
    for fold, (tr, va) in enumerate(gkf.split(X, labels, groups)):
        print(f"  {elapsed()} fold {fold}: starting ({len(tr):,} train / {len(va):,} val rows)", flush=True)
        dtr = lgb.Dataset(X[tr], labels[tr], feature_name=list(feat_df.columns))
        dva = lgb.Dataset(X[va], labels[va], reference=dtr)
        m = lgb.train(LGB_PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50), lgb.log_evaluation(LGB_LOG_PERIOD)])
        oof[va] = m.predict(X[va], num_iteration=m.best_iteration)
        models.append(m)
        print(f"  {elapsed()} fold {fold}: best_iter={m.best_iteration}  "
              f"val_logloss={m.best_score['valid_0']['binary_logloss']:.4f}", flush=True)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(oof, labels)
    oof_cal = iso.transform(oof)

    imp = np.mean([m.feature_importance(importance_type="gain") for m in models], axis=0)
    print("  Top features (mean gain across folds):")
    for fn, fi in sorted(zip(feat_df.columns, imp), key=lambda x: -x[1])[:12]:
        print(f"    {fn}: {fi:.0f}")

    return models, iso, oof_cal, labels


def tune_thresholds(cand, oof_cal, gt_dict, s1, sample_size=THRESH_TUNE_SAMPLE):
    pred_dict = build_pred_dict(cand, oof_cal)
    all_s1 = s1["entity_id"].values
    rng = np.random.RandomState(42)
    sample_ids = (rng.choice(all_s1, size=sample_size, replace=False)
                 if sample_size and len(all_s1) > sample_size else all_s1)
    val_gt = {s: gt_dict.get(s, set()) for s in sample_ids}
    sample_ids = list(sample_ids)

    t_first, f05_single = best_single_threshold(val_gt, pred_dict, sample_ids)
    print(f"  single-threshold: t={t_first:.3f}  macro F0.5={f05_single:.4f}")

    best_t_rest, best_f05_2 = t_first, f05_single
    for t_rest in np.arange(0.10, t_first, 0.02):
        s = macro_f05_two(val_gt, pred_dict, t_first, t_rest, sample_ids)
        if s > best_f05_2:
            best_f05_2, best_t_rest = s, t_rest
    print(f"  two-threshold:    t_first={t_first:.3f} t_rest={best_t_rest:.3f}  macro F0.5={best_f05_2:.4f}")
    return t_first, best_t_rest, best_f05_2

def blocking_quality(cand, gt_dict, s1_ids, n_s23):
    """Pair Completeness (PC, i.e. blocking recall) and Reduction Ratio (RR).

    PC = fraction of true match pairs that survived blocking — the hard
    ceiling on final recall (methodology doc 1.2): the matcher can never
    recover a true pair blocking already dropped.
    RR = fraction of the full S1 x S23 comparison space blocking avoided
    computing at all.
    Neither of these is "precision" — blocking is deliberately over-inclusive,
    so most candidates NOT being true matches is expected and fine at this
    stage. Precision only starts to matter once the matcher + threshold turn
    candidates into final predictions — see macro_score_report below.
    """
    pairs = set(zip(cand["s1_id"].values, cand["s23_id"].values))
    tp = total_true = 0
    for sid in s1_ids:
        M = gt_dict.get(sid, set())
        if not M:
            continue
        total_true += len(M)
        tp += sum(1 for m in M if (sid, m) in pairs)
    pc = tp / total_true if total_true else float("nan")
    full_space = len(s1_ids) * n_s23
    rr = 1 - (len(cand) / full_space) if full_space else float("nan")
    print(f"  Blocking PC (recall ceiling): {pc:.4f}  ({tp:,}/{total_true:,} true pairs kept)")
    print(f"  Blocking RR (reduction ratio): {rr:.6f}  ({len(cand):,} candidates vs {full_space:,} full pairs)")
    return pc, rr


def macro_score_report(gt_dict, pred_dict, t_first, t_rest, all_s1_ids, use_two_thresh=True):
    """Full scoring report on entities with known ground truth (a held-out
    training slice — the real test set has no labels; per the PDF, "hold out
    a validation split from the training data and score it yourself").

    Macro F0.5 is computed exactly as specified (per-entity F0.5, singletons
    scoring 1.0/0.0, averaged over ALL entities) — this estimates your
    leaderboard score. Precision and recall are additionally reported over
    non-singleton entities only, where they are well-defined in the usual
    sense; singleton accuracy separately reports how often you correctly
    predicted "no match" (any prediction on a true singleton is automatically
    a false positive, so this is effectively precision on that slice).
    """
    f05s, ns_precs, ns_recs = [], [], []
    n_singleton = n_singleton_correct = n_nonsingleton = 0

    for s1id in all_s1_ids:
        M = gt_dict.get(s1id, set())
        plist = sorted(pred_dict.get(s1id, []), key=lambda x: -x[1])
        if use_two_thresh:
            P = set() if (not plist or plist[0][1] < t_first) else {sid for sid, sc in plist if sc >= t_rest}
        else:
            P = {sid for sid, sc in plist if sc >= t_first}

        if not M:
            n_singleton += 1
            correct = not P
            n_singleton_correct += correct
            f05s.append(1.0 if correct else 0.0)
        else:
            n_nonsingleton += 1
            tp = len(P & M)
            p = tp / len(P) if P else 0.0
            r = tp / len(M)
            f05s.append(f05(p, r))
            ns_precs.append(p); ns_recs.append(r)

    macro_f = float(np.mean(f05s))
    macro_p = float(np.mean(ns_precs)) if ns_precs else float("nan")
    macro_r = float(np.mean(ns_recs)) if ns_recs else float("nan")
    singleton_acc = n_singleton_correct / n_singleton if n_singleton else float("nan")

    print(f"  Macro F0.5 (official metric, {len(all_s1_ids):,} entities):  {macro_f:.4f}")
    print(f"  Non-singleton macro precision:  {macro_p:.4f}   ({n_nonsingleton:,} entities with true matches)")
    print(f"  Non-singleton macro recall:     {macro_r:.4f}")
    print(f"  Singleton accuracy:             {singleton_acc:.4f}   ({n_singleton_correct:,}/{n_singleton:,} correctly predicted empty)")
    return macro_f, macro_p, macro_r, singleton_acc

# ==============================================================================
# 6. Run the pipeline on the training data
# ==============================================================================

# --- notebook cell 17 ---
print("="*70); print("ENTITY RESOLUTION — TRAIN"); print("="*70)

s1 = load_source(os.path.join(TRAIN_DIR, "train_source1.tsv"))
s2 = load_source(os.path.join(TRAIN_DIR, "train_source2.tsv"))
s3 = load_source(os.path.join(TRAIN_DIR, "train_source3.tsv"))
print(f"S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}")

gt_dict = load_gt_dict(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"))

if SAMPLE_S1:
    print(f"*** SAMPLE_S1={SAMPLE_S1}: debug subsample ***")
    gt_full = pd.read_csv(os.path.join(TRAIN_DIR, "train_ground_truth.tsv"), sep="\t", dtype=str).fillna("")
    sids = s1["entity_id"].sample(min(SAMPLE_S1, len(s1)), random_state=42).values
    s1 = s1[s1["entity_id"].isin(sids)].reset_index(drop=True)
    gt_full = gt_full[gt_full["source1_entity_id"].isin(sids)]
    matched = set()
    for mids in gt_full["matched_entity_ids"]:
        if mids.strip():
            matched.update(mids.split(","))
    bg2 = s2.sample(min(len(s2), SAMPLE_S1*5), random_state=42)
    bg3 = s3.sample(min(len(s3), SAMPLE_S1*5), random_state=42)
    s2 = pd.concat([s2[s2["entity_id"].isin(matched)], bg2]).drop_duplicates("entity_id").reset_index(drop=True)
    s3 = pd.concat([s3[s3["entity_id"].isin(matched)], bg3]).drop_duplicates("entity_id").reset_index(drop=True)
    del gt_full, bg2, bg3; gc.collect()
    print(f"  -> S1: {len(s1):,}  S2: {len(s2):,}  S3: {len(s3):,}")

s23 = pd.concat([s2, s3], ignore_index=True)
del s2, s3; gc.collect()

cand_train = run_blocking(s1, s23)
blocking_quality(cand_train, gt_dict, s1["entity_id"].values, len(s23))

feat_train = compute_features(cand_train, s1, s23)

models, iso, oof_cal, labels = train_cv(feat_train, cand_train, gt_dict, n_folds=N_FOLDS)
t_first, t_rest, oof_f05 = tune_thresholds(cand_train, oof_cal, gt_dict, s1)

for i, m in enumerate(models):
    m.save_model(os.path.join(MODEL_DIR, f"lgbm_fold{i}.txt"))
with open(os.path.join(MODEL_DIR, "iso_thr.pkl"), "wb") as f:
    pickle.dump({"iso": iso, "t_first": t_first, "t_rest": t_rest,
                "feature_names": list(feat_train.columns)}, f)

print(f"\n{elapsed()} Model training + saving done.")

# ==============================================================================
# 6a. Sanity check — v4 vectorized features vs. v3 reference loop
# ==============================================================================

# --- notebook cell 19 ---
def _reference_features_loop(idx_sample, cand, s1, s23):
    """v3-style per-pair Python loop — used only here, to verify the v4
    vectorized features reproduce identical values on a random subsample.
    Not used anywhere in the main pipeline; intentionally slow."""
    s1i  = cand["s1i"].values[idx_sample]
    s23i = cand["s23i"].values[idx_sample]
    nc1 = s1["name_core"].values[s1i];   nc2 = s23["name_core"].values[s23i]
    nt1 = s1["name_trade"].values[s1i];  nt2 = s23["name_trade"].values[s23i]
    a1  = s1["addr_norm"].values[s1i];   a2  = s23["addr_norm"].values[s23i]
    num_re = re.compile(r"\b\d+\b")

    def jac(sa, sb):
        u = sa | sb
        return len(sa & sb) / len(u) if u else 0.0
    def ov(sa, sb):
        m = min(len(sa), len(sb))
        return len(sa & sb) / m if m else 0.0
    def ngrams(s, k):
        return frozenset(s[i:i+k] for i in range(len(s)-k+1)) if len(s) >= k else frozenset()

    out = {k: [] for k in ["name_jaccard_tok", "name_overlap", "name_jac3", "name_jac4",
                            "addr_jaccard_tok", "addr_overlap", "addr_jac3",
                            "addr_num_overlap", "dba_trade_jaccard"]}
    for i in range(len(idx_sample)):
        t1, t2 = frozenset(nc1[i].split()), frozenset(nc2[i].split())
        out["name_jaccard_tok"].append(jac(t1, t2))
        out["name_overlap"].append(ov(t1, t2))
        out["name_jac3"].append(jac(ngrams(nc1[i], 3), ngrams(nc2[i], 3)))
        out["name_jac4"].append(jac(ngrams(nc1[i], 4), ngrams(nc2[i], 4)))
        ta1, ta2 = frozenset(a1[i].split()), frozenset(a2[i].split())
        out["addr_jaccard_tok"].append(jac(ta1, ta2))
        out["addr_overlap"].append(ov(ta1, ta2))
        out["addr_jac3"].append(jac(ngrams(a1[i], 3), ngrams(a2[i], 3)))
        n1s, n2s = frozenset(num_re.findall(a1[i])), frozenset(num_re.findall(a2[i]))
        out["addr_num_overlap"].append(len(n1s & n2s))
        best = out["name_jaccard_tok"][-1]
        if nt1[i]:
            best = max(best, jac(frozenset(nt1[i].split()), t2))
        if nt2[i]:
            best = max(best, jac(t1, frozenset(nt2[i].split())))
        out["dba_trade_jaccard"].append(best)
    return pd.DataFrame(out)


print(f"\n{elapsed()} ─── SANITY CHECK (v4 vectorized vs. v3 loop) ───")
_rng = np.random.RandomState(0)
_sample_n = min(5000, len(cand_train))
_idx_sample = _rng.choice(len(cand_train), size=_sample_n, replace=False)
_ref = _reference_features_loop(_idx_sample, cand_train, s1, s23)

_all_ok = True
for col in _ref.columns:
    diff = np.abs(feat_train[col].values[_idx_sample] - _ref[col].values)
    ok = diff.max() < 1e-5
    _all_ok &= ok
    print(f"  {col:20s} max_abs_diff={diff.max():.2e}  [{'OK' if ok else 'MISMATCH'}]")
print("  " + ("All vectorized features match the v3 reference loop." if _all_ok
              else "MISMATCH DETECTED -- do not trust the vectorized features until this is resolved."))

# ==============================================================================
# 6b. Validation score report — mirrors the official macro F0.5 metric
# ==============================================================================

# --- notebook cell 21 ---
pred_dict_oof = build_pred_dict(cand_train, oof_cal)
macro_score_report(gt_dict, pred_dict_oof, t_first, t_rest, s1["entity_id"].values, use_two_thresh=True)

del s1, s23, cand_train, feat_train, pred_dict_oof; gc.collect()
print(f"\n{elapsed()} Training done. OOF macro F0.5 (two-threshold): {oof_f05:.4f}")

# ==============================================================================
# 7. Inference on the test set
# ==============================================================================

# --- notebook cell 23 ---
# If you're resuming a fresh session with only the trained models on disk,
# re-run cells 1-13 (imports through the training-function definitions) first
# — those define load_source, run_blocking, compute_features, etc. The
# imports below only prevent an `os`/`pickle`/`lgb` NameError if you land on
# this cell first; they don't recreate the pipeline functions.
import os, gc, pickle
import numpy as np
import pandas as pd
import lightgbm as lgb

USE_PYARROW_ENGINE = globals().get("USE_PYARROW_ENGINE", True)  # v4 config flag, safe default on resume

def score_candidates(models, iso, feat_df):
    X = feat_df.values.astype(np.float32)
    raw = np.mean([m.predict(X, num_iteration=m.best_iteration) for m in models], axis=0)
    return iso.transform(raw)

def write_outputs(cand, scores, t_first, t_rest, s1, out_dir):
    # v4: single global sort + boundary-slice pass instead of a per-S1-entity
    # groupby/sort_values loop — identical output files, no per-group pandas
    # call overhead at full test-set scale.
    s1_ids_all = s1["entity_id"].values
    s1i = cand["s1i"].values
    s1_id_arr = cand["s1_id"].values
    s23_id_arr = cand["s23_id"].values

    order = np.lexsort((-scores, s1i))
    s1i_s, s1id_s, s23id_s, sc_s = s1i[order], s1_id_arr[order], s23_id_arr[order], scores[order]
    _, start = np.unique(s1i_s, return_index=True)
    bounds = np.append(start, len(s1i_s))

    match_dict, cand_dict = {}, {}
    for k in range(len(start)):
        lo, hi = bounds[k], bounds[k+1]
        sid = s1id_s[lo]
        ids, scs = s23id_s[lo:hi], sc_s[lo:hi]
        cand_dict[sid] = list(dict.fromkeys(ids))
        if len(ids) and scs[0] >= t_first:
            keep = ids[scs >= t_rest]
        else:
            keep = ids[:0]
        match_dict[sid] = list(dict.fromkeys(keep))

    mp = os.path.join(out_dir, "matching_results.tsv")
    cp = os.path.join(out_dir, "candidate_pairs.tsv")
    with open(mp, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sidv in s1_ids_all:
            f.write(f"{sidv}\t{','.join(match_dict.get(sidv, []))}\n")
    with open(cp, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sidv in s1_ids_all:
            f.write(f"{sidv}\t{','.join(cand_dict.get(sidv, []))}\n")

    nm = sum(1 for v in match_dict.values() if v)
    tp = sum(len(v) for v in match_dict.values())
    print(f"  Matched entities: {nm:,}/{len(s1_ids_all):,}  Total predicted pairs: {tp:,}")
    print(f"  Output: {mp}")
    print(f"  Output: {cp}")
    return mp, cp


print("="*70); print("ENTITY RESOLUTION — TEST INFERENCE"); print("="*70)

with open(os.path.join(MODEL_DIR, "iso_thr.pkl"), "rb") as f:
    saved = pickle.load(f)
models = [lgb.Booster(model_file=os.path.join(MODEL_DIR, p))
          for p in sorted(os.listdir(MODEL_DIR)) if p.startswith("lgbm_fold")]
iso, t_first, t_rest = saved["iso"], saved["t_first"], saved["t_rest"]
print(f"  Loaded {len(models)} fold models, t_first={t_first:.3f} t_rest={t_rest:.3f}")

s1t = load_source(os.path.join(TEST_DIR, "test_source1.tsv"))
s2t = load_source(os.path.join(TEST_DIR, "test_source2.tsv"))
s3t = load_source(os.path.join(TEST_DIR, "test_source3.tsv"))
print(f"S1: {len(s1t):,}  S2: {len(s2t):,}  S3: {len(s3t):,}")
s23t = pd.concat([s2t, s3t], ignore_index=True)
del s2t, s3t; gc.collect()

cand_test = run_blocking(s1t, s23t)
feat_test = compute_features(cand_test, s1t, s23t)
# guard against any train/test feature-column mismatch before scoring
feat_test = feat_test[saved["feature_names"]]
scores_test = score_candidates(models, iso, feat_test)
mp, cp = write_outputs(cand_test, scores_test, t_first, t_rest, s1t, OUTPUT_DIR)

print(f"\n{elapsed()} Inference done.")

# ==============================================================================
# 8. Inline sanity validation
# ==============================================================================

# --- notebook cell 25 ---
import os
import pandas as pd

def validate_outputs(matching_path, candidate_path, s1t_path, s2t_path, s3t_path):
    issues = []
    s1_ids = set(pd.read_csv(s1t_path, sep="\t", dtype=str)["entity_id"])
    s2_ids = set(pd.read_csv(s2t_path, sep="\t", dtype=str)["entity_id"])
    s3_ids = set(pd.read_csv(s3t_path, sep="\t", dtype=str)["entity_id"])
    valid_targets = s2_ids | s3_ids

    mdf = pd.read_csv(matching_path, sep="\t", dtype=str).fillna("")
    seen = set()
    for _, row in mdf.iterrows():
        sid = row["source1_entity_id"]
        if sid in seen:
            issues.append(f"duplicate source1_entity_id: {sid}")
        seen.add(sid)
        ids = [x for x in row["matched_entity_ids"].split(",") if x]
        if len(ids) != len(set(ids)):
            issues.append(f"duplicate ids within row for {sid}")
        bad = [x for x in ids if x not in valid_targets]
        if bad:
            issues.append(f"{sid}: invalid target ids {bad[:3]}")
    missing = s1_ids - seen
    if missing:
        issues.append(f"{len(missing)} S1 test entities missing from matching_results.tsv")

    if not issues:
        print("Inline validation: PASS (no issues found)")
    else:
        print(f"Inline validation found {len(issues)} issue(s):")
        for msg in issues[:30]:
            print(" -", msg)
    return issues

validate_outputs(
    os.path.join(OUTPUT_DIR, "matching_results.tsv"),
    os.path.join(OUTPUT_DIR, "candidate_pairs.tsv"),
    os.path.join(TEST_DIR, "test_source1.tsv"),
    os.path.join(TEST_DIR, "test_source2.tsv"),
    os.path.join(TEST_DIR, "test_source3.tsv"),
)

# ==============================================================================
# 8b. Official validator (`utils/validate_submission.py`)
# ==============================================================================

# --- notebook cell 27 ---
import importlib.util

def load_validator():
    hits = (glob.glob("/kaggle/input/**/validate_submission.py", recursive=True)
            or glob.glob(os.path.join(BASE_DIR, "**", "validate_submission.py"), recursive=True)
            or glob.glob("utils/validate_submission.py"))
    if not hits:
        raise FileNotFoundError(
            "validate_submission.py not found — expected it under the mounted "
            "dataset's utils/ folder (Kaggle) or ./utils/ (SageMaker/local)."
        )
    spec = importlib.util.spec_from_file_location("validate_submission", hits[0])
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print(f"  Loaded validator from: {hits[0]}")
    return mod

_validator = load_validator()
_errors, _warnings = _validator.validate(mp, cp, TEST_DIR, check_ids=False)

print()
for w in _warnings:
    print(f"WARNING: {w}")
if _errors:
    print(f"FAIL — {len(_errors)} issue(s) to fix before submitting:")
    for i, e in enumerate(_errors, 1):
        print(f"  {i}. {e}")
else:
    print("PASS — no blocking issues found. Safe to submit.")

# ==============================================================================
# 9. Next steps (also tracked in the methodology doc, Part 13)
# ==============================================================================

# ==============================================================================
# 10. Running this on a remote box via CLI (no Jupyter kernel required)
# ==============================================================================
