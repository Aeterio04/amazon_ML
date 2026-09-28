#!/usr/bin/env python3
"""
run_test_inference_chunked.py — Robust, Chunked Test Inference for Approach A
Amazon ML Challenge 2026: Business Entity Resolution

Solves the monolithic OOM and time-out bottlenecks by:
1. Loading saved model artifacts: `iso_thr.pkl` and `lgbm_fold*.txt`.
2. Utilizing pre-computed S2+S3 (12M records) or reading TSVs via PyArrow.
3. Processing Test S1 (1,732,544 rows) in streaming chunks of 200,000 queries.
4. Auto-checkpointing each chunk to disk — fully resumable if interrupted.
5. Emitting official `matching_results.tsv` and `candidate_pairs.tsv`.
"""

import os, sys, time, gc, glob, pickle, re, unicodedata, warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

N_CORES = os.cpu_count() or 8
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, str(N_CORES))

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import normalize as sk_normalize
from sparse_dot_topn import sp_matmul_topn
import lightgbm as lgb
from rapidfuzz import fuzz, process as rf_process
from rapidfuzz.distance import Levenshtein as Lev

warnings.filterwarnings("ignore")

# ─── PATHS & CONFIG ───
BASE_DIR = os.getcwd()
if os.path.isdir("/kaggle/input"):
    def find_dir(filename, search_root="/kaggle/input"):
        hits = glob.glob(os.path.join(search_root, "**", filename), recursive=True)
        return os.path.dirname(hits[0]) if hits else None
    TEST_DIR = find_dir("test_source1.tsv") or "/kaggle/input/dataset/test"
    MODEL_DIR = find_dir("iso_thr.pkl") or "/kaggle/input/approach_a_models"
    OUTPUT_DIR = "/kaggle/working/output"
else:
    TEST_DIR = os.path.join(BASE_DIR, "data", "student_resource", "dataset", "test")
    MODEL_DIR = os.path.join(BASE_DIR, "models")
    OUTPUT_DIR = os.path.join(BASE_DIR, "output", "approach_a_inference")

CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

BLOCK_TOP_K = 50
BLOCK_MIN_SIM = 0.12
CHUNK_SIZE_S1 = 200_000   # Outer streaming chunk size
SPARSE_CHUNK_SIZE = 12_000  # Inner dot-product chunk size

# ─── NORMALIZATION ───
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
_punct_re = re.compile(r"[^a-z0-9\s]")
_ws_re    = re.compile(r"\s+")

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
    parts = re.split(r"\bdba\b", t)
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
    t = re.sub(r"#+(\d)", r"\1", t)
    t = _punct_re.sub(" ", t)
    t = _ws_re.sub(" ", t).strip()
    out = [STREET_ABBR.get(tok, US_STATES.get(tok, tok)) for tok in t.split()]
    return " ".join(out)

def load_source(path):
    print(f"Loading {path}...")
    cols = ["entity_id", "business_name", "business_address", "country"]
    try:
        df = pd.read_csv(path, sep="\t", dtype=str, usecols=cols, engine="pyarrow").fillna("")
    except Exception:
        df = pd.read_csv(path, sep="\t", dtype=str, usecols=cols).fillna("")
    name_res = [norm_name(t) for t in df["business_name"]]
    df["name_full"] = [r[0] for r in name_res]
    df["name_core"] = [r[1] for r in name_res]
    df["name_trade"] = [r[2] for r in name_res]
    df["addr_norm"] = [norm_addr(t) for t in df["business_address"]]
    df["combined"] = df["name_core"] + " " + df["addr_norm"]
    df["country_norm"] = df["country"].map(lambda x: x.strip().lower() if x.strip() else "__unknown__")
    return df[["entity_id", "name_full", "name_core", "name_trade", "addr_norm", "combined", "country", "country_norm"]]

def build_country_buckets(s1, s23):
    s1_c = s1["country_norm"].values
    s23_c = s23["country_norm"].values
    countries = sorted(set(s1_c.tolist()) | set(s23_c.tolist()))
    unknown_s23_idx = np.where(s23_c == "__unknown__")[0].astype(np.int32)
    has_unknown = len(unknown_s23_idx) > 0
    buckets = []
    for c in countries:
        s1_idx = np.where(s1_c == c)[0].astype(np.int32)
        if len(s1_idx) == 0: continue
        if c == "__unknown__":
            buckets.append({"s1_idx": s1_idx, "exact_s23_idx": np.arange(len(s23), dtype=np.int32), "shared_unknown": False})
        else:
            exact_idx = np.where(s23_c == c)[0].astype(np.int32)
            if len(exact_idx) == 0 and not has_unknown: continue
            buckets.append({"s1_idx": s1_idx, "exact_s23_idx": exact_idx, "shared_unknown": has_unknown})
    return {"unknown_s23_idx": unknown_s23_idx, "buckets": buckets}

def tfidf_block_route(s1, s23, text_col, vec, top_k, min_sim):
    s1_text = s1[text_col].values
    s23_text = s23[text_col].values
    bucket_info = build_country_buckets(s1, s23)
    buckets = bucket_info["buckets"]
    unknown_idx = bucket_info["unknown_s23_idx"]
    min_sim_strict = float(np.nextafter(np.float32(min_sim), np.float32(-np.inf)))

    unknown_mat = sk_normalize(vec.transform(s23_text[unknown_idx])) if len(unknown_idx) else None
    out_s1, out_s23, out_sim = [], [], []

    for b in buckets:
        s1_idx, exact_idx, shared_unknown = b["s1_idx"], b["exact_s23_idx"], b["shared_unknown"]
        exact_mat = sk_normalize(vec.transform(s23_text[exact_idx])) if len(exact_idx) else None
        if shared_unknown and unknown_mat is not None:
            if exact_mat is not None:
                s23_mat = sparse.vstack([exact_mat, unknown_mat], format="csr")
                s23_idx = np.concatenate([exact_idx, unknown_idx])
                order = np.argsort(s23_idx, kind="stable")
                s23_idx = s23_idx[order]
                s23_mat = s23_mat[order]
            else:
                s23_mat, s23_idx = unknown_mat, unknown_idx
        else:
            s23_mat, s23_idx = exact_mat, exact_idx

        if s23_mat is None or s23_mat.shape[0] == 0: continue
        s23_mat_T = s23_mat.T.tocsr()

        for ci in range(0, len(s1_idx), SPARSE_CHUNK_SIZE):
            chunk_g = s1_idx[ci:ci+SPARSE_CHUNK_SIZE]
            chunk_m = sk_normalize(vec.transform(s1_text[chunk_g]))
            C = sp_matmul_topn(chunk_m, s23_mat_T, top_n=top_k, threshold=min_sim_strict, sort=True, n_threads=N_CORES).tocoo()
            if len(C.row):
                out_s1.append(chunk_g[C.row.astype(np.int32)])
                out_s23.append(s23_idx[C.col.astype(np.int32)])
                out_sim.append(C.data.astype(np.float32))
    if out_s1:
        return np.concatenate(out_s1), np.concatenate(out_s23), np.concatenate(out_sim)
    return np.array([], dtype=np.int32), np.array([], dtype=np.int32), np.array([], dtype=np.float32)

def run_blocking(s1, s23, vec_name, vec_addr, vec_comb):
    n23 = len(s23)
    n_i, n_c, n_v = tfidf_block_route(s1, s23, "name_core", vec_name, BLOCK_TOP_K, BLOCK_MIN_SIM)
    a_i, a_c, a_v = tfidf_block_route(s1, s23, "addr_norm", vec_addr, max(5, BLOCK_TOP_K//2), BLOCK_MIN_SIM)
    c_i, c_c, c_v = tfidf_block_route(s1, s23, "combined", vec_comb, max(5, BLOCK_TOP_K//2), BLOCK_MIN_SIM)

    def make_key(i1, i23):
        return i1.astype(np.int64) * (n23 + 1) + i23.astype(np.int64)

    df_n = pd.DataFrame({"key": make_key(n_i, n_c), "nsim": n_v})
    df_a = pd.DataFrame({"key": make_key(a_i, a_c), "asim": a_v})
    df_c = pd.DataFrame({"key": make_key(c_i, c_c), "csim": c_v})

    cand = df_n.merge(df_a, on="key", how="outer").merge(df_c, on="key", how="outer")
    for col in ["nsim", "asim", "csim"]:
        cand[col] = cand[col].fillna(0.0).astype(np.float32)

    cand["s1i"] = (cand["key"] // (n23 + 1)).astype(np.int32)
    cand["s23i"] = (cand["key"] % (n23 + 1)).astype(np.int32)
    cand = cand.drop(columns="key")
    cand["s1_id"] = s1["entity_id"].values[cand["s1i"].values]
    cand["s23_id"] = s23["entity_id"].values[cand["s23i"].values]
    return cand

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

def compute_features(cand, s1, s23):
    s1i, s23i = cand["s1i"].values, cand["s23i"].values
    nc1, nc2 = s1["name_core"].values[s1i], s23["name_core"].values[s23i]
    nf1, nf2 = s1["name_full"].values[s1i], s23["name_full"].values[s23i]
    a1,  a2  = s1["addr_norm"].values[s1i],  s23["addr_norm"].values[s23i]
    c1,  c2  = s1["country"].values[s1i],    s23["country"].values[s23i]
    s23_ids = cand["s23_id"].values
    n = len(cand)

    feat = {}
    feat["nsim"] = cand["nsim"].values.astype(np.float32)
    feat["asim"] = cand["asim"].values.astype(np.float32)
    feat["csim"] = cand["csim"].values.astype(np.float32)
    feat["max_sim"] = np.maximum(np.maximum(feat["nsim"], feat["asim"]), feat["csim"])

    n_inter, n_sz1, n_sz2 = _binary_intersection(s1["name_core"].values, s23["name_core"].values, s1i, s23i, analyzer="word", token_pattern=r"\S+")
    feat["name_jaccard_tok"], feat["name_overlap"] = _jac_ov(n_inter, n_sz1, n_sz2)

    n3i, n3s1, n3s2 = _binary_intersection(s1["name_core"].values, s23["name_core"].values, s1i, s23i, analyzer="char", ngram_range=(3, 3))
    feat["name_jac3"], _ = _jac_ov(n3i, n3s1, n3s2)
    n4i, n4s1, n4s2 = _binary_intersection(s1["name_core"].values, s23["name_core"].values, s1i, s23i, analyzer="char", ngram_range=(4, 4))
    feat["name_jac4"], _ = _jac_ov(n4i, n4s1, n4s2)

    a_inter, a_sz1, a_sz2 = _binary_intersection(s1["addr_norm"].values, s23["addr_norm"].values, s1i, s23i, analyzer="word", token_pattern=r"\S+")
    feat["addr_jaccard_tok"], feat["addr_overlap"] = _jac_ov(a_inter, a_sz1, a_sz2)
    a3i, a3s1, a3s2 = _binary_intersection(s1["addr_norm"].values, s23["addr_norm"].values, s1i, s23i, analyzer="char", ngram_range=(3, 3))
    feat["addr_jac3"], _ = _jac_ov(a3i, a3s1, a3s2)
    num_inter, _, _ = _binary_intersection(s1["addr_norm"].values, s23["addr_norm"].values, s1i, s23i, analyzer="word", token_pattern=r"\b\d+\b")
    feat["addr_num_overlap"] = num_inter

    d1i, d1s1, d1s2 = _binary_intersection(s1["name_trade"].values, s23["name_core"].values, s1i, s23i, analyzer="word", token_pattern=r"\S+")
    d1_j, _ = _jac_ov(d1i, d1s1, d1s2)
    d2i, d2s1, d2s2 = _binary_intersection(s1["name_core"].values, s23["name_trade"].values, s1i, s23i, analyzer="word", token_pattern=r"\S+")
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

    wratio = np.zeros(n, np.float32); tsort = np.zeros(n, np.float32)
    tset = np.zeros(n, np.float32); partial = np.zeros(n, np.float32); levn = np.zeros(n, np.float32)
    addr_ts = np.zeros(n, np.float32); addr_te = np.zeros(n, np.float32)

    for i in range(n):
        wratio[i] = fuzz.WRatio(nc1[i], nc2[i]) / 100.0
        tsort[i] = fuzz.token_sort_ratio(nc1[i], nc2[i]) / 100.0
        tset[i] = fuzz.token_set_ratio(nc1[i], nc2[i]) / 100.0
        partial[i] = fuzz.partial_ratio(nc1[i], nc2[i]) / 100.0
        levn[i] = Lev.normalized_similarity(nc1[i], nc2[i])
        addr_ts[i] = fuzz.token_sort_ratio(a1[i], a2[i]) / 100.0
        addr_te[i] = fuzz.token_set_ratio(a1[i], a2[i]) / 100.0

    feat.update(dict(
        name_wratio=wratio, name_tsort=tsort, name_tset=tset,
        name_partial=partial, name_lev_norm=levn,
        addr_tsort=addr_ts, addr_tset=addr_te
    ))

    tmp = pd.DataFrame({"s1_id": cand["s1_id"].values, "s23_id": cand["s23_id"].values, "max_sim": feat["max_sim"]})
    tmp["rank_in_cand"] = tmp.groupby("s1_id")["max_sim"].rank(ascending=False, method="first")
    tmp["top_sim"] = tmp.groupby("s1_id")["max_sim"].transform("max")
    tmp["n_cand_for_entity"] = tmp.groupby("s1_id")["max_sim"].transform("size")
    tmp["reverse_rank"] = tmp.groupby("s23_id")["max_sim"].rank(ascending=False, method="first")

    feat["rank_in_cand"] = tmp["rank_in_cand"].values.astype(np.float32)
    feat["gap_to_top"] = (tmp["top_sim"] - tmp["max_sim"]).values.astype(np.float32)
    feat["n_cand_for_entity"] = tmp["n_cand_for_entity"].values.astype(np.float32)
    feat["reverse_rank"] = tmp["reverse_rank"].values.astype(np.float32)
    feat["mutual_top1"] = ((tmp["rank_in_cand"] == 1) & (tmp["reverse_rank"] == 1)).values.astype(np.float32)

    s1_nf = s1["name_core"].value_counts()
    s1_af = s1["addr_norm"].value_counts()
    feat["name_freq_s1"] = pd.Series(nc1).map(s1_nf).fillna(0).values.astype(np.float32)
    feat["addr_freq_s1"] = pd.Series(a1).map(s1_af).fillna(0).values.astype(np.float32)

    return pd.DataFrame(feat)

def main():
    print("=" * 70)
    print("APPROACH A: STREAMING TEST INFERENCE WITH CHECKPOINTING")
    print("=" * 70)

    # 1. Load trained models
    print(f"Loading models from {MODEL_DIR}...")
    with open(os.path.join(MODEL_DIR, "iso_thr.pkl"), "rb") as f:
        saved = pickle.load(f)
    iso, t_first, t_rest = saved["iso"], saved["t_first"], saved["t_rest"]
    feature_names = saved["feature_names"]

    model_files = sorted(glob.glob(os.path.join(MODEL_DIR, "lgbm_fold*.txt")))
    models = [lgb.Booster(model_file=f) for f in model_files]
    print(f"Loaded {len(models)} LightGBM folds. Thresholds: t_first={t_first:.3f}, t_rest={t_rest:.3f}")

    # 2. Load target S2 + S3
    s23_parquet = os.path.join(OUTPUT_DIR, "s23_norm.parquet")
    if os.path.exists(s23_parquet):
        print(f"Loading normalized S23 from parquet: {s23_parquet}")
        s23 = pd.read_parquet(s23_parquet)
    else:
        s2 = load_source(os.path.join(TEST_DIR, "test_source2.tsv"))
        s3 = load_source(os.path.join(TEST_DIR, "test_source3.tsv"))
        s23 = pd.concat([s2, s3], ignore_index=True)
        del s2, s3; gc.collect()
        s23.to_parquet(s23_parquet, index=False)
    print(f"Total S23 target records: {len(s23):,}")

    # Fit vectorizers on S23 once
    print("Fitting blocking vectorizers...")
    vec_name = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=200_000, sublinear_tf=True, dtype=np.float32)
    vec_name.fit(s23["name_core"])
    vec_addr = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=150_000, sublinear_tf=True, dtype=np.float32)
    vec_addr.fit(s23["addr_norm"])
    vec_comb = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), max_features=200_000, sublinear_tf=True, dtype=np.float32)
    vec_comb.fit(s23["combined"])

    # 3. Stream S1 in Chunks
    s1_full = load_source(os.path.join(TEST_DIR, "test_source1.tsv"))
    total_s1 = len(s1_full)
    n_chunks = -(-total_s1 // CHUNK_SIZE_S1)
    print(f"Total Test S1: {total_s1:,} queries across {n_chunks} chunks (chunk_size={CHUNK_SIZE_S1:,})")

    all_matches = {}
    all_cands = {}

    for chunk_idx in range(n_chunks):
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"chunk_{chunk_idx}_results.pkl")
        lo = chunk_idx * CHUNK_SIZE_S1
        hi = min(total_s1, (chunk_idx + 1) * CHUNK_SIZE_S1)
        s1_chunk = s1_full.iloc[lo:hi].reset_index(drop=True)

        if os.path.exists(ckpt_path):
            print(f"[Chunk {chunk_idx+1}/{n_chunks}] Loading from checkpoint {ckpt_path}...")
            with open(ckpt_path, "rb") as f:
                res = pickle.load(f)
            all_matches.update(res["matches"])
            all_cands.update(res["cands"])
            continue

        print(f"\n[Chunk {chunk_idx+1}/{n_chunks}] Processing rows {lo:,} to {hi:,} ({len(s1_chunk):,} entities)...")
        t0 = time.time()
        cand_chunk = run_blocking(s1_chunk, s23, vec_name, vec_addr, vec_comb)
        print(f"  Blocking done: {len(cand_chunk):,} candidate pairs in {time.time()-t0:.1f}s")

        t_f = time.time()
        feat_chunk = compute_features(cand_chunk, s1_chunk, s23)
        feat_chunk = feat_chunk[feature_names]
        print(f"  Features done in {time.time()-t_f:.1f}s")

        X = feat_chunk.values.astype(np.float32)
        raw_scores = np.mean([m.predict(X) for m in models], axis=0)
        cal_scores = iso.transform(raw_scores)
        cand_chunk["score"] = cal_scores

        match_dict, cand_dict = {}, {}
        grouped = cand_chunk.groupby("s1_id")
        for sid, grp in grouped:
            grp_sorted = grp.sort_values("score", ascending=False)
            ids = grp_sorted["s23_id"].tolist()
            scs = grp_sorted["score"].tolist()
            cand_dict[sid] = list(dict.fromkeys(ids))
            if scs and scs[0] >= t_first:
                keep = [i for i, s in zip(ids, scs) if s >= t_rest]
                match_dict[sid] = list(dict.fromkeys(keep))
            else:
                match_dict[sid] = []

        for sid in s1_chunk["entity_id"]:
            if sid not in match_dict:
                match_dict[sid] = []
                cand_dict[sid] = []

        # Checkpoint to disk
        with open(ckpt_path, "wb") as f:
            pickle.dump({"matches": match_dict, "cands": cand_dict}, f, protocol=pickle.HIGHEST_PROTOCOL)

        all_matches.update(match_dict)
        all_cands.update(cand_dict)
        print(f"  Chunk {chunk_idx+1}/{n_chunks} completed and checkpointed!")

    # 4. Write Final Output TSVs
    print("\nWriting final TSVs...")
    mp_path = os.path.join(OUTPUT_DIR, "matching_results.tsv")
    cp_path = os.path.join(OUTPUT_DIR, "candidate_pairs.tsv")

    s1_order = s1_full["entity_id"].values
    with open(mp_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in s1_order:
            f.write(f"{sid}\t{','.join(all_matches.get(sid, []))}\n")

    with open(cp_path, "w", encoding="utf-8") as f:
        f.write("source1_entity_id\tcandidate_entity_ids\n")
        for sid in s1_order:
            f.write(f"{sid}\t{','.join(all_cands.get(sid, []))}\n")

    print(f"SUCCESS: Generated {mp_path} and {cp_path}")


if __name__ == "__main__":
    main()
