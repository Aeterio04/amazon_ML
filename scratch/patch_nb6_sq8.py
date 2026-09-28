import json
from pathlib import Path

nb_path = Path("notebooks/06_stage6_test_submission_v2.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find the code cell containing run_gpu_faiss
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "def run_gpu_faiss" in src:
            # Replace run_gpu_faiss with the SQ8 memory-safe implementation
            old_start = src.find("    dim = 384")
            old_end = src.find("    # Merge candidates with strict country matching")
            
            new_code = """    dim = 384
    nlist = 4096
    nprobe = 64
    
    def load_any_emb(filepath, dim=384):
        filepath = Path(filepath)
        try:
            if filepath.stat().st_size > 1000 and not str(filepath).endswith('.tmp'):
                return np.load(filepath, mmap_mode='r')
        except Exception:
            pass
        size = filepath.stat().st_size
        n_rows = size // (dim * 2)
        return np.memmap(filepath, dtype=np.float16, mode='r', shape=(n_rows, dim))

    def run_gpu_faiss(channel_name, s1_path, s2_path, s3_path):
        print(f"  Building GPU SQ8 index for {channel_name} channel...")
        t0 = time.time()
        
        a2 = load_any_emb(s2_path)
        a3 = load_any_emb(s3_path)
        n_a2 = len(a2)
        n_a3 = len(a3)
        n_target = n_a2 + n_a3
        print(f"    Targets: S2={n_a2:,}, S3={n_a3:,} | Total={n_target:,}")
        
        # 1. Scalar Quantization (SQ8): reduces index memory from 15.3 GB to 3.8 GB
        quantizer = faiss.IndexFlatIP(dim)
        cpu_index = faiss.IndexIVFScalarQuantizer(
            quantizer, dim, nlist, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
        )
        
        # 2. Train on 200k samples (100k from S2, 100k from S3)
        rng = np.random.default_rng(42)
        idx2 = rng.choice(n_a2, size=min(100000, n_a2), replace=False)
        idx3 = rng.choice(n_a3, size=min(100000, n_a3), replace=False)
        train_data = np.vstack([a2[idx2].astype(np.float32), a3[idx3].astype(np.float32)])
        
        print(f"    Training SQ8 IVF quantizer on {len(train_data):,} samples...")
        cpu_index.train(train_data)
        del train_data
        gc.collect()
        
        # 3. Transfer to GPU (fits easily in 3.8 GB VRAM)
        res = faiss.StandardGpuResources()
        res.setTempMemory(256 * 1024 * 1024)
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.nprobe = nprobe
        del cpu_index
        gc.collect()
        
        # 4. Stream S2 and S3 in 500k chunks directly to GPU
        print(f"    Adding S2 ({n_a2:,}) to GPU index...")
        for s in range(0, n_a2, 500000):
            e = min(s + 500000, n_a2)
            gpu_index.add(np.asarray(a2[s:e], dtype=np.float32))
            
        print(f"    Adding S3 ({n_a3:,}) to GPU index...")
        for s in range(0, n_a3, 500000):
            e = min(s + 500000, n_a3)
            gpu_index.add(np.asarray(a3[s:e], dtype=np.float32))
            
        del a2, a3
        gc.collect()
        
        print(f"    Index ready ({gpu_index.ntotal:,} vectors in GPU VRAM). Searching...")
        
        # 5. Batched search to keep GPU memory flat
        s1_embs = load_any_emb(s1_path)
        all_sims = []
        all_indices = []
        query_chunk = 200000
        for q in range(0, len(s1_embs), query_chunk):
            qe = min(q + query_chunk, len(s1_embs))
            q_batch = np.asarray(s1_embs[q:qe], dtype=np.float32)
            sb, ib = gpu_index.search(q_batch, TOP_K)
            all_sims.append(sb)
            all_indices.append(ib)
            
        sims = np.vstack(all_sims)
        indices = np.vstack(all_indices)
        print(f"    {channel_name} search finished in {time.time()-t0:.1f}s")
        del gpu_index, res, s1_embs, all_sims, all_indices
        gc.collect()
        return indices, sims

    name_idx, name_sim = run_gpu_faiss(\"Name\", TEST_S1_NAME_EMB, TEST_S2_NAME_EMB, TEST_S3_NAME_EMB)
    
    if HAS_ADDR_EMBS:
        addr_idx, addr_sim = run_gpu_faiss(\"Addr\", TEST_S1_ADDR_EMB, TEST_S2_ADDR_EMB, TEST_S3_ADDR_EMB)
    else:
        addr_idx, addr_sim = None, None
        
"""
            src = src[:old_start] + new_code + src[old_end:]
            cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + [src.split("\n")[-1]]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Patch applied successfully to 06_stage6_test_submission_v2.ipynb!")
