import json
from pathlib import Path

nb_path = Path("notebooks/06_stage6_test_submission_v2.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find Cell 2 (Discovery cell) and Cell 5 (FAISS cell)
for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "TEST_S3_ADDR_EMB = find_file('test_s3_addr_emb.npy')" in src:
            src = src.replace(
                "TEST_S3_ADDR_EMB = find_file('test_s3_addr_emb.npy')",
                "TEST_S3_ADDR_EMB = find_file('test_s3_addr_emb.npy')\nif TEST_S3_ADDR_EMB is None or TEST_S3_ADDR_EMB.stat().st_size < 1_000_000:\n    tmp_s3 = find_file('test_s3_addr_emb.npy.tmp')\n    if tmp_s3 and tmp_s3.stat().st_size > 1_000_000:\n        print(f'  [Auto-Recover] Found complete .tmp file: {tmp_s3}')\n        TEST_S3_ADDR_EMB = tmp_s3"
            )
            cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + [src.split("\n")[-1]]

        if "def run_gpu_faiss" in src:
            helper = """    def load_any_emb(filepath, dim=384):
        filepath = Path(filepath)
        try:
            if filepath.stat().st_size > 1000 and not str(filepath).endswith('.tmp'):
                return np.load(filepath, mmap_mode='r')
        except Exception:
            pass
        size = filepath.stat().st_size
        n_rows = size // (dim * 2)
        return np.memmap(filepath, dtype=np.float16, mode='r', shape=(n_rows, dim))

"""
            src = src.replace("    def run_gpu_faiss", helper + "    def run_gpu_faiss")
            src = src.replace("tgt_embs = np.load(tgt_path, mmap_mode='r')", "tgt_embs = load_any_emb(tgt_path)")
            src = src.replace("a2 = np.load(s2_path, mmap_mode='r')", "a2 = load_any_emb(s2_path)")
            src = src.replace("a3 = np.load(s3_path, mmap_mode='r')", "a3 = load_any_emb(s3_path)")
            src = src.replace("s1_embs = np.load(s1_path, mmap_mode='r')", "s1_embs = load_any_emb(s1_path)")
            cell["source"] = [line + "\n" for line in src.split("\n")[:-1]] + [src.split("\n")[-1]]

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Stage 6 v2 patched successfully!")
