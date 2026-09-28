import json
from pathlib import Path

p = Path('notebooks/06_stage6_test_submission_v2.ipynb')
with open(p, 'r', encoding='utf-8') as f:
    nb = json.load(f)

cell5 = nb['cells'][5]
src = ''.join(cell5['source'])

src = src.replace('query_chunk = 200000', 'query_chunk = 10000')
src = src.replace('res.setTempMemory(256 * 1024 * 1024)', 'res.setTempMemory(512 * 1024 * 1024)')

old_train = """        rng = np.random.default_rng(42)
        idx2 = rng.choice(n_a2, size=min(100000, n_a2), replace=False)
        idx3 = rng.choice(n_a3, size=min(100000, n_a3), replace=False)
        train_data = np.vstack([a2[idx2].astype(np.float32), a3[idx3].astype(np.float32)])"""

new_train = """        print(f"    Sampling 200k vectors for IVF quantizer training...")
        s2_step = max(1, n_a2 // 100000)
        s3_step = max(1, n_a3 // 100000)
        t_s2 = np.asarray(a2[0 : 100000 * s2_step : s2_step], dtype=np.float32)
        t_s3 = np.asarray(a3[0 : 100000 * s3_step : s3_step], dtype=np.float32)
        train_data = np.vstack([t_s2, t_s3])
        del t_s2, t_s3"""

src = src.replace(old_train, new_train)

old_del = """        del gpu_index, res, s1_embs, all_sims, all_indices
        gc.collect()
        return indices, sims"""

new_del = """        del gpu_index, res, s1_embs, all_sims, all_indices
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        return indices, sims"""

src = src.replace(old_del, new_del)
src = src.replace("del name_idx, name_sim; gc.collect()", "del name_idx, name_sim, addr_idx, addr_sim; gc.collect()")

cell5['source'] = [line + '\n' for line in src.split('\n')]
if cell5['source'] and cell5['source'][-1] == '\n':
    cell5['source'].pop()

with open(p, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Successfully updated 06_stage6_test_submission_v2.ipynb")
