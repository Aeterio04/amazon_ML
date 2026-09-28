import json
from pathlib import Path

p = Path("notebooks/06_stage6_test_submission_v2.ipynb")
with open(p, "r", encoding="utf-8") as f:
    nb = json.load(f)

# 1. Fix Cell 2: Add TEST_TGT_NAME_EMB and TEST_TGT_ADDR_EMB
cell2_src = "".join(nb['cells'][2]['source'])
old_emb_block = """# 4. Dense Embeddings
TEST_S1_NAME_EMB = find_file('test_s1_name_emb.npy')
TEST_S1_ADDR_EMB = find_file('test_s1_addr_emb.npy')
TEST_S2_NAME_EMB = find_file('test_s2_name_emb.npy')
TEST_S2_ADDR_EMB = find_file('test_s2_addr_emb.npy')
TEST_S3_NAME_EMB = find_file('test_s3_name_emb.npy')
TEST_S3_ADDR_EMB = find_file('test_s3_addr_emb.npy')"""

new_emb_block = """# 4. Dense Embeddings
TEST_S1_NAME_EMB = find_file('test_s1_name_emb.npy')
TEST_S1_ADDR_EMB = find_file('test_s1_addr_emb.npy')
TEST_TGT_NAME_EMB = find_file('test_tgt_name_emb.npy')
TEST_TGT_ADDR_EMB = find_file('test_tgt_addr_emb.npy')
TEST_S2_NAME_EMB = find_file('test_s2_name_emb.npy')
TEST_S2_ADDR_EMB = find_file('test_s2_addr_emb.npy')
TEST_S3_NAME_EMB = find_file('test_s3_name_emb.npy')
TEST_S3_ADDR_EMB = find_file('test_s3_addr_emb.npy')
if TEST_S3_ADDR_EMB is None:
    tmp_s3 = find_file('test_s3_addr_emb.npy.tmp')
    if tmp_s3 and tmp_s3.stat().st_size > 1_000_000:
        TEST_S3_ADDR_EMB = tmp_s3"""

cell2_src = cell2_src.replace(old_emb_block, new_emb_block)
nb['cells'][2]['source'] = [line + '\n' for line in cell2_src.strip().split('\n')]

# 2. Fix Cell 4: Safe reference to all embedding variables
cell4_src = "".join(nb['cells'][4]['source'])
old_c4_check = """# Check embedding presence
HAS_NAME_EMBS = (TEST_S1_NAME_EMB is not None and (
    TEST_TGT_NAME_EMB is not None or (TEST_S2_NAME_EMB is not None and TEST_S3_NAME_EMB is not None)
))
HAS_ADDR_EMBS = (TEST_S1_ADDR_EMB is not None and (
    TEST_TGT_ADDR_EMB is not None or (TEST_S2_ADDR_EMB is not None and TEST_S3_ADDR_EMB is not None)
))"""

new_c4_check = """# Check embedding presence safely
s1_name_emb = globals().get('TEST_S1_NAME_EMB')
s1_addr_emb = globals().get('TEST_S1_ADDR_EMB')
tgt_name_emb = globals().get('TEST_TGT_NAME_EMB')
tgt_addr_emb = globals().get('TEST_TGT_ADDR_EMB')
s2_name_emb = globals().get('TEST_S2_NAME_EMB')
s2_addr_emb = globals().get('TEST_S2_ADDR_EMB')
s3_name_emb = globals().get('TEST_S3_NAME_EMB')
s3_addr_emb = globals().get('TEST_S3_ADDR_EMB')

HAS_NAME_EMBS = (s1_name_emb is not None and (
    tgt_name_emb is not None or (s2_name_emb is not None and s3_name_emb is not None)
))
HAS_ADDR_EMBS = (s1_addr_emb is not None and (
    tgt_addr_emb is not None or (s2_addr_emb is not None and s3_addr_emb is not None)
))"""

cell4_src = cell4_src.replace(old_c4_check, new_c4_check)
nb['cells'][4]['source'] = [line + '\n' for line in cell4_src.strip().split('\n')]

with open(p, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2, ensure_ascii=False)

print("Successfully fixed NameError in Cell 2 and Cell 4 of notebooks/06_stage6_test_submission_v2.ipynb")
