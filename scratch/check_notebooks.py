import json
from pathlib import Path

notebooks = [
    'notebooks/01b_stage1_test_embeddings.ipynb',
    'notebooks/04_stage4_test_cross_encoder.ipynb',
    'notebooks/06_stage6_test_submission_v2.ipynb'
]

for nb in notebooks:
    p = Path(nb)
    assert p.exists(), f"Missing: {nb}"
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"PASS: {nb} ({len(data['cells'])} cells)")
