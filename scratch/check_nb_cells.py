import json
from pathlib import Path

nb_path = Path("notebooks/06_stage6_test_submission_v2.ipynb")
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print(f"Total cells in notebook: {len(nb['cells'])}")

# Cell 7 is cell index 6 (0-indexed) or second to last
# Let's inspect cell types and headings
for i, c in enumerate(nb['cells']):
    src = "".join(c.get('source', []))
    first_line = src.split('\n')[0] if src else ""
    print(f"Cell {i}: {first_line[:70]}")

# Cell 6 is index 6, Cell 7 is index 7
