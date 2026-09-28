# Amazon ML Challenge 2026: Business Entity Resolution

Entry guide. Source: the official problem statement walkthrough. Anything not stated there is marked as an inference or an open question.

---

## 1. One-line summary

Given business records from three independent, noisy sources, decide which records describe the same real-world business. For every Source 1 entity, output the list of matching record IDs from Sources 2 and 3 (possibly empty).

---

## 2. Background

- Scenario: a business signs up on Amazon Business. Name and address are captured at signup.
- To enrich that business, extra data is pulled from other vendors. Each vendor has its own formats and conventions.
- These external sources share **no common identifier** with Amazon's record.
- The challenge deliberately restricts data to **business name and address**. Those are the only usable fields.

---

## 3. The core difficulty

The same business is written differently in each source. Examples from the PS:

- Name variants: "Acme Robotics Inc." vs other forms of the same name.
- Address abbreviations.
- Addresses referenced via a nearby landmark instead of a full address.

So neither exact name match nor exact address match is reliable.

---

## 4. The three sources

| Source | Role | Quality |
|---|---|---|
| Source 1 | Reference list | Clean, deduplicated |
| Source 2 | To be reconciled against Source 1 | Noisy fragments |
| Source 3 | To be reconciled against Source 1 | Noisy fragments |

Every record has: an **ID**, a **noisy name**, and an **address**.

Matching direction: every prediction is anchored on a Source 1 entity. You are not asked to link Source 2 to Source 3 directly.

---

## 5. Match cardinality

For each Source 1 entity, the number of matches across Sources 2 and 3 can be:

- **Many** records
- **Exactly one** record
- **None** (a *singleton*)

The submission must handle all three cases.

---

## 6. Pipeline the PS describes

### Step 1: Blocking

- Comparing all Source 1 records against all Source 2 and 3 records is too expensive at scale.
- Blocking sorts records into buckets using a cheap key built from **both name and address**, so likely matches land in the same bucket.
- Records can group either through a similar name or through a shared address.
- Records in a block become the **candidate matches** for the entity.
- Blocking is tuned for **recall**. Buckets will contain lookalikes:
  - a business with a similar name at a different address
  - a different business that shares an address
- Output of this stage: **candidate pairs**.

### Step 2: Matching model

- A model scores each candidate pair and keeps only true matches, discarding the rest.
- Output of this stage: **final matching results**.

### Why blocking comes first

You cannot match a record you never considered. Blocking recall is the **ceiling** on final recall. The PS explicitly says to invest here first.

---

## 7. Data

Two datasets:

**Training set**
- Records from all three sources.
- Ground truth labels file.

**Test set**
- Same three sources, no labels.
- This is what predictions are generated for.

### Ground truth / label format

- One row per Source 1 entity.
- The Source 1 ID maps to a **comma-separated list** of all matching IDs (from Sources 2 and 3).
- The list is **empty** when the entity matches nothing.
- This is the same format you submit.

### File format

- All files are **tab-separated (TSV)**.
- Read with an explicit tab separator, otherwise columns will not parse. In pandas: `pd.read_csv(path, sep="\t")`.

---

## 8. Deliverables

### During the challenge (leaderboard)

- Upload a single file: `matching_results.tsv`
- One row per Source 1 entity with its predicted matches.
- This is the **only file scored** on the leaderboard.

### At the close of the challenge (final archive)

One archive containing:

1. `matching_results.tsv` (final matches)
2. `candidate_pairs.tsv` (output of your blocking stage, before the model narrowed it down)
3. Complete, runnable pipeline
4. Methodology document describing the approach

Notes:
- `candidate_pairs.tsv` is **not scored**, but is used to **audit blocking quality**.
- Packages of top teams are **reviewed in detail** before final rankings are confirmed. Reproducibility matters.
- Run the **provided validation script** before every submission so a formatting error does not cost a submission.

---

## 9. Scoring

- Metric: **macro F0.5**.
- F0.5 weights precision twice as heavily as recall.
- Practical meaning: wrongly merging two different businesses costs roughly twice as much as missing a true match. **When in doubt, do not merge.**

Formula for reference:

```
F0.5 = (1 + 0.5^2) * P * R / (0.5^2 * P + R)
     = 1.25 * P * R / (0.25 * P + R)
```

### Singleton scoring (critical)

| Situation | Predicted | Score for that entity |
|---|---|---|
| Entity has no true match | Empty list | **1** |
| Entity has no true match | Any match | **0** |

Correctly identifying no-match entities is worth as much as finding matches for the others. A model that always finds something to match will be punished on singletons.

Inference (not stated explicitly): "macro" plus the per-entity singleton rule implies the score is computed per Source 1 entity and then averaged over entities. Confirm against the validation script or scoring notes if available.

---

## 10. Rules

- **Pure ML challenge.** External databases, APIs, and lookups are **strictly prohibited**.
- Use **only the provided data**.

---

## 11. Recommendations from the PS

1. **Understand singletons.** Predicting empty correctly earns a full 1; any wrong match earns 0.
2. **Invest in blocking first.** It sets the recall ceiling.
3. **Look for region-specific patterns** in both names and addresses.

---

## 12. What follows from the above (inferences)

These are derived from the PS, not stated in it.

- Precision-leaning decision threshold. F0.5 plus the singleton rule both push toward abstaining under uncertainty.
- Blocking needs at least two independent routes (name-based and address-based), since records can group by either. The PS says the block key is built from both.
- Blocking should be measured on its own: recall of true pairs within the candidate set, and candidate set size. Both are visible in the training labels.
- Validating on the training set with the same metric is possible because labels are provided. Singleton rate in training is worth checking early.
- Region-specific hint suggests inspecting the address formats and name conventions in the data before choosing normalization.

---

## 13. Open questions to resolve from the data / validation script

- Exact column names and schema of each source file.
- Exact required column names and ordering for `matching_results.tsv` and `candidate_pairs.tsv`.
- Whether a Source 1 row's match list mixes Source 2 and Source 3 IDs in one field (the PS says one comma-separated list carries the full match set, so likely yes) and how IDs are prefixed to tell them apart.
- Whether a Source 2 or 3 record can match more than one Source 1 entity, or is exclusive.
- Size of each source, and the proportion of singletons in training.
- Submission limits and leaderboard rules (not covered in the PS video).

---

## 14. Checklist

- [ ] Load all files with `sep="\t"`
- [ ] Inspect schema, sizes, singleton rate in training labels
- [ ] Build blocking, measure recall of true pairs and candidate count
- [ ] Build matcher, evaluate with macro F0.5 on training
- [ ] Tune threshold toward precision, verify singleton handling
- [ ] Generate `matching_results.tsv`, run validation script
- [ ] Save `candidate_pairs.tsv`
- [ ] Keep pipeline runnable end to end, write methodology doc
- [ ] Confirm no external data or APIs anywhere in the pipeline
