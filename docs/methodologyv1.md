# Amazon ML Challenge 2026: Methodology Research Doc

Companion to the entry guide. Covers the internals of each candidate approach, step by step, scoped for a 72-hour hackathon.

Conventions used throughout:

- `S1`, `S2`, `S3` are the three sources. `e` is an S1 entity. `r` is an S2 or S3 record.
- A **pair** is `(e, r)`. `C(e)` is the candidate set for `e` (output of blocking). `M(e)` is the true match set. `P(e)` is the predicted set.
- **[Inference]** marks reasoning that is not stated in the problem statement (PS).
- **[Unverified]** marks details of a cited paper that come from general knowledge of the method and were not re-checked against the paper text. Check the paper before writing them into the methodology document.
- Time estimates are person-hours for one person, not team-hours.

---

## Part 1. Problem formalization

### 1.1 The task as a function

For each `e` in S1, output `P(e) ⊆ S2 ∪ S3`. The score is computed per entity and averaged.

Per-entity score **[Inference, from "macro" plus the singleton rule; confirm with the validation script]**:

```
if M(e) is empty:      score(e) = 1 if P(e) is empty else 0
else:                  score(e) = F0.5(P(e), M(e))
                                = 1.25 * p * r / (0.25 * p + r)
                       where p = |P ∩ M| / |P|   (0 if P empty)
                             r = |P ∩ M| / |M|
```

Consequences worth internalizing:

- If `M(e)` is non-empty and `P(e)` is empty, the score is 0 (recall is 0).
- If `M(e)` is empty and `P(e)` has one wrong record, the score is 0. A single false positive on a singleton costs the whole entity.
- If `M(e)` has 5 records and you return 4 correct ones, `p = 1`, `r = 0.8`, score = `1.25*0.8 / (0.25 + 0.8)` = 0.952. Missing one costs little.
- If you return those 4 plus 1 wrong one, `p = 0.8`, `r = 0.8`, score = 0.8. One false positive costs about four times more than one miss here.

That asymmetry is the whole design constraint: **be confident before you emit anything**.

### 1.2 Why this decomposes into stages

Full comparison is `|S1| × (|S2| + |S3|)` pairs. With tens of thousands of records per source that is hundreds of millions to billions of pairs, too many to score with any real model. So:

1. **Blocking** cuts this to a candidate set `C(e)` of maybe 10-200 records per entity. Cheap, recall-oriented.
2. **Matching** scores each candidate pair. Expensive, precision-oriented.
3. **Decision** turns scores into `P(e)`, including the option to output nothing.

Final recall is bounded by blocking recall:

```
final_recall <= blocking_recall = |{true pairs in candidate set}| / |{all true pairs}|
```

### 1.3 Where each approach plugs in

| Approach | Blocking | Matcher | Decision |
|---|---|---|---|
| A. Features + GBDT | n-gram index + address keys | LightGBM | threshold |
| B. Fellegi-Sunter | rule-based blocking | EM-estimated weights | probability threshold |
| C. Embeddings | ANN over embeddings + address keys | GBDT/MLP on similarities | threshold |
| D. Cross-encoder | borrowed from A or C | fine-tuned transformer | threshold |
| E. Cluster layer | borrowed | borrowed | cluster-level scoring and constraints |

Stages are interchangeable. This is what makes the team split workable: swap blocker or matcher between teammates late in the hackathon.

---

## Part 2. Shared foundations

Build these once, share them, and do not let each teammate reinvent them.

### 2.1 Data audit (first 2-3 hours)

Before any modeling, answer these from the training data. Each answer changes a design choice.

1. **Sizes**: rows in S1, S2, S3 (train and test). Sets the compute budget and whether an all-pairs brute-force baseline is feasible for sanity checks.
2. **Singleton rate**: fraction of S1 entities with empty match lists. Sets the base rate for abstention.
3. **Match-count distribution**: histogram of `|M(e)|`. If most non-singletons have 1-3 matches, a top-k cap works. If some have 50, it does not.
4. **Source split**: for each entity, how many matches from S2 vs S3. Tells you whether the two sources have different noise profiles and might need separate models or a source-indicator feature.
5. **Exclusivity**: does any S2/S3 record appear in more than one `M(e)`? If never, you can enforce one-to-one from the record side. If sometimes, you cannot.
6. **Noise taxonomy**: sample 100 true pairs and 100 hard negatives by eye. Tag the noise types: abbreviation, legal-suffix change, word reorder, typo, missing word, transliteration, landmark-only address, missing address, different unit or floor. The tag frequencies drive which features and normalizations matter.
7. **Field coverage**: how often are name or address empty or very short in each source? Short strings ("ABC") produce false matches and need special handling.
8. **Address structure**: is there a consistent order (building, street, locality, city, postal code)? Are there postal-code-like tokens? The PS's "region-specific patterns" hint suggests locality or postal tokens are strong, possibly the strongest, signals. **[Inference]**
9. **Duplicate structure inside S2 and S3**: how often do multiple S2 records refer to the same entity? Approach E depends on this.

### 2.2 Text normalization

Goal: make equal things look equal, without destroying information. Apply as a pure function so it is identical in train and test.

**Name normalization**

1. Unicode normalize (NFKD), strip accents, lowercase.
2. Replace punctuation with spaces, collapse whitespace.
3. Handle `&` vs `and`, and hyphen or dot splitting (`A.B.C.` vs `ABC`).
4. **Legal-form tokens** (inc, ltd, pvt, llp, llc, co, corp, and regional equivalents): do not delete blindly. Keep two versions: `name_full` and `name_core` (legal forms stripped). Features and blocking use `name_core`. Legal form can be a weak positive or negative signal, so keep it as a separate feature.
5. **Data-derived abbreviation map**: from training true pairs, align tokens and count substitutions (`traders` ↔ `trdrs`, `enterprises` ↔ `ent`). Keep substitutions above a frequency threshold. This is more reliable than a hand-written list and picks up region-specific patterns automatically.
6. **Token frequency weights**: compute IDF over all name tokens. Generic tokens ("services", "solutions", "and", "the") carry little information; rare tokens carry a lot. Use this weighting in similarity features and in blocking.

**Address normalization**

1. Same unicode/case/punctuation cleanup.
2. Standardize common street-type and directional abbreviations (rd→road, st→street, nr→near, opp→opposite, flr→floor). Again, derive from data where possible.
3. **Token typing**: tag tokens as numeric (house/plot/floor number), postal-code-like (fixed-length digit string), or word. Numeric and postal tokens are high precision: two records sharing a postal code and a house number and a rare name token are almost surely the same business. Two records at the same address with different names are the hard case.
4. **Landmark handling**: tokens after "near", "opposite", "behind", "beside" describe a landmark, not the address itself. Split the address into `core` and `landmark` parts. Match on core first, use landmark as a secondary signal. **[Inference]**: the PS says landmarks appear, but not how; inspect samples.
5. Keep the raw string too. Never overwrite the original; the methodology audit may look at it.

### 2.3 Blocking theory

Blocking is quality-controlled by two numbers (Papadakis et al. use equivalent terms, pairs completeness and reduction ratio):

```
Pair completeness (PC) = |true pairs in candidates| / |true pairs|          (recall)
Reduction ratio    (RR) = 1 - |candidates| / (|S1| * (|S2| + |S3|))         (savings)
```

You want PC near 1 with a candidate count small enough for the matcher. A rough target **[Inference]**: PC above 0.97 with 20-100 candidates per entity. Measure it on training labels every time the blocker changes.

Standard blocking families:

- **Standard (key) blocking**: candidates share an exact key (e.g., first three letters of name plus postal code). Very cheap. Fragile to typos in the key.
- **Sorted neighborhood**: sort by a key, slide a window. Tolerates small key changes, misses reordering.
- **q-gram / token inverted index**: index every token or character n-gram; candidates share enough (weighted) grams. Tolerates typos and reordering. This is the workhorse.
- **MinHash + LSH**: hash token or n-gram sets so similar sets collide. Scales to huge data; less precise than an exact inverted index at this scale.
- **Embedding + ANN**: represent each record as a dense vector; retrieve nearest neighbors. Handles semantic variants. See Approach C.

**Multiple routes, unioned.** The PS says records can group by similar name or by shared address. Run at least two independent routes, one on name and one on address, and take the union. A pair only needs to be caught by one route. The union raises PC; the candidate count grows roughly additively, which is acceptable.

Suggested minimum routes:

1. Name route: char 3-5-gram TF-IDF cosine, top-k per entity.
2. Address route: address-token TF-IDF cosine, or exact keys on (postal token, house number).
3. Combined route: cosine on `name + address` concatenated. Catches cases where each field alone is weak.

### 2.4 Evaluation harness

Write this before any model.

1. **Metric**: implement the per-entity score in 1.1 exactly. Test it with hand-built cases (singleton predicted empty, singleton predicted non-empty, partial matches).
2. **Splitting**: split by **S1 entity**, never by pair. All pairs touching an entity's matches go to the same fold. Splitting by pair leaks: the same S2 record can appear in train and validation. Use 5-fold grouped cross-validation for GBDT-type models; a single grouped hold-out for expensive neural ones.
3. **Blocking report**: PC, candidates per entity (mean, p95, max), and the list of missed true pairs. Read the misses; they tell you what the blocker does not understand.
4. **Threshold report**: score vs threshold curve, split into singleton and non-singleton entities.
5. **Error dump**: top false positives and false negatives with both records printed. This is where the most improvement comes from.

**Train/inference distribution match.** The matcher must be trained on candidate pairs produced by the same blocker used at inference. If you train on random negatives but infer on blocker-selected lookalikes, the model has never seen a hard negative and precision collapses. Always generate training negatives from `C(e) \ M(e)`.

### 2.5 The decision layer

The matcher outputs `s(e, r)`, ideally a calibrated probability. The decision rule turns scores into `P(e)`.

**Basic rule**: `P(e) = { r in C(e) : s(e, r) >= t }`.

**Why the optimal threshold is high.** F0.5 penalizes false positives more than false negatives, and singletons make any false positive fatal. A calibrated probability of 0.5 is not good enough to emit. Tune `t` on grouped validation predictions by maximizing the macro per-entity score directly. Expect `t` well above 0.5, commonly 0.7-0.9 **[Inference, verify on your data]**.

**Refinements, in order of expected payoff:**

1. **Two thresholds**: a stricter `t_first` for the top-scoring candidate of an entity, and a looser `t_rest` for additional candidates once the top one is confirmed. Reasoning: if the best candidate is not convincing, the entity is probably a singleton and emitting anything risks a zero. If the best is convincing, the remaining candidates are probably siblings of a confirmed match. **[Inference]**
2. **Margin rule**: if the top candidate for `e` scores barely above the runner-up and they look like different businesses, abstain.
3. **Record-side conflict resolution**: if the same `r` is a candidate for multiple entities and exclusivity holds (audit item 5), assign it only to its highest-scoring `e`, and drop it if the margin is small.
4. **Calibration**: for GBDT and neural scores, fit isotonic regression or Platt scaling on grouped validation predictions so thresholds mean the same thing across folds and models. Needed anyway to ensemble.

Everything above is tuned on validation. Freeze the thresholds, then run test once.

---

## Part 3. Approach A: fuzzy features + gradient boosting

Time: 10-14h to a working submission, about 24h tuned. Hardware: CPU is enough.

Idea in one sentence: build a rich set of hand-designed similarity features for each candidate pair, then let a gradient-boosted tree model learn how to combine them.

### A1. Normalize

As in 2.2. Produce `name_full`, `name_core`, `addr_core`, `addr_landmark`, plus token lists and IDF weights.

### A2. Blocking with TF-IDF and sparse matrix products

**How TF-IDF n-gram blocking works internally**

1. Split each name into character n-grams (for example 3- to 5-grams within word boundaries, `char_wb` in scikit-learn). "acme" becomes `acm`, `cme`, `acme` and so on.
2. Build a vocabulary over all records from S1, S2, S3 together. Fitting the vectorizer on unlabeled text from all sources is unsupervised and does not use labels.
3. Each record becomes a sparse vector. Each entry is `tf * idf`, where `idf = log(N / df)`. Rare n-grams get high weight; common ones ("ing", "ser") get low weight.
4. L2-normalize every vector. Then cosine similarity is just a dot product.
5. Compute `S1_matrix @ S23_matrix.T`. This sparse product only touches n-grams that occur in both records, so it is far cheaper than dense comparison. Do it in row chunks (for example 1,000 S1 rows at a time) to bound memory.
6. For each S1 row keep the top-k columns (k of 30-100) above a minimum similarity floor.

**Why char n-grams and not just words.** A typo, a dropped vowel, or a merged or split word ("Tech Nova" vs "Technova") destroys word-level matching but leaves most character n-grams intact.

**Address route.** Same machinery on address word tokens (and a second one on n-grams if addresses are heavily abbreviated). In addition, add exact-key candidates: any record sharing a postal-like token and a house-number token with `e`. This catches addresses that TF-IDF undervalues because numbers are short.

**Combined route.** Cosine on the concatenation `name_core + " " + addr_core` with separate vectorizers per field and weighted sum: `sim = a * sim_name + (1 - a) * sim_addr`. Tune `a` for PC on training labels.

**Union and dedupe** the three routes. Record which route(s) produced each pair; route membership is itself a useful matcher feature.

**Tuning k.** Plot PC against k. Pick the smallest k where the curve flattens. If PC saturates at 0.90, read the missed pairs: those are cases neither route understands (usually landmark-only addresses or heavily different names), and they set your ceiling.

### A3. Feature engineering

Group features by what they capture. A reasonable first set is 40-70 features.

**Name features**

- Exact equality of `name_full`; of `name_core`.
- Jaro-Winkler, normalized Levenshtein similarity on `name_core`.
- Token-based: token-set ratio, token-sort ratio, partial ratio (RapidFuzz implements these).
- Jaccard of token sets; Jaccard of character 3-gram sets.
- **IDF-weighted overlap**: `sum(idf(t) for t in shared) / sum(idf(t) for t in union)`. This is more informative than plain Jaccard because sharing "zenith" matters and sharing "services" does not.
- Count and max-IDF of shared rare tokens (rare = IDF above a percentile).
- First-token equality (business names are often identified by their first word).
- Acronym match: initials of one name equal the other, in either direction.
- Phonetic equality (Soundex or Metaphone of first token). Cheap, catches spelling variants.
- Length features: both lengths, absolute difference, ratio.
- Legal-form agreement flag: both have one and it matches, both have one and differs, one missing.

**Address features**

- Token Jaccard; IDF-weighted overlap; TF-IDF cosine on `addr_core`.
- Postal token: equal / different / one missing (three-state; trees handle this via categorical or missing values).
- House-number token equal, and number of shared numeric tokens.
- Locality-token overlap (last few word tokens of the core address).
- Landmark overlap, and whether either side has a landmark at all.
- Address missing on either side (flags).

**Interaction and context features**

- `name_sim * addr_sim` products (trees can learn interactions but explicit products help with small trees).
- **Address multiplicity**: how many S1 entities share this address (or its postal + number key). A shared address (a mall, a tech park) is weak evidence. High multiplicity should reduce the value of an address match, and the model can learn that if it sees the feature.
- **Name frequency**: how many S1 entities have a near-identical name. A very common name is weak evidence (chains, franchises).
- **Rank features**: rank of `r` in `C(e)` by cosine; gap between this candidate's cosine and the top candidate's; number of candidates for `e`. These tell the model how ambiguous the neighborhood is.
- **Reverse rank**: rank of `e` among all S1 entities that have `r` as a candidate. Mutual best matches (rank 1 both ways) are strongly predictive of a true match.
- **Source indicator** (S2 vs S3), since noise profiles differ.
- Blocking route flags.

### A4. Build the training set

- One row per candidate pair from the training blocker output. Label 1 if `r in M(e)`, else 0.
- Expect heavy imbalance (for example 1 positive to 30 negatives). Do not rebalance aggressively; if you do downsample, keep the hardest negatives (highest cosine) and recalibrate afterward.
- Train with grouped K-fold by S1 entity. Keep out-of-fold predictions for threshold tuning.

### A5. How LightGBM works, briefly

Gradient boosting builds an additive model `F(x) = sum_m eta * tree_m(x)`.

1. Start with a constant prediction (log-odds of the base rate).
2. Compute, for each training row, the gradient `g_i` and Hessian `h_i` of the log-loss with respect to the current score.
3. Fit a small regression tree to these (in the second-order form, split gain for a candidate split is `1/2 * [G_L^2/(H_L + lambda) + G_R^2/(H_R + lambda) - G^2/(H + lambda)] - gamma`, where G and H are sums of `g` and `h` in the left, right, and parent nodes).
4. Leaf values are Newton steps `-G/(H + lambda)`, scaled by the learning rate `eta`.
5. Add the tree, recompute gradients, repeat.

LightGBM specifics that matter here: features are **histogram-binned** (up to 255 bins), which makes split search fast; trees grow **leaf-wise** (best-first), controlled by `num_leaves` and `min_data_in_leaf`; **missing values** are handled natively by learning a default direction per split, which is useful for empty addresses.

Starting hyperparameters (adjust with grouped CV and early stopping): `learning_rate` 0.05, `num_leaves` 31, `min_data_in_leaf` 20-50, `feature_fraction` 0.8, `bagging_fraction` 0.8, up to a few thousand rounds with early stopping on the validation fold.

Alternative objective: **LambdaRank** grouped by entity, which optimizes candidate ordering per entity rather than absolute pair probability. It gives you a better top-1 but not a probability, so you still need a second stage or calibration to decide whether to abstain. Start with binary log-loss.

### A6. Decision

Calibrate out-of-fold probabilities (isotonic), then tune thresholds as in 2.5, including the two-threshold rule.

### A7. Failure modes and fixes

| Failure | Symptom | Fix |
|---|---|---|
| Chain or franchise, same name at different addresses | high name similarity, false positives | address-weighted features, name-frequency feature |
| Shared address (mall, tech park), different businesses | high address similarity, false positives | address-multiplicity feature, require some name evidence |
| Generic names | false positives | IDF-weighted overlap, rare-token count |
| Landmark-only address | address features near zero for true matches | rely on name plus landmark tokens; check that PC covers these |
| Very short names | unstable similarity | length-aware features, higher threshold when either name is short |
| Source noise differs | one source has lower precision | source indicator, per-source thresholds |

### A8. Time plan

| Hours | Work |
|---|---|
| 0-4 | shared harness, audit (Part 2) |
| 4-8 | normalization, three-route blocking, blocking report |
| 8-14 | features, first LightGBM, grouped CV, first threshold |
| 14-24 | error analysis loop, extra features, calibration, two-threshold rule |
| 24+ | ensembling with other teammates' scores, packaging |

---

## Part 4. Approach B: Fellegi-Sunter probabilistic linkage

Time: 6-10h to working, about 16h tuned. Hardware: CPU.

Idea in one sentence: for every field comparison, estimate how much more likely that agreement pattern is among true matches than among non-matches, and add up the evidence as log-odds.

### B1. The model

For a candidate pair, compute a comparison vector `gamma = (gamma_1, ..., gamma_K)`. Each `gamma_k` is a discrete **comparison level** for one comparison (for example name similarity: exact / very close / close / different).

Two parameters per level:

```
m_k(l) = P(gamma_k = l | pair is a match)
u_k(l) = P(gamma_k = l | pair is a non-match)
```

Assume conditional independence across `k` (the naive Bayes assumption). Let `lambda` be the prior probability that a random candidate pair is a match. Then the log2 match weight of a level is:

```
w_k(l) = log2( m_k(l) / u_k(l) )
```

and the pair's total weight and probability are:

```
W = log2( lambda / (1 - lambda) ) + sum_k w_k(gamma_k)
P(match | gamma) = 2^W / (1 + 2^W)
```

Reading it: a level that is common among matches and rare among non-matches (for example exact postal-code agreement) has a large positive weight. A level that is common among non-matches and rare among matches (complete name disagreement) has a large negative weight. The prior sets the starting point.

The original Fellegi-Sunter decision rule uses two thresholds: above the upper one is a match, below the lower one is a non-match, in between goes to clerical review. Here there is no reviewer, so map the middle zone to **abstain**, which is what the F0.5 metric wants anyway.

### B2. Designing comparisons and levels

Design is the main manual work. Levels must be ordered by decreasing agreement and should be roughly mutually exclusive.

**Name comparison (example)**

1. `name_core` exact match
2. Jaro-Winkler >= 0.95
3. Jaro-Winkler >= 0.88
4. IDF-weighted token overlap >= 0.6
5. Share at least one rare token
6. else

**Address comparisons (example, one comparison each)**

- Postal-like token: exact / different / null
- House number: exact / different / null
- Address token overlap: >= 0.7 / >= 0.4 / >= 0.1 / else
- Landmark token overlap: any / none / null

**Null handling.** A missing field should get its own level with `m = u` so its weight is zero: absence of information neither helps nor hurts.

**Term-frequency adjustment.** Exact agreement on a rare value is more informative than on a common one. Instead of one `u` for "exact name-token agreement", use `u(value) proportional to frequency of value in the data`. This is the same intuition as IDF, built into the probability. Splink implements this.

### B3. Estimating the parameters

There are two routes. Use both.

**Route 1: unsupervised EM (works without labels).**

- **E-step**: with current `m`, `u`, `lambda`, compute for every candidate pair `p_i = P(match | gamma_i)`.
- **M-step**: update
  ```
  m_k(l) = sum_i p_i * 1[gamma_ik = l] / sum_i p_i
  u_k(l) = sum_i (1 - p_i) * 1[gamma_ik = l] / sum_i (1 - p_i)
  lambda = mean(p_i)
  ```
- Repeat until the parameters stop changing.

Two practical details, as implemented in Splink (per the Splink documentation, general description; [Unverified] against the current version): `u` is estimated by sampling random pairs and assuming almost all of them are non-matches, which is close to true when the datasets are large; and EM for `m` is run in sessions where the candidate pairs are restricted by a blocking rule. A field used in the blocking rule cannot have its own `m` estimated in that session (every pair agrees on it), so run a second session blocked on a different field and combine.

**Route 2: supervised counting (uses the training labels).** You have labels, so estimate directly: `m_k(l)` is the fraction of true pairs at level `l`, and `u_k(l)` is the fraction of non-matching candidate pairs at level `l`. No EM needed. This is usually more accurate here. Use EM to sanity check, or as the fallback for fields with little supervision.

### B4. Scoring and decision

Sum the weights, convert to a probability, then apply the same thresholding as in 2.5. Because the conditional independence assumption is violated (name and address are correlated; several name-similarity levels are correlated with each other), the raw probabilities are overconfident. Calibrate them on grouped validation data before thresholding.

### B5. Blocking in this approach

Splink-style blocking rules are equality conditions: two records are candidates if they agree on a rule (for example same postal token, or same first name token plus same locality). Rules are OR-ed. Because equality rules are brittle to typos, add fuzzy blocking keys or reuse the TF-IDF blocker from A2 as the candidate generator, then score with FS. Since the stages are interchangeable, this is the sensible combination.

### B6. Using the library vs hand-rolling

Splink gives EM, term-frequency adjustment, and diagnostics out of the box. The hand-rolled version (comparison levels plus counting plus log-odds sum) is about 100-200 lines and has no library dependency, which helps with the reproducibility and rules review. The library is a gray area under "only the provided data"; confirm with organizers. **[Inference]** A hand-rolled FS scorer is the safer default; use Splink for quick diagnostics.

### B7. Strengths, weaknesses, and where it fits

- Strength: interpretable, fast, gives per-field weights you can read and sanity check, works with few labels.
- Weakness: discretized comparisons lose information; the independence assumption; every new similarity idea needs manual level design.
- Best use in the team: a strong low-cost baseline, and a source of **features for the GBDT in A** (total weight and per-comparison weights as columns).

### B8. Time plan

| Hours | Work |
|---|---|
| 0-4 | shared harness, audit |
| 4-8 | comparison and level design, supervised `m`/`u` counting |
| 8-12 | blocking route, scoring, calibration, thresholds |
| 12-16 | EM comparison, term-frequency adjustment, error analysis, export weights as features |

---

## Part 5. Approach C: embedding retrieval and rerank

Time: 20-28h to working, about 36h tuned. Hardware: GPU helps for the neural variant (Colab or Kaggle T4 class is enough). The subword-embedding variant runs on CPU.

Idea in one sentence: map each record to a vector so that records of the same business land close together, retrieve nearest neighbors as candidates, then rerank them with a small model.

### C1. What to embed

Three representations per record, each usable separately:

1. `name_core`
2. `addr_core` (plus landmark part as a separate string if useful)
3. A joint string such as `name | address`

Separate embeddings let you compute separate similarities and feed both to the matcher, which is more informative than one blended number.

### C2. Encoder choices, in order of risk

1. **TF-IDF + truncated SVD (LSA).** Take the sparse n-gram vectors from A2 and project to 128-256 dimensions. No neural training, CPU only, strong baseline for typo-heavy text. Weakness: no semantics beyond n-gram co-occurrence.
2. **Subword embeddings trained on the provided data** (fastText-style: a word vector is the sum of its character n-gram vectors). It is trained only on the challenge data, so it sits cleanly inside the "use only the provided data" rule, and it handles typos and unseen spellings because words share n-grams. Average the vectors over tokens, weighted by IDF, to get a record vector.
3. **Small pretrained sentence encoder, fine-tuned contrastively** (a MiniLM-class model). Strongest, but it uses externally pretrained weights. **[Inference]** This is a gray area under the rules; confirm before relying on it, and prefer 1 or 2 if organizers say no.

### C3. Contrastive fine-tuning (option 3) internals

The model is a **bi-encoder**: one shared encoder `f` maps text to a vector; similarity is cosine between two vectors. Records are encoded independently, so all vectors can be precomputed.

Encoder: subword tokenizer, transformer layers, then **mean pooling** over token outputs, then L2 normalization.

Training data: `(anchor, positive)` pairs from training labels, anchor = S1 record text, positive = a matched S2/S3 record text.

Loss, **in-batch negatives (InfoNCE)**. For a batch of `B` pairs, treat every other positive in the batch as a negative for anchor `i`:

```
L_i = - log( exp(cos(a_i, p_i) / tau) / sum_j exp(cos(a_i, p_j) / tau) )
```

`tau` is a temperature (about 0.05, equivalently a scale of about 20). Bigger batches give more negatives per step and generally a better embedding space.

**Hard negatives.** In-batch negatives are mostly easy (random businesses). Add, for each anchor, one or two hard negatives taken from the blocker's high-similarity non-matches (`C(e) \ M(e)`). These are the lookalikes the model must learn to push away. Append them to the denominator of the loss.

**Data amplification.** Generate extra synthetic positives by corrupting S1 records: drop a token, abbreviate a word, inject a typo, reorder tokens. Also useful for S1 singletons, which have no labeled positive: a noisy copy of a singleton is a valid positive for itself. This is the same self-supervised idea DeepBlocker uses **[Unverified in detail]**.

Cost: a few epochs on tens of thousands of pairs at short sequence length (under 64 tokens) is on the order of tens of minutes on a T4. Measure, do not trust this number.

### C4. Retrieval with ANN

Given precomputed vectors, retrieve for each S1 vector its top-k neighbors among S2/S3 vectors.

- **Exact inner-product search** (FAISS `IndexFlatIP` or plain matrix multiply on normalized vectors). If `|S1| * |S23|` is up to about 10^9-10^10 dot products of 256 dims, this is feasible on a GPU and just about on CPU in chunks. Try exact search first: it has no recall loss to tune.
- **IVF (inverted file)**: cluster vectors with k-means into `nlist` cells; at query time search only the `nprobe` closest cells. Faster, recall depends on `nprobe`.
- **HNSW**: a layered proximity graph; queries walk the graph greedily. Very fast, high recall, more memory.

Report PC as a function of `k`. Union the embedding candidates with the address-key candidates from A2, since embeddings alone can blur numbers (see C7).

### C5. Rerank

For each candidate pair compute: cosine on name embeddings, cosine on address embeddings, cosine on joint embeddings, rank and gap features as in A3, plus the hand features from A3. Train a LightGBM on these (the same machinery as A5). Alternative: an MLP over `[|u - v|, u * v]` of the two joint embeddings (the classification head used in Sentence-BERT-style training).

### C6. Relation to DeepBlocker

The DeepBlocker paper explores exactly this design space for blocking: tuple embeddings (built by aggregating word embeddings or by a sequence model), self-supervised training that needs no labels, and nearest-neighbor retrieval. Its reported results, per the abstract: eight label-free solutions, the best ones beat prior deep-learning and non-deep-learning blockers on dirty and textual data, and a hybrid of the two families did better still. Take from it: (a) a label-free embedding is a legitimate blocker, (b) always union with a non-neural route. The internals of individual variants (autoencoder architecture, cross-tuple training) are **[Unverified]** here; read Sections 3-4 of the paper before describing them in the methodology document.

### C7. Failure modes

- **Numbers blur.** Embeddings can treat "12" and "21", or two similar postal codes, as nearly the same. Never rely on the embedding alone for house numbers and postal tokens; keep the exact-token features and the exact-key blocking route.
- **Franchises**: same name, different branch. Name embeddings say "match"; only address evidence separates them.
- **Anisotropy and hubness**: a few vectors become nearest neighbors of many queries. Counter with the reverse-rank feature in A3.
- **Overfitting to seen entities**: if the encoder memorizes training names, validation on unseen entities (grouped split) shows it.

### C8. Time plan

| Hours | Work |
|---|---|
| 0-4 | shared harness, audit |
| 4-10 | LSA or fastText baseline retrieval, PC curves, union with address keys |
| 10-22 | contrastive fine-tune with hard negatives, exact or IVF retrieval |
| 22-32 | reranker, calibration, thresholds |
| 32-36 | ensembling or handoff of candidate set to other approaches |

---

## Part 6. Approach D: fine-tuned cross-encoder (Ditto-style)

Time: 24-36h to working, about 48h tuned. Hardware: GPU required for fine-tuning and inference at this scale.

Idea in one sentence: feed both records into one transformer together so every token of one can attend to every token of the other, then classify the pair as match or non-match.

### D1. Input serialization

Ditto turns each record into text with attribute markers and joins the pair into one sequence:

```
[CLS] [COL] name [VAL] acme robotics inc [COL] address [VAL] 12 mg road ...
[SEP] [COL] name [VAL] acme robotics [COL] address [VAL] 12 m g rd ... [SEP]
```

The task is then **sequence-pair classification**, the same architecture used for natural-language-inference fine-tuning. The Ditto abstract states this framing, and reports that a straightforward application of pretrained language models already outperforms earlier deep entity matchers, and that the added optimizations improve it further.

### D2. What happens inside

1. **Tokenization**: WordPiece or similar subwords. Unknown spellings are split into known pieces.
2. **Embeddings**: each token gets a token embedding plus position embedding plus segment embedding (which record it belongs to).
3. **Transformer layers**: each layer applies multi-head self-attention, where every token attends to all tokens in both records, then a feed-forward block. Attention lets the model learn alignments such as "rd" with "road", or an initial with a full word, without anyone writing that rule.
4. **Classification**: the final hidden state of `[CLS]` goes through a linear layer and softmax over two classes; the loss is cross-entropy.

**Bi-encoder vs cross-encoder trade-off.** The cross-encoder compares at token level, so it is usually more accurate on ambiguous pairs. It cannot precompute anything: each pair costs a full forward pass. So it must run only on blocked candidates, never on all pairs.

### D3. Ditto's three optimizations, adapted

From the abstract: domain-knowledge injection by highlighting important input pieces, summarization of strings that are too long, and data augmentation to create harder training examples. How to adapt each **[details are Unverified; check the paper]**:

1. **Domain knowledge**: wrap or tag high-value tokens, such as postal-like tokens and house numbers, with marker tokens so the model attends to them. This directly addresses the number-blur problem.
2. **Summarization**: if serialized pairs exceed the max length, keep the tokens with the highest TF-IDF weight. Addresses can be long; set max length to 64-128 subword tokens and measure how often truncation happens.
3. **Augmentation**: create harder positives by span deletion, token or span shuffling, attribute deletion (drop the address, keep the name), and swapping the order of the two records. Adds robustness to missing fields and ordering noise.

### D4. Training set

- **Positives**: true pairs present in the candidate set.
- **Hard negatives**: `C(e) \ M(e)`, especially the highest-scoring lookalikes from the shared blocker.
- **Singleton candidates as negatives**: essential. Their top candidates are exactly the lookalikes the model must reject to earn the singleton score.
- Sample a ratio such as 1 positive to 3-10 negatives per epoch; keep the validation set at the natural candidate ratio so thresholds transfer.
- Grouped split by S1 entity.

Typical fine-tuning settings (general practice, not from the paper): AdamW, learning rate 2e-5 to 5e-5, batch size 32, 3-8 epochs, 10% warmup, mixed precision. Model: DistilBERT or MiniLM class for speed. If the data contains non-Latin scripts, use a multilingual small model. Weights are externally pretrained; same rules caveat as C2.

### D5. Inference cost and cascading

Number of forward passes equals the number of candidate pairs. On a T4 with fp16 and short sequences, a distilled model handles very roughly a couple of thousand pairs per second **[measure this; do not plan on it]**. To cut cost, cascade: score everything with the cheap GBDT from A, and run the transformer only on pairs above a low GBDT threshold. Because the cascade keeps everything the cheap model finds plausible, it costs little recall and saves most of the compute.

### D6. Calibration and stacking

Softmax probabilities from fine-tuned transformers tend to be overconfident. Apply temperature scaling (divide logits by a scalar `T` fit on grouped validation data to minimize log-loss) or isotonic regression. Then feed the transformer probability as one feature into the GBDT from A alongside the hand-built features, which include reliable exact-number features. This stacked model is usually stronger than either alone. To avoid leakage, the transformer feature for training rows must be **out-of-fold**, which means training the transformer K times, or at minimum using a held-out split for the stacker. Given 72 hours, use a single grouped hold-out for the stacker.

### D7. Failure modes

- **Number handling** (see D3).
- **Memorization**: high training accuracy, weaker on unseen entities. Grouped validation reveals it; augmentation and early stopping help.
- **Slow iteration**: each experiment costs a training run. Freeze data and hyperparameters early, and use a small subset for debugging.
- **Label noise** in the training matches will show as confident false positives in error analysis; inspect them before assuming the model is wrong.

### D8. Time plan

| Hours | Work |
|---|---|
| 0-8 | shared harness, audit, take the shared blocker (from A2) |
| 8-14 | serialization, dataset builder, first fine-tune on a subset |
| 14-28 | full training with hard negatives and augmentation, grouped validation |
| 28-36 | cascade inference, calibration |
| 36-48 | stack with A's features, thresholds |

---

## Part 7. Approach E: cluster-level (collective) matching

Time: 16-24h as a layer on top of a pair scorer. It cannot start until some approach (A, B, C, or D) produces a table of `(e, r, score)`.

Idea in one sentence: S2 and S3 records are fragments of the same businesses, so decide at the level of groups of fragments, letting confident fragments vouch for weaker ones and letting unsupported fragments be rejected.

### E1. Why pairwise scoring is not enough

Pairwise scoring treats `(e, r1)` and `(e, r2)` independently. But if `r1` and `r2` are obviously the same business, then their entity assignments should agree. Evidence should flow between them. **[Inference]**

Two gains:

- **Recall without precision loss**: a fragment with a mediocre score to `e`, but strongly tied to a fragment confidently matched to `e`, can be included.
- **Precision**: a lone fragment that scores high against `e` but has no support from its own cluster is riskier, and is more likely a lookalike.

### E2. Step 1: cluster S2 and S3 records among themselves

1. **Within-set blocking**: run the same blocker on `(S2 ∪ S3)` against itself to get candidate pairs.
2. **Pair scorer for record-record pairs.** Labels come for free from the training data: two records are positive if they belong to the same `M(e)`. Negatives: pairs from different entities' match sets. Note that records belonging to no `M(e)` have no reliable label, so avoid using them as certain negatives unless the audit shows they are truly distinct businesses. **[Inference]**
3. **Clustering algorithm.** Options, from riskiest to safest:
   - **Connected components** at a threshold. Fast, but suffers from **chaining**: A~B and B~C merges A and C even when A and C differ. A single false link can merge two businesses.
   - **Average-linkage agglomerative clustering** with a distance cutoff. A cluster merges only if the average similarity across all cross pairs is high, which resists chaining.
   - **Mutual-kNN graph then components**: keep an edge only when each is among the other's top neighbors.
   Given the F0.5 penalty on false merges, use average-linkage or mutual-kNN with a conservative cutoff.

### E3. Step 2: score `(entity, cluster)` pairs

For an entity `e` and cluster `K` of records, aggregate the existing pair scores `s(e, r)` for `r in K`:

- max, mean, min of `s(e, r)`
- number and fraction of members with `s(e, r)` above a threshold
- cluster size, and cluster coherence (mean within-cluster similarity)
- name and address similarity between `e` and a **cluster representative** (for example the member with the highest average within-cluster similarity, or a token-union "centroid" record)

Train a second-stage model (small GBDT) to predict "cluster `K` belongs to `e`", with labels from training data and grouped splits. The predicted set `P(e)` is then the union of accepted clusters, optionally with individual members filtered by their own score.

### E4. Step 3: conflicts and constraints

If the audit shows a record can belong to at most one entity (exclusivity), enforce it. Build a bipartite graph between entities and clusters with edge weights equal to the cluster-level score, and solve a maximum-weight assignment, subject to a minimum score for an edge to exist. `scipy.optimize.linear_sum_assignment` (Hungarian-style) solves this. Run it per connected component of the candidate graph, not globally, since most components are tiny.

If exclusivity does not hold, use the simpler rule: accept `(e, K)` when its score exceeds the threshold, with the margin rule from 2.5.

### E5. Consistency checks

- If `r1` is in `P(e)` and `r2` is very strongly the same business as `r1`, but `r2` scores very low with `e`, flag the conflict: either drop both or send the pair to the stricter threshold. Conflicts often reveal a wrong cluster.
- Cap cluster sizes at a plausible maximum from the training match-count histogram (audit item 3). An oversized cluster is likely a chained merge.

### E6. Failure modes

- **Amplified errors.** One wrong cluster contributes several false positives at once, and a false positive on a singleton zeroes it. The cutoff must be conservative, and the cluster-level score must be trained on realistic (blocker-produced) data.
- **Circularity.** The record-record scorer and the entity-record scorer share features; check that validation is grouped so that neither sees the labels of held-out entities.
- **Sparse gains.** If audit item 9 shows few entities have multiple fragments, this layer buys little. Check before investing.

### E7. Time plan

| Hours | Work |
|---|---|
| 0-8 | audit item 9, within-set blocker, record-record labels from training matches |
| 8-16 | record-record scorer, clustering with average linkage, cluster-quality report |
| 16-22 | cluster-level features and second-stage model |
| 22-24 | assignment constraint, ablation against the base pair scorer |

---

## Part 8. Combining approaches

The team's decision rule is "whichever works best gets submitted." A hybrid usually beats every member, and the pieces are interchangeable if the interfaces are fixed early.

### 8.1 Fixed interfaces

Agree on these in the first hours:

- `candidates.parquet`: `e_id, r_id, source, route_flags`
- `scores_<method>.parquet`: `e_id, r_id, p_calibrated` (out-of-fold for training rows)
- `oof` versus `test` naming, and a common list of fold assignments by `e_id`

With this, anyone's blocker or scorer plugs into anyone else's pipeline.

### 8.2 Blocking union

Candidate sets from different teammates' blockers can be unioned to raise PC. The final `candidate_pairs.tsv` for the archive should be this union and must be reproducible from code. Keep the candidate count in check: the union feeds the expensive matchers.

**Dependency note.** Approaches D and E need a candidate set early. Have one person build the A2 blocker and publish it by roughly hour 8, so the D and E owners are not blocked.

### 8.3 Ensembling scores

- **Weighted average of calibrated probabilities**, or of logits, with weights tuned on the grouped validation set. Simple, robust.
- **Stacking**: a small GBDT or logistic regression on the out-of-fold scores of each method plus a few meta-features (rank, gap, candidate count). Stronger, but it needs out-of-fold scores for every base model. If a base model has none, use a single grouped hold-out for the stacker instead.

Then apply the decision layer from 2.5 once, on the final score.

---

## Part 9. 72-hour team schedule

| Hours | Team-level goal |
|---|---|
| 0-4 | shared harness, audit, interfaces agreed; each person reads their approach |
| 4-36 | parallel implementation, each person to a scored `scores_<method>.parquet` on the shared folds |
| 36-44 | compare on the same folds and the same metric; pick the top pieces |
| 44-58 | combine (blocking union, ensemble, cluster layer), tune the decision layer |
| 58-68 | one end-to-end run from raw data to `matching_results.tsv`; write `candidate_pairs.tsv`; validation script |
| 68-72 | buffer, methodology document, packaging, final submission |

The archive components (`candidate_pairs.tsv`, runnable pipeline, methodology document) are required at the close of the challenge. **[Open]** Whether they are due at the 72h mark or later is not in the PS; check. If they are due with the final submission, start the methodology document at hour 44, not hour 68.

---

## Part 10. Pitfalls checklist

1. **Leakage**: any split that is not grouped by S1 entity. Also fitting anything supervised (abbreviation maps from matched pairs, target encoding) on validation rows.
2. **Train/inference mismatch**: negatives not drawn from the same blocker used at inference.
3. **Threshold overfit**: with few entities per fold, the optimal threshold is noisy. Use out-of-fold predictions across all folds together to pick one threshold, and prefer the flat region of the curve over the sharpest peak.
4. **Distribution shift**: the test singleton rate and noise level may differ from training. **[Inference]** Compare the distribution of top-candidate scores on train and test; a large shift means thresholds need to be more conservative.
5. **Reproducibility**: fixed seeds, pinned `requirements.txt`, one command that runs everything, no downloads at runtime beyond what the rules allow, no reliance on files outside the archive.
6. **Rules compliance**: no external data, APIs or lookups anywhere. Decide early, in writing, whether pretrained weights and third-party libraries are acceptable, and record the answer in the methodology document.
7. **Format**: run the provided validation script on every submission file.
8. **Audit trail**: `candidate_pairs.tsv` must be written at the blocking stage, before the matcher filters, not reconstructed afterward.
9. **Empty predictions are valid output.** Make sure the writer emits a row with an empty list for every abstained entity, in the required format. Check the validation script for the exact form.

---

## Part 11. Reading notes for the four papers

What was checked (from paper abstracts and metadata) and what was not.

1. **Ditto: Deep Entity Matching with Pre-Trained Language Models** (Li, Li, Suhara, Doan, Tan; PVLDB 14(1), 2021; arXiv 2004.00584). Checked: sequence-pair classification framing, the three optimizations (domain-knowledge highlighting, summarization, augmentation), reported 96.5% F1 on company datasets of 789K and 412K records. Read in full for: exact serialization, the augmentation operators, and hyperparameters. Feeds Approach D.
2. **Deep Learning for Blocking in Entity Matching: A Design Space Exploration** (Thirumuruganathan et al.; PVLDB 14(11), 2021; DeepBlocker). Checked: eight label-free solutions, best ones beat prior methods on dirty and textual data, hybrid of DL and non-DL blocking performs better. Read in full for: the architectures and training objectives. Feeds Approach C and the blocking-union design.
3. **Splink: Free software for probabilistic record linkage at scale** (Linacre et al.; Int. J. Population Data Science 7(3), 2022; doi 10.23889/ijpds.v7i3.1794). Checked: EM estimation of a Fellegi-Sunter model building on FastLink's implementation, no training data required. Read for: term-frequency adjustment and the blocking-rule handling during EM. Foundational reference for the model itself: Fellegi and Sunter (1969), which was not fetched here. Feeds Approach B.
4. **Blocking and Filtering Techniques for Entity Resolution: A Survey** (Papadakis, Skoutas, Thanos, Palpanas; ACM Computing Surveys 53(2), 2020). Checked: bibliographic details only. Use it as the catalogue of blocking methods and the definition of PC and RR when writing the methodology document.

---

## Part 12. Open questions that affect design

- Are pretrained model weights and third-party libraries (Splink, RapidFuzz, LightGBM, FAISS) permitted under "only the provided data"? Likely yes for libraries. Weights are less clear. Ask.
- Exclusivity of S2/S3 records (audit item 5). Decides whether the assignment constraint in E4 is valid.
- Script and language of the records (audit item 6). Decides normalization, phonetic features, and the encoder choice in C and D.
- Whether the archive is due at the deadline or afterward (Part 9).
- Exact scoring definition for entities with matches but partial predictions: confirm the per-entity F0.5 in 1.1 against the scoring notes or the validation script.
