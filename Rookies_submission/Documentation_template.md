# ML Challenge 2026: Business Entity Resolution Solution

**Team Name:** Rookies
**Team Members:** Swaraj, Subhajit, Tanmay, Abhishek
**Submission Date:** 2 October 2026

---

## 1. Executive Summary

Our solution is a **recall-first blocking pipeline followed by metric-aware assignment**.
Nine DuckDB blocking schemes (name, phonetic, postcode and address keys, each restricted to
an entity's rarest keys and joined only within its country) propose the top 100 candidates
per S1 entity. A multilingual sentence encoder, fine-tuned on our own training matches, adds
the matches that no spelling-based key can reach. A two-stage LightGBM scores the pairs. Each
S1 entity's match set is then chosen to maximize its **exact expected F0.5**, and a global
repair step gives every S2/S3 record at most one owner. Final 5-fold cross-validated macro
F0.5: **0.9431** on 33,128 held-out training entities.

---

## 2. Methodology

### 2.1 Problem Analysis

| EDA finding (training data) | What it meant for us |
|---|---|
| 2.21M S1 entities against a 10.3M-record S2+S3 pool; 3.46 true matches per entity on average (max 11); 5.6% singletons | All-pairs comparison (≈ 2×10¹³ pairs) is impossible. **Blocking recall is a hard ceiling**: an entity whose matches are never proposed scores 0 whatever the model does. |
| **Matches are exclusive.** All 7.64M matched ids are unique, so a pool record belongs to at most one S1, and 74% of the pool is owned. | This is an assignment problem. The final step enforces one owner per record. |
| F0.5 weighs precision ×2, and every entity counts equally | One wrong guess on a small entity costs about as much as missing its whole set. The set size must be decided per entity, and "predict nothing" has to be a real option. |
| **Noise patterns.** Nine Indic scripts. Garbled romanizations: `grin lajistiks` = green logistics, `yunaited vencars` = united ventures, `vijaya siva kansaltansi` = vijay shiv consultancy. OCR digits: `5hakti`, `f0od`. Typos. | We need transliteration, digit-in-word repair, sound-alike keys, and a learned similarity that reads native scripts. |
| **Name variations.** Legal-suffix variants (`pvt ltd` / `private limited` / `S.A.R.L.`), website-style names (`acme.com`), `formerly X` / `d/b/a` aliases, renamed businesses at the same address | Suffixes are canonicalized and kept out of the core name, and domain tokens are squashed. Some schemes key on the address alone. |
| **Address variations and missing fields.** Abbreviations (`rd`, `st`, `bd`), house numbers, 6-digit PIN or 5-digit ZIP codes, and many records without any address | Addresses are parsed into words, numbers and postcode. A 5-digit US house number can parse as a ZIP code, so the postcode doubles as a house number in scheme G. Every feature handles a missing address. |
| **Generic and chain names.** Many S1 entities share a name (branches), and same-name pool records often belong to a different S1. | Name similarity alone over-merges. We added rival features, which ask whether another S1 with this name fits the record better. |
| **France exists only in test** (259k S1 entities, 15% of test), with zero labels | Country is an open label. Every join, document frequency and search is per country, and nothing is hard-coded to US or India. |

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Multi-scheme lexical/phonetic/address blocking ∪ fine-tuned dense
retrieval → two-stage LightGBM classifier → per-entity expected-F0.5 set selection → global
exclusivity repair.

**Core Innovation:**
1. **Fair-slot union of nine blocking schemes.** Each scheme ranks its own candidates. The
   union orders all candidates by their best rank within any scheme, so one broad scheme
   cannot crowd out the precise ones inside the top-100 cut (+0.03 validation F0.5).
2. **Embedding retrieval as a tenth "blocking scheme".** An encoder fine-tuned with a
   contrastive loss on our own matches proposes its top-10 neighbours that DuckDB never
   proposed. On held-out entities this adds **4,732 true pairs** (+4.3%) that lexical
   blocking missed.
3. **Rival-aware features.** Computed over the full S1 file, these measure whether a different
   S1 with the same name is a better owner of the record.
4. **Exact expected-F0.5 selection.** For each entity we compute the full distribution of hits
   and pick the k (including 0) that maximizes the expected metric, instead of using a
   probability threshold.

### 2.3 Pipeline and notebook responsibilities

The submission is produced by four Kaggle notebooks. Each one hands its output to the next as
a Kaggle dataset, which keeps every session within Kaggle's limits. The same code ships as
four scripts plus an `er/` package (Appendix A).

| # | Notebook (script) | Hardware | Responsibility | Output used downstream |
|---|---|---|---|---|
| 1 | `er_v9_kaggle.ipynb` (`01_main_pipeline.py`) | CPU, 4 vCPU | **The main pipeline.** Normalizes all six source files (Stage 0); builds the DuckDB pool index and runs **blocking schemes A–H** (top-100 per S1, Section 3); computes the 52 pair features; trains LightGBM #1 with isotonic calibration on a 5% entity sample of train; scores all of test in 24 resumable shards. | `lgbm.txt`, `calib.npz`, `train_candidates`, `train_features`, `test_candidates_shNN` (**the blocking set**), `test_probs_shNN`, `test_sums_shNN` |
| 2 | `ann_embed_kaggle.ipynb` (`02_embed_original.py`) | GPU T4 ×2 | Encodes every record's raw `"name \| address"` with the **original** multilingual MiniLM and runs an exact per-country top-K cosine search (top 100 for the train sample, top 50 for test). It also reports the blocking gate: DuckDB vs embeddings vs their union. | `ann/ann_train`, `ann/ann_test` (similarity score and rank per pair) |
| 3 | `ann_finetune_kaggle.ipynb` (`03_embed_finetuned.py`) | GPU T4 ×2 | **Fine-tunes** the encoder on 800k (S1, true S2/S3) training pairs, excluding the validation sample. It checks recall@1/@10 before and after, re-encodes, and runs the same per-country search. | `ann_ft/ann_train`, `ann_ft/ann_test` |
| 4 | `er_final_tuned_kaggle.ipynb` (`04_final_selection.py`) | CPU | **Final assembly.** Writes `candidate_pairs.tsv` from step 1's shards. Adds the fine-tuned encoder's top-10 extras to the DuckDB candidates. Trains LightGBM #2 on held-out probabilities plus embedding features and picks its setting by 5-fold CV. Chooses each set by exact expected F0.5, repairs exclusivity globally, and writes both TSVs. | `matching_results.tsv`, `candidate_pairs.tsv` |

Two earlier notebooks, `er_finish` and `er_final`, are not needed for reproduction. They
introduced the second-stage selector and compared blends: DuckDB alone; DuckDB rescored with
embedding similarity; DuckDB + embedding extras ("B"); and embeddings alone. Variant **B**
won, and step 4 tunes only B.

---

## 3. Candidate Generation (Blocking)

Blocking decides the ceiling of the whole system, so most of our development time went into
it. Every change to blocking was measured on the training sample against the **full** pool
(the pool is never subsampled, since a smaller pool inflates recall) before any model was
retrained.

### 3.1 Stage 0: keys that blocking can join on

Each record is normalized once into parquet (`er/normalize.py`):
- Unicode NFKC; zero-width characters removed; `&` → `and`; dotted acronyms joined (`S.A.R.L.` → `sarl`).
- **Indic → Latin transliteration** of nine scripts, run by run (IAST plus script-specific fixes and a small hand-written phrase map such as `प्राइवेट` → private). A self-test raises if any script yields an empty name.
- OCR digit repair, applied only inside words that mix letters and digits (`f0od` → food); generated `id 12345` tails removed.
- Legal forms (US, Indian, French and other European) mapped to a canonical suffix and kept out of the **core name tokens**. `name_key` = sorted unique core tokens.
- Addresses become **address words** (abbreviations expanded: `rd` → road, `bd` → boulevard), **address numbers** (house and unit), and a **postcode** (6-digit PIN or 5-digit ZIP).

### 3.2 Blocking keys used

For each split we build one **pool index** over S2+S3 in DuckDB. For every key type it stores
the key's document frequency (df) and IDF = ln(1 + N/df), **per country**. Each S1 entity then
queries the index with only its **rarest keys**, so a common word like `delhi` or `services`
never explodes the join. Every scheme ranks its own candidates (`rk`) and keeps at most its cap.

| Scheme | Key (joined within the same country) | Which S1 keys are used | Ranked by | Cap per S1 | Miss pattern it targets |
|---|---|---|---|---|---|
| **A** exact name | `name_key` (sorted core tokens) | the whole key | same postcode first | 300 | clean duplicates, word-order changes, suffix variants |
| **A2** squashed name | core tokens with web tokens (`com`, `www`, `in`, …) removed, concatenated | length ≥ 4 | id | 300 | `acme.com` vs `acme`, `blue pub` vs `bluepub` |
| **B** rare name token | single name token | the 3 rarest, df ≤ 1000 | Σ IDF of shared tokens | 100 | partial names, extra or missing words |
| **C** consonant skeleton | token with `sh→s`, `ph→f`, `c→k`, vowels removed | the 3 rarest, df ≤ 1000 | Σ IDF | 100 | vowel and spelling noise in romanizations |
| **D** postcode + name | (postcode, name token) | every token, when a postcode exists | number of shared tokens | none | same-place records with partial names |
| **E** rare address tokens | address word | the 4 rarest, df ≤ 300; **≥ 2 must be shared** | Σ IDF | 100 | renamed or rebranded businesses at the same address |
| **F** name-token pairs | `tokA\|tokB` from the 4 rarest name tokens | the 6 rarest pairs, df ≤ 2000 | Σ IDF | 100 | names made only of common words (`sai traders`) |
| **G** house number + street word | `number\|address word` (≤ 3 numbers + postcode × the 4 rarest words) | the 6 rarest keys, df ≤ 100 | Σ IDF | 20 | address-only matches with a garbled or renamed name |
| **H** name word × address word | `name tok\|address word` (3 rarest × 3 rarest) | the 6 rarest keys, df ≤ 50 | Σ IDF | 20 | names whose words are each too common alone, but rare together with the street |
| **ANN** fine-tuned embeddings | cosine of `"name \| address"` vectors, exact search | top-10 neighbours not already proposed by A–H | cosine | 10 | transliteration and garbling (`kansaltansi` → consultancy), native-script names |

The pool side builds the same keys (rarest tokens, pairs, G/H keys) once. The test pool is
indexed once and reused by all 24 shards.

### 3.3 From nine scheme lists to one candidate set: the fair-slot union

1. **Union** all scheme outputs per (S1, candidate). For each pair we keep which schemes found it (`by_a` … `by_h`), its best IDF score, the shared-key count and its **best within-scheme rank** (`best_srk`).
2. **Containment** = the pair's IDF score / the S1 name's total IDF. This is the share of the S1 name's information the candidate covers.
3. **Order** each S1's candidates by `best_srk` ascending (all schemes' #1s first, then all #2s, …), then exact or squashed name agreement, then containment, shared keys and IDF score.
4. **Cut at the top 100 per S1.**

Before the fair-slot rule we ordered candidates by one global score. A broad scheme such as B
then filled the list and pushed out the few, precise G and H hits. Ordering by best rank within
any scheme fixed that: validation F0.5 rose from 0.887 to 0.917 with the same schemes.

The ANN extras are added after the cut. They are scored by the second-stage model like every
other candidate.

### 3.4 Candidate pairs generated

| Test set | Pairs |
|---|---|
| DuckDB blocking (schemes A–H, top-100 per S1) | 155,797,826 |
| + embedding-found pairs that were selected as matches, listed in `candidate_pairs.tsv` | 158,709 |
| **Total in `candidate_pairs.tsv`** | **155,956,535** (90.0 per S1; every one of the 1,732,544 S1 rows is present) |
| Pairs scored by the second-stage model (DuckDB pairs with p₁ ≥ 0.01 + all embedding top-10 extras) | 16,347,458 |

All of A–H operate within a single country. The reduction is from the ≈ 1.7×10¹³ pairs of an
all-pairs comparison over the test set to 1.6×10⁸, a factor of about 10⁵.

### 3.5 How we ensured true matches were not lost

- **We measure recall before modelling.** Every blocking change logs, on the 5% training sample against the full pool:
  - uncapped generation recall;
  - recall@10/30/50/100/200;
  - the number of candidates per entity;
  - the **ceiling F0.5**: the score of a perfect classifier restricted to the candidates.
- **Each scheme targets a logged miss pattern.** `inspect_misses` prints true pairs that blocking missed. G and H exist because many misses shared 3–4 address words that were individually too common for scheme E. C, F and the embeddings target transliteration and common-word names.
- **Rarity filters work per S1, not globally.** Every S1 always keeps its rarest keys, so no entity is left with zero keys just because its words are common. Only those keys are capped by df.
- **The cap was raised from 30 to 100 per S1** after measuring recall@30 = 0.668 against recall@100 = 0.766 (v1 schemes).
- **Fair slots** stop a broad scheme from evicting precise hits at the cut.
- **The embedding top-10 extras** recover what no lexical key can see. On the held-out entities, DuckDB alone holds 108,919 true pairs with p₁ ≥ 0.01, and adding the extras raises this to 113,651.
- **Safety rails, not silent truncation.** A pre-flight pair-count estimate runs before every join and aborts above a 60M-pair budget. Spill and memory are capped. Test shards are resumable.

| Blocking version | Generation (uncapped) | Recall @ cut | Ceiling F0.5 | Cut |
|---|---|---|---|---|
| v1: A–F | 0.798 | 0.668 | 0.793 | top-30 |
| v2: A–F | 0.798 | 0.766 | 0.876 | top-100 |
| v3: + G | 0.937 | 0.889 | 0.945 | top-100 |
| v4: + H | 0.957 | 0.907 | 0.954 | top-100 |
| v6+: fair-slot union | — | — | — | top-100 |
| final: + fine-tuned embedding top-10 extras | — | +4.3% true pairs in the scored set | — | top-100 + 10 |

---

## 4. Matching Model

**Features used.** Stage 1 has 52 pair features, vectorized with DuckDB and
`rapidfuzz.process.cpdist`. A slow reference implementation must agree with them before
training proceeds.
- **Name features (19):**
  - token-set, token-sort and partial ratio; Jaro-Winkler; 12-character prefix ratio; token Jaccard;
  - IDF-weighted coverage from both sides; length ratio; token-count difference; exact-key match;
  - legal-suffix agreement and conflict;
  - **sound-alike skeleton** Jaccard and ratio (`kansaltansi` ≈ consultancy);
  - count and maximum IDF of the **leftover words** on each side, i.e. words with no exact or sound-alike counterpart.
- **Address features (7):** token-set ratio; token Jaccard; postcode match; both have a postcode; house-number overlap; both have numbers; either address empty.
- **Other:**
  - *Blocking provenance (14):* which of the 9 schemes found the pair, number of schemes, IDF score, shared keys, containment, union rank.
  - *Group context (5):* candidates per S1, rank fraction, score ratio and margin to the entity's best, is-top.
  - *Rival S1 (5), over the full S1 file:*
    - how many S1 entities share this exact name, counted on each side;
    - the best address fit of a **different** S1 with the candidate's exact name;
    - this pair's address fit minus that rival's;
    - whether the rival shares a house number.
  - *Context (2):* source (S2/S3) and same country.
- **Stage 2 features (22):**
  - from LightGBM #1's probability p₁ across the entity's whole candidate list: p₁, rank, list size, E[m] = Σp₁, max, second-best, p₁ relative to the max, gaps to the top/previous/next candidate, cumulative and remaining mass, count above 0.5, logit;
  - from the **fine-tuned embedding**: cosine, rank, gap to the best neighbour, best neighbour score, found-by-DuckDB flag;
  - from the **original embedding**: cosine, rank, gap.

**Model type:** two LightGBM binary classifiers (MIT licence).
- **LightGBM #1**:
  - Settings: 63 leaves, `min_data_in_leaf` 100, learning rate 0.05, up to 600 rounds with early stopping on average precision, `deterministic=True`.
  - Data: trained on a 5% entity-hash sample of train S1 (≈ 110k entities), blocked against the full train pool. The split is by entity: 70% train, 15% **isotonic calibration**, 15% validation.
  - Pair-level validation: ROC-AUC ≈ 0.999, PR-AUC ≈ 0.99.
- **LightGBM #2** (second stage):
  - Trained only on #1's **out-of-sample** (calibration + validation) entities: 33,128 entities and 301,357 rows, of which 147,938 are embedding-only extras and 113,651 are true pairs.
  - Five settings, with and without the original-embedding columns, were compared by 5-fold CV by entity. A setting replaces the reference only if it wins on the mean **and** in ≥ 4 of 5 folds.
  - Winner: **7 leaves, 800 trees, `min_data_in_leaf` 300, learning rate 0.05, with the original-embedding columns** (CV table in Appendix B).
- **Encoder** used for retrieval: `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters), in its original form and fine-tuned by us (in-batch contrastive loss, single-country batches, 800k pairs, fp16).

**Threshold selection method:** there is no global threshold. Instead we optimize the expected
F0.5 exactly for each entity.
- **The expected score of each cut.** Pairs are treated as independent Bernoulli(p). Take an entity's candidates sorted by p. For each cut k ≤ 30, the hits among the top k and the true matches below the cut each follow a Poisson-binomial distribution, which we compute exactly by convolution:
  `E[F0.5(k)] = Σ_h Σ_r P(h)·P(r) · 1.25h / (k + 0.25(h + r))`.
- **The no-match option.** `E[F0.5(0)] = Π(1 − pᵢ)`, the probability that the entity is a singleton.
- **The choice.** The k with the highest expectation is chosen.
- **Exclusivity repair** runs once, globally, over all of test:
  - A record selected by several S1 entities stays with the highest-probability claimant.
  - The losers re-run their selection without it.
  - This is iterated until no record is shared.
  - On test it resolved **82,749 contested records**, and the final file assigns **0 records to more than one S1**.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro): 0.9431.** This is the 5-fold CV over 33,128 held-out training entities (not used to train LightGBM #1 or the encoder), blocked against the full train pool. It is the final configuration: DuckDB candidates + fine-tuned embedding extras, LightGBM #2 "7 leaves, 800 trees" with the original-embedding columns, and the exact expected-F0.5 cut. The per-fold gains over the reference setting are all positive: +0.0010, +0.0022, +0.0014, +0.0009 and +0.0017.
- **Development history** (validation macro F0.5, one change at a time):
  - 0.74: v1 baseline
  - 0.816: top-100 cut
  - 0.880: + scheme G
  - 0.887: + scheme H
  - 0.917: fair-slot union
  - 0.920: + rival-S1 features
  - 0.9416: + embedding extras and the second stage (reference setting)
  - **0.9431**: tuned second stage

  Earlier versions were scored on the 15% validation fold; the last two use the 5-fold CV above.
- **Test predictions:**
  - 1,732,544 S1 rows, of which 4.9% are predicted empty; mean set size 3.34. The training truth has 5.6% singletons and 3.46 matches per entity.
  - 5,780,761 predicted pairs.
  - These distributions are close to training, which suggests the model is not mis-calibrated on the unseen country (France).

**Where the score is lost** (error analysis on validation, before the embedding stage; share
of the 1.0 maximum):

| Category | Score lost |
|---|---|
| correct but incomplete set | 0.035 |
| wrong match(es) included | 0.018 |
| true singleton given a match | 0.014 |
| blocking found none of the true matches | 0.008 |
| model predicted nothing | 0.005 |

- **Common false positives (wrong merges):**
  - **Same-name branches of a chain:** a pool record with an identical name but another branch's address, which actually belongs to a different S1. Rival features and exclusivity reduce this, but address-less copies remain ambiguous.
  - **Generic names with no address** (`iu producer service`, `zinet |`). These match several S1 entities equally well.
  - **True singletons** with a near-duplicate name in the pool.
  - Some **label noise**: identical name and address labelled as different entities.
- **Common false negatives (missed matches):**
  - **Heavily garbled transliterations** of Indian names (`vijaya siva kansaltansi` = vijay shiv consultancy) that share no token or skeleton. The fine-tuned embeddings were added for these.
  - **Address-less records with a renamed or abbreviated name**, where no address key can fire.
  - **Conservative cuts on large sets:** the model is confident about 3–4 matches but not the 5th or 6th. "Correct but incomplete" is the largest loss bucket, and it is the price of F0.5's precision weighting.
  - **Never-blocked pairs** whose words are all common and which have no shared address word.

---

## 6. Conclusion

Most of the score comes from candidate generation. Nine targeted blocking schemes, the
fair-slot union and a 100-candidate cut raised the reachable ceiling from 0.79 to 0.95, and
validation F0.5 from 0.74 to 0.92. A fine-tuned multilingual encoder then recovered matches
that no spelling-based key could see. The rest came from precision: rival-aware features, a
second-stage model that sees the entity's whole candidate list, exact expected-F0.5 set
selection and one-owner-per-record assignment, for a final CV macro F0.5 of 0.9431. The key
lessons were to measure the blocking ceiling before touching the model, to never subsample
the pool, and to treat "no match" as a first-class prediction.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/` is self-contained. Exact commands are in `README.md` and
pinned versions in `requirements.txt`.

```
src/
  01_main_pipeline.py     step 1, CPU  — Stage 0, blocking A–H, features, LightGBM #1, test shards
  02_embed_original.py    step 2, GPU  — original-encoder retrieval
  03_embed_finetuned.py   step 3, GPU  — encoder fine-tuning + retrieval
  04_final_selection.py   step 4, CPU  — second stage, selection, exclusivity, both TSVs
  er/
    config.py      paths (env-overridable), CONFIG (all blocking caps and df ceilings)
    normalize.py   Stage 0 normalization + self-test
    blocking.py    pool index, schemes A, A2, B–H, fair-slot union, recall diagnostics
    features.py    52 pair features (vectorized) + slow reference + equivalence check
    model.py       LightGBM #1, isotonic calibration, validation report, shard scoring
    select.py      v9 set selection and exclusivity repair
    submit.py      sharded test run (resumable), validator call
    dense.py       sentence encoder, exact per-country search, contrastive fine-tuning
    stage2.py      LightGBM #2 features, exact expected-F0.5 cut, fast exclusivity repair
    metric.py      F0.5 / macro F0.5
  notebooks/       the four Kaggle notebooks exactly as run for this submission
```

**Entry points.** Run `01` → `02` and `03` (independent, GPU) → `04`. The last step writes
`$ER_RUN_DIR/final/matching_results.tsv` and `candidate_pairs.tsv`, which are the files in
`output/`.

### B. Additional Results

**Second-stage selection (5-fold CV by entity, 33,128 held-out entities; final Kaggle run).**
"vs reference" is the gain over the reference setting `B / current` (15 leaves, 300 trees).

| Setting | Macro F0.5 | vs reference | Per-fold gain |
|---|---|---|---|
| B / current (15 leaves, 300 trees), the reference | 0.9416 | — | — |
| B / 31 leaves, 600 trees | 0.9419 | +0.0003 | −0.0000 +0.0010 −0.0003 −0.0008 +0.0018 |
| B / 63 leaves, lr .03, 1000 trees | 0.9415 | −0.0001 | −0.0006 +0.0010 −0.0003 −0.0018 +0.0011 |
| B / 31 leaves, lr .03, 1000 trees, L2 5 | 0.9419 | +0.0003 | −0.0001 +0.0007 +0.0005 −0.0017 +0.0018 |
| B / 7 leaves, 800 trees | 0.9421 | +0.0005 | +0.0000 +0.0013 +0.0005 +0.0002 +0.0003 |
| B + original-embedding columns / 15 leaves, 300 trees | 0.9428 | +0.0012 | +0.0012 +0.0018 +0.0014 +0.0004 +0.0009 |
| **B + original-embedding columns / 7 leaves, 800 trees (WINNER)** | **0.9431** | **+0.0015** | +0.0010 +0.0022 +0.0014 +0.0009 +0.0017 |

The two embeddings disagree usefully. The original encoder's score is present on 38% of the
rows and adds +0.0012 on top of the fine-tuned one, and it improves every fold.

**Final test run (step 4 on Kaggle CPU, 38 minutes):**
- 16,347,458 pairs scored
- 82,749 contested records resolved
- 158,709 embedding-found matches added to the candidate file
- 0 records with two owners

**Approaches that did not help:**
- *Sibling-support features* (similarity of a candidate to the entity's other candidates) lowered validation F0.5 from 0.920 to 0.917 and were removed. They raised wrong picks, because same-name records are often other branches owned by a different S1.
- *A hand-weighted rank tier* for the union (exact name first) scored 0.882 against 0.887 and was reverted in favour of the fair-slot rule.
- *Deeper second-stage models* (63 leaves, 1000 trees) did not beat the shallow reference. The second stage needs context, not capacity.

**Compliance:**
- No external data, gazetteers, geocoding or APIs. The only vocabularies are short hand-written maps in `normalize.py`.
- Models: LightGBM (MIT) and paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M parameters), well under the 8B limit.
- `candidate_pairs.tsv` lists the DuckDB blocking set for every test S1, plus every embedding-retrieved pair that was selected as a match. Every predicted match appears in it.
