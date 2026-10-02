# Business Entity Resolution: reproduction guide

This folder regenerates `matching_results.tsv` and `candidate_pairs.tsv` from the
competition's train/test TSVs. Nothing else is needed: no external data, no APIs, and no
pretrained weights other than one Apache-2.0 sentence encoder, which is downloaded from the
Hugging Face hub on first use.

## Pipeline at a glance

```
raw TSVs ─► 01 main pipeline (CPU) ──────────────────────────────┐
            normalize ─► DuckDB blocking A–H (top-100/S1)         │ work/: lgbm.txt, calib.npz,
            ─► 52 pair features ─► LightGBM + isotonic            │ train_features, train_candidates,
            ─► sharded test scoring                               │ test_{candidates,probs,sums}_shNN
                                                                  ▼
raw TSVs ─► 02 original encoder (GPU) ─► ann/ann_{train,test}  ─► 04 final selection (CPU)
raw TSVs ─► 03 fine-tuned encoder (GPU) ─► ann_ft/ann_{train,test} ─►  second-stage LightGBM,
                                                                     exact expected-F0.5 sets,
                                                                     one owner per record
                                                                  ─► final/matching_results.tsv
                                                                     final/candidate_pairs.tsv
```

Steps 2 and 3 depend only on the raw data, so they can run in parallel with step 1.
Step 4 needs all three.

## Folder layout

```
src/
  01_main_pipeline.py     step 1  (notebooks/er_v9_kaggle.ipynb)
  02_embed_original.py    step 2  (notebooks/ann_embed_kaggle.ipynb, RUN_TEST = True)
  03_embed_finetuned.py   step 3  (notebooks/ann_finetune_kaggle.ipynb)
  04_final_selection.py   step 4  (notebooks/er_final_tuned_kaggle.ipynb)
  er/
    config.py     paths (env-overridable), CONFIG, row-count checks
    normalize.py  Stage 0: Unicode/Indic transliteration, legal suffixes, address parts, self-test
    blocking.py   Stage 1: DuckDB pool index, schemes A, A2, B–H, fair-slot union, top-k cut
    features.py   Stage 2: pair features (vectorized), with a slow reference + equivalence check
    model.py      Stage 3: LightGBM + isotonic calibration, val report, test shard scoring
    select.py     Stage 4–5 (v9): expected-F0.5 set selection, exclusivity repair
    submit.py     sharded test run, the plain v9 submission, validator call
    dense.py      embedding retrieval + contrastive fine-tuning (GPU)
    stage2.py     second-stage model, exact expected-F0.5 cut, fast exclusivity repair
    metric.py     F0.5 / macro F0.5
  notebooks/      the four Kaggle notebooks exactly as they were run for the submission
```

The `.py` files contain the notebook code with the cell glue removed. Function bodies are
unchanged; global state is passed as arguments instead. Paths come from environment variables
instead of `/kaggle/...`. There is one functional change: `dense.sample_ids` reads its query
result with `fetchall()` instead of `.arrow()`, because DuckDB ≥ 1.4 returns a
`RecordBatchReader` from `.arrow()`. The sample itself is the same.

## Environment

| | Steps 1 and 4 | Steps 2 and 3 |
|---|---|---|
| Hardware | 4 vCPU, ~30 GB RAM, ≥ 20 GB free disk | 1–2 NVIDIA GPUs (we used Kaggle T4 ×2); runs on CPU too, but slowly |
| OS | Linux (Kaggle) for the full run | Linux (Kaggle) |

`requirements.txt` pins the versions with which all four scripts were smoke-tested end to end
on a synthetic miniature dataset in the competition's format, using Python 3.13. The full-size
run that produced the submission used the Kaggle notebooks in `src/notebooks/`.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# GPU steps: install the CUDA build of torch that matches your driver, e.g.
# pip install torch==<version in requirements.txt> --index-url https://download.pytorch.org/whl/cu121
```

## Data and paths

Every path is set through an environment variable. The defaults match the Kaggle layout.

| Variable | Meaning | Default |
|---|---|---|
| `ER_DATA_DIR` | folder holding `train/` and `test/` | Kaggle competition input, else `./dataset` |
| `ER_RUN_DIR` | where all outputs go | `/kaggle/working`, else `./run` |
| `ER_WORK_DIR` | step-1 artefacts | `$ER_RUN_DIR/work` |
| `ER_ANN_DIR` / `ER_ANN_FT_DIR` | step-2 / step-3 retrieval | `$ER_RUN_DIR/ann`, `$ER_RUN_DIR/ann_ft` |
| `ER_FT_MODEL_DIR` | fine-tuned encoder | `$ER_RUN_DIR/ft_model` |
| `ER_FINAL_DIR` | the submission files | `$ER_RUN_DIR/final` |
| `ER_VALIDATOR` | `validate_submission.py` | next to the dataset folder; skipped if absent |

## Run it end to end

```bash
cd src
export ER_DATA_DIR=/path/to/dataset ER_RUN_DIR=/path/to/run

python 01_main_pipeline.py        # CPU. Resumable: re-running skips finished test shards
python 02_embed_original.py       # GPU
python 03_embed_finetuned.py      # GPU (independent of step 2)
python 04_final_selection.py      # CPU. Writes $ER_RUN_DIR/final/{matching_results,candidate_pairs}.tsv
```

`$ER_RUN_DIR/final/` then holds the two files that go into `output/` of the submission zip.
Step 4 ends by running the official validator (it must print `PASS`) and by counting S2/S3
records assigned to more than one S1 (it must print 0). On a 30 GB machine the validator can
be killed for lack of memory (`exit code -9`), because step 4 still holds its data. In that
case run it on its own afterwards:

```bash
python validate_submission.py --matching $ER_RUN_DIR/final/matching_results.tsv \
    --candidate $ER_RUN_DIR/final/candidate_pairs.tsv --test-dir $ER_DATA_DIR/test
```

`python 01_main_pipeline.py --dev` runs the train split only and stops after the validation
report. It is the fast loop for checking blocking and model changes.

### Running on Kaggle (how the submission was produced)

Each step was a separate Kaggle notebook. A later step reads an earlier step's output as an
attached dataset, which keeps every session under Kaggle's 12-hour limit.

1. `notebooks/er_v9_kaggle.ipynb`: CPU, set `RUN_TEST = True`, Save & Run All. Its output
   (`work/`) becomes the "v9 output" dataset.
2. `notebooks/ann_embed_kaggle.ipynb`: GPU T4 ×2, set `RUN_TEST = True`. Output: `ann/`.
3. `notebooks/ann_finetune_kaggle.ipynb`: GPU T4 ×2. Output: `ann_ft/`.
4. `notebooks/er_final_tuned_kaggle.ipynb`: CPU, with the competition data and the outputs of
   steps 1–3 attached. It writes `/kaggle/working/final/`.

The scripts accept the same split: point `ER_WORK_DIR`, `ER_ANN_DIR` and `ER_ANN_FT_DIR` at
the attached read-only datasets and set `ER_RUN_DIR` to `/kaggle/working`.

## What each step checks before it trusts its own output

- Stage 0 runs a normalization self-test and raises if any of the 9 Indic scripts
  transliterates to an empty name. It also compares row counts with the published sizes.
  Stage 0 output is versioned (`NORM_VERSION`), so a normalizer change can never reuse stale
  parquet files.
- `check_features` recomputes 200 entities with the slow row-by-row reference and raises if the
  vectorized features differ.
- Every DuckDB join is preceded by a pair-count estimate and aborts above `pair_budget`. DuckDB
  spill is capped.
- `04_final_selection.py` asserts that the fast exclusivity repair leaves every record with
  one owner, and that the exact expected-F0.5 cut behaves on trivial cases.

## Reproducibility notes

- LightGBM runs with `deterministic=True` and fixed seeds. The train/calib/val split and the
  5 CV folds are hashes of the entity id, so they do not depend on file order.
- The 5% training sample is `hash(entity_id) % 1000 < 50` in DuckDB. Every step must therefore
  use the same DuckDB version, which `requirements.txt` pins.
- The encoder fine-tuning (step 3) runs in fp16 on the GPU and is not bit-for-bit
  deterministic. A rerun gives near-identical but not identical embedding scores, so a few
  borderline matches can flip. The second-stage setting is chosen by a fixed rule (better mean
  and better in ≥ 4 of 5 folds), so the choice itself is stable.
