"""Step 4 of 4 (CPU). Second stage + the submitted files, as run in notebooks/er_final_tuned_kaggle.ipynb.

Inputs: WORK_DIR from step 1, ANN_FT_DIR from step 3 (main embedding) and, optionally, ANN_DIR
from step 2 (the original embedding, offered to the model as extra columns).

1. candidate_pairs.tsv from step 1's per-shard DuckDB candidates.
2. Held-out (calib + val fold) probabilities of the step-1 model on the s1_sample.
3. Variant B: DuckDB candidates + the fine-tuned embedding's top-10 records DuckDB never proposed,
   re-scored by a second-stage LightGBM on group-context + embedding features. Several LightGBM
   settings (and "+orig embedding columns") are compared by 5-fold CV by entity; one replaces the
   reference setting only if it is better on average AND in at least 4 of 5 folds.
4. The winner is refit on all held-out rows, applied to test; sets are chosen by exact expected
   F0.5, every S2/S3 record is kept by one S1 only. Written to FINAL_DIR and validated.
"""
import gc
import shutil
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from er.config import (ANN_DIR, ANN_FT_DIR, DATA_DIR, FINAL_DIR, PROB_FLOOR, SOURCE_COLUMNS,
                       TMP_DIR, WORK_DIR)
from er.metric import macro_f05
from er.model import fold_of
from er.stage2 import (ANN_FEATURES, ORIG_FEATURES, PARAM_GRID, S2_FEATURES, add_orig,
                       build_candidate_pairs, fit, label, repair_fast, s2_features, select_exact,
                       self_test, with_ann)
from er.submit import duplicate_records, validate

ANN_K = 10
T_START = time.perf_counter()


def lap(label_):
    print(f"  [{label_}: {(time.perf_counter() - T_START) / 60:.1f} min elapsed]")


def main():
    for d in (FINAL_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
    assert (WORK_DIR / "lgbm.txt").exists(), f"{WORK_DIR}/lgbm.txt missing: run step 1 first"
    EMB = {k: d for k, d in (("ft", ANN_FT_DIR), ("orig", ANN_DIR))
           if (d / "ann_train.parquet").exists() and (d / "ann_test.parquet").exists()}
    assert EMB, "no embedding retrieval found: run step 3 (and optionally step 2) first"
    print(f"v9 run: {WORK_DIR}")
    for k, d in EMB.items():
        print(f"embeddings '{k}': {d}")

    # ---- candidate_pairs.tsv (first, while memory is empty) ----
    n_shards = build_candidate_pairs(WORK_DIR, FINAL_DIR / "candidate_pairs.tsv")
    print(f"candidate_pairs.tsv from {n_shards} shards")
    lap("candidates")
    self_test()

    # ---- held-out probabilities from the step-1 model (calib + val folds) ----
    booster = lgb.Booster(model_file=str(WORK_DIR / "lgbm.txt"))
    cal = np.load(WORK_DIR / "calib.npz")
    calibrate = lambda raw: np.interp(raw, cal["x"], cal["y"])   # = IsotonicRegression.predict (clip)
    FEATURES = booster.feature_name()

    t0 = time.perf_counter()
    pf = pq.ParquetFile(WORK_DIR / "train_features.parquet")
    parts = []
    for batch in pf.iter_batches(batch_size=1_000_000, columns=["s1_id", "cand_id"] + FEATURES):
        b = batch.to_pandas()
        b = b[fold_of(b["s1_id"].to_numpy()) > 0]
        if len(b):
            p = calibrate(booster.predict(b[FEATURES]))
            parts.append(pd.DataFrame({"s1_id": b["s1_id"].to_numpy(), "cand_id": b["cand_id"].to_numpy(), "p": p}))
    held = pd.concat(parts, ignore_index=True)
    del parts

    gt = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False, na_values=[])
    ents = set(held["s1_id"].unique())
    truth = {s: (set(m.split(",")) if m else set())
             for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]) if s in ents}
    pos = {(s, c) for s, cs in truth.items() for c in cs}
    held["y"] = [int(k in pos) for k in zip(held["s1_id"].to_numpy(), held["cand_id"].to_numpy())]

    # mimic the test files: E[m] from all candidates, pairs kept only above the floor
    held_em = held.groupby("s1_id")["p"].sum()
    held = held[held["p"] >= PROB_FLOOR].reset_index(drop=True)
    print(f"held-out entities {len(truth):,}  pairs above floor {len(held):,}  "
          f"positives {held['y'].sum():,}  ({time.perf_counter() - t0:.0f}s)")
    lap("held-out")

    # ---- features for every variant ----
    EMB_MAIN = "ft" if "ft" in EMB else next(iter(EMB))
    known, extra, _ = with_ann(held, EMB[EMB_MAIN] / "ann_train.parquet", [WORK_DIR / "train_candidates.parquet"], ANN_K)
    b_rows = label(pd.concat([known, extra], ignore_index=True), pos)
    print(f"  B_{EMB_MAIN}: {len(b_rows):,} rows ({len(extra):,} embedding-only, {b_rows['y'].sum():,} true)")
    del known, extra
    B_D = s2_features(b_rows, held_em)
    B_FEATS = S2_FEATURES + ANN_FEATURES
    USE_ORIG = EMB_MAIN == "ft" and "orig" in EMB
    if USE_ORIG:
        BO_D = s2_features(add_orig(b_rows, EMB["orig"] / "ann_train.parquet"), held_em)
        BO_FEATS = B_FEATS + ORIG_FEATURES
        print(f"  original-embedding score present on {BO_D['o_score'].notna().mean():.0%} of rows")
    del b_rows
    gc.collect()
    lap("features")

    # ---- 5-fold CV of every variant ----
    all_ents = np.array(sorted(truth))
    cv_fold = pd.util.hash_array(all_ents, hash_key="1" * 16) % 5
    fold_of_ent = dict(zip(all_ents, cv_fold))
    REF = "B / current (15 leaves, 300 trees)"
    CANDIDATES = {f"B / {n}": (B_D, B_FEATS, p, r) for n, (p, r) in PARAM_GRID.items()}
    scores = {n: [] for n in CANDIDATES}
    t0 = time.perf_counter()
    for f in range(5):
        te_truth = {s_: truth[s_] for s_ in all_ents[cv_fold == f]}
        cv = B_D["s1_id"].map(fold_of_ent)
        for name, (d, feats, params, rounds) in CANDIDATES.items():
            tr, te = d[cv != f], d[cv == f]
            q = te[["s1_id", "cand_id"]].assign(p=fit(tr, feats, params, rounds).predict(te[feats]))
            scores[name].append(macro_f05(select_exact(q), te_truth))
        print(f"  fold {f + 1}/5 done ({(time.perf_counter() - t0) / 60:.1f} min)")
    best_grid = max((n for n in scores), key=lambda n: np.mean(scores[n]))
    if USE_ORIG:
        # the original embedding's score as extra columns, with the current and the best settings
        for label_, gname in (("B+orig / current (15 leaves, 300 trees)", "current (15 leaves, 300 trees)"),
                              (f"B+orig / {best_grid[4:]}", best_grid[4:])):
            if label_ in CANDIDATES:
                continue
            p, r = PARAM_GRID[gname]
            CANDIDATES[label_] = (BO_D, BO_FEATS, p, r)
            scores[label_] = []
            cvo = BO_D["s1_id"].map(fold_of_ent)
            for f in range(5):
                te_truth = {s_: truth[s_] for s_ in all_ents[cv_fold == f]}
                tr, te = BO_D[cvo != f], BO_D[cvo == f]
                q = te[["s1_id", "cand_id"]].assign(p=fit(tr, BO_FEATS, p, r).predict(te[BO_FEATS]))
                scores[label_].append(macro_f05(select_exact(q), te_truth))
            print(f"  {label_} done ({(time.perf_counter() - t0) / 60:.1f} min)")

    ref = np.array(scores[REF])
    print(f"\n  {'setting':44s} {'macro F0.5':>10s} {'vs current':>11s}   per fold")
    for name, sc in scores.items():
        print(f"  {name:44s} {np.mean(sc):10.4f} {np.mean(sc) - ref.mean():+11.4f}   "
              + " ".join(f"{x:+.4f}" for x in np.array(sc) - ref))
    wins = {n: np.mean(sc) - ref.mean() for n, sc in scores.items()
            if n != REF and np.mean(sc) > ref.mean() and (np.array(sc) > ref).sum() >= 4}
    WINNER = max(wins, key=wins.get) if wins else REF
    print(f"\nWINNER: {WINNER}" + (f"  (+{wins[WINNER]:.4f} vs the current submission's settings)"
                                   if WINNER in wins else "  (= current submission's settings)"))
    lap("cv")

    # ---- apply the winner to test, write, free memory, validate ----
    WIN_D, WIN_FEATS, WIN_PARAMS, WIN_ROUNDS = CANDIDATES[WINNER]
    model = fit(WIN_D, WIN_FEATS, WIN_PARAMS, WIN_ROUNDS)
    del CANDIDATES, B_D, WIN_D, held
    if USE_ORIG:
        del BO_D
    gc.collect()

    probs = pd.concat([pd.read_parquet(f) for f in sorted(WORK_DIR.glob("test_probs*.parquet"))], ignore_index=True)
    sums = pd.concat([pd.read_parquet(f) for f in sorted(WORK_DIR.glob("test_sums*.parquet"))],
                     ignore_index=True).groupby("s1_id")["e_m"].sum()
    known, extra, _ = with_ann(probs, EMB[EMB_MAIN] / "ann_test.parquet", sorted(WORK_DIR.glob("test_candidates*.parquet")), ANN_K)
    rows, em, extra_pairs = pd.concat([known, extra], ignore_index=True), sums, extra[["s1_id", "cand_id"]]
    del known, extra, probs
    gc.collect()
    if WINNER.startswith("B+orig"):
        rows = add_orig(rows, EMB["orig"] / "ann_test.parquet")
    print(f"test rows {len(rows):,}")
    t = s2_features(rows, em)
    q, rule = t[["s1_id", "cand_id"]].assign(p=model.predict(t[WIN_FEATS])), select_exact
    del t, rows
    gc.collect()
    lap("test scored")

    pred = rule(q)
    pred, contested = repair_fast(pred, q, rule)
    print(f"  contested records resolved {contested:,}")
    del q
    gc.collect()
    lap("test selected")

    s1_ids = pd.read_csv(DATA_DIR / "test" / "test_source1.tsv", sep="\t", header=0, names=SOURCE_COLUMNS,
                         usecols=["entity_id"], dtype=str)["entity_id"].tolist()
    with open(FINAL_DIR / "matching_results.tsv", "w") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for s in s1_ids:
            fh.write(f"{s}\t{','.join(sorted(pred.get(s, ())))}\n")

    # a match the embeddings found must also be listed as a candidate
    chosen = {(s, c) for s, cs in pred.items() for c in cs}
    add = {}
    for s, c in zip(extra_pairs["s1_id"].to_numpy(), extra_pairs["cand_id"].to_numpy()):
        if (s, c) in chosen:
            add.setdefault(s, []).append(c)
    src_path = FINAL_DIR / "candidate_pairs.tsv"
    tmp_path = TMP_DIR / "candidate_pairs.tsv"
    with open(src_path) as src, open(tmp_path, "w") as dst:
        dst.write(next(src))
        for line in src:
            s, c = line.rstrip("\n").split("\t", 1)
            if s in add:
                have = set(filter(None, c.strip('"').split(",")))
                c = ",".join(sorted(have | set(add[s])))
            dst.write(f"{s}\t{c}\n")
    shutil.move(tmp_path, src_path)
    print(f"  added {sum(len(v) for v in add.values()):,} embedding-found matches to candidate_pairs.tsv")

    k = np.array([len(pred.get(s, ())) for s in s1_ids])
    print(f"  wrote {len(s1_ids):,} rows: empty {np.mean(k == 0):.1%}  mean set size {k.mean():.2f}")
    del pred, extra_pairs, sums
    gc.collect()                                # the validator needs the memory
    validate(FINAL_DIR)
    print(f"records assigned to more than one S1: {duplicate_records(FINAL_DIR):,}")
    lap("done")
    print(f"\nsubmit {FINAL_DIR}  (setting: {WINNER})")


if __name__ == "__main__":
    main()
