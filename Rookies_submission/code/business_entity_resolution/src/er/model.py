"""Stage 3: LightGBM pair scorer (70/15/15 by S1 entity) + isotonic calibration, and its val report."""
import numpy as np
import pandas as pd

from .config import CONFIG, PROB_FLOOR, WORK_DIR
from .features import FEATURE_COLUMNS
from .metric import f05, load_ground_truth, macro_f05
from .select import repair_exclusivity, select_global_threshold, select_sets

MODEL_PATH, CALIB_PATH = WORK_DIR / "lgbm.txt", WORK_DIR / "calib.npz"

def fold_of(s1_ids):
    h = pd.util.hash_array(np.asarray(s1_ids), hash_key="0" * 16) % 100
    return np.where(h < 70, 0, np.where(h < 85, 1, 2))

def train_model(split="train"):
    import lightgbm as lgb
    from sklearn.isotonic import IsotonicRegression
    from sklearn.metrics import average_precision_score, roc_auc_score

    feats = pd.read_parquet(WORK_DIR / f"{split}_features.parquet")
    truth_all = load_ground_truth(set(feats["s1_id"].unique()))
    pos = {(s, c) for s, cs in truth_all.items() for c in cs}
    feats["y"] = [int((s, c) in pos) for s, c in
                  zip(feats["s1_id"].to_numpy(), feats["cand_id"].to_numpy())]
    feats["fold"] = fold_of(feats["s1_id"].to_numpy())
    print(f"  pairs {len(feats):,}  positives {feats.y.sum():,} ({100*feats.y.mean():.2f}%)")
    tr, ca, va = feats[feats.fold == 0], feats[feats.fold == 1], feats[feats.fold == 2]
    print(f"  entities  train {tr.s1_id.nunique():,}  calib {ca.s1_id.nunique():,}  val {va.s1_id.nunique():,}")

    dtr = lgb.Dataset(tr[FEATURE_COLUMNS], tr["y"])
    dca = lgb.Dataset(ca[FEATURE_COLUMNS], ca["y"], reference=dtr)
    booster = lgb.train(
        dict(objective="binary", metric="average_precision", learning_rate=0.05,
             num_leaves=63, min_data_in_leaf=100, feature_fraction=0.9,
             bagging_fraction=0.8, bagging_freq=1, seed=42, num_threads=CONFIG["threads"],
             verbosity=-1, deterministic=True),
        dtr, num_boost_round=600, valid_sets=[dca],
        callbacks=[lgb.early_stopping(40, verbose=False), lgb.log_evaluation(200)])
    print(f"  best iteration {booster.best_iteration}")

    raw_ca = booster.predict(ca[FEATURE_COLUMNS], num_iteration=booster.best_iteration)
    iso = IsotonicRegression(out_of_bounds="clip").fit(raw_ca, ca["y"])
    raw_va = booster.predict(va[FEATURE_COLUMNS], num_iteration=booster.best_iteration)
    va = va.copy(); va["p"] = iso.predict(raw_va)
    print(f"  val ROC-AUC {roc_auc_score(va.y, raw_va):.4f}  PR-AUC {average_precision_score(va.y, raw_va):.4f}")
    booster.save_model(str(MODEL_PATH), num_iteration=booster.best_iteration)
    np.savez(CALIB_PATH, x=iso.X_thresholds_, y=iso.y_thresholds_)
    return booster, iso, va


def val_report(booster, va):
    """Val macro F0.5 under each selection rule, top features, and where the score is lost."""
    truth = load_ground_truth(set(va["s1_id"].unique()))
    best_thr, best_score = 0.5, -1.0
    for thr in np.arange(0.05, 0.96, 0.05):
        s = macro_f05(select_global_threshold(va, thr), truth)
        if s > best_score:
            best_thr, best_score = thr, s
    sel = select_sets(va)
    exp_score = macro_f05(sel, truth)
    repaired, conflicts = repair_exclusivity(sel, va)
    rep_score = macro_f05(repaired, truth)

    print("=== VAL MACRO F_0.5 ===")
    print(f"  global threshold (best {best_thr:.2f})   {best_score:.4f}")
    print(f"  expected-F_0.5 selection          {exp_score:.4f}  ({exp_score-best_score:+.4f})")
    print(f"  + exclusivity repair              {rep_score:.4f}  ({rep_score-exp_score:+.4f}, {conflicts:,} contested)")
    imp = pd.Series(booster.feature_importance("gain"), index=FEATURE_COLUMNS).sort_values(ascending=False)
    print("\n=== TOP FEATURES (gain) ===")
    for k, v in imp.head(15).items():
        print(f"  {k:<18s} {v:12,.0f}")

    # ---- error analysis (read-only) ----
    cset = va.groupby("s1_id")["cand_id"].apply(set).to_dict()
    s1meta = pd.read_parquet(WORK_DIR / "train_s1.parquet", columns=["entity_id", "country"],
                             filters=[("entity_id", "in", list(truth))]).set_index("entity_id")
    rows = []
    for sid, t in truth.items():
        p = repaired.get(sid, set()); c = cset.get(sid, set())
        m, k, h, found = len(t), len(p), len(p & t), len(t & c)
        if m == 0:        cat = "singleton OK" if k == 0 else "singleton given a match"
        elif found == 0:  cat = "blocking found none"
        elif k == 0:      cat = "model predicted nothing"
        elif k > h:       cat = "has wrong match(es)"
        elif h < m:       cat = "correct but incomplete"
        else:             cat = "perfect"
        rows.append((sid, cat, 1 - f05(p, t), s1meta["country"].get(sid, "?")))
    E = pd.DataFrame(rows, columns=["s1_id", "cat", "loss", "country"])
    t = E.groupby("cat").agg(entities=("s1_id", "size"), score_lost=("loss", "sum"))
    t["score_lost"] /= len(E)
    print(t.sort_values("score_lost", ascending=False).to_string(float_format=lambda x: f"{x:.4f}"))
    print(E.groupby("country").agg(entities=("s1_id", "size"),
          macro_f=("loss", lambda x: 1 - x.mean())).to_string(float_format=lambda x: f"{x:.4f}"))


def predict_shard(split, suffix, booster, iso):
    feats = pd.read_parquet(WORK_DIR / f"{split}_features{suffix}.parquet")
    p = iso.predict(booster.predict(feats[FEATURE_COLUMNS]))
    out = pd.DataFrame({"s1_id": feats.s1_id, "cand_id": feats.cand_id, "p": p})
    sums = out.groupby("s1_id", sort=False)["p"].sum().rename("e_m").reset_index()
    kept = out[out.p >= PROB_FLOOR]
    kept.to_parquet(WORK_DIR / f"{split}_probs{suffix}.parquet", index=False)
    sums.to_parquet(WORK_DIR / f"{split}_sums{suffix}.parquet", index=False)
    print(f"  {suffix}: scored {len(out):,} -> kept {len(kept):,}, {len(sums):,} entities")
