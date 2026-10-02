"""Stage 4-5 of the v9 run: expected-F0.5 set selection and global exclusivity repair."""
import numpy as np
import pandas as pd


def select_sets(df, prob_col="p", e_m=None):
    d = df[["s1_id", "cand_id", prob_col]].sort_values(
        ["s1_id", prob_col], ascending=[True, False], kind="stable")
    p = d[prob_col].to_numpy(np.float64)
    g = d.groupby("s1_id", sort=False)
    k = g.cumcount().to_numpy() + 1
    cum_p = g[prob_col].cumsum().to_numpy(np.float64)
    e_m_arr = (g[prob_col].transform("sum").to_numpy(np.float64) if e_m is None
               else d["s1_id"].map(e_m).fillna(0.0).to_numpy(np.float64))
    d["score"] = 1.25 * cum_p / (k + 0.25 * e_m_arr)
    d["k"] = k
    d["log1mp"] = np.log1p(-np.clip(p, 0.0, 1.0 - 1e-12))
    best_idx = d.groupby("s1_id", sort=False)["score"].idxmax()
    best = d.loc[best_idx, ["s1_id", "score", "k"]].set_index("s1_id")
    score0 = np.exp(d.groupby("s1_id", sort=False)["log1mp"].sum())
    kstar = best["k"].where(~(score0 > best["score"]), 0)
    d["kstar"] = d["s1_id"].map(kstar).to_numpy()
    sel = d[d["k"] <= d["kstar"]]
    out = {sid: set() for sid in kstar.index}
    for sid, cid in zip(sel["s1_id"].to_numpy(), sel["cand_id"].to_numpy()):
        out[sid].add(cid)
    return out

def select_global_threshold(df, thr, prob_col="p"):
    out = {sid: set() for sid in df["s1_id"].unique()}
    hit = df[df[prob_col] >= thr]
    for sid, cid in zip(hit["s1_id"].to_numpy(), hit["cand_id"].to_numpy()):
        out[sid].add(cid)
    return out


def repair_exclusivity(pred, df, prob_col="p", rounds=3, e_m=None):
    prob = {(s, c): p for s, c, p in zip(df["s1_id"].to_numpy(),
                                        df["cand_id"].to_numpy(), df[prob_col].to_numpy())}
    total = 0
    for _ in range(rounds):
        owners = {}
        for sid, cands in pred.items():
            for cid in cands:
                owners.setdefault(cid, []).append(sid)
        contested = {c: s for c, s in owners.items() if len(s) > 1}
        if not contested:
            break
        total += len(contested)
        drop = set()
        for cid, sids in contested.items():
            winner = max(sids, key=lambda s: prob.get((s, cid), 0.0))
            drop.update((s, cid) for s in sids if s != winner)
        surviving = df[~pd.MultiIndex.from_arrays([df["s1_id"], df["cand_id"]]).isin(drop)]
        pred = select_sets(surviving, prob_col, e_m)
    return pred, total
