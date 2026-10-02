"""Second stage: re-score the v9 probabilities together with embedding retrieval, then pick each
S1's set by exact expected F0.5 and enforce one owner per S2/S3 record."""
import shutil

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow as pa

from .config import DATA_DIR, SOURCE_COLUMNS, TMP_DIR


def best_cut(ps, cap=30):
    """Exact expected F0.5 of predicting the top k, for every k; returns the best k.
    Pairs are treated as independent; ps must be sorted descending."""
    ps = ps[:cap]
    n = len(ps)
    suffix = [None] * (n + 1)
    suffix[n] = np.ones(1)
    for i in range(n - 1, -1, -1):
        suffix[i] = np.convolve(suffix[i + 1], [1.0 - ps[i], ps[i]])
    best_k, best = 0, suffix[0][0]                 # k = 0 scores 1 only when there is no match
    prefix = np.ones(1)
    for k in range(1, n + 1):
        prefix = np.convolve(prefix, [1.0 - ps[k - 1], ps[k - 1]])
        h = np.arange(k + 1)[:, None]
        r = np.arange(n - k + 1)[None, :]
        e = prefix @ (1.25 * h / (k + 0.25 * (h + r))) @ suffix[k]
        if e > best:
            best_k, best = k, e
    return best_k

def select_exact(df, e_m=None):
    d = df.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable")
    out = {}
    sids, cids, ps = d["s1_id"].to_numpy(), d["cand_id"].to_numpy(), d["p"].to_numpy(np.float64)
    starts = np.flatnonzero(np.r_[True, sids[1:] != sids[:-1]])
    ends = np.r_[starts[1:], len(sids)]
    for a, b in zip(starts, ends):
        k = best_cut(ps[a:b])
        out[sids[a]] = set(cids[a:a + k])
    return out

def repair_fast(pred, df, select, rounds=5):
    """A record can belong to one S1 only: keep it with the highest-probability claimant.
    Removals accumulate, and only the entities that lost a record are re-selected."""
    total = 0
    for _ in range(rounds):
        sel = pd.DataFrame([(s, c) for s, cs in pred.items() for c in cs], columns=["s1_id", "cand_id"])
        dup = sel[sel["cand_id"].duplicated(keep=False)]
        if dup.empty:
            break
        total += dup["cand_id"].nunique()
        d = dup.merge(df[["s1_id", "cand_id", "p"]], on=["s1_id", "cand_id"], how="left").fillna({"p": 0.0})
        d = d.sort_values(["cand_id", "p", "s1_id"], ascending=[True, False, True])
        losers = d[d.duplicated("cand_id", keep="first")]
        df = df[~pd.MultiIndex.from_arrays([df["s1_id"], df["cand_id"]]).isin(
            pd.MultiIndex.from_arrays([losers["s1_id"], losers["cand_id"]]))]
        affected = set(losers["s1_id"])
        redo = select(df[df["s1_id"].isin(affected)])
        for s in affected:
            pred[s] = redo.get(s, set())
    return pred, total

def self_test():
    # the fast repair must leave every record with one owner on a small random case
    rng = np.random.default_rng(1)
    df = pd.DataFrame({"s1_id": [f"s{i // 4}" for i in range(400)],
                       "cand_id": [f"c{x}" for x in rng.integers(0, 150, 400)], "p": rng.random(400)})
    df = df.drop_duplicates(["s1_id", "cand_id"])
    fast, _ = repair_fast(select_exact(df), df, select_exact)
    seen = [c for cs in fast.values() for c in cs]
    assert len(seen) == len(set(seen)), "fast repair left a record with two owners"
    assert best_cut(np.array([0.99])) == 1 and best_cut(np.array([0.05])) == 0
    print("selection checks PASS")


# ---- stage-2 features ----
S2_FEATURES = ["p", "rank", "n", "e_m", "p_max", "p_2nd", "p_rel", "gap_top", "gap_prev",
               "gap_next", "cum_p", "rest_p", "n_over_half", "logit_p"]
ANN_FEATURES = ["ann_score", "ann_rank", "in_duck", "ann_top", "ann_gap"]
ORIG_FEATURES = ["o_score", "o_rank", "o_gap"]
S2_PARAMS = dict(objective="binary", learning_rate=0.05, num_leaves=15, min_data_in_leaf=200,
                 feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                 seed=7, deterministic=True, verbosity=-1, num_threads=4)
S2_ROUNDS = 300
PARAM_GRID = {
    "current (15 leaves, 300 trees)": (S2_PARAMS, 300),
    "31 leaves, 600 trees": ({**S2_PARAMS, "num_leaves": 31, "min_data_in_leaf": 100}, 600),
    "63 leaves, lr .03, 1000 trees": ({**S2_PARAMS, "num_leaves": 63, "min_data_in_leaf": 50,
                                       "learning_rate": 0.03}, 1000),
    "31 leaves, lr .03, 1000, l2 5": ({**S2_PARAMS, "num_leaves": 31, "min_data_in_leaf": 100,
                                       "learning_rate": 0.03, "lambda_l2": 5.0, "feature_fraction": 0.8}, 1000),
    "7 leaves, 800 trees": ({**S2_PARAMS, "num_leaves": 7, "min_data_in_leaf": 300}, 800),
}

def s2_features(df, e_m):
    d = df.sort_values(["s1_id", "p"], ascending=[True, False], kind="stable").reset_index(drop=True)
    g = d.groupby("s1_id", sort=False)["p"]
    d["rank"] = g.cumcount() + 1
    d["n"] = g.transform("size")
    d["e_m"] = d["s1_id"].map(e_m).fillna(0.0)
    d["p_max"] = g.transform("max")
    d["p_2nd"] = d["s1_id"].map(d.loc[d["rank"] == 2].set_index("s1_id")["p"]).fillna(0.0)
    d["p_rel"] = d["p"] / d["p_max"].clip(lower=1e-9)
    d["gap_top"] = d["p_max"] - d["p"]
    d["gap_prev"] = (g.shift(1) - d["p"]).fillna(0.0)
    d["gap_next"] = (d["p"] - g.shift(-1)).fillna(d["p"])
    d["cum_p"] = g.cumsum()
    d["rest_p"] = d["e_m"] - d["p"]
    d["n_over_half"] = (d["p"] > 0.5).groupby(d["s1_id"], sort=False).transform("sum")
    d["logit_p"] = np.log(np.clip(d["p"], 1e-6, 1 - 1e-6) / np.clip(1 - d["p"], 1e-6, 1))
    for c in d.columns:
        if d[c].dtype == np.float64:
            d[c] = d[c].astype(np.float32)          # halves the memory of the feature table
    return d

def fit(d, feats, params=None, rounds=None):
    return lgb.train(params or S2_PARAMS, lgb.Dataset(d[feats], d["y"]), num_boost_round=rounds or S2_ROUNDS)

def label(d, pos):
    d["y"] = [int(k in pos) for k in zip(d["s1_id"].to_numpy(), d["cand_id"].to_numpy())]
    return d


# ---- joining the embedding retrieval ----
def add_orig(rows, ann_path):
    """The original (not fine-tuned) embedding's score and rank for each row, NaN if absent."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='10GB'; SET temp_directory='{TMP_DIR}'; SET preserve_insertion_order=false")
    con.register("r", pa.Table.from_pandas(rows[["s1_id", "cand_id"]], preserve_index=False))
    o = con.sql(f"""
        WITH ents AS (SELECT DISTINCT s1_id FROM r),
             a AS (SELECT x.s1_id, x.cand_id, x.ann_score, x.ann_rank FROM read_parquet('{ann_path}') x JOIN ents USING (s1_id)),
             t AS (SELECT s1_id, MAX(ann_score) AS top FROM a GROUP BY 1)
        SELECT r.s1_id, r.cand_id, a.ann_score AS o_score, a.ann_rank::FLOAT AS o_rank, t.top - a.ann_score AS o_gap
        FROM r LEFT JOIN a USING (s1_id, cand_id) LEFT JOIN t USING (s1_id)""").df()
    con.close()
    return rows.merge(o, on=["s1_id", "cand_id"], how="left")

def with_ann(pairs, ann_path, cand_files, k):
    """DuckDB pairs + embedding score/rank (NaN outside the embedding list); embedding top-k
    records that were never DuckDB candidates (p = 0); and the embedding top-k on their own."""
    con = duckdb.connect()
    con.execute(f"SET memory_limit='10GB'; SET temp_directory='{TMP_DIR}'; SET preserve_insertion_order=false")
    con.register("pairs_in", pa.Table.from_pandas(pairs[["s1_id", "cand_id", "p"]], preserve_index=False))
    con.execute(f"""
        CREATE TABLE ents AS SELECT DISTINCT s1_id FROM pairs_in;
        CREATE TABLE annx AS SELECT a.s1_id, a.cand_id, a.ann_score, a.ann_rank
                             FROM read_parquet('{ann_path}') a JOIN ents USING (s1_id);
        CREATE TABLE top1 AS SELECT s1_id, MAX(ann_score) AS ann_top FROM annx GROUP BY 1;""")
    known = con.sql("""SELECT p.s1_id, p.cand_id, p.p, a.ann_score, a.ann_rank::FLOAT AS ann_rank, 1 AS in_duck, t.ann_top
                       FROM pairs_in p LEFT JOIN annx a USING (s1_id, cand_id) LEFT JOIN top1 t USING (s1_id)""").df()
    extra = con.sql(f"""SELECT a.s1_id, a.cand_id, 0.0 AS p, a.ann_score, a.ann_rank::FLOAT AS ann_rank, 0 AS in_duck, t.ann_top
                        FROM annx a ANTI JOIN read_parquet({[str(f) for f in cand_files]}) c USING (s1_id, cand_id)
                        JOIN top1 t USING (s1_id) WHERE a.ann_rank <= {k}""").df()
    alone = con.sql(f"""SELECT a.s1_id, a.cand_id, a.ann_score AS p, a.ann_score, a.ann_rank::FLOAT AS ann_rank,
                               0 AS in_duck, t.ann_top
                        FROM annx a JOIN top1 t USING (s1_id) WHERE a.ann_rank <= {k}""").df()
    con.close()
    for d in (known, extra, alone):
        d["ann_gap"] = d["ann_top"] - d["ann_score"]
    return known, extra, alone


# ---- submission files ----
def build_candidate_pairs(work, out_path):
    """candidate_pairs.tsv from the per-shard DuckDB candidate files, in test_source1 order."""
    tmp = TMP_DIR / "cand"
    tmp.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET memory_limit='12GB'; SET temp_directory='{tmp}'; SET preserve_insertion_order=false")
    aggs = []
    for f in sorted(work.glob("test_candidates*.parquet")):
        a = tmp / f"agg_{f.stem}.parquet"
        con.execute(f"""COPY (SELECT s1_id, string_agg(cand_id, ',' ORDER BY cand_id) AS c
                             FROM read_parquet('{f}') GROUP BY s1_id) TO '{a}' (FORMAT PARQUET)""")
        aggs.append(str(a))
    ids = pd.read_csv(DATA_DIR / "test" / "test_source1.tsv", sep="\t", header=0, names=SOURCE_COLUMNS,
                      usecols=["entity_id"], dtype=str)["entity_id"]
    con.register("s1ord", pa.table({"entity_id": pa.array(ids.tolist(), pa.string()),
                                    "i": pa.array(np.arange(len(ids)), pa.int64())}))
    # an entity without candidates gets NULL, which the CSV writer leaves as an empty field
    con.execute(f"""COPY (SELECT s.entity_id AS source1_entity_id, a.c AS candidate_entity_ids
                          FROM s1ord s LEFT JOIN read_parquet({aggs}) a ON a.s1_id = s.entity_id
                          ORDER BY s.i)
                    TO '{out_path}' (FORMAT CSV, DELIMITER '\t', HEADER)""")
    con.close()
    shutil.rmtree(tmp, ignore_errors=True)
    return len(aggs)
