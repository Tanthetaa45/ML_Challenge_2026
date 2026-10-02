"""Stage 2: pair features.

The fast path (DuckDB set features + rapidfuzz cpdist string scores) is what runs.
`pair_features` is the original row-by-row version, kept only as the reference that
`check_features` compares the fast version against.
"""
import math
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler

from .blocking import connect
from .config import WORK_DIR

FEATURE_COLUMNS = [
    "n_token_set", "n_token_sort", "n_partial", "n_jaro", "n_prefix", "n_jacc",
    "n_idf_cover_s1", "n_idf_cover_c", "n_len_ratio", "n_tok_diff", "n_exact_key",
    "n_suffix_agree", "n_suffix_conflict",
    "n_skel_jacc", "n_skel_ratio", "n_extra_cnt", "n_extra_max_idf",
    "n_missing_cnt", "n_missing_max_idf",
    "a_token_set", "a_jacc", "a_post_match", "a_post_both", "a_num_overlap",
    "a_num_both", "a_empty_either",
    "b_by_a", "b_by_a2", "b_by_b", "b_by_c", "b_by_d", "b_by_e", "b_by_f", "b_by_g", "b_by_h",
    "b_n_schemes", "b_idf_score", "b_shared", "b_containment", "b_rank",
    "g_n_cands", "g_rank_frac", "g_score_ratio", "g_score_margin", "g_is_top",
    "c_source", "c_same_country",
    "x_s1_name_dup", "x_c_name_s1cnt", "x_rival_addr", "x_rival_addr_adv", "x_rival_num",
]
IDX = {c: i for i, c in enumerate(FEATURE_COLUMNS)}
N_FEATURES = len(FEATURE_COLUMNS)

# Sound-alike skeleton ('kansaltansi' ~ 'consultancy').
_SKEL_RULES = [(re.compile(p), r) for p, r in [
    (r"ph", "f"), (r"sh", "s"), (r"ck", "k"), (r"c(?=[eiy])", "s"), (r"c", "k"),
    (r"q", "k"), (r"x", "ks"), (r"z", "s"), (r"w", "v"), (r"j", "g"), (r"h", ""),
    (r"[aeiouy]", ""), (r"(.)\1+", r"\1")]]

@lru_cache(maxsize=2_000_000)
def skel(tok):
    for pat, rep in _SKEL_RULES:
        tok = pat.sub(rep, tok)
    return tok

_UNSEEN_IDF = 16.0

def _leftovers(a, b, b_sk):
    return [t for t in a if t not in b and not (len(skel(t)) >= 2 and skel(t) in b_sk)]

def _jacc(a, b):
    return len(a & b) / len(a | b) if a and b else 0.0

def pair_features(df, idf):
    out = np.zeros((len(df), N_FEATURES), dtype=np.float32); I = IDX
    for i, r in enumerate(df.itertuples(index=False)):
        sn, cn = r.s_name_text or "", r.c_name_text or ""
        st = set(r.s_name_tokens) if r.s_name_tokens is not None else set()
        ct = set(r.c_name_tokens) if r.c_name_tokens is not None else set()
        country = r.s_country
        out[i, I["n_token_set"]] = fuzz.token_set_ratio(sn, cn) / 100.0
        out[i, I["n_token_sort"]] = fuzz.token_sort_ratio(sn, cn) / 100.0
        out[i, I["n_partial"]] = fuzz.partial_ratio(sn, cn) / 100.0
        out[i, I["n_jaro"]] = JaroWinkler.similarity(sn, cn)
        out[i, I["n_prefix"]] = fuzz.QRatio(sn[:12], cn[:12]) / 100.0
        out[i, I["n_jacc"]] = _jacc(st, ct)
        ws = sum(idf.get((country, t), 0.0) for t in st & ct)
        w_s = sum(idf.get((country, t), 0.0) for t in st)
        w_c = sum(idf.get((country, t), 0.0) for t in ct)
        out[i, I["n_idf_cover_s1"]] = ws / w_s if w_s else 0.0
        out[i, I["n_idf_cover_c"]] = ws / w_c if w_c else 0.0
        ls, lc = len(sn), len(cn)
        out[i, I["n_len_ratio"]] = min(ls, lc) / max(ls, lc) if max(ls, lc) else 0.0
        out[i, I["n_tok_diff"]] = abs(len(st) - len(ct))
        out[i, I["n_exact_key"]] = float(r.s_name_key == r.c_name_key and r.s_name_key != "")
        ssuf, csuf = set((r.s_name_suffix or "").split()), set((r.c_name_suffix or "").split())
        out[i, I["n_suffix_agree"]] = float(bool(ssuf & csuf))
        out[i, I["n_suffix_conflict"]] = float(bool(ssuf) and bool(csuf) and not (ssuf & csuf))
        s_sk = {k for k in (skel(t) for t in st) if len(k) >= 2}
        c_sk = {k for k in (skel(t) for t in ct) if len(k) >= 2}
        out[i, I["n_skel_jacc"]] = _jacc(s_sk, c_sk)
        out[i, I["n_skel_ratio"]] = fuzz.ratio(" ".join(sorted(s_sk)), " ".join(sorted(c_sk))) / 100.0
        extra = _leftovers(ct, st, s_sk)
        missing = _leftovers(st, ct, c_sk)
        out[i, I["n_extra_cnt"]] = len(extra)
        out[i, I["n_extra_max_idf"]] = max((idf.get((country, t), _UNSEEN_IDF) for t in extra), default=0.0)
        out[i, I["n_missing_cnt"]] = len(missing)
        out[i, I["n_missing_max_idf"]] = max((idf.get((country, t), _UNSEEN_IDF) for t in missing), default=0.0)
        sa, ca = r.s_addr_text or "", r.c_addr_text or ""
        sat = set(r.s_addr_tokens) if r.s_addr_tokens is not None else set()
        cat = set(r.c_addr_tokens) if r.c_addr_tokens is not None else set()
        out[i, I["a_token_set"]] = fuzz.token_set_ratio(sa, ca) / 100.0 if sa and ca else 0.0
        out[i, I["a_jacc"]] = _jacc(sat, cat)
        sp, cp_ = r.s_postcode or "", r.c_postcode or ""
        out[i, I["a_post_match"]] = float(sp != "" and sp == cp_)
        out[i, I["a_post_both"]] = float(sp != "" and cp_ != "")
        snum = set(r.s_addr_nums) if r.s_addr_nums is not None else set()
        cnum = set(r.c_addr_nums) if r.c_addr_nums is not None else set()
        out[i, I["a_num_overlap"]] = _jacc(snum, cnum)
        out[i, I["a_num_both"]] = float(bool(snum) and bool(cnum))
        out[i, I["a_empty_either"]] = float(r.s_addr_empty or r.c_addr_empty)
        a, a2, b, c, d, e, f = r.by_a, r.by_a2, r.by_b, r.by_c, r.by_d, r.by_e, r.by_f
        for nm, v in (("b_by_a", a), ("b_by_a2", a2), ("b_by_b", b), ("b_by_c", c),
                      ("b_by_d", d), ("b_by_e", e), ("b_by_f", f)):
            out[i, I[nm]] = v
        out[i, I["b_n_schemes"]] = a + a2 + b + c + d + e + f
        out[i, I["b_idf_score"]] = r.idf_score or 0.0
        out[i, I["b_shared"]] = r.shared or 0
        out[i, I["b_containment"]] = r.containment or 0.0
        out[i, I["b_rank"]] = r.rnk
        out[i, I["c_source"]] = 2.0 if str(r.cand_id).startswith("S2-") else 3.0
        out[i, I["c_same_country"]] = float(r.s_country == r.c_country)
        out[i, I["x_s1_name_dup"]] = r.x_s1_dup
        out[i, I["x_c_name_s1cnt"]] = r.x_c_s1cnt
        rj = r.x_rival_jacc
        if pd.isna(rj):
            out[i, I["x_rival_addr"]] = np.nan
            out[i, I["x_rival_addr_adv"]] = np.nan
            out[i, I["x_rival_num"]] = np.nan
        else:
            out[i, I["x_rival_addr"]] = rj
            out[i, I["x_rival_addr_adv"]] = rj - out[i, I["a_jacc"]]
            out[i, I["x_rival_num"]] = r.x_rival_num
    return out

def add_group_features(feats, s1_ids):
    """Rank/margin within each entity's own candidate list, in place."""
    base = feats[:, IDX["n_token_set"]] * 0.6 + feats[:, IDX["n_idf_cover_s1"]] * 0.4
    g = pd.DataFrame({"sid": s1_ids, "score": base}).groupby("sid")["score"]
    n = g.transform("size").to_numpy(np.float32)
    mx = g.transform("max").to_numpy(np.float32)
    rank = g.rank(ascending=False, method="first").to_numpy(np.float32)
    feats[:, IDX["g_n_cands"]] = n
    feats[:, IDX["g_rank_frac"]] = rank / np.maximum(n, 1)
    feats[:, IDX["g_score_ratio"]] = np.where(mx > 0, base / mx, 0.0)
    feats[:, IDX["g_score_margin"]] = base - mx
    feats[:, IDX["g_is_top"]] = (rank == 1).astype(np.float32)


# ---- fast features ----
GROUP_FEATURES = ["g_n_cands", "g_rank_frac", "g_score_ratio", "g_score_margin", "g_is_top"]
STRING_FEATURES = ["n_token_set", "n_token_sort", "n_partial", "n_jaro", "n_prefix",
                   "n_skel_ratio", "a_token_set"]
NAN_FEATURES = ["x_rival_addr", "x_rival_addr_adv", "x_rival_num"]   # NaN = no rival S1
SQL_FEATURES = [c for c in FEATURE_COLUMNS if c not in STRING_FEATURES + GROUP_FEATURES]

def _pool_files(split):
    return [str(WORK_DIR / f"{split}_s2.parquet"), str(WORK_DIR / f"{split}_s3.parquet")]

def load_idf(con, split):
    path = WORK_DIR / f"{split}_idf.parquet"
    if not path.exists():
        pool = _pool_files(split)
        con.execute(f"""COPY (
            WITH ct AS (SELECT country, UNNEST(name_tokens) AS tok FROM read_parquet({pool}))
            SELECT country, tok,
                   LN(1.0 + (SELECT COUNT(*) FROM read_parquet({pool})) / COUNT(*)) AS idf
            FROM ct GROUP BY country, tok
        ) TO '{path}' (FORMAT PARQUET)""")
    return path

def load_skel(con, split):
    """Sound-alike skeleton for every name token of the split, computed once."""
    path = WORK_DIR / f"{split}_skel.parquet"
    if not path.exists():
        files = [str(WORK_DIR / f"{split}_s{i}.parquet") for i in (1, 2, 3)]
        toks = con.sql(f"SELECT DISTINCT tok FROM (SELECT UNNEST(name_tokens) AS tok "
                       f"FROM read_parquet({files}))").fetchnumpy()["tok"].tolist()
        pq.write_table(pa.table({"tok": toks, "sk": [skel(t) for t in toks]}), path)
    return path

def _J(a, b):
    return (f"CASE WHEN len({a}) = 0 OR len({b}) = 0 THEN 0.0 ELSE "
            f"len(list_intersect({a}, {b}))::DOUBLE / len(list_distinct(list_concat({a}, {b}))) END")

def _leftover_sql(name, id_col, toks, other, other_sk_tbl, other_id):
    # words of one side with no exact or sound-alike counterpart on the other
    return f"""
    CREATE OR REPLACE TABLE {name} AS
    SELECT x.s1_id, x.cand_id, COUNT(*) AS n, MAX(coalesce(t.idf, {_UNSEEN_IDF})) AS mx
    FROM (SELECT b.s1_id, b.cand_id, b.s_country AS country, b.{other} AS other, o.sk AS osk,
                 UNNEST(b.{toks}) AS tok
          FROM base b LEFT JOIN {other_sk_tbl} o ON o.id = b.{other_id}) x
    LEFT JOIN skmap m ON m.tok = x.tok
    LEFT JOIN tokidf t ON t.country = x.country AND t.tok = x.tok
    WHERE NOT list_contains(x.other, x.tok)
      AND NOT coalesce(length(m.sk) >= 2 AND list_contains(x.osk, m.sk), false)
    GROUP BY x.s1_id, x.cand_id;"""

def _sk_sql(name, id_col, toks):
    return f"""
    CREATE OR REPLACE TABLE {name} AS
    SELECT x.id, list_distinct(list(m.sk)) AS sk
    FROM (SELECT DISTINCT id, tok FROM (
            SELECT id, UNNEST({toks}) AS tok FROM (SELECT DISTINCT {id_col} AS id, {toks} FROM base))) x
    JOIN skmap m ON m.tok = x.tok
    WHERE length(m.sk) >= 2
    GROUP BY x.id;"""

def _feature_setup(con, split, suffix):
    """Tables shared by all parts of one build."""
    cand = WORK_DIR / f"{split}_candidates{suffix}.parquet"
    s1p = WORK_DIR / f"{split}_s1.parquet"
    pool = _pool_files(split)
    idf_path, skel_path = load_idf(con, split), load_skel(con, split)
    con.execute(f"""
    CREATE OR REPLACE TABLE tokidf AS SELECT country, tok, idf FROM read_parquet('{idf_path}');
    CREATE OR REPLACE TABLE skmap AS SELECT tok, sk FROM read_parquet('{skel_path}');
    CREATE OR REPLACE TABLE candall AS
    SELECT s1_id, cand_id, by_a, by_a2, by_b, by_c, by_d, by_e, by_f, by_g, by_h,
           idf_score, shared, containment, rnk
    FROM read_parquet('{cand}');
    CREATE OR REPLACE TABLE ssub AS
    SELECT * FROM read_parquet('{s1p}') WHERE entity_id IN (SELECT DISTINCT s1_id FROM candall);
    CREATE OR REPLACE TABLE psub AS
    SELECT * FROM read_parquet({pool}) WHERE entity_id IN (SELECT DISTINCT cand_id FROM candall);
    CREATE OR REPLACE TABLE s1all AS
    SELECT entity_id, country, name_key, list_distinct(addr_tokens) AS at, list_distinct(addr_nums) AS an
    FROM read_parquet('{s1p}') WHERE name_key <> '';
    CREATE OR REPLACE TABLE s1cnt AS
    SELECT country, name_key, COUNT(*)::INT AS n FROM s1all GROUP BY 1, 2;
    """)
    return con.sql("SELECT COUNT(*) FROM candall").fetchone()[0]

def _feature_query(con, part, n_parts):
    con.execute(f"""
    CREATE OR REPLACE TABLE base AS
    SELECT c.s1_id, c.cand_id, c.by_a, c.by_a2, c.by_b, c.by_c, c.by_d, c.by_e, c.by_f, c.by_g, c.by_h,
           c.idf_score, c.shared, c.containment, c.rnk,
           s.country AS s_country, p.country AS c_country,
           coalesce(s.name_key, '') AS s_key, coalesce(p.name_key, '') AS c_key,
           coalesce(s.name_text, '') AS s_name, coalesce(p.name_text, '') AS c_name,
           coalesce(s.addr_text, '') AS s_addr, coalesce(p.addr_text, '') AS c_addr,
           list_distinct(s.name_tokens) AS st, list_distinct(p.name_tokens) AS ct,
           list_distinct(s.addr_tokens) AS sat, list_distinct(p.addr_tokens) AS cat,
           list_distinct(s.addr_nums) AS snum, list_distinct(p.addr_nums) AS cnum,
           list_filter(string_split(coalesce(s.name_suffix, ''), ' '), x -> x <> '') AS ssuf,
           list_filter(string_split(coalesce(p.name_suffix, ''), ' '), x -> x <> '') AS csuf,
           coalesce(s.postcode, '') AS s_post, coalesce(p.postcode, '') AS c_post,
           (s.addr_empty OR p.addr_empty) AS addr_empty_either
    FROM (SELECT * FROM candall WHERE hash(s1_id) % {n_parts} = {part}) c
    JOIN ssub s ON s.entity_id = c.s1_id
    JOIN psub p ON p.entity_id = c.cand_id;

    CREATE OR REPLACE TABLE w_s AS
    SELECT x.s1_id, SUM(t.idf) AS w
    FROM (SELECT s1_id, country, UNNEST(st) AS tok
          FROM (SELECT DISTINCT s1_id, s_country AS country, st FROM base)) x
    JOIN tokidf t ON t.country = x.country AND t.tok = x.tok GROUP BY x.s1_id;

    CREATE OR REPLACE TABLE w_c AS
    SELECT x.cand_id, SUM(t.idf) AS w
    FROM (SELECT cand_id, country, UNNEST(ct) AS tok
          FROM (SELECT DISTINCT cand_id, s_country AS country, ct FROM base)) x
    JOIN tokidf t ON t.country = x.country AND t.tok = x.tok GROUP BY x.cand_id;

    CREATE OR REPLACE TABLE w_pair AS
    SELECT x.s1_id, x.cand_id, SUM(t.idf) AS w
    FROM (SELECT s1_id, cand_id, s_country AS country, UNNEST(list_intersect(st, ct)) AS tok FROM base) x
    JOIN tokidf t ON t.country = x.country AND t.tok = x.tok GROUP BY x.s1_id, x.cand_id;

    {_sk_sql('s_sk', 's1_id', 'st')}
    {_sk_sql('c_sk', 'cand_id', 'ct')}
    {_leftover_sql('extra', 'cand_id', 'ct', 'st', 's_sk', 's1_id')}
    {_leftover_sql('missing', 's1_id', 'st', 'ct', 'c_sk', 'cand_id')}

    -- Rival fit, once per candidate (not per pair): the best and second-best address fit
    -- among S1 businesses with exactly the candidate's name. For a pair, the rival is the
    -- best one unless that is the pair's own S1, then the second best.
    CREATE OR REPLACE TABLE rtop AS
    SELECT cand_id,
           MAX(CASE WHEN rk = 1 THEN entity_id END) AS e1,
           MAX(CASE WHEN rk = 1 THEN jac END) AS j1,
           MAX(CASE WHEN rk = 2 THEN jac END) AS j2,
           MAX(cnt) AS cnt, MAX(nnum) AS nnum
    FROM (
        SELECT cand_id, entity_id, jac,
               ROW_NUMBER() OVER (PARTITION BY cand_id ORDER BY jac DESC, entity_id) AS rk,
               COUNT(*) OVER (PARTITION BY cand_id) AS cnt,
               SUM(num) OVER (PARTITION BY cand_id) AS nnum
        FROM (
            SELECT c.cand_id, r.entity_id,
                   COALESCE(len(list_intersect(r.at, c.cat))::DOUBLE
                       / NULLIF(len(list_distinct(list_concat(r.at, c.cat))), 0), 0.0) AS jac,
                   CASE WHEN len(list_intersect(r.an, c.cnum)) > 0 THEN 1 ELSE 0 END AS num
            FROM (SELECT DISTINCT cand_id, c_country, c_key, cat, cnum FROM base WHERE c_key <> '') c
            JOIN s1cnt k ON k.country = c.c_country AND k.name_key = c.c_key AND k.n <= 50
            JOIN s1all r ON r.country = c.c_country AND r.name_key = c.c_key
        )
    ) WHERE rk <= 2
    GROUP BY cand_id;

    CREATE OR REPLACE TABLE rival AS
    SELECT s1_id, cand_id,
           CASE WHEN e1 = s1_id THEN j2 ELSE j1 END AS rival_jacc,
           (nnum - (self_in AND self_num)::INT > 0)::INT AS rival_num
    FROM (
        SELECT b.s1_id, b.cand_id, t.e1, t.j1, t.j2, t.cnt, t.nnum,
               (b.s_key = b.c_key AND b.s_country = b.c_country) AS self_in,
               (len(list_intersect(b.snum, b.cnum)) > 0) AS self_num
        FROM base b JOIN rtop t ON t.cand_id = b.cand_id
    ) WHERE cnt - self_in::INT > 0;
    """)
    return f"""
    SELECT b.s1_id, b.cand_id, b.s_name, b.c_name, b.s_addr, b.c_addr,
      coalesce(array_to_string(list_sort(ss.sk), ' '), '') AS s_skstr,
      coalesce(array_to_string(list_sort(cs.sk), ' '), '') AS c_skstr,
      {_J('st', 'ct')} AS n_jacc,
      coalesce(coalesce(wp.w, 0) / NULLIF(ws.w, 0), 0) AS n_idf_cover_s1,
      coalesce(coalesce(wp.w, 0) / NULLIF(wc.w, 0), 0) AS n_idf_cover_c,
      CASE WHEN greatest(length(s_name), length(c_name)) = 0 THEN 0.0
           ELSE least(length(s_name), length(c_name))::DOUBLE
                / greatest(length(s_name), length(c_name)) END AS n_len_ratio,
      abs(len(st) - len(ct)) AS n_tok_diff,
      (s_key = c_key AND s_key <> '')::INT AS n_exact_key,
      (len(list_intersect(ssuf, csuf)) > 0)::INT AS n_suffix_agree,
      (len(ssuf) > 0 AND len(csuf) > 0 AND len(list_intersect(ssuf, csuf)) = 0)::INT AS n_suffix_conflict,
      {_J('ss.sk', 'cs.sk')} AS n_skel_jacc,
      coalesce(ex.n, 0) AS n_extra_cnt, coalesce(ex.mx, 0) AS n_extra_max_idf,
      coalesce(mi.n, 0) AS n_missing_cnt, coalesce(mi.mx, 0) AS n_missing_max_idf,
      {_J('sat', 'cat')} AS a_jacc,
      (s_post <> '' AND s_post = c_post)::INT AS a_post_match,
      (s_post <> '' AND c_post <> '')::INT AS a_post_both,
      {_J('snum', 'cnum')} AS a_num_overlap,
      (len(snum) > 0 AND len(cnum) > 0)::INT AS a_num_both,
      addr_empty_either::INT AS a_empty_either,
      by_a AS b_by_a, by_a2 AS b_by_a2, by_b AS b_by_b, by_c AS b_by_c, by_d AS b_by_d,
      by_e AS b_by_e, by_f AS b_by_f, by_g AS b_by_g, by_h AS b_by_h,
      by_a + by_a2 + by_b + by_c + by_d + by_e + by_f + by_g + by_h AS b_n_schemes,
      coalesce(idf_score, 0) AS b_idf_score, coalesce(shared, 0) AS b_shared,
      coalesce(containment, 0) AS b_containment, rnk AS b_rank,
      CASE WHEN starts_with(b.cand_id, 'S2-') THEN 2 ELSE 3 END AS c_source,
      (s_country = c_country)::INT AS c_same_country,
      coalesce(ks.n, 1) AS x_s1_name_dup, coalesce(kc.n, 0) AS x_c_name_s1cnt,
      rv.rival_jacc AS x_rival_addr,
      rv.rival_jacc - ({_J('sat', 'cat')}) AS x_rival_addr_adv,
      rv.rival_num::DOUBLE AS x_rival_num
    FROM base b
    LEFT JOIN w_pair wp ON wp.s1_id = b.s1_id AND wp.cand_id = b.cand_id
    LEFT JOIN w_s ws ON ws.s1_id = b.s1_id
    LEFT JOIN w_c wc ON wc.cand_id = b.cand_id
    LEFT JOIN s_sk ss ON ss.id = b.s1_id
    LEFT JOIN c_sk cs ON cs.id = b.cand_id
    LEFT JOIN extra ex ON ex.s1_id = b.s1_id AND ex.cand_id = b.cand_id
    LEFT JOIN missing mi ON mi.s1_id = b.s1_id AND mi.cand_id = b.cand_id
    LEFT JOIN s1cnt ks ON ks.country = b.s_country AND ks.name_key = b.s_key
    LEFT JOIN s1cnt kc ON kc.country = b.c_country AND kc.name_key = b.c_key
    LEFT JOIN rival rv ON rv.s1_id = b.s1_id AND rv.cand_id = b.cand_id
    ORDER BY b.s1_id, b.cand_id"""

def _cp(a, b, scorer, scale=100.0):
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32) / scale

def _string_features(df, feats):
    I = IDX
    sn, cn = df["s_name"].tolist(), df["c_name"].tolist()
    sa, ca = df["s_addr"].tolist(), df["c_addr"].tolist()
    feats[:, I["n_token_set"]] = _cp(sn, cn, fuzz.token_set_ratio)
    feats[:, I["n_token_sort"]] = _cp(sn, cn, fuzz.token_sort_ratio)
    feats[:, I["n_partial"]] = _cp(sn, cn, fuzz.partial_ratio)
    feats[:, I["n_jaro"]] = _cp(sn, cn, JaroWinkler.normalized_similarity, 1.0)
    feats[:, I["n_prefix"]] = _cp([s[:12] for s in sn], [c[:12] for c in cn], fuzz.QRatio)
    feats[:, I["n_skel_ratio"]] = _cp(df["s_skstr"].tolist(), df["c_skstr"].tolist(), fuzz.ratio)
    both = (df["s_addr"].to_numpy() != "") & (df["c_addr"].to_numpy() != "")
    feats[:, I["a_token_set"]] = np.where(both, _cp(sa, ca, fuzz.token_set_ratio), 0.0)

def build_features(split, suffix="", chunk=500_000, part_pairs=1_500_000):
    con = connect(mem_gb=8)           # the test run keeps a second DuckDB open
    t0 = time.perf_counter()
    n_pairs = _feature_setup(con, split, suffix)
    n_parts = max(1, math.ceil(n_pairs / part_pairs))
    print(f"  setup {time.perf_counter() - t0:.1f}s  {n_pairs:,} pairs in {n_parts} part(s)")

    out = WORK_DIR / f"{split}_features{suffix}.parquet"
    schema = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string())]
                       + [(c, pa.float32()) for c in FEATURE_COLUMNS])
    writer = pq.ParquetWriter(out, schema, compression="zstd")
    sql_idx = [IDX[c] for c in SQL_FEATURES]
    fill = np.array([c not in NAN_FEATURES for c in SQL_FEATURES])
    total = 0

    def emit(df):
        nonlocal total
        feats = np.zeros((len(df), N_FEATURES), np.float32)
        vals = df[SQL_FEATURES].to_numpy(np.float32)
        vals[:, fill] = np.nan_to_num(vals[:, fill])
        feats[:, sql_idx] = vals
        _string_features(df, feats)
        add_group_features(feats, df["s1_id"].to_numpy())
        writer.write_table(pa.table(
            {"s1_id": pa.array(df["s1_id"]), "cand_id": pa.array(df["cand_id"]),
             **{c: pa.array(feats[:, j], pa.float32()) for j, c in enumerate(FEATURE_COLUMNS)}},
            schema=schema))
        total += len(df)
        print(f"    {total:>12,} pairs ({total / (time.perf_counter() - t0):,.0f}/s)", end="\r", flush=True)

    try:
        for part in range(n_parts):       # a part holds whole entities, so group features stay exact
            query = _feature_query(con, part, n_parts)
            carry = None
            for batch in con.execute(query).fetch_record_batch(chunk):
                df = batch.to_pandas()
                if carry is not None:
                    df = pd.concat([carry, df], ignore_index=True)
                tail = df["s1_id"].to_numpy() == df["s1_id"].iloc[-1]
                carry, df = df[tail], df[~tail]
                if len(df):
                    emit(df.reset_index(drop=True))
            if carry is not None and len(carry):
                emit(carry.reset_index(drop=True))
    finally:
        writer.close()
    print(f"\n  {total:,} pairs in {time.perf_counter() - t0:.1f}s -> {out.name}")
    con.close()
    if total != n_pairs:
        raise RuntimeError(f"feature rows {total:,} != candidate pairs {n_pairs:,}")
    return out

def check_features(split="train", n_ent=200):
    """Recompute a sample of entities with the slow reference; raise on any difference."""
    fast = pd.read_parquet(WORK_DIR / f"{split}_features.parquet")
    ent = fast["s1_id"].drop_duplicates()
    ids = ent.sample(min(n_ent, len(ent)), random_state=0).tolist()
    cand, s1p, pool = (WORK_DIR / f"{split}_candidates.parquet", WORK_DIR / f"{split}_s1.parquet",
                       _pool_files(split))
    con = connect(mem_gb=8)
    t = pq.read_table(load_idf(con, split)).to_pydict()
    idf = dict(zip(zip(t["country"], t["tok"]), t["idf"]))
    ids_tbl = pa.table({"s1_id": pa.array(ids, pa.string())})
    con.register("ids", ids_tbl)
    con.execute(f"CREATE OR REPLACE TABLE csub AS SELECT c.* FROM read_parquet('{cand}') c JOIN ids USING (s1_id)")
    con.execute(f"""CREATE OR REPLACE TABLE s1cnt AS SELECT country, name_key, COUNT(*)::INT AS n
        FROM read_parquet('{s1p}') WHERE name_key <> '' GROUP BY 1, 2""")
    con.execute(f"""CREATE OR REPLACE TABLE rival AS
        SELECT c.s1_id, c.cand_id,
               MAX(COALESCE(
                   len(list_intersect(list_distinct(r.addr_tokens), list_distinct(p.addr_tokens)))::DOUBLE
                   / NULLIF(len(list_distinct(list_concat(r.addr_tokens, p.addr_tokens))), 0), 0.0)) AS rival_jacc,
               MAX(CASE WHEN len(list_intersect(r.addr_nums, p.addr_nums)) > 0 THEN 1 ELSE 0 END) AS rival_num
        FROM csub c
        JOIN read_parquet({pool}) p ON p.entity_id = c.cand_id
        JOIN s1cnt k ON k.country = p.country AND k.name_key = p.name_key AND k.n <= 50
        JOIN read_parquet('{s1p}') r ON r.country = p.country AND r.name_key = p.name_key
                                     AND r.entity_id <> c.s1_id
        GROUP BY 1, 2""")
    df = con.execute(f"""
        SELECT c.s1_id, c.cand_id, c.by_a, c.by_a2, c.by_b, c.by_c, c.by_d, c.by_e,
               c.by_f, c.idf_score, c.shared, c.containment, c.rnk,
               s.country s_country, s.name_key s_name_key, s.name_text s_name_text,
               s.name_tokens s_name_tokens, s.name_suffix s_name_suffix,
               s.addr_text s_addr_text, s.addr_tokens s_addr_tokens,
               s.addr_nums s_addr_nums, s.postcode s_postcode, s.addr_empty s_addr_empty,
               p.country c_country, p.name_key c_name_key, p.name_text c_name_text,
               p.name_tokens c_name_tokens, p.name_suffix c_name_suffix,
               p.addr_text c_addr_text, p.addr_tokens c_addr_tokens,
               p.addr_nums c_addr_nums, p.postcode c_postcode, p.addr_empty c_addr_empty,
               COALESCE(ks.n, 1) x_s1_dup, COALESCE(kc.n, 0) x_c_s1cnt,
               rv.rival_jacc x_rival_jacc, rv.rival_num x_rival_num
        FROM csub c
        JOIN read_parquet('{s1p}') s ON s.entity_id = c.s1_id
        JOIN read_parquet({pool}) p ON p.entity_id = c.cand_id
        LEFT JOIN s1cnt ks ON ks.country = s.country AND ks.name_key = s.name_key
        LEFT JOIN s1cnt kc ON kc.country = p.country AND kc.name_key = p.name_key
        LEFT JOIN rival rv ON rv.s1_id = c.s1_id AND rv.cand_id = c.cand_id
        ORDER BY c.s1_id, c.cand_id""").df()
    con.close()
    ref = pair_features(df, idf)
    add_group_features(ref, df["s1_id"].to_numpy())
    ref = pd.DataFrame(ref, columns=FEATURE_COLUMNS).assign(s1_id=df["s1_id"].values,
                                                            cand_id=df["cand_id"].values)
    m = ref.merge(fast, on=["s1_id", "cand_id"], suffixes=("_r", "_f"))
    # the reference predates the G/H flags
    skip = {"b_by_g", "b_by_h", "b_n_schemes"}
    bad = []
    for c in FEATURE_COLUMNS:
        a, b = m[f"{c}_r"].to_numpy(np.float64), m[f"{c}_f"].to_numpy(np.float64)
        same = np.isclose(a, b, atol=1e-3) | (np.isnan(a) & np.isnan(b))
        frac = 1.0 - same.mean()
        if frac > 0:
            print(f"  {c:18s} differs on {frac:.2%}")
        if frac > 0.001 and c not in skip:
            bad.append(c)
    print(f"  checked {len(m):,} pairs of {len(ids)} entities (reference rows {len(ref):,})")
    if bad or len(m) != len(ref):
        raise RuntimeError(f"fast features differ from the reference: {bad}")
    print("  EQUIVALENT")
