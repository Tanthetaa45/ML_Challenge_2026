"""Stage 1: DuckDB blocking. Schemes A, A2, B-H, each kept to its rank `rk`; fair-slot union."""
import shutil
import time
from pathlib import Path

import duckdb

from .config import CONFIG, DATA_DIR, GROUND_TRUTH, WORK_DIR
from .metric import f05, load_ground_truth

DOMAIN_TOKENS = "('com','www','net','org','co','in','io','biz','info','html')"
SQUASH_SQL = f"array_to_string(list_filter(name_tokens, t -> t NOT IN {DOMAIN_TOKENS}), '')"
SKELETON_SQL = ("regexp_replace(regexp_replace(regexp_replace(regexp_replace("
                "tok,'sh','s','g'),'ph','f','g'),'c','k','g'),'[aeiou]','','g')")
# Scheme G: up to 3 address numbers plus the postcode (5-digit US house numbers
# get parsed as ZIP codes, so the postcode doubles as a house number).
GNUM_SQL = ("CASE WHEN postcode <> '' THEN list_append(list_slice(addr_nums, 1, 3), postcode) "
            "ELSE list_slice(addr_nums, 1, 3) END")

def connect(threads=None, mem_gb=None, spill_gb=None, db=None):
    """Hard spill ceiling, so a bad config fails instead of filling the disk."""
    threads = threads or CONFIG["threads"]; mem_gb = mem_gb or CONFIG["mem_gb"]
    spill_gb = spill_gb or CONFIG["spill_gb"]
    con = duckdb.connect(str(db) if db else ":memory:")
    tmp = WORK_DIR / "duckdb_tmp"; tmp.mkdir(parents=True, exist_ok=True)
    free_gb = shutil.disk_usage(tmp).free / 2**30
    if free_gb < spill_gb + 2:
        spill_gb = max(1, int(free_gb) - 2)
        print(f"  ! only {free_gb:.1f}GB free, capping spill at {spill_gb}GB")
    con.execute(f"SET threads={threads}")
    con.execute(f"SET memory_limit='{mem_gb}GB'")
    con.execute(f"SET temp_directory='{tmp}'")
    con.execute(f"SET max_temp_directory_size='{spill_gb}GB'")
    con.execute("SET preserve_insertion_order=false")
    return con

def estimate_pairs(con, table, label, budget=None):
    budget = budget or CONFIG["pair_budget"]
    est = int(con.sql(f"SELECT COALESCE(SUM(df),0) FROM {table}").fetchone()[0] or 0)
    print(f"  estimate {label:<10s} {est:>14,} pairs  [{'OK' if est <= budget else 'OVER BUDGET'}]")
    if est > budget:
        raise RuntimeError(f"{label}: ~{est:,} pairs exceeds budget {budget:,}.")
    return est

def _step(con, label, sql):
    t = time.perf_counter(); con.execute(sql)
    print(f"  {label:<22s} {time.perf_counter() - t:6.1f}s", flush=True)


def build_pool(con, split):
    pool = [str(WORK_DIR / f"{split}_s2.parquet"), str(WORK_DIR / f"{split}_s3.parquet")]
    con.execute(f"""CREATE OR REPLACE VIEW cand AS
        SELECT entity_id, country, name_key, name_tokens, addr_tokens, postcode, addr_nums
        FROM read_parquet({pool})""")
    n_cand = con.sql("SELECT COUNT(*) FROM cand").fetchone()[0]
    print(f"  pool {n_cand:,} records")

    _step(con, "name tokens", """
        CREATE OR REPLACE TABLE cand_tok AS
        SELECT entity_id, country, UNNEST(name_tokens) AS tok FROM cand""")
    _step(con, "token df", f"""
        CREATE OR REPLACE TABLE tokdf AS
        SELECT country, tok, COUNT(*)::BIGINT AS df, LN(1.0 + {n_cand}.0 / COUNT(*)) AS idf
        FROM cand_tok GROUP BY country, tok""")
    _step(con, "squash keys", f"""
        CREATE OR REPLACE TABLE cand_keys AS
        SELECT entity_id, country, {SQUASH_SQL} AS squash FROM cand""")
    _step(con, "skeletons", f"""
        CREATE OR REPLACE TABLE cand_skel AS
        SELECT entity_id, country, skel FROM (
            SELECT entity_id, country, {SKELETON_SQL} AS skel FROM cand_tok
        ) WHERE length(skel) >= 3;
        CREATE OR REPLACE TABLE skeldf AS
        SELECT country, skel, COUNT(*)::BIGINT AS df, LN(1.0 + {n_cand}.0 / COUNT(*)) AS idf
        FROM cand_skel GROUP BY country, skel;""")
    _step(con, "address tokens", f"""
        CREATE OR REPLACE TABLE cand_atok AS
        SELECT entity_id, country, UNNEST(addr_tokens) AS atok FROM cand;
        CREATE OR REPLACE TABLE atokdf AS
        SELECT country, atok, COUNT(*)::BIGINT AS df, LN(1.0 + {n_cand}.0 / COUNT(*)) AS idf
        FROM cand_atok GROUP BY country, atok;""")
    _step(con, "token pairs", f"""
        CREATE OR REPLACE TABLE cand_rare AS
        SELECT entity_id, country, tok FROM (
            SELECT c.entity_id, c.country, c.tok,
                   ROW_NUMBER() OVER (PARTITION BY c.entity_id ORDER BY t.df ASC, c.tok) AS rk
            FROM cand_tok c JOIN tokdf t ON t.country = c.country AND t.tok = c.tok
        ) WHERE rk <= {CONFIG['pair_tokens']};
        CREATE OR REPLACE TABLE cand_pair AS
        SELECT a.entity_id, a.country, a.tok || '|' || b.tok AS pk
        FROM cand_rare a JOIN cand_rare b ON b.entity_id = a.entity_id AND b.tok > a.tok;
        CREATE OR REPLACE TABLE pairdf AS
        SELECT country, pk, COUNT(*)::BIGINT AS df, LN(1.0 + {n_cand}.0 / COUNT(*)) AS idf
        FROM cand_pair GROUP BY country, pk;""")
    _step(con, "addr num keys (G)", f"""
        CREATE OR REPLACE TABLE cand_g AS
        SELECT DISTINCT t.entity_id, t.country, n.num || '|' || t.atok AS gk
        FROM (
            SELECT entity_id, country, atok FROM (
                SELECT c.entity_id, c.country, c.atok,
                       ROW_NUMBER() OVER (PARTITION BY c.entity_id ORDER BY a.df ASC, c.atok) AS rk
                FROM cand_atok c JOIN atokdf a ON a.country = c.country AND a.atok = c.atok
                WHERE length(c.atok) >= 3
            ) WHERE rk <= {CONFIG['g_tokens']}
        ) t
        JOIN (SELECT entity_id, UNNEST({GNUM_SQL}) AS num FROM cand) n ON n.entity_id = t.entity_id;
        CREATE OR REPLACE TABLE gdf AS
        SELECT country, gk, COUNT(*)::BIGINT AS df, LN(1.0 + {n_cand}.0 / COUNT(*)) AS idf
        FROM cand_g GROUP BY country, gk;""")
    _step(con, "name x addr keys (H)", f"""
        CREATE OR REPLACE TABLE cand_h AS
        SELECT DISTINCT n.entity_id, n.country, n.tok || '|' || a.atok AS hk
        FROM (
            SELECT entity_id, country, tok FROM (
                SELECT c.entity_id, c.country, c.tok,
                       ROW_NUMBER() OVER (PARTITION BY c.entity_id ORDER BY td.df ASC, c.tok) AS rk
                FROM cand_tok c JOIN tokdf td ON td.country = c.country AND td.tok = c.tok
                WHERE length(c.tok) >= 3 AND c.tok NOT IN {DOMAIN_TOKENS}
            ) WHERE rk <= {CONFIG['h_name_tokens']}
        ) n
        JOIN (
            SELECT entity_id, atok FROM (
                SELECT c.entity_id, c.atok,
                       ROW_NUMBER() OVER (PARTITION BY c.entity_id ORDER BY ad.df ASC, c.atok) AS rk
                FROM cand_atok c JOIN atokdf ad ON ad.country = c.country AND ad.atok = c.atok
                WHERE length(c.atok) >= 3
            ) WHERE rk <= {CONFIG['h_addr_tokens']}
        ) a ON a.entity_id = n.entity_id;
        CREATE OR REPLACE TABLE hdf AS
        SELECT country, hk, COUNT(*)::BIGINT AS df, LN(1.0 + {n_cand}.0 / COUNT(*)) AS idf
        FROM cand_h GROUP BY country, hk;""")
    return n_cand


def build_shard(con, split, s1_filter, suffix="", top_k=None):
    cfg, top_k = CONFIG, top_k or CONFIG["top_k"]
    s1p = str(WORK_DIR / f"{split}_s1.parquet")
    con.execute(f"""CREATE OR REPLACE VIEW s1 AS
        SELECT entity_id, country, name_key, name_tokens, addr_tokens, postcode, addr_nums
        FROM read_parquet('{s1p}') {s1_filter}""")
    n_s1 = con.sql("SELECT COUNT(*) FROM s1").fetchone()[0]
    print(f"  S1 rows {n_s1:,}")

    _step(con, "s1 rare tokens", f"""
        CREATE OR REPLACE TABLE s1_rare AS
        SELECT entity_id, country, tok, df, idf FROM (
            SELECT s.entity_id, s.country, t.tok, t.df, t.idf,
                   ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY t.df ASC, t.tok) AS rk
            FROM (SELECT entity_id, country, UNNEST(name_tokens) AS tok FROM s1) s
            JOIN tokdf t ON t.country = s.country AND t.tok = s.tok
        ) WHERE rk <= {cfg['rare_per_s1']} AND df <= {cfg['max_token_df']}""")
    estimate_pairs(con, "s1_rare", "scheme B")

    _step(con, "s1 skeletons", f"""
        CREATE OR REPLACE TABLE s1_skel AS
        SELECT entity_id, country, skel, df, idf FROM (
            SELECT s.entity_id, s.country, k.skel, k.df, k.idf,
                   ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY k.df ASC, k.skel) AS rk
            FROM (SELECT entity_id, country, {SKELETON_SQL} AS skel
                  FROM (SELECT entity_id, country, UNNEST(name_tokens) AS tok FROM s1)) s
            JOIN skeldf k ON k.country = s.country AND k.skel = s.skel
            WHERE length(s.skel) >= 3
        ) WHERE rk <= {cfg['rare_per_s1']} AND df <= {cfg['skel_max_df']}""")
    estimate_pairs(con, "s1_skel", "scheme C")

    _step(con, "s1 rare addr", f"""
        CREATE OR REPLACE TABLE s1_arare AS
        SELECT entity_id, country, atok, df, idf FROM (
            SELECT s.entity_id, s.country, t.atok, t.df, t.idf,
                   ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY t.df ASC, t.atok) AS rk
            FROM (SELECT entity_id, country, UNNEST(addr_tokens) AS atok FROM s1) s
            JOIN atokdf t ON t.country = s.country AND t.atok = s.atok
        ) WHERE rk <= {cfg['addr_rare_per_s1']} AND df <= {cfg['addr_max_df']}""")
    estimate_pairs(con, "s1_arare", "scheme E")

    _step(con, "s1 token pairs", f"""
        CREATE OR REPLACE TABLE s1_rare4 AS
        SELECT entity_id, country, tok FROM (
            SELECT s.entity_id, s.country, s.tok,
                   ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY t.df ASC, s.tok) AS rk
            FROM (SELECT entity_id, country, UNNEST(name_tokens) AS tok FROM s1) s
            JOIN tokdf t ON t.country = s.country AND t.tok = s.tok
        ) WHERE rk <= {cfg['pair_tokens']};
        CREATE OR REPLACE TABLE s1_pair AS
        SELECT entity_id, country, pk, df, idf FROM (
            SELECT sp.entity_id, sp.country, sp.pk, d.df, d.idf,
                   ROW_NUMBER() OVER (PARTITION BY sp.entity_id ORDER BY d.df ASC, sp.pk) AS rk
            FROM (SELECT a.entity_id, a.country, a.tok || '|' || b.tok AS pk
                  FROM s1_rare4 a JOIN s1_rare4 b ON b.entity_id = a.entity_id AND b.tok > a.tok) sp
            JOIN pairdf d ON d.country = sp.country AND d.pk = sp.pk
        ) WHERE rk <= {cfg['pair_per_s1']} AND df <= {cfg['pair_max_df']}""")
    estimate_pairs(con, "s1_pair", "scheme F")

    _step(con, "s1 addr num keys", f"""
        CREATE OR REPLACE TABLE s1_g AS
        SELECT entity_id, country, gk, df, idf FROM (
            SELECT k.entity_id, k.country, k.gk, d.df, d.idf,
                   ROW_NUMBER() OVER (PARTITION BY k.entity_id ORDER BY d.df ASC, k.gk) AS rk
            FROM (
                SELECT DISTINCT t.entity_id, t.country, n.num || '|' || t.atok AS gk
                FROM (
                    SELECT entity_id, country, atok FROM (
                        SELECT s.entity_id, s.country, s.atok,
                               ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY a.df ASC, s.atok) AS rk
                        FROM (SELECT entity_id, country, UNNEST(addr_tokens) AS atok FROM s1) s
                        JOIN atokdf a ON a.country = s.country AND a.atok = s.atok
                        WHERE length(s.atok) >= 3
                    ) WHERE rk <= {cfg['g_tokens']}
                ) t
                JOIN (SELECT entity_id, UNNEST({GNUM_SQL}) AS num FROM s1) n ON n.entity_id = t.entity_id
            ) k JOIN gdf d ON d.country = k.country AND d.gk = k.gk
        ) WHERE rk <= {cfg['g_per_s1']} AND df <= {cfg['g_max_df']}""")
    estimate_pairs(con, "s1_g", "scheme G")

    _step(con, "s1 name x addr keys", f"""
        CREATE OR REPLACE TABLE s1_h AS
        SELECT entity_id, country, hk, df, idf FROM (
            SELECT k.entity_id, k.country, k.hk, d.df, d.idf,
                   ROW_NUMBER() OVER (PARTITION BY k.entity_id ORDER BY d.df ASC, k.hk) AS rk
            FROM (
                SELECT DISTINCT n.entity_id, n.country, n.tok || '|' || a.atok AS hk
                FROM (
                    SELECT entity_id, country, tok FROM (
                        SELECT s.entity_id, s.country, s.tok,
                               ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY td.df ASC, s.tok) AS rk
                        FROM (SELECT entity_id, country, UNNEST(name_tokens) AS tok FROM s1) s
                        JOIN tokdf td ON td.country = s.country AND td.tok = s.tok
                        WHERE length(s.tok) >= 3 AND s.tok NOT IN {DOMAIN_TOKENS}
                    ) WHERE rk <= {cfg['h_name_tokens']}
                ) n
                JOIN (
                    SELECT entity_id, atok FROM (
                        SELECT s.entity_id, s.atok,
                               ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY ad.df ASC, s.atok) AS rk
                        FROM (SELECT entity_id, country, UNNEST(addr_tokens) AS atok FROM s1) s
                        JOIN atokdf ad ON ad.country = s.country AND ad.atok = s.atok
                        WHERE length(s.atok) >= 3
                    ) WHERE rk <= {cfg['h_addr_tokens']}
                ) a ON a.entity_id = n.entity_id
            ) k JOIN hdf d ON d.country = k.country AND d.hk = k.hk
        ) WHERE rk <= {cfg['h_per_s1']} AND df <= {cfg['h_max_df']}""")
    estimate_pairs(con, "s1_h", "scheme H")

    # ---- the schemes; each keeps its internal rank rk ----
    _step(con, "A exact key", f"""
        CREATE OR REPLACE TABLE pairs_a AS
        SELECT s1_id, cand_id, rk FROM (
            SELECT s.entity_id AS s1_id, c.entity_id AS cand_id,
                   ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY
                       (c.postcode <> '' AND c.postcode = s.postcode) DESC, c.entity_id) AS rk
            FROM s1 s JOIN cand c ON c.country = s.country AND c.name_key = s.name_key
            WHERE s.name_key <> ''
        ) WHERE rk <= {cfg['key_cap']}""")
    _step(con, "A2 squash", f"""
        CREATE OR REPLACE TABLE pairs_a2 AS
        SELECT s1_id, cand_id, rk FROM (
            SELECT s.entity_id AS s1_id, c.entity_id AS cand_id,
                   ROW_NUMBER() OVER (PARTITION BY s.entity_id ORDER BY c.entity_id) AS rk
            FROM (SELECT entity_id, country, {SQUASH_SQL} AS squash FROM s1) s
            JOIN cand_keys c ON c.country = s.country AND c.squash = s.squash
            WHERE length(s.squash) >= 4
        ) WHERE rk <= {cfg['key_cap']}""")

    def ranked(tag, key_tbl, key_col, cand_tbl, cap, having=""):
        _step(con, f"{tag} scheme", f"""
            CREATE OR REPLACE TABLE pairs_{tag} AS
            SELECT s1_id, cand_id, idf_score, shared, rk FROM (
                SELECT s1_id, cand_id, idf_score, shared,
                       ROW_NUMBER() OVER (PARTITION BY s1_id
                           ORDER BY idf_score DESC, shared DESC, cand_id) AS rk
                FROM (
                    SELECT r.entity_id AS s1_id, c.entity_id AS cand_id,
                           SUM(r.idf) AS idf_score, COUNT(*)::INT AS shared
                    FROM {key_tbl} r JOIN {cand_tbl} c
                      ON c.country = r.country AND c.{key_col} = r.{key_col}
                    GROUP BY 1, 2 {having}
                )
            ) WHERE rk <= {cap}""")

    ranked("b", "s1_rare", "tok", "cand_tok", cfg["scheme_cap"])
    ranked("c", "s1_skel", "skel", "cand_skel", cfg["scheme_cap"])
    ranked("e", "s1_arare", "atok", "cand_atok", cfg["scheme_cap"], "HAVING COUNT(*) >= 2")
    ranked("f", "s1_pair", "pk", "cand_pair", cfg["scheme_cap"])
    ranked("g", "s1_g", "gk", "cand_g", cfg["g_cap"])
    ranked("h", "s1_h", "hk", "cand_h", cfg["h_cap"])

    _step(con, "D postcode", """
        CREATE OR REPLACE TABLE pairs_d AS
        SELECT s1_id, cand_id, shared,
               ROW_NUMBER() OVER (PARTITION BY s1_id ORDER BY shared DESC, cand_id) AS rk
        FROM (
            SELECT s1_id, cand_id, COUNT(*)::INT AS shared FROM (
                SELECT s.entity_id AS s1_id, c.entity_id AS cand_id
                FROM (SELECT entity_id, country, postcode, UNNEST(name_tokens) AS tok
                      FROM s1 WHERE postcode <> '') s
                JOIN (SELECT entity_id, country, postcode, UNNEST(name_tokens) AS tok
                      FROM cand WHERE postcode <> '') c
                  ON c.country = s.country AND c.postcode = s.postcode AND c.tok = s.tok
            ) GROUP BY 1, 2
        )""")

    _step(con, "s1 idf norm", """
        CREATE OR REPLACE TABLE s1_norm AS
        SELECT s.entity_id, SUM(COALESCE(t.idf, 0.0)) AS total_idf
        FROM (SELECT entity_id, country, UNNEST(name_tokens) AS tok FROM s1) s
        LEFT JOIN tokdf t ON t.country = s.country AND t.tok = s.tok
        GROUP BY 1""")
    # Fair slots: a candidate's priority is its BEST rank inside any scheme (all #1s,
    # then all #2s, ...), so one scheme can't crowd out the others.
    _step(con, "union (uncapped)", """
        CREATE OR REPLACE TABLE upairs AS
        SELECT u.s1_id, u.cand_id, u.by_a, u.by_a2, u.by_b, u.by_c, u.by_d, u.by_e, u.by_f,
               u.by_g, u.by_h, u.idf_score, u.shared,
               u.idf_score / NULLIF(n.total_idf, 0) AS containment,
               ROW_NUMBER() OVER (PARTITION BY u.s1_id ORDER BY
                   u.best_srk ASC, GREATEST(u.by_a, u.by_a2) DESC,
                   u.idf_score / NULLIF(n.total_idf, 0) DESC,
                   u.shared DESC, u.idf_score DESC, u.cand_id) AS rnk
        FROM (
            SELECT s1_id, cand_id, MAX(by_a) by_a, MAX(by_a2) by_a2, MAX(by_b) by_b,
                   MAX(by_c) by_c, MAX(by_d) by_d, MAX(by_e) by_e, MAX(by_f) by_f,
                   MAX(by_g) by_g, MAX(by_h) by_h,
                   MAX(idf_score) idf_score, MAX(shared) shared, MIN(srk) best_srk
            FROM (
                SELECT s1_id, cand_id, 1 by_a, 0 by_a2, 0 by_b, 0 by_c, 0 by_d, 0 by_e, 0 by_f,
                       0 by_g, 0 by_h, 0.0::DOUBLE idf_score, 0 shared, rk srk FROM pairs_a
                UNION ALL SELECT s1_id,cand_id,0,1,0,0,0,0,0,0,0,0.0,0,rk FROM pairs_a2
                UNION ALL SELECT s1_id,cand_id,0,0,1,0,0,0,0,0,0,idf_score,shared,rk FROM pairs_b
                UNION ALL SELECT s1_id,cand_id,0,0,0,1,0,0,0,0,0,idf_score,shared,rk FROM pairs_c
                UNION ALL SELECT s1_id,cand_id,0,0,0,0,1,0,0,0,0,0.0,shared,rk FROM pairs_d
                UNION ALL SELECT s1_id,cand_id,0,0,0,0,0,1,0,0,0,idf_score,shared,rk FROM pairs_e
                UNION ALL SELECT s1_id,cand_id,0,0,0,0,0,0,1,0,0,idf_score,shared,rk FROM pairs_f
                UNION ALL SELECT s1_id,cand_id,0,0,0,0,0,0,0,1,0,idf_score,shared,rk FROM pairs_g
                UNION ALL SELECT s1_id,cand_id,0,0,0,0,0,0,0,0,1,idf_score,shared,rk FROM pairs_h
            ) GROUP BY s1_id, cand_id
        ) u LEFT JOIN s1_norm n ON n.entity_id = u.s1_id""")
    _step(con, "top-k cut", f"CREATE OR REPLACE TABLE candidates AS SELECT * FROM upairs WHERE rnk <= {top_k}")

    out = WORK_DIR / f"{split}_candidates{suffix}.parquet"
    con.execute(f"COPY candidates TO '{out}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    for t in ("pairs_a", "pairs_a2", "pairs_b", "pairs_c", "pairs_d", "pairs_e",
              "pairs_f", "pairs_g", "pairs_h", "upairs", "candidates"):
        print(f"  {t:<12s} {con.sql(f'SELECT COUNT(*) FROM {t}').fetchone()[0]:>13,}")
    return out


# ---- training sample ----
DEV_DB = WORK_DIR / "er_train.duckdb"

def block_train_sample():
    """Block the s1_sample per mille of train S1 against the FULL train pool, in 2% slices.
    File-backed DuckDB: the pool index with G/H keys does not fit 16 GB in memory."""
    for suf in ("", ".wal"):
        Path(f"{DEV_DB}{suf}").unlink(missing_ok=True)
    con = connect(db=DEV_DB)
    t0 = time.perf_counter()
    print("--- pool index (built once) ---")
    build_pool(con, "train")

    S, STEP = CONFIG["s1_sample"], 20
    parts = []
    con.execute("CREATE OR REPLACE TABLE upairs_all (s1_id VARCHAR, cand_id VARCHAR, rnk BIGINT)")
    for lo in range(0, S, STEP):
        hi = min(lo + STEP, S)
        print(f"\n--- slice: S1 hash {lo}-{hi} per mille ---")
        parts.append(str(build_shard(con, "train",
            f"WHERE hash(entity_id) % 1000 >= {lo} AND hash(entity_id) % 1000 < {hi}",
            suffix=f"_p{lo:03d}")))
        con.execute("INSERT INTO upairs_all SELECT s1_id, cand_id, rnk FROM upairs")
    cand_path = WORK_DIR / "train_candidates.parquet"
    con.execute(f"COPY (SELECT * FROM read_parquet({parts})) TO '{cand_path}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    for p in parts:
        Path(p).unlink()

    # diagnostics below must see the whole sample, not only the last slice
    con.execute(f"""CREATE OR REPLACE VIEW s1 AS SELECT * FROM read_parquet('{WORK_DIR / "train_s1.parquet"}')
                    WHERE hash(entity_id) % 1000 < {S}""")
    con.execute("CREATE OR REPLACE TABLE upairs AS SELECT * FROM upairs_all")
    con.execute(f"CREATE OR REPLACE TABLE candidates AS SELECT * FROM read_parquet('{cand_path}')")
    print(f"\ntotal {time.perf_counter() - t0:.1f}s -> {cand_path.name}")
    return con


def diagnose(con, ks=(10, 30, 50, 100, 200, 500)):
    gt_path = str(DATA_DIR / "train" / GROUND_TRUTH)
    con.execute(f"""
        CREATE OR REPLACE TABLE gt AS
        SELECT g.s1_id, TRIM(g.cand_id) AS cand_id FROM (
            SELECT source1_entity_id AS s1_id,
                   UNNEST(STRING_SPLIT(matched_entity_ids, ',')) AS cand_id
            FROM read_csv('{gt_path}', delim='\t', header=true,
                 columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
            WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
        ) g JOIN s1 ON s1.entity_id = g.s1_id""")
    cols = ", ".join(f"SUM(CASE WHEN rnk<={k} THEN 1 ELSE 0 END)*1.0/COUNT(*) AS r{k}" for k in ks)
    row = con.sql(f"""
        WITH m AS (SELECT g.s1_id, g.cand_id, u.rnk FROM gt g
                   LEFT JOIN upairs u ON u.s1_id=g.s1_id AND u.cand_id=g.cand_id)
        SELECT COUNT(*), SUM(CASE WHEN rnk IS NOT NULL THEN 1 ELSE 0 END)*1.0/COUNT(*), {cols}
        FROM m""").fetchone()
    print(f"  true pairs            {row[0]:,}")
    print(f"  uncapped (generation) {row[1]:.4f}")
    for k, v in zip(ks, row[2:]):
        print(f"  recall@{k:<4d}          {v:.4f}")

def blocking_report(candidates: dict, truth: dict):
    n = len(truth); tp = cp = hits = 0
    rec = perfect = zero = ceil_sum = 0.0
    n_single = 0
    for sid, tset in truth.items():
        cand = candidates.get(sid, set()); hit = tset & cand
        cp += len(cand); tp += len(tset); hits += len(hit)
        if tset:
            r = len(hit) / len(tset); rec += r; perfect += (r == 1.0); zero += (r == 0.0)
        else:
            n_single += 1; rec += 1.0; perfect += 1.0
        ceil_sum += f05(hit, tset)
    print(f"  entities              {n:,}")
    print(f"  mean candidates/ent   {cp / n:.1f}")
    print(f"  recall (micro)        {hits / tp:.4f}")
    print(f"  entities zero recall  {zero / max(1, n - n_single):.4f}")
    print(f"  CEILING macro F_0.5   {ceil_sum / n:.4f}")

def inspect_misses(con, n=15):
    pool = [str(WORK_DIR / "train_s2.parquet"), str(WORK_DIR / "train_s3.parquet")]
    s1p = str(WORK_DIR / "train_s1.parquet")
    rows = con.sql(f"""
        WITH miss AS (SELECT g.s1_id, g.cand_id FROM gt g
                      LEFT JOIN upairs u ON u.s1_id=g.s1_id AND u.cand_id=g.cand_id
                      WHERE u.s1_id IS NULL)
        SELECT s.name_text, p.name_text, s.name_tokens, p.name_tokens, s.country,
               s.addr_tokens, p.addr_tokens
        FROM miss m JOIN read_parquet('{s1p}') s ON s.entity_id=m.s1_id
        JOIN read_parquet({pool}) p ON p.entity_id=m.cand_id
        USING SAMPLE {n} ROWS""").fetchall()
    for sn, cn, st, ct, country, sat, cat in rows:
        ov = set(st or ()) & set(ct or ()); aov = set(sat or ()) & set(cat or ())
        print(f"  [{country:6s}] {sn!r:36s} vs {cn!r:36s} shared={sorted(ov)} addr={sorted(aov)[:4]}")


def report_and_close(con):
    """Read-only blocking diagnostics on the training sample, then free the disk for the test run."""
    print("=== RECALL@K ===");  diagnose(con)
    rows = con.sql("SELECT s1_id, cand_id FROM candidates").fetchnumpy()
    cands = {}
    for s, c in zip(rows["s1_id"], rows["cand_id"]):
        cands.setdefault(s, set()).add(c)
    s1_ids = set(con.sql("SELECT entity_id FROM s1").fetchnumpy()["entity_id"])
    print("\n=== BLOCKING REPORT ===");  blocking_report(cands, load_ground_truth(s1_ids))
    inspect_misses(con)
    con.close()
    for suf in ("", ".wal"):
        Path(f"{DEV_DB}{suf}").unlink(missing_ok=True)
