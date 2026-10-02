"""Sharded test inference (resumable), the plain v9 submission, and the validator call."""
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .blocking import build_pool, build_shard, connect
from .config import CONFIG, DATA_DIR, OUTPUT_DIR, VALIDATOR, WORK_DIR
from .features import build_features
from .model import predict_shard
from .select import repair_exclusivity, select_sets


def run_test(booster, iso, shards=None, resume=True):
    shards = shards or CONFIG["test_shards"]
    if not resume:
        for pat in ("test_probs*", "test_sums*", "test_candidates*"):
            for f in WORK_DIR.glob(f"{pat}.parquet"):
                f.unlink()
    for f in WORK_DIR.glob("test_features*.parquet"):
        f.unlink()

    def done(sfx):
        return all((WORK_DIR / f"test_{k}{sfx}.parquet").exists() for k in ("probs", "sums", "candidates"))

    todo = [i for i in range(shards) if not done(f"_sh{i:02d}")]
    if not todo:
        print("all shards already complete - go straight to assemble()")
        return
    print(f"shards to run: {todo}")
    dbfile = WORK_DIR / "er_test.duckdb"
    for suf in ("", ".wal"):
        Path(f"{dbfile}{suf}").unlink(missing_ok=True)
    con = connect(db=dbfile)
    print("--- pool index (once, on disk) ---"); build_pool(con, "test")
    for i in todo:
        sfx = f"_sh{i:02d}"
        t_shard = time.perf_counter()
        print(f"\n--- shard {i+1}/{shards} ---")
        build_shard(con, "test", f"WHERE hash(entity_id) % {shards} = {i}", sfx)
        for t in ("upairs", "candidates", "s1_rare", "s1_skel", "s1_arare", "s1_rare4", "s1_pair",
                  "s1_g", "s1_h", "s1_norm", "pairs_a", "pairs_a2", "pairs_b", "pairs_c",
                  "pairs_d", "pairs_e", "pairs_f", "pairs_g", "pairs_h"):
            con.execute(f"DROP TABLE IF EXISTS {t}")
        con.execute("CHECKPOINT")
        build_features("test", sfx)
        predict_shard("test", sfx, booster, iso)
        (WORK_DIR / f"test_features{sfx}.parquet").unlink(missing_ok=True)
        cpath = WORK_DIR / f"test_candidates{sfx}.parquet"
        pq.write_table(pq.read_table(cpath, columns=["s1_id", "cand_id"]), cpath, compression="zstd")
        print(f"  shard {time.perf_counter() - t_shard:.0f}s  free disk {shutil.disk_usage(WORK_DIR).free/2**30:.1f} GB")
    con.close()
    for suf in ("", ".wal"):
        Path(f"{dbfile}{suf}").unlink(missing_ok=True)


def assemble(split="test"):
    """The plain v9 submission (stage-1 model only) -> OUTPUT_DIR. The final submission is
    written by 04_final_selection.py; this one is kept as the reference it must beat."""
    probs = pd.concat([pd.read_parquet(f) for f in
                       sorted(WORK_DIR.glob(f"{split}_probs*.parquet"))], ignore_index=True)
    sums = pd.concat([pd.read_parquet(f) for f in sorted(WORK_DIR.glob(f"{split}_sums*.parquet"))],
                     ignore_index=True).groupby("s1_id", sort=False)["e_m"].sum()
    print(f"  probs {len(probs):,}  entities with candidates {len(sums):,}")
    pred = select_sets(probs, "p", sums)
    pred, conflicts = repair_exclusivity(pred, probs, "p", 3, sums)
    print(f"  contested ids resolved: {conflicts:,}")

    s1_path = WORK_DIR / f"{split}_s1.parquet"
    all_s1 = pq.read_table(s1_path, columns=["entity_id"])["entity_id"].to_pylist()
    with open(OUTPUT_DIR / "matching_results.tsv", "w") as fh:
        fh.write("source1_entity_id\tmatched_entity_ids\n")
        for sid in all_s1:
            fh.write(f"{sid}\t{','.join(sorted(pred.get(sid, ())))}\n")

    # candidate_pairs.tsv via DuckDB: ~170M pairs would not fit in a Python dict
    cand_files = [str(f) for f in sorted(WORK_DIR.glob(f"{split}_candidates*.parquet"))]
    con = connect(mem_gb=12)
    con.execute(f"""COPY (
        SELECT s.entity_id AS source1_entity_id,
               string_agg(c.cand_id, ',' ORDER BY c.cand_id) AS candidate_entity_ids
        FROM read_parquet('{s1_path}', file_row_number = true) s
        LEFT JOIN (SELECT s1_id, cand_id FROM read_parquet({cand_files})) c ON c.s1_id = s.entity_id
        GROUP BY s.entity_id, s.file_row_number
        ORDER BY s.file_row_number
    ) TO '{OUTPUT_DIR / "candidate_pairs.tsv"}' (FORMAT CSV, DELIMITER '\t', HEADER)""")
    con.close()
    n = sum(1 for s in all_s1 if pred.get(s))
    print(f"  wrote both TSVs: {len(all_s1):,} rows, {n:,} with >=1 match ({100*n/len(all_s1):.1f}%)")


def validate(folder):
    """The organisers' validate_submission.py; must print PASS before any upload."""
    if not VALIDATOR.exists():
        print(f"validator not found at {VALIDATOR} (set ER_VALIDATOR): skipped")
        return None
    r = subprocess.run([sys.executable, str(VALIDATOR),
                        "--matching", str(folder / "matching_results.tsv"),
                        "--candidate", str(folder / "candidate_pairs.tsv"),
                        "--test-dir", str(DATA_DIR / "test")], capture_output=True, text=True)
    print(r.stdout[-2500:] or r.stderr[-2000:])
    print(f"{folder.name}: exit code {r.returncode}  ->  {'PASS' if r.returncode == 0 else 'FAIL'}")
    return r.returncode == 0


def duplicate_records(folder):
    """Records given to more than one S1. The truth never does this, so each one is a wrong match."""
    seen, dup = set(), 0
    with open(folder / "matching_results.tsv") as fh:
        next(fh)
        for line in fh:
            for c in line.rstrip("\n").split("\t")[1].split(","):
                if c:
                    dup += c in seen
                    seen.add(c)
    return dup
