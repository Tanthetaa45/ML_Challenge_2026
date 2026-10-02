"""Embedding retrieval (GPU): encode "name | address" with a multilingual sentence encoder and,
for every S1 record, take the most similar S2/S3 records IN THE SAME COUNTRY by exact cosine
search. Also the contrastive fine-tuning of the encoder on our own training matches.

Output format: ann_{split}.parquet = (s1_id, cand_id, ann_score, ann_rank).
"""
import gc
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from .config import CONFIG, DATA_DIR, SOURCE_COLUMNS

# Apache-2.0, 118M params. ER_ENCODER only exists to smoke-test the code with a smaller model.
BASE_MODEL = os.environ.get("ER_ENCODER", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
S1_SAMPLE = CONFIG["s1_sample"]   # the same per-mille sample as the main pipeline
K_TRAIN, K_TEST = 100, 50
BATCH, MAX_LEN = 512, 64
N_PAIRS, FT_BATCH, FT_LR, FT_SCALE = 800_000, 256, 3e-5, 20.0
QUERY_BATCH, POOL_CHUNK = 1024, 1_000_000

DEVICES = [f"cuda:{i}" for i in range(torch.cuda.device_count())] or ["cpu"]


def describe_devices():
    print(f"devices {DEVICES}  duckdb {duckdb.__version__}  torch {torch.__version__}")
    for d in DEVICES:
        if d.startswith("cuda"):
            print(f"  {d}: {torch.cuda.get_device_name(d)}  {torch.cuda.get_device_properties(d).total_memory / 2**30:.1f} GB")


# ---- canonical text ----
# Lowercased raw text, not the Stage-0 normalized text: the encoder reads Indic scripts
# natively, and transliterating first would throw that away.
def read_source(split, n):
    df = pd.read_csv(DATA_DIR / split / f"{split}_source{n}.tsv", sep="\t", header=0,
                     names=SOURCE_COLUMNS, dtype=str, keep_default_na=False, na_values=[])
    name = df["business_name"].str.lower().str.split().str.join(" ")
    addr = df["business_address"].str.lower().str.split().str.join(" ")
    addr = addr.where(addr != "", "unknown address")
    return pd.DataFrame({"entity_id": df["entity_id"], "country": df["country"],
                         "text": name + " | " + addr})

def sample_ids(ids, per_mille):
    """Same sample as the main pipeline: DuckDB hash of the id string."""
    con = duckdb.connect()
    con.register("ids", pa.table({"entity_id": pa.array(ids, pa.string())}))
    keep = con.sql(f"SELECT entity_id FROM ids WHERE hash(entity_id) % 1000 < {per_mille}").fetchall()
    con.close()
    return {r[0] for r in keep}


# ---- encoder (one copy per GPU, fp16) ----
_models = {}

def load_models(model_name):
    if model_name not in _models:
        from sentence_transformers import SentenceTransformer
        _models.clear()                       # one encoder in VRAM at a time
        ms = []
        for d in DEVICES:
            m = SentenceTransformer(model_name, device=d)
            m.max_seq_length = MAX_LEN
            if d.startswith("cuda"):
                m.half()
            ms.append(m)
        _models[model_name] = ms
    return _models[model_name]

def encode(texts, label, model_name, chunk=100_000):
    """L2-normalized fp16 embeddings; the texts are split across the GPUs."""
    models = load_models(model_name)
    dim = models[0].get_sentence_embedding_dimension()
    out = np.empty((len(texts), dim), np.float16)
    parts = np.array_split(np.arange(len(texts)), len(models))
    done, lock, t0 = [0], threading.Lock(), time.perf_counter()

    def run(g):
        idx = parts[g]
        for s in range(0, len(idx), chunk):
            sl = idx[s:s + chunk]
            v = models[g].encode([texts[j] for j in sl], batch_size=BATCH, convert_to_numpy=True,
                                 normalize_embeddings=True, show_progress_bar=False)
            out[sl] = v.astype(np.float16)
            with lock:
                done[0] += len(sl)
                rate = done[0] / (time.perf_counter() - t0)
                print(f"    {label}: {done[0]:>11,}/{len(texts):,}  {rate:,.0f}/s  "
                      f"eta {(len(texts) - done[0]) / rate / 60:5.1f} min", end="\r", flush=True)

    with ThreadPoolExecutor(len(models)) as ex:
        list(ex.map(run, range(len(models))))
    print(f"\n  {label}: {len(texts):,} in {(time.perf_counter() - t0) / 60:.1f} min")
    return out


# ---- exact top-K search, per country, on the GPU ----
def search(q, p, k):
    """Top-k inner product of every row of q against p. Query batches are split across GPUs."""
    k = min(k, len(p))
    scores = np.empty((len(q), k), np.float32)
    index = np.empty((len(q), k), np.int64)
    batches = list(range(0, len(q), QUERY_BATCH))

    def run(g):
        dev = DEVICES[g]
        dt = torch.float16 if dev.startswith("cuda") else torch.float32
        P = torch.from_numpy(p).to(dev, dt)
        for qs in batches[g::len(DEVICES)]:
            Q = torch.from_numpy(q[qs:qs + QUERY_BATCH]).to(dev, dt)
            best_s = best_i = None
            for ps in range(0, len(P), POOL_CHUNK):
                sc = Q @ P[ps:ps + POOL_CHUNK].T
                s, i = sc.topk(min(k, sc.shape[1]), dim=1)   # top-k in fp16: a float copy would need 4 GB
                s = s.float()
                i += ps
                if best_s is not None:
                    s, j = torch.cat([best_s, s], 1).topk(k, dim=1)
                    i = torch.cat([best_i, i], 1).gather(1, j)
                best_s, best_i = s, i
            scores[qs:qs + len(Q)] = best_s.cpu().numpy()
            index[qs:qs + len(Q)] = best_i.cpu().numpy()
        del P
        if dev.startswith("cuda"):
            torch.cuda.empty_cache()

    with ThreadPoolExecutor(len(DEVICES)) as ex:
        list(ex.map(run, range(len(DEVICES))))
    return scores, index

def run_split(split, k, model_name, out_dir, per_mille=None):
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)
    pool = pd.concat([read_source(split, 2), read_source(split, 3)], ignore_index=True)
    s1 = read_source(split, 1)
    if per_mille is not None:
        s1 = s1[s1["entity_id"].isin(sample_ids(s1["entity_id"].tolist(), per_mille))].reset_index(drop=True)
    print(f"{split}: S1 queries {len(s1):,}  pool {len(pool):,}")

    pool_emb = encode(pool["text"].tolist(), f"{split} pool", model_name)
    s1_emb = encode(s1["text"].tolist(), f"{split} S1", model_name)

    out = out_dir / f"ann_{split}.parquet"
    schema = pa.schema([("s1_id", pa.string()), ("cand_id", pa.string()),
                        ("ann_score", pa.float32()), ("ann_rank", pa.int16())])
    writer = pq.ParquetWriter(out, schema, compression="zstd")
    total = 0
    for country in sorted(s1["country"].unique()):
        qi = np.flatnonzero(s1["country"].to_numpy() == country)
        pi = np.flatnonzero(pool["country"].to_numpy() == country)
        if len(pi) == 0:
            print(f"  {country}: no pool records, skipped")
            continue
        t = time.perf_counter()
        sc, ix = search(s1_emb[qi], pool_emb[pi], k)
        kk = sc.shape[1]
        pool_ids = pa.array(pool["entity_id"].to_numpy()[pi], pa.string())
        s1_ids = pa.array(s1["entity_id"].to_numpy()[qi], pa.string())
        for s in range(0, len(qi), 200_000):          # write in slices to bound memory
            e = min(s + 200_000, len(qi))
            n = e - s
            writer.write_table(pa.table({
                "s1_id": s1_ids.take(pa.array(np.repeat(np.arange(s, e), kk))),
                "cand_id": pool_ids.take(pa.array(ix[s:e].ravel())),
                "ann_score": pa.array(sc[s:e].ravel(), pa.float32()),
                "ann_rank": pa.array(np.tile(np.arange(1, kk + 1, dtype=np.int16), n)),
            }, schema=schema))
        total += len(qi) * kk
        print(f"  {country}: {len(qi):,} queries x {len(pi):,} records  top-{kk} "
              f"in {time.perf_counter() - t:.0f}s")
    writer.close()
    del pool_emb, s1_emb
    gc.collect()
    print(f"{split}: {total:,} rows -> {out}  ({(time.perf_counter() - t0) / 60:.1f} min)")
    return out


# ---- the gate: DuckDB alone vs embeddings alone vs the union (read-only diagnostics) ----
def f05_ceiling_sql(found_col):
    # a perfect model predicts exactly the found true matches: 1.25h/(h+0.25m); singletons score 1
    return f"""AVG(CASE WHEN m = 0 THEN 1.0 WHEN {found_col} = 0 THEN 0.0
                        ELSE 1.25 * {found_col} / ({found_col} + 0.25 * m) END)"""

def evaluate_gate(ann_path, duck_path, ks=(10, 20, 50, 100)):
    con = duckdb.connect()
    gt_path = DATA_DIR / "train" / "train_ground_truth.tsv"
    con.execute(f"""
        CREATE TABLE ann AS SELECT * FROM read_parquet('{ann_path}');
        CREATE TABLE duck AS SELECT s1_id, cand_id FROM read_parquet('{duck_path}');
        CREATE TABLE ents AS SELECT DISTINCT s1_id FROM ann;
        CREATE TABLE gt AS
        SELECT g.s1_id, TRIM(g.cand_id) AS cand_id FROM (
            SELECT source1_entity_id AS s1_id, UNNEST(STRING_SPLIT(matched_entity_ids, ',')) AS cand_id
            FROM read_csv('{gt_path}', delim='\t', header=true,
                 columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
            WHERE matched_entity_ids IS NOT NULL AND matched_entity_ids <> ''
        ) g JOIN ents USING (s1_id);
        CREATE TABLE m AS
        SELECT s1_id, COUNT(*) AS m FROM gt GROUP BY 1;""")
    n_ann = con.sql("SELECT COUNT(*) FROM ents").fetchone()[0]
    n_duck = con.sql("SELECT COUNT(DISTINCT s1_id) FROM duck").fetchone()[0]
    print(f"  S1 entities: embeddings {n_ann:,}  DuckDB {n_duck:,}")
    if abs(n_ann - n_duck) > 0.01 * n_duck:
        raise RuntimeError("the two files do not cover the same S1 sample (check s1_sample)")

    flags = ", ".join(f"MAX(CASE WHEN a.ann_rank <= {k} THEN 1 ELSE 0 END) AS a{k}" for k in ks)
    con.execute(f"""CREATE TABLE hit AS
        SELECT g.s1_id, g.cand_id,
               MAX(CASE WHEN d.cand_id IS NOT NULL THEN 1 ELSE 0 END) AS d, {flags}
        FROM gt g
        LEFT JOIN duck d ON d.s1_id = g.s1_id AND d.cand_id = g.cand_id
        LEFT JOIN ann a ON a.s1_id = g.s1_id AND a.cand_id = g.cand_id
        GROUP BY 1, 2""")
    rows = [("DuckDB A-H (top 100)", "d")]
    rows += [(f"embeddings alone (top {k})", f"a{k}") for k in ks]
    rows += [(f"union: DuckDB + embed top {k}", f"GREATEST(d, a{k})") for k in ks]
    print(f"\n  {'configuration':34s} {'recall':>8s} {'delta':>8s} {'ceiling F0.5':>13s} {'added cands/ent':>16s}")
    base = None
    for label, expr in rows:
        r = con.sql(f"SELECT AVG({expr}) FROM hit").fetchone()[0]
        ceil = con.sql(f"""
            SELECT {f05_ceiling_sql('f')} FROM (
                SELECT e.s1_id, COALESCE(m.m, 0) AS m, COALESCE(h.f, 0) AS f
                FROM ents e LEFT JOIN m USING (s1_id)
                LEFT JOIN (SELECT s1_id, SUM({expr}) AS f FROM hit GROUP BY 1) h USING (s1_id))""").fetchone()[0]
        added = ""
        if label.startswith("union"):
            k = int(label.rsplit(" ", 1)[1])
            n_new = con.sql(f"""SELECT COUNT(*) FROM ann a LEFT JOIN duck d USING (s1_id, cand_id)
                                WHERE a.ann_rank <= {k} AND d.cand_id IS NULL""").fetchone()[0]
            added = f"{n_new / n_ann:16.1f}"
        base = r if base is None else base
        delta = f"{r - base:+8.4f}" if label.startswith("union") else ""
        print(f"  {label:34s} {r:8.4f} {delta:>8s} {ceil:13.4f} {added}")

    k = 50 if 50 in ks else ks[-1]
    part = con.sql(f"""SELECT SUM(d * a{k}), SUM(d * (1 - a{k})), SUM((1 - d) * a{k}),
                              SUM((1 - d) * (1 - a{k})), COUNT(*) FROM hit""").fetchone()
    both, d_only, a_only, neither, n = part
    print(f"\n  true matches, embeddings at top {k}:  both {both / n:.4f}  DuckDB only {d_only / n:.4f}  "
          f"embeddings only {a_only / n:.4f}  neither {neither / n:.4f}")
    dist = con.sql(f"""SELECT a.ann_rank FROM hit h JOIN ann a USING (s1_id, cand_id)
                       WHERE h.d = 0""").fetchnumpy()["ann_rank"]
    if len(dist):
        print("  embeddings-only hits by rank: " + "  ".join(
            f"<= {r}: {(dist <= r).mean():.0%}" for r in (5, 10, 20, 50)))
    con.close()
    return a_only / n

def show_embedding_only(ann_path, duck_path, n=30):
    """The true matches only the embeddings found - are they genuine?"""
    s1 = read_source("train", 1).set_index("entity_id")["text"]
    pool = pd.concat([read_source("train", 2), read_source("train", 3)]).set_index("entity_id")["text"]
    con = duckdb.connect()
    gt_path = DATA_DIR / "train" / "train_ground_truth.tsv"
    rows = con.sql(f"""
        WITH gt AS (
            SELECT s1_id, TRIM(c) AS cand_id FROM (
                SELECT source1_entity_id AS s1_id, UNNEST(STRING_SPLIT(matched_entity_ids, ',')) AS c
                FROM read_csv('{gt_path}', delim='\t', header=true,
                     columns={{'source1_entity_id':'VARCHAR','matched_entity_ids':'VARCHAR'}})
                WHERE matched_entity_ids <> ''))
        SELECT a.s1_id, a.cand_id, a.ann_rank, a.ann_score
        FROM read_parquet('{ann_path}') a
        JOIN gt USING (s1_id, cand_id)
        ANTI JOIN read_parquet('{duck_path}') d USING (s1_id, cand_id)
        USING SAMPLE {n} ROWS""").fetchall()
    con.close()
    for sid, cid, rank, score in rows:
        print(f"  rank {rank:3d} cos {score:.3f}  {s1.get(sid, '?')[:60]!r}\n"
              f"  {'':19s}{pool.get(cid, '?')[:60]!r}")

def gate(ann_path, duck_path):
    if not duck_path.exists():
        print(f"{duck_path} not found: gate skipped (it is a diagnostic only)")
        return
    gain = evaluate_gate(ann_path, duck_path)
    show_embedding_only(ann_path, duck_path)
    print(f"\nGATE: union recall gain at top 50 = {gain:+.4f}  ->  "
          + ("PROCEED" if gain >= 0.025 else "STOP" if gain < 0.015 else "BORDERLINE: inspect the examples"))


# ---- fine-tuning on our own matches ----
def training_pairs():
    """One true partner per S1. The s1_sample used by the gate and the blend is EXCLUDED."""
    t0 = time.perf_counter()
    gt = pd.read_csv(DATA_DIR / "train" / "train_ground_truth.tsv", sep="\t", dtype=str,
                     keep_default_na=False, na_values=[])
    gt = gt[gt["matched_entity_ids"] != ""]
    s1_all = read_source("train", 1).set_index("entity_id")
    pool_all = pd.concat([read_source("train", 2), read_source("train", 3)]).set_index("entity_id")["text"]
    held_out = sample_ids(s1_all.index.tolist(), S1_SAMPLE)

    rng = random.Random(0)
    rows = []
    for sid, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        rows.append((sid, rng.choice(m.split(","))))
    pairs = pd.DataFrame(rows, columns=["s1_id", "cand_id"])
    pairs["a"] = pairs["s1_id"].map(s1_all["text"])
    pairs["b"] = pairs["cand_id"].map(pool_all)
    pairs["country"] = pairs["s1_id"].map(s1_all["country"])
    pairs = pairs.dropna()
    check = pairs[pairs["s1_id"].isin(held_out)].groupby("country").head(2_500)   # never trained on
    train_pairs = pairs[~pairs["s1_id"].isin(held_out)].sample(frac=1.0, random_state=0).head(N_PAIRS)
    print(f"training pairs {len(train_pairs):,}  held-out check pairs {len(check):,}  "
          f"({time.perf_counter() - t0:.0f}s)")
    print(train_pairs.groupby("country").size().to_string())
    return train_pairs, check

def country_batches(df, size, seed=0):
    """Single-country batches: the other records in a batch are the negatives."""
    out = []
    for _, g in df.groupby("country"):
        g = g.sample(frac=1.0, random_state=seed)
        for i in range(0, len(g) - size + 1, size):
            out.append((g["a"].iloc[i:i + size].tolist(), g["b"].iloc[i:i + size].tolist()))
    random.Random(seed).shuffle(out)
    return out

def retrieval_check(check, model_name, label):
    """For each held-out S1 text, is its true partner the nearest among all partners of that country?"""
    res = []
    for c, g in check.groupby("country"):
        qa = torch.from_numpy(encode(g["a"].tolist(), f"{label} {c} S1", model_name).astype(np.float32))
        qb = torch.from_numpy(encode(g["b"].tolist(), f"{label} {c} pool", model_name).astype(np.float32))
        rank = (qa @ qb.T).argsort(dim=1, descending=True)
        hit = rank == torch.arange(len(g))[:, None]
        res.append((c, len(g), hit[:, :1].any(1).float().mean().item(), hit[:, :10].any(1).float().mean().item()))
    for c, n, r1, r10 in res:
        print(f"  {label:6s} {c:7s} n={n:5d}  recall@1 {r1:.3f}  recall@10 {r10:.3f}")

def _to_device(features, dev):
    # newer sentence-transformers versions put non-tensor entries in the batch too
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in features.items()}

def finetune(base, out_dir, batches, lr=FT_LR, scale=FT_SCALE):
    """In-batch-negatives contrastive loss (both directions), one GPU, fp16, plain PyTorch."""
    from sentence_transformers import SentenceTransformer
    dev = DEVICES[0]
    cuda = dev.startswith("cuda")
    model = SentenceTransformer(base, device=dev)
    model.max_seq_length = MAX_LEN
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps, warm = len(batches), max(1, len(batches) // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (steps - s) / max(1, steps - warm)))
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    model.train()
    t0, run = time.perf_counter(), 0.0
    for step, (a_txt, b_txt) in enumerate(batches, 1):
        fa = _to_device(model.tokenize(a_txt), dev)
        fb = _to_device(model.tokenize(b_txt), dev)
        with torch.autocast(device_type="cuda" if cuda else "cpu", dtype=torch.float16, enabled=cuda):
            ea = model(fa)["sentence_embedding"]
            eb = model(fb)["sentence_embedding"]
        ea, eb = F.normalize(ea.float(), dim=1), F.normalize(eb.float(), dim=1)
        sim = ea @ eb.T * scale
        labels = torch.arange(len(sim), device=dev)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2   # both directions
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step > 1 else loss.item()
        if step % 100 == 0 or step == steps:
            rate = step / (time.perf_counter() - t0)
            print(f"    step {step:>6,}/{steps:,}  loss {run:.3f}  {rate:.1f} steps/s  "
                  f"eta {(steps - step) / rate / 60:5.1f} min", end="\r", flush=True)
    print(f"\n  fine-tuned in {(time.perf_counter() - t0) / 60:.1f} min")
    model.eval()
    model.save(str(out_dir))
    del model, opt
    if cuda:
        torch.cuda.empty_cache()
