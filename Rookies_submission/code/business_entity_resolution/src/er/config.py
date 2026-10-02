"""Paths and settings shared by every stage.

Every path can be overridden with an environment variable, so the same code runs on Kaggle
(defaults below) or on any Linux machine:
  ER_DATA_DIR    competition dataset folder (holds train/ and test/)
  ER_RUN_DIR     where everything is written (default /kaggle/working, else ./run)
  ER_WORK_DIR, ER_ANN_DIR, ER_ANN_FT_DIR, ER_FT_MODEL_DIR, ER_FINAL_DIR
                 point one stage at another run's output (e.g. an attached Kaggle dataset)
  ER_VALIDATOR   validate_submission.py (default: next to the dataset folder)
"""
import os
from pathlib import Path


def _first_existing(*paths):
    for p in paths:
        if Path(p).exists():
            return Path(p)
    return Path(paths[-1])


def _env_path(name, default):
    return Path(os.environ[name]) if os.environ.get(name) else Path(default)


DATA_DIR = _env_path("ER_DATA_DIR", _first_existing(
    "/kaggle/input/datasets/satwiksps/amazon-ml-challenge-2026/dataset",
    "/kaggle/input/amazon-ml-challenge-2026/dataset", "dataset"))
RUN_DIR = _env_path("ER_RUN_DIR", "/kaggle/working" if Path("/kaggle/working").exists() else "run")

WORK_DIR = _env_path("ER_WORK_DIR", RUN_DIR / "work")             # stage 0-3 artefacts, lgbm.txt
OUTPUT_DIR = RUN_DIR / "output"                                   # the plain v9 submission
ANN_DIR = _env_path("ER_ANN_DIR", RUN_DIR / "ann")                # original encoder retrieval
ANN_FT_DIR = _env_path("ER_ANN_FT_DIR", RUN_DIR / "ann_ft")       # fine-tuned encoder retrieval
FT_MODEL_DIR = _env_path("ER_FT_MODEL_DIR", RUN_DIR / "ft_model")
FINAL_DIR = _env_path("ER_FINAL_DIR", RUN_DIR / "final")          # the submitted files
TMP_DIR = RUN_DIR / "tmp"
VALIDATOR = _env_path("ER_VALIDATOR", DATA_DIR.parent / "validate_submission.py")

SOURCES = [("train", 1, "train_source1.tsv"), ("train", 2, "train_source2.tsv"),
           ("train", 3, "train_source3.tsv"), ("test", 1, "test_source1.tsv"),
           ("test", 2, "test_source2.tsv"), ("test", 3, "test_source3.tsv")]
GROUND_TRUTH = "train_ground_truth.tsv"
# train_source1.tsv's header starts with a bare tab, so files are always read with names=
SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_ROWS = {"train_s1": 2_206_821, "train_s2": 5_034_616, "train_s3": 5_285_603,
                 "test_s1": 1_732_544, "test_s2": 4_887_273, "test_s3": 5_082_316}

# Bump whenever normalize_name / normalize_address change: Stage 0 then rebuilds.
NORM_VERSION = 3

CONFIG = dict(
    # Stage 0
    prepare_limit=None,
    workers=max(1, (os.cpu_count() or 4) - 1),

    # Stage 1
    s1_sample=50,            # per-mille of train S1 used to train/validate (50 = 5%)
    rare_per_s1=3, max_token_df=1000, skel_max_df=1000,
    addr_rare_per_s1=4, addr_max_df=300,
    pair_tokens=4, pair_per_s1=6, pair_max_df=2000,
    key_cap=300, scheme_cap=100, top_k=100,
    pair_budget=60_000_000,
    # scheme G: house number + street word
    g_tokens=4, g_per_s1=6, g_max_df=100, g_cap=20,
    # scheme H: name word x address word
    h_name_tokens=3, h_addr_tokens=3, h_per_s1=6, h_max_df=50, h_cap=20,

    # Kaggle: 4 vCPU, ~30 GB RAM, ~19.5 GB disk
    threads=4, mem_gb=14, spill_gb=8,
    test_shards=24,
)

# Pairs below this calibrated probability are not stored for test (E[m] still uses all of them).
PROB_FLOOR = 0.01


def make_dirs():
    for d in (WORK_DIR, OUTPUT_DIR, TMP_DIR):
        d.mkdir(parents=True, exist_ok=True)
