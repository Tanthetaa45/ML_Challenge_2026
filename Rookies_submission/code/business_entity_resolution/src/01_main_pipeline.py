"""Step 1 of 4 (CPU). The main pipeline, as run in notebooks/er_v9_kaggle.ipynb.

normalize -> block (A-H, fair slots) -> features (DuckDB + rapidfuzz cpdist)
-> LightGBM + isotonic -> expected-F0.5 sets -> exclusivity -> sharded test inference.

Writes to WORK_DIR: lgbm.txt, calib.npz, train_candidates.parquet, train_features.parquet,
test_{candidates,probs,sums}_shNN.parquet (what step 4 consumes), and the plain v9
submission to OUTPUT_DIR. Test inference is resumable: re-running skips finished shards.

    python 01_main_pipeline.py            # full run (train + test)
    python 01_main_pipeline.py --dev      # train split only, stops after the val report
"""
import argparse
import os
import shutil

from er import blocking, features, metric, model, normalize, submit
from er.config import CONFIG, DATA_DIR, OUTPUT_DIR, WORK_DIR, make_dirs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", action="store_true", help="train split only; stop after the val report")
    ap.add_argument("--shards", type=int, default=CONFIG["test_shards"])
    args = ap.parse_args()

    make_dirs()
    print(f"data   {DATA_DIR}\nwork   {WORK_DIR}\noutput {OUTPUT_DIR}")
    print(f"cpus {os.cpu_count()}  free disk {shutil.disk_usage(WORK_DIR).free / 2**30:.1f} GB  dev={args.dev}")

    normalize.self_test()                     # raises if transliteration is broken
    metric.self_test()
    normalize.run_stage0(("train",) if args.dev else ("train", "test"))

    con = blocking.block_train_sample()
    blocking.report_and_close(con)

    features.build_features("train")
    features.check_features("train")          # fast features must equal the reference

    booster, iso, va = model.train_model()
    model.val_report(booster, va)
    if args.dev:
        return

    submit.run_test(booster, iso, args.shards)
    submit.assemble()
    submit.validate(OUTPUT_DIR)


if __name__ == "__main__":
    main()
