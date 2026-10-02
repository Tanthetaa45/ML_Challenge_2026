"""Step 2 of 4 (GPU). Embedding retrieval with the original encoder, as run in
notebooks/ann_embed_kaggle.ipynb (with RUN_TEST = True).

paraphrase-multilingual-MiniLM-L12-v2 (Apache-2.0, 118M params) encodes "name | address";
exact per-country cosine top-K on the GPU.
  ANN_DIR/ann_train.parquet  the s1_sample of train S1, top 100 each
  ANN_DIR/ann_test.parquet   every test S1, top 50 each
Step 4 uses these as the "orig" embedding columns. Needs only the raw data; the gate printout
also reads WORK_DIR/train_candidates.parquet from step 1 if it exists.
"""
from er import dense
from er.config import ANN_DIR, WORK_DIR


def main():
    dense.describe_devices()
    ann_train = dense.run_split("train", dense.K_TRAIN, dense.BASE_MODEL, ANN_DIR, per_mille=dense.S1_SAMPLE)
    dense.gate(ann_train, WORK_DIR / "train_candidates.parquet")
    dense.run_split("test", dense.K_TEST, dense.BASE_MODEL, ANN_DIR)


if __name__ == "__main__":
    main()
