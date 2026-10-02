"""Step 3 of 4 (GPU). Fine-tuned embedding retrieval, as run in notebooks/ann_finetune_kaggle.ipynb.

1. Fine-tune paraphrase-multilingual-MiniLM-L12-v2 on our own training matches: each example is
   an S1 record and one of its true S2/S3 records ("united ventures | lajpat nagar delhi" ->
   "yunaited vencars | lajpat nagar"). In-batch contrastive loss over single-country batches.
   The s1_sample that steps 1 and 4 validate on is EXCLUDED from fine-tuning.
2. Before/after recall@1 / @10 on held-out pairs.
3. Re-encode, exact per-country search -> ANN_FT_DIR/ann_train.parquet, ANN_FT_DIR/ann_test.parquet.
Step 4 uses these as the main ("ft") embedding. Independent of step 2; can run in parallel.
"""
from er import dense
from er.config import ANN_FT_DIR, FT_MODEL_DIR, WORK_DIR


def main():
    dense.describe_devices()
    train_pairs, check = dense.training_pairs()
    dense.retrieval_check(check, dense.BASE_MODEL, "before")
    dense.finetune(dense.BASE_MODEL, FT_MODEL_DIR, dense.country_batches(train_pairs, dense.FT_BATCH))
    ft = str(FT_MODEL_DIR)
    dense.retrieval_check(check, ft, "after")

    ann_train = dense.run_split("train", dense.K_TRAIN, ft, ANN_FT_DIR, per_mille=dense.S1_SAMPLE)
    dense.gate(ann_train, WORK_DIR / "train_candidates.parquet")
    dense.run_split("test", dense.K_TEST, ft, ANN_FT_DIR)


if __name__ == "__main__":
    main()
