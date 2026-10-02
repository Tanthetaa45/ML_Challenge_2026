"""F0.5 = 1.25*h / (k + 0.25*m); empty vs empty = 1; any prediction on a singleton = 0."""
import pandas as pd

from .config import DATA_DIR, GROUND_TRUTH


def f05(pred: set, true: set) -> float:
    k, m = len(pred), len(true)
    if k == 0 and m == 0:
        return 1.0
    if k == 0 or m == 0:
        return 0.0
    h = len(pred & true)
    return 0.0 if h == 0 else 1.25 * h / (k + 0.25 * m)

def macro_f05(pred: dict, truth: dict) -> float:
    if not truth:
        return 0.0
    return sum(f05(pred.get(s, set()), t) for s, t in truth.items()) / len(truth)

def load_ground_truth(s1_ids=None) -> dict:
    g = pd.read_csv(DATA_DIR / "train" / GROUND_TRUTH, sep="\t",
                    dtype=str, keep_default_na=False, na_values=[])
    out = {}
    for sid, matched in zip(g["source1_entity_id"], g["matched_entity_ids"]):
        if s1_ids is not None and sid not in s1_ids:
            continue
        out[sid] = set(matched.split(",")) if matched else set()
    return out

def self_test():
    ex = f05({"S2-00047", "S2-00193", "S3-00812"}, {"S2-00047", "S3-00812"})
    assert abs(ex - 0.714) < 0.001, ex
    assert f05(set(), set()) == 1.0 and f05({"x"}, set()) == 0.0 and f05({"a"}, {"a"}) == 1.0
    print(f"metric checks PASS (README example = {ex:.3f})")
