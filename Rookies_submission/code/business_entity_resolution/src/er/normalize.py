"""Stage 0: raw TSV -> normalized parquet (transliteration, name/address parts)."""
import multiprocessing as mp
import re
import time
import unicodedata
from dataclasses import dataclass, field

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rapidfuzz import fuzz
# Imported here so a missing package fails now, not silently inside Stage 0.
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

from .config import CONFIG, DATA_DIR, EXPECTED_ROWS, NORM_VERSION, SOURCE_COLUMNS, SOURCES, WORK_DIR

_INDIC =re.compile(r"[ऀ-ൿ]")
_indic_run_re = re.compile(r"([ऀ-ൿ]+)")
_SCRIPT_BLOCKS = ((0x0900, 0x097F, "DEVANAGARI"), (0x0980, 0x09FF, "BENGALI"),
                  (0x0A00, 0x0A7F, "GURMUKHI"), (0x0A80, 0x0AFF, "GUJARATI"),
                  (0x0B00, 0x0B7F, "ORIYA"), (0x0B80, 0x0BFF, "TAMIL"),
                  (0x0C00, 0x0C7F, "TELUGU"), (0x0C80, 0x0CFF, "KANNADA"),
                  (0x0D00, 0x0D7F, "MALAYALAM"))

def _script_of(run):
    cp = ord(run[0])
    for lo, hi, name in _SCRIPT_BLOCKS:
        if lo <= cp <= hi:
            return name
    return "DEVANAGARI"

_DEVA_PHRASE_MAP = {
    "प्राइवेट": "private", "प्रा": "private", "लिमिटेड": "limited", "लिमिडेट": "limited",
    "लि": "limited", "कंपनी": "company", "कम्पनी": "company", "एंड": "and", "एण्ड": "and",
    "इंडिया": "india", "भारत": "india", "सर्विसेज": "services", "सर्विस": "service",
    "इंटरप्राइजेज": "enterprises", "एंटरप्राइजेज": "enterprises", "उद्योग": "udyog",
    "ट्रेडर्स": "traders", "टेक्नोलॉजीज": "technologies", "सॉल्यूशंस": "solutions",
    "इंडस्ट्रीज": "industries", "मार्केटिंग": "marketing", "फाइनेंस": "finance",
    "स्टोर्स": "stores", "स्टोर": "store",
}
_deva_phrase_re = re.compile("|".join(sorted(map(re.escape, _DEVA_PHRASE_MAP), key=len, reverse=True)))

_IAST_FIXUPS = (("ph", "f"), ("mg", "ng"))
_TAMIL_FIXUPS = (("dh", "t"), ("gh", "k"), ("bh", "p"))
_TRANSLIT_SUFFIX_RULES = (
    (re.compile(r"^limi"), "limited"),
    (re.compile(r"^(?:praiv|piraiv|privat|praiw)"), "private"),
    (re.compile(r"^(?:ailaail|elel|elail)"), "llp"),
    (re.compile(r"^(?:korpor|corpor)"), "corporation"),
    (re.compile(r"^(?:kampani|kampeni|kampan)"), "company"),
)

def fold_ascii(text):
    """Strip diacritics. Safe only AFTER transliteration."""
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ascii")

def _fix_indic_word(word, script):
    word = word.replace("ṃ", "n").replace("ṁ", "n")   # anusvara, before folding turns it into m
    word = fold_ascii(word).lower()
    if script == "TAMIL":
        for a, b in _TAMIL_FIXUPS:
            word = word.replace(a, b)
    for a, b in _IAST_FIXUPS:
        word = word.replace(a, b)
    if script == "DEVANAGARI" and len(word) > 3 and word.endswith("a"):
        word = word[:-1]
    for pattern, canon in _TRANSLIT_SUFFIX_RULES:
        if pattern.match(word):
            return canon
    return word

def _transliterate_indic(text):
    """Indic -> Latin, run by run, so Latin words already present are untouched."""
    text = _deva_phrase_re.sub(lambda m: " " + _DEVA_PHRASE_MAP[m.group(0)] + " ", text)
    if not _INDIC.search(text):
        return text
    out = []
    for i, part in enumerate(_indic_run_re.split(text)):
        if i % 2 == 0:
            out.append(part)
            continue
        script = _script_of(part)
        try:
            latin = transliterate(part, getattr(sanscript, script), sanscript.IAST)
        except Exception:
            out.append(part)
            continue
        out.append(" ".join(_fix_indic_word(w, script) for w in latin.split()))
    return "".join(out)

LEGAL_SUFFIX_CANON = {
    "corporation": "corp", "corp": "corp", "corpn": "corp",
    "incorporated": "inc", "inc": "inc",
    "private": "pvt", "pvt": "pvt", "pte": "pvt",
    "limited": "ltd", "ltd": "ltd", "ltda": "ltd",
    "llp": "llp", "llc": "llc", "lp": "lp",
    "company": "co", "co": "co", "cos": "co",
    "sarl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa", "eurl": "eurl", "snc": "snc",
    "gmbh": "gmbh", "plc": "plc", "ag": "ag", "bv": "bv", "nv": "nv",
    "partnership": "partnership", "associates": "associates", "assoc": "associates",
    "holdings": "holdings", "group": "group", "trust": "trust", "society": "society",
}
NOISE_TOKENS = {"and", "the", "of", "a", "an"}
ADDR_ABBREV = {
    "rd": "road", "road": "road", "st": "street", "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue", "blvd": "boulevard",
    "boulevard": "boulevard", "bd": "boulevard", "ln": "lane", "lane": "lane",
    "dr": "drive", "drive": "drive", "ct": "court", "court": "court", "pl": "place",
    "place": "place", "sq": "square", "square": "square", "hwy": "highway",
    "highway": "highway", "pkwy": "parkway", "parkway": "parkway", "apt": "apartment",
    "apartment": "apartment", "ste": "suite", "suite": "suite", "bldg": "building",
    "building": "building", "flr": "floor", "fl": "floor", "floor": "floor",
    "nr": "near", "near": "near", "opp": "opposite", "opposite": "opposite",
    "n": "north", "north": "north", "s": "south", "south": "south", "e": "east",
    "east": "east", "w": "west", "west": "west", "ne": "northeast", "nw": "northwest",
    "se": "southeast", "sw": "southwest", "marg": "marg", "nagar": "nagar",
    "colony": "colony", "sector": "sector", "phase": "phase", "block": "block",
    "gali": "gali", "chowk": "chowk", "po": "postoffice", "ps": "policestation",
    "dist": "district", "tq": "taluk", "taluk": "taluk", "tehsil": "taluk",
    "r": "rue", "rue": "rue",
}
ADDR_NOISE = {"apartment", "suite", "floor", "building", "unit", "no", "number",
              "near", "opposite", "postoffice", "policestation"}

_word_re = re.compile(r"[a-z0-9]+")
_amp_re = re.compile(r"&amp;|&#38;|&")
_zerowidth_re = re.compile(r"[​-‍﻿]")
_acronym_re = re.compile(r"\b[a-z](?:\.[a-z])+\.?")
_pin6_re, _zip5_re = re.compile(r"\b\d{6}\b"), re.compile(r"\b\d{5}\b")
_LEET = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "6": "g", "7": "t", "8": "b"})

def _unleet(tok):
    """'f0od'->'food', '6lobal'->'global'. Only words that MIX letters and digits."""
    if any(ch.isdigit() for ch in tok) and any(ch.isalpha() for ch in tok):
        return tok.translate(_LEET)
    return tok

def _pre_clean(raw):
    if raw is None:
        return ""
    text = str(raw)
    if not text or text == "nan":
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _zerowidth_re.sub("", text)
    if _INDIC.search(text):
        text = _transliterate_indic(text)
    text = _amp_re.sub(" and ", text)
    text = fold_ascii(text).lower()
    return _acronym_re.sub(lambda m: m.group(0).replace(".", ""), text)

@dataclass(slots=True)
class NameParts:
    core: list = field(default_factory=list)
    suffix: list = field(default_factory=list)
    key: str = ""
    @property
    def text(self): return " ".join(self.core)

def normalize_name(raw):
    text = _pre_clean(raw)
    if not text:
        return NameParts()
    toks = _word_re.findall(text)
    skip = set()   # "... id 37752" is a generated record number, not part of the name
    for j, tok in enumerate(toks):
        if tok == "id" and j + 1 < len(toks) and toks[j + 1].isdigit():
            skip.update((j, j + 1))
    core, suffix = [], []
    for j, tok in enumerate(toks):
        if j in skip:
            continue
        tok = _unleet(tok)
        canon = LEGAL_SUFFIX_CANON.get(tok)
        if canon is not None:
            if canon not in suffix:
                suffix.append(canon)
            continue
        if tok in NOISE_TOKENS:
            continue
        core.append(tok)
    if not core and suffix:
        core = list(suffix)
    return NameParts(core, suffix, " ".join(sorted(set(core))))

@dataclass(slots=True)
class AddressParts:
    tokens: list = field(default_factory=list)
    numbers: list = field(default_factory=list)
    postcode: str = ""
    is_empty: bool = True
    @property
    def text(self): return " ".join(self.tokens)

def normalize_address(raw):
    text = _pre_clean(raw)
    if not text.strip():
        return AddressParts()
    m = _pin6_re.search(text) or _zip5_re.search(text)
    postcode = m.group(0) if m else ""
    tokens, numbers = [], []
    for tok in _word_re.findall(text):
        if tok == postcode:
            continue
        if tok.isdigit() or (any(c.isdigit() for c in tok) and any(c.isalpha() for c in tok)):
            if tok not in numbers:
                numbers.append(tok)
            continue
        tok = ADDR_ABBREV.get(tok, tok)
        if tok in NOISE_TOKENS or tok in ADDR_NOISE:
            continue
        tokens.append(tok)
    return AddressParts(tokens, numbers, postcode, False)


# ---- self-test: the Stage 0 gate (raises on failure) ----
FUZZY_MIN = 60
NAME_CASES = [
    ("Ram Marketing Private Limited", "राम मार्केटिंग प्राइवेट लिमिटेड", "key"),
    ("मॉडर्न फाइनेंस", "Modern Finance", "fuzzy"),
    ("श्री साई इंफ्राटेक", "Shri Sai Infratech", "fuzzy"),
    ("Smith & Sons Corp", "Smith and Sons Corporation", "key"),
    ("Smith & Sons", "Smith Sons", "key"),
    ("Café Déjà Inc", "Cafe Deja Incorporated", "key"),
    ("Acme Traders Pvt Ltd", "Traders Acme Private Limited", "key"),
    ("Marina Ecole France Sarl", "Marina Ecole France S.A.R.L.", "key"),
    ("Zephay Labs Inc", "Zephyr Labs Inc", "differ"),
    ("Prime Money", "Prime Motors", "differ"),
]
INDIC_SAMPLES = [
    ("Devanagari", "राम मार्केटिंग प्राइवेट लिमिटेड"), ("Devanagari", "मॉडर्न फाइनेंस"),
    ("Malayalam", "സിൽവർ കൺസൾട്ടൻസി പ്രൈവറ്റ് ലിമിറ്റഡ്"),
    ("Gujarati", "આલ્ફા ફાઉન્ડેશન પ્રાઇવેટ લિમિટેડ"),
    ("Kannada", "ರಾಮ್ ಬಿಸಿನೆಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"),
    ("Tamil", "ஸ்மார்ட் எக்ஸ்போர்ட்ஸ் பிரைவேட் லிமிடெட்"),
    ("Bengali", "রেড টেক প্রাইভেট লিমিটেড"),
    ("Telugu", "లోటస్ మార్కెటింగ్ ప్రైవేట్ లిమిటెడ్"),
    ("Gurmukhi", "ਸ਼ਿਵਮ ਗੋਲਡਨ ਪ੍ਰੋਡਿਊਸਰ ਐਲਐਲਪੀ"), ("Odia", "ସୁପର୍ ମ୍ୟାନେଜମେଣ୍ଟ୍"),
]

def self_test():
    fails = 0
    print("=== name normalization ===")
    for a, b, mode in NAME_CASES:
        na, nb = normalize_name(a), normalize_name(b)
        same = na.key == nb.key and na.key != ""
        sim = fuzz.token_set_ratio(na.text, nb.text)
        ok = same if mode == "key" else (sim >= FUZZY_MIN if mode == "fuzzy" else not same)
        fails += not ok
        print(f"  [{'ok ' if ok else 'FAIL'}] {mode:6s} {na.key!r} | {nb.key!r}  sim={sim:.0f}")
    print("\n=== non-empty guarantee, all 9 Indic scripts ===")
    for script, raw in INDIC_SAMPLES:
        p = normalize_name(raw)
        fails += not p.core
        print(f"  [{'ok ' if p.core else 'FAIL'}] {script:11s} core={p.core} suffix={p.suffix}")
    print(f"\n{'PASS' if fails == 0 else f'{fails} FAILURE(S)'}")
    if fails:
        raise RuntimeError(f"Stage 0 self-test: {fails} failure(s). Do not run Stage 0.")


# ---- Stage 0 driver (skips sources already built at this NORM_VERSION) ----
SCHEMA = pa.schema([
    ("entity_id", pa.string()), ("country", pa.string()), ("source", pa.int8()),
    ("name_key", pa.string()), ("name_text", pa.string()),
    ("name_tokens", pa.list_(pa.string())), ("name_suffix", pa.string()),
    ("n_core", pa.int16()), ("addr_text", pa.string()),
    ("addr_tokens", pa.list_(pa.string())), ("addr_nums", pa.list_(pa.string())),
    ("postcode", pa.string()), ("addr_empty", pa.bool_()),
])

def _normalize_batch(batch):
    out = []
    for raw_name, raw_addr in batch:
        n, a = normalize_name(raw_name), normalize_address(raw_addr)
        out.append((n.key, n.text, n.core, " ".join(n.suffix), len(n.core),
                    a.text, a.tokens, a.numbers, a.postcode, a.is_empty))
    return out

def prepare_source(tsv, split, source, out, pool, workers, limit=None):
    tmp = out.with_suffix(".tmp")      # renamed only on success: a crash never leaves a half file
    writer = pq.ParquetWriter(tmp, SCHEMA, compression="zstd")
    total, t0 = 0, time.perf_counter()
    try:
        for chunk in pd.read_csv(tsv, sep="\t", header=0, names=SOURCE_COLUMNS,
                                 chunksize=200_000, dtype=str, keep_default_na=False,
                                 na_values=[], nrows=limit):
            rows = list(zip(chunk["business_name"], chunk["business_address"]))
            size = max(1, len(rows) // workers + 1)
            parts = pool.map(_normalize_batch, [rows[i:i + size] for i in range(0, len(rows), size)])
            norm = [r for p in parts for r in p]
            writer.write_table(pa.table({
                "entity_id": pa.array(chunk["entity_id"].tolist(), pa.string()),
                "country": pa.array(chunk["country"].tolist(), pa.string()),
                "source": pa.array([source] * len(chunk), pa.int8()),
                "name_key": pa.array([r[0] for r in norm], pa.string()),
                "name_text": pa.array([r[1] for r in norm], pa.string()),
                "name_tokens": pa.array([r[2] for r in norm], pa.list_(pa.string())),
                "name_suffix": pa.array([r[3] for r in norm], pa.string()),
                "n_core": pa.array([r[4] for r in norm], pa.int16()),
                "addr_text": pa.array([r[5] for r in norm], pa.string()),
                "addr_tokens": pa.array([r[6] for r in norm], pa.list_(pa.string())),
                "addr_nums": pa.array([r[7] for r in norm], pa.list_(pa.string())),
                "postcode": pa.array([r[8] for r in norm], pa.string()),
                "addr_empty": pa.array([r[9] for r in norm], pa.bool_()),
            }, schema=SCHEMA))
            total += len(chunk)
    finally:
        writer.close()
    tmp.rename(out)
    print(f"  {split}/s{source}: {total:>10,} rows in {time.perf_counter() - t0:5.1f}s")

def run_stage0(splits):
    stamp = WORK_DIR / "norm_version.txt"
    tag = f"{NORM_VERSION}:{CONFIG['prepare_limit']}"
    if not stamp.exists() or stamp.read_text().strip() != tag:
        stale = [*WORK_DIR.glob("*.parquet"), *WORK_DIR.glob("*.tmp"), *WORK_DIR.glob("*.duckdb*"),
                 WORK_DIR / "lgbm.txt", WORK_DIR / "calib.npz"]
        for f in stale:
            f.unlink(missing_ok=True)
        print(f"  stage 0 invalidated -> cleared work/ (now {tag})")

    with mp.Pool(CONFIG["workers"]) as pool:
        for split, source, name in SOURCES:
            if split not in splits:
                continue
            out = WORK_DIR / f"{split}_s{source}.parquet"
            if out.exists():
                print(f"  {split}/s{source}: exists ({pq.ParquetFile(out).metadata.num_rows:,} rows), skip")
                continue
            prepare_source(DATA_DIR / split / name, split, source, out, pool,
                           CONFIG["workers"], CONFIG["prepare_limit"])
    stamp.write_text(tag)

    print("\nrow-count check / empty-name rate:")
    with duckdb.connect() as c0:
        for split, src, _ in SOURCES:
            if split not in splits:
                continue
            p = WORK_DIR / f"{split}_s{src}.parquet"
            n, e = c0.sql(f"SELECT COUNT(*), SUM((n_core = 0)::INT) FROM '{p}'").fetchone()
            exp = EXPECTED_ROWS[f"{split}_s{src}"]
            ok = CONFIG["prepare_limit"] is not None or n == exp
            print(f"  {split}/s{src} {n:>11,} rows {'OK' if ok else f'MISMATCH (expected {exp:,})'}"
                  f"  empty names {100 * e / n:.4f}% {'OK' if e / n <= 1e-4 else 'CHECK'}")
