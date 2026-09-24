"""Stage 0 — text normalization.

- Transliterate Devanagari before ASCII-folding. `encode('ASCII','ignore')` was
  deleting it outright, blanking ~11.6% of S3 names.
- Drop `&`/`and` instead of rewriting it. `\\b&\\b` never matched (`&` is not a
  word char), so "&" and "and" stayed different.
- Normalize name and address separately, never concatenated — one combined
  vectorizer conflates the two signals and inflates same-address false merges.
- Keep canonical legal suffixes separate from core tokens, as their own feature.
- Extract address slots (postcode, house numbers) instead of one token bag.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Indic script handling
# --------------------------------------------------------------------------- #

# The data is not just Devanagari: S3 also carries Malayalam, Gujarati, Kannada,
# Tamil, Bengali, Odia, Telugu and Gurmukhi. Their blocks are contiguous, so one
# range matches any of them; the per-run script is resolved from the code point.
_INDIC = re.compile(r"[ऀ-ൿ]")
_indic_run_re = re.compile(r"([ऀ-ൿ]+)")

_SCRIPT_BLOCKS = (
    (0x0900, 0x097F, "DEVANAGARI"),
    (0x0980, 0x09FF, "BENGALI"),
    (0x0A00, 0x0A7F, "GURMUKHI"),
    (0x0A80, 0x0AFF, "GUJARATI"),
    (0x0B00, 0x0B7F, "ORIYA"),
    (0x0B80, 0x0BFF, "TAMIL"),
    (0x0C00, 0x0C7F, "TELUGU"),
    (0x0C80, 0x0CFF, "KANNADA"),
    (0x0D00, 0x0D7F, "MALAYALAM"),
)


def _script_of(run: str) -> str:
    cp = ord(run[0])
    for lo, hi, name in _SCRIPT_BLOCKS:
        if lo <= cp <= hi:
            return name
    return "DEVANAGARI"


_DEVANAGARI = re.compile(r"[ऀ-ॿ]")

# IAST would give "praiveta"/"limiteda", which share few n-grams with
# "private"/"limited". Mapping these first lands them on the Latin token.
_DEVA_PHRASE_MAP = {
    "प्राइवेट": "private",
    "प्रा": "private",
    "लिमिटेड": "limited",
    "लिमिडेट": "limited",
    "लि": "limited",
    "कंपनी": "company",
    "कम्पनी": "company",
    "एंड": "and",
    "एण्ड": "and",
    "इंडिया": "india",
    "भारत": "india",
    "सर्विसेज": "services",
    "सर्विस": "service",
    "इंटरप्राइजेज": "enterprises",
    "एंटरप्राइजेज": "enterprises",
    "उद्योग": "udyog",
    "ट्रेडर्स": "traders",
    "टेक्नोलॉजीज": "technologies",
    "सॉल्यूशंस": "solutions",
    "इंडस्ट्रीज": "industries",
    "मार्केटिंग": "marketing",
    "फाइनेंस": "finance",
    "स्टोर्स": "stores",
    "स्टोर": "store",
}

_deva_phrase_re = re.compile(
    "|".join(sorted(map(re.escape, _DEVA_PHRASE_MAP), key=len, reverse=True))
)


# IAST -> the spelling Indian business names actually use in Latin script.
_IAST_FIXUPS = (
    ("ṃ", "n"),  # anusvara: "iṃ" -> "in", matching "infratech" not "imfratech"
    ("ṁ", "n"),
    ("ph", "f"),      # फ is written "f" far more often than "ph"
    ("mg", "ng"),     # "marketimg" -> "marketing"
)

# Tamil has no voiced/aspirated stops, so IAST renders them as the aspirated
# form: "limidhedh" for "limited". Undoing that recovers most of the name.
_TAMIL_FIXUPS = (("dh", "t"), ("gh", "k"), ("bh", "p"))

# Suffix rules applied ONLY to transliterated tokens, where the transliteration
# is too lossy for an exact lookup ("limiteda", "limirrad", "limitet", ...).
# Restricting them to transliterated text matters: a blanket `^limi` rule would
# also rewrite a genuine Latin name like "Limitless".
_TRANSLIT_SUFFIX_RULES = (
    (re.compile(r"^limi"), "limited"),
    (re.compile(r"^(?:praiv|piraiv|privat|praiw)"), "private"),
    (re.compile(r"^(?:ailaail|elel|elail)"), "llp"),
    (re.compile(r"^(?:korpor|corpor)"), "corporation"),
    (re.compile(r"^(?:kampani|kampeni|kampan)"), "company"),
)


def _fix_indic_word(word: str, script: str) -> str:
    # Fold first: IAST emits "ḍh"/"ā", so the fixups and suffix rules below would
    # otherwise never match their plain-ASCII patterns.
    word = fold_ascii(word).lower()
    if script == "TAMIL":
        for src, dst in _TAMIL_FIXUPS:
            word = word.replace(src, dst)
    for src, dst in _IAST_FIXUPS:
        word = word.replace(src, dst)
    # Drop the inherent final vowel: "rama" -> "ram". Devanagari only — it is the
    # dominant script here and the case we validated. Applying it to the others
    # ate real vowels ("alpha" -> "alf", "siva" -> "siv"); leaving them intact
    # keeps them closer to the Latin spelling, and the prefix-based suffix rules
    # below match with or without the trailing "a".
    if script == "DEVANAGARI" and len(word) > 3 and word.endswith("a"):
        word = word[:-1]
    for pattern, canon in _TRANSLIT_SUFFIX_RULES:
        if pattern.match(word):
            return canon
    return word


def _transliterate_indic(text: str) -> str:
    """Indic scripts -> Latin, run by run.

    Only Indic runs go through IAST and the fixups, so Latin words already
    present (from the phrase map or a mixed-script name) are left untouched.
    """
    text = _deva_phrase_re.sub(lambda m: " " + _DEVA_PHRASE_MAP[m.group(0)] + " ", text)
    if not _INDIC.search(text):
        return text
    try:
        from indic_transliteration import sanscript
        from indic_transliteration.sanscript import transliterate
    except Exception:
        return text  # never let a transliteration failure blank a record

    out: list[str] = []
    for i, part in enumerate(_indic_run_re.split(text)):
        if i % 2 == 0:  # not Indic
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


def fold_ascii(text: str) -> str:
    """Strip diacritics. Safe only *after* transliteration."""
    return unicodedata.normalize("NFKD", text).encode("ASCII", "ignore").decode("ascii")


# --------------------------------------------------------------------------- #
# Token vocabularies
# --------------------------------------------------------------------------- #

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
    "rd": "road", "road": "road",
    "st": "street", "str": "street", "street": "street",
    "ave": "avenue", "av": "avenue", "avenue": "avenue",
    "blvd": "boulevard", "boulevard": "boulevard", "bd": "boulevard",
    "ln": "lane", "lane": "lane",
    "dr": "drive", "drive": "drive",
    "ct": "court", "court": "court",
    "pl": "place", "place": "place",
    "sq": "square", "square": "square",
    "hwy": "highway", "highway": "highway",
    "pkwy": "parkway", "parkway": "parkway",
    "apt": "apartment", "apartment": "apartment",
    "ste": "suite", "suite": "suite",
    "bldg": "building", "building": "building",
    "flr": "floor", "fl": "floor", "floor": "floor",
    "nr": "near", "near": "near",
    "opp": "opposite", "opposite": "opposite",
    "n": "north", "north": "north", "s": "south", "south": "south",
    "e": "east", "east": "east", "w": "west", "west": "west",
    "ne": "northeast", "nw": "northwest", "se": "southeast", "sw": "southwest",
    "marg": "marg", "nagar": "nagar", "colony": "colony", "sector": "sector",
    "phase": "phase", "block": "block", "gali": "gali", "chowk": "chowk",
    "po": "postoffice", "ps": "policestation", "dist": "district",
    "tq": "taluk", "taluk": "taluk", "tehsil": "taluk",
    "r": "rue", "rue": "rue",
}

# Unit/suite words that should not drive a match on their own.
ADDR_NOISE = {"apartment", "suite", "floor", "building", "unit", "no", "number",
              "near", "opposite", "postoffice", "policestation"}

_word_re = re.compile(r"[a-z0-9]+")
_amp_re = re.compile(r"&amp;|&#38;|&")
_zerowidth_re = re.compile(r"[​-‍﻿]")
# "s.a.r.l." / "u.s.a" -> one token, instead of shattering into single letters.
_acronym_re = re.compile(r"\b[a-z](?:\.[a-z])+\.?")


def _pre_clean(raw) -> str:
    """unicode -> latin -> lowercase, shared by name and address."""
    if raw is None:
        return ""
    text = str(raw)
    if not text or text == "nan":
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = _zerowidth_re.sub("", text)  # ZWNJ/ZWJ appear inside Indic names
    if _INDIC.search(text):
        text = _transliterate_indic(text)
    text = _amp_re.sub(" and ", text)
    text = fold_ascii(text).lower()
    return _acronym_re.sub(lambda m: m.group(0).replace(".", ""), text)


# --------------------------------------------------------------------------- #
# Names
# --------------------------------------------------------------------------- #

@dataclass(slots=True)
class NameParts:
    core: list[str] = field(default_factory=list)     # discriminative tokens
    suffix: list[str] = field(default_factory=list)   # canonical legal suffixes
    key: str = ""                                     # order-invariant block key

    @property
    def text(self) -> str:
        return " ".join(self.core)


def normalize_name(raw) -> NameParts:
    text = _pre_clean(raw)
    if not text:
        return NameParts()

    core: list[str] = []
    suffix: list[str] = []
    for tok in _word_re.findall(text):
        canon = LEGAL_SUFFIX_CANON.get(tok)
        if canon is not None:
            if canon not in suffix:
                suffix.append(canon)
            continue
        if tok in NOISE_TOKENS:
            continue
        core.append(tok)

    # A name made only of a legal suffix would otherwise block to nothing.
    if not core and suffix:
        core = list(suffix)

    return NameParts(core=core, suffix=suffix, key=" ".join(sorted(set(core))))


# --------------------------------------------------------------------------- #
# Addresses
# --------------------------------------------------------------------------- #

_pin6_re = re.compile(r"\b\d{6}\b")
_zip5_re = re.compile(r"\b\d{5}\b")


@dataclass(slots=True)
class AddressParts:
    tokens: list[str] = field(default_factory=list)    # expanded, de-noised words
    numbers: list[str] = field(default_factory=list)   # house / plot / door numbers
    postcode: str = ""                                 # 6-digit PIN or 5-digit ZIP
    is_empty: bool = True

    @property
    def text(self) -> str:
        return " ".join(self.tokens)


def normalize_address(raw) -> AddressParts:
    text = _pre_clean(raw)
    if not text.strip():
        return AddressParts()

    # Pull the postcode before tokenizing so it is not read as a house number.
    postcode = ""
    m = _pin6_re.search(text) or _zip5_re.search(text)
    if m:
        postcode = m.group(0)

    tokens: list[str] = []
    numbers: list[str] = []
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

    return AddressParts(tokens=tokens, numbers=numbers, postcode=postcode, is_empty=False)


# --------------------------------------------------------------------------- #
# Self-test — the Stage 0 gate from PLAN.md
# --------------------------------------------------------------------------- #

#   "key"   -> blocking keys must be identical (scheme A will catch it)
#   "fuzzy" -> keys may differ, but similarity must clear FUZZY_MIN (scheme C's job).
#              Transliteration is inherently lossy, so demanding exact keys here
#              would be a test that lies to us.
#   "differ"-> must NOT share a key; guards against over-normalizing
FUZZY_MIN = 60

_NAME_CASES = [
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

_ADDR_CASES = [
    "1795 Westchester Drive, High Point, NC",
    "KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi",
    "No 10 Enkay Square, 448A, Udyog Vihar Phase V, Gurugram, Gurgaon, HR",
    "63 R. DE DIEPPE, LILLE, Hauts-de-France",
    "Near SBI ATM, Opp. Bus Stand, Nashik 422001",
    "",
]


def _self_test() -> int:
    failures = 0

    from rapidfuzz.fuzz import token_set_ratio

    print("=== name normalization ===")
    for a, b, mode in _NAME_CASES:
        na, nb = normalize_name(a), normalize_name(b)
        same_key = na.key == nb.key and na.key != ""
        sim = token_set_ratio(na.text, nb.text)
        if mode == "key":
            ok, note = same_key, "key match"
        elif mode == "fuzzy":
            ok, note = sim >= FUZZY_MIN, f"sim={sim:.0f} (>= {FUZZY_MIN})"
        else:
            ok, note = not same_key, "keys differ"
        failures += not ok
        print(f"  [{'ok ' if ok else 'FAIL'}] {mode:6s} {note}")
        print(f"         {a!r:44s} -> {na.key!r}")
        print(f"         {b!r:44s} -> {nb.key!r}")

    print("\n=== non-empty guarantee, all Indic scripts (the notebook's fatal bug) ===")
    indic_samples = [
        ("Devanagari", "राम मार्केटिंग प्राइवेट लिमिटेड"),
        ("Devanagari", "मॉडर्न फाइनेंस"),
        ("Malayalam", "സിൽവർ കൺസൾട്ടൻസി പ്രൈവറ്റ് ലിമിറ്റഡ്"),
        ("Gujarati", "આલ્ફા ફાઉન્ડેશન પ્રાઇવેટ લિમિટેડ"),
        ("Kannada", "ರಾಮ್ ಬಿಸಿನೆಸ್ ಪ್ರೈವೇಟ್ ಲಿಮಿಟೆಡ್"),
        ("Tamil", "ஸ்மார்ட் எக்ஸ்போர்ட்ஸ் பிரைவேட் லிமிடெட்"),
        ("Bengali", "রেড টেক প্রাইভেট লিমিটেড"),
        ("Telugu", "లోటస్ మార్కెటింగ్ ప్రైవేట్ లిమిటెడ్"),
        ("Gurmukhi", "ਸ਼ਿਵਮ ਗੋਲਡਨ ਪ੍ਰੋਡਿਊਸਰ ਐਲਐਲਪੀ"),
        ("Odia", "ସୁପର୍ ମ୍ୟାନେଜମେଣ୍ଟ୍"),
    ]
    for script, raw in indic_samples:
        parts = normalize_name(raw)
        ok = bool(parts.core)
        failures += not ok
        print(f"  [{'ok ' if ok else 'FAIL'}] {script:11s} core={parts.core} suffix={parts.suffix}")

    print("\n=== address normalization ===")
    for raw in _ADDR_CASES:
        p = normalize_address(raw)
        print(f"  {raw!r}")
        print(f"     tokens={p.tokens} nums={p.numbers} pin={p.postcode!r} empty={p.is_empty}")

    print(f"\n{'PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
