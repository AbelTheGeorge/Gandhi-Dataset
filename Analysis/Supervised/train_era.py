"""CWMG era prediction with ModernBERT, with dates in the text masked.

Usage: python train_era.py
"""
import collections
import re
import sys

import pandas as pd

import cwmg_common as C

ERA_BOUNDS = [
    (1884, 1914, "1884-1914 (South Africa)"),
    (1915, 1924, "1915-1924 (Emergence)"),
    (1925, 1934, "1925-1934 (Civil Disobedience)"),
    (1935, 1944, "1935-1944 (Wardha Era)"),
    (1945, 1948, "1945-1948 (Independence)"),
]
YEAR_MIN, YEAR_MAX = 1884, 1948

SPEC = C.TaskSpec("era", "Diachronic era prediction", [n for _, _, n in ERA_BOUNDS], "masked")

_MISSING_TOKENS = {"", "none", "unknown", "<null>", "nan", "null", "-"}
_PREFIX_RE = re.compile(r"^(ON OR BEFORE|ON OR AFTER|ON OR AROUND)\s+")
_DAY_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})$")
_MONTH_RE = re.compile(r"^(\d{2})-(\d{4})$")
_YEAR_RE = re.compile(r"^(\d{4})$")
_YEAR_TOKEN = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


def parse_cwmg_date(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None, "missing"
    s = str(raw).strip()
    if s.lower() in _MISSING_TOKENS:
        return None, "missing"
    s = s.upper()
    m = _PREFIX_RE.match(s)
    body = s[m.end():].strip() if m else s
    for rx, res, grp in [(_DAY_RE, "day", 3), (_MONTH_RE, "month", 2), (_YEAR_RE, "year", 1)]:
        m = rx.match(body)
        if m:
            return int(m.group(grp)), res
    return None, "unparsed"


def era_index(year):
    if year is None:
        return None
    for i, (lo, hi, _) in enumerate(ERA_BOUNDS):
        if lo <= year <= hi:
            return i
    return None


def prepare(df):
    parsed = df["document_date"].apply(parse_cwmg_date)
    df["year"] = [p[0] for p in parsed]
    df["date_resolution"] = [p[1] for p in parsed]
    df["era"] = df["year"].apply(era_index)
    work = df[(df["clean_text"] != "") & df["era"].notnull()].copy()
    work["label"] = work["era"].astype(int)

    covered = correct = with_true_year = 0
    for text, year, label in zip(work["clean_text"], work["year"].astype(int), work["label"]):
        found = [int(t) for t in _YEAR_TOKEN.findall(text)]
        with_true_year += year in found
        in_range = [y for y in found if YEAR_MIN <= y <= YEAR_MAX]
        if in_range:
            covered += 1
            correct += era_index(collections.Counter(in_range).most_common(1)[0][0]) == label
    n = len(work)
    diag = {
        "documents_total": int(len(df)),
        "documents_missing_date": int((df["date_resolution"] == "missing").sum()),
        "documents_unparsed_date": int((df["date_resolution"] == "unparsed").sum()),
        "documents_out_of_range_year": int((df["year"].notnull() & df["era"].isnull()).sum()),
        "documents_empty_contents": int((df["clean_text"] == "").sum()),
        "documents_usable": int(n),
        "date_resolution_counts": {k: int(v) for k, v in df["date_resolution"].value_counts().items()},
        "leakage_probe": {
            "fraction_with_true_year_in_text": with_true_year / n,
            "regex_probe_coverage": covered / n,
            "regex_probe_accuracy_on_covered": correct / covered if covered else 0.0,
            "regex_probe_accuracy_overall": correct / n,
        },
    }
    for k, v in diag.items():
        print(f"  {k}: {v}")

    work["text"] = work["clean_text"].apply(C.mask_dates)
    return work[["doc_id", "volume", "section", "text", "label"]], diag


if __name__ == "__main__":
    sys.exit(C.run_task(SPEC, prepare))
