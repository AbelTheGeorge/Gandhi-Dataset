"""CWMG translation detection with ModernBERT: composed in English or translated.

Usage: python train_translation.py
"""
import sys

import cwmg_common as C

SPEC = C.TaskSpec("translation", "Translation detection", ["Translated", "Original English"], "raw")

_MISSING = {"", "none", "nan", "unknown", "null", "<null>", "-"}


def prepare(df):
    lang = df["original_language"].astype(str).str.strip().str.lower()
    df["label_opt"] = lang.map(lambda l: None if l in _MISSING else (1 if l == "english" else 0))
    work = df[(df["clean_text"] != "") & df["label_opt"].notnull()].copy()
    work["label"] = work["label_opt"].astype(int)
    diag = {
        "documents_total": int(len(df)),
        "documents_unknown_language": int(df["label_opt"].isnull().sum()),
        "documents_empty_contents": int((df["clean_text"] == "").sum()),
        "documents_usable": int(len(work)),
        "translated_source_languages": {
            str(k): int(v) for k, v in work.loc[work["label"] == 0, "original_language"].value_counts().items()},
    }
    for k, v in diag.items():
        print(f"  {k}: {v}")
    work["text"] = work["clean_text"]
    return work[["doc_id", "volume", "section", "text", "label"]], diag


if __name__ == "__main__":
    sys.exit(C.run_task(SPEC, prepare))
