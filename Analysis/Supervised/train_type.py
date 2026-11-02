"""CWMG document-type classification with ModernBERT.

Usage: python train_type.py
"""
import sys

import cwmg_common as C

SPEC = C.TaskSpec("type", "Document type classification",
                  ["ARTICLE", "LETTER", "PROSE", "TELEGRAM", "TRANSCRIPT"], "raw")


def prepare(df):
    idx = {n: i for i, n in enumerate(SPEC.class_names)}
    label_str = df["document_type"].astype(str).str.strip().str.upper()
    df["label_opt"] = label_str.map(idx)
    work = df[(df["clean_text"] != "") & df["label_opt"].notnull()].copy()
    work["label"] = work["label_opt"].astype(int)
    diag = {
        "documents_total": int(len(df)),
        "documents_empty_contents": int((df["clean_text"] == "").sum()),
        "documents_unrecognised_type": int(df["label_opt"].isnull().sum()),
        "unrecognised_type_values": label_str[df["label_opt"].isnull()].value_counts().to_dict(),
        "documents_usable": int(len(work)),
    }
    for k, v in diag.items():
        print(f"  {k}: {v}")
    work["text"] = work["clean_text"]
    return work[["doc_id", "volume", "section", "text", "label"]], diag


if __name__ == "__main__":
    sys.exit(C.run_task(SPEC, prepare))
