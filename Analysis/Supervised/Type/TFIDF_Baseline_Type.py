
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score)

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42



GANDHI_PATH = "/Enter/Path/here"

PATH_CANDIDATES = [
    GANDHI_PATH,
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "The-Collected-Works-Mahatma-Gandhi.json"),
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..",
                 "The-Collected-Works-Mahatma-Gandhi.json"),
    "./The-Collected-Works-Mahatma-Gandhi.json",
]

CONDITIONS_TO_RUN = ["raw"]

SEED = 42
MAX_FEATURES = 300_000
NGRAM_RANGE = (1, 2)
MIN_DF = 3
C_VALUE = 1.0
TOP_K_FEATURES = 20

CLASS_NAMES = ["ARTICLE", "LETTER", "PROSE", "TELEGRAM", "TRANSCRIPT"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
NUM_LABELS = len(CLASS_NAMES)


def resolve_path(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    raise FileNotFoundError(
        "Could not locate The-Collected-Works-Mahatma-Gandhi.json. "
        "Set GANDHI_PATH at the top of this script. Tried:\n  "
        + "\n  ".join(str(c) for c in candidates)
    )


def resolve_results_root():
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")


RESULTS_ROOT = resolve_results_root()
RESULTS_DIR = os.path.join(RESULTS_ROOT, "tfidf_type")
os.makedirs(RESULTS_DIR, exist_ok=True)



def clean_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if isinstance(text, list):
        text = " ".join(str(item) for item in text if item is not None)
    return str(text).strip()


_MONTHS = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
)
_ORD = r"(?:st|nd|rd|th)?"

_DATE_PATTERNS = [
    (re.compile(r"\b\d{1,2}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{2,4}\b"), "<DATE>"),
    (re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}{_ORD}\s*,?\s*\d{{4}}\b", re.I), "<DATE>"),
    (re.compile(rf"\b\d{{1,2}}{_ORD}\s+{_MONTHS}\.?\s*,?\s*\d{{4}}\b", re.I), "<DATE>"),
    (re.compile(rf"\b{_MONTHS}\.?\s*,?\s*\d{{4}}\b", re.I), "<DATE>"),
    (re.compile(rf"\b\d{{1,2}}{_ORD}\s+{_MONTHS}\b", re.I), "<DATE>"),
    (re.compile(rf"\b{_MONTHS}\.?\s+\d{{1,2}}{_ORD}\b", re.I), "<DATE>"),
    (re.compile(r"\b(?:1[6-9]\d{2}|20\d{2})\b"), "<YEAR>"),
]


def mask_dates(text):
    for rx, rep in _DATE_PATTERNS:
        text = rx.sub(rep, text)
    return re.sub(r"\s{2,}", " ", text).strip()


def apply_condition(text, condition):
    if condition == "masked":
        return mask_dates(text)
    if condition == "raw":
        return text
    raise ValueError(f"Unknown condition: {condition!r}")



print("=" * 70)
print("CWMG - DOCUMENT TYPE: TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 70)

data_path = resolve_path(PATH_CANDIDATES)
print(f"Reading: {data_path}")
with open(data_path, "r", encoding="utf-8") as f:
    records = json.load(f)

df = pd.DataFrame(records)
df["clean_text"] = df["contents"].apply(clean_text)
df["label_str"] = df["document_type"].astype(str).str.strip().str.upper()
df["label_opt"] = df["label_str"].map(CLASS_TO_IDX)

work = df[(df["clean_text"] != "") & (df["label_opt"].notnull())].copy()
work["label"] = work["label_opt"].astype(int)

texts_all = work["clean_text"].tolist()
labels = np.array(work["label"].tolist(), dtype=np.int64)

print(f"Total documents  : {len(df):,}")
print(f"Usable documents : {len(work):,}")
for idx, name in enumerate(CLASS_NAMES):
    n = int((labels == idx).sum())
    print(f"  [{idx}] {name:<12s} : {n:,}  ({100*n/len(work):.1f}%)")



idx_all = np.arange(len(labels))
idx_train, idx_temp = train_test_split(
    idx_all, test_size=0.20, random_state=SEED, stratify=labels
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.50, random_state=SEED, stratify=labels[idx_temp]
)
y_train, y_test = labels[idx_train], labels[idx_test]
print(f"\nTrain {len(idx_train):,} | Val {len(idx_val):,} | Test {len(idx_test):,}")



dummy = DummyClassifier(strategy="most_frequent").fit(
    np.zeros((len(idx_train), 1)), y_train
)
dummy_preds = dummy.predict(np.zeros((len(idx_test), 1)))
dummy_acc = accuracy_score(y_test, dummy_preds)
dummy_f1 = f1_score(y_test, dummy_preds, average="macro", zero_division=0)
print(f"\nMajority-class floor: accuracy {dummy_acc:.4f} | macro F1 {dummy_f1:.4f}")



def run_condition(condition):
    print("\n" + "#" * 70)
    print(f"# CONDITION: {condition.upper()}")
    print("#" * 70)

    texts = [apply_condition(t, condition) for t in texts_all]
    X_train_txt = [texts[i] for i in idx_train]
    X_test_txt = [texts[i] for i in idx_test]

    vectorizer = TfidfVectorizer(
        lowercase=True, sublinear_tf=True, ngram_range=NGRAM_RANGE,
        min_df=MIN_DF, max_features=MAX_FEATURES, strip_accents="unicode",
        token_pattern=r"(?u)\b\w[\w<>]*\b",
    )
    Xtr = vectorizer.fit_transform(X_train_txt)
    Xte = vectorizer.transform(X_test_txt)
    print(f"TF-IDF matrix: {Xtr.shape[0]:,} x {Xtr.shape[1]:,}")

    counts = np.bincount(y_train, minlength=NUM_LABELS).astype(np.float64)
    weights = np.sqrt(len(y_train) / (NUM_LABELS * np.maximum(counts, 1)))
    class_weight = {i: float(weights[i]) for i in range(NUM_LABELS)}

    clf = LogisticRegression(
        C=C_VALUE, max_iter=2000, solver="liblinear",
        class_weight=class_weight, random_state=SEED,
    )
    print("Fitting logistic regression...")
    clf.fit(Xtr, y_train)

    preds = clf.predict(Xte)
    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  Test Macro F1 : {macro_f1:.4f}")

    report = classification_report(
        y_test, preds, labels=list(range(NUM_LABELS)),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    print(f"\n{report}")

    with open(os.path.join(RESULTS_DIR, f"classification_report_{condition}.txt"), "w") as f:
        f.write(f"Condition: {condition}\n\n{report}")
    pd.DataFrame({
        "true_label": [CLASS_NAMES[i] for i in y_test],
        "predicted_label": [CLASS_NAMES[i] for i in preds],
    }).to_csv(os.path.join(RESULTS_DIR, f"test_predictions_{condition}.csv"), index=False)

    feature_names = np.array(vectorizer.get_feature_names_out())
    rows = []
    print(f"\n  Top {TOP_K_FEATURES} features per document type:")
    for i, name in enumerate(CLASS_NAMES):
        coefs = clf.coef_[i] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        top = np.argsort(coefs)[::-1][:TOP_K_FEATURES]
        print(f"    {name:<12s}: {', '.join(feature_names[top][:12])}")
        for rank, j in enumerate(top, 1):
            rows.append({"document_type": name, "rank": rank,
                         "feature": feature_names[j], "coefficient": float(coefs[j])})
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, f"top_features_{condition}.csv"), index=False
    )
    print("\n    ^ Salutations, sign-offs and speaker turns confirm that these")
    print("      categories are defined largely by surface form.")

    cm = confusion_matrix(y_test, preds, labels=list(range(NUM_LABELS)))
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                linewidths=0.5, linecolor="lightgrey")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"TF-IDF + LR, document type - {condition} (Macro F1 {macro_f1:.4f})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{condition}.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{condition}.png"), dpi=300)
    plt.close(fig)

    return {"model": "TF-IDF + LogisticRegression", "condition": condition,
            "test_accuracy": acc, "test_macro_f1": macro_f1,
            "test_weighted_f1": weighted_f1, "n_test": int(len(idx_test)),
            "n_features": int(Xtr.shape[1])}


summaries = [{
    "model": "Majority class", "condition": "-", "test_accuracy": dummy_acc,
    "test_macro_f1": dummy_f1,
    "test_weighted_f1": f1_score(y_test, dummy_preds, average="weighted", zero_division=0),
    "n_test": int(len(idx_test)), "n_features": 0,
}]

for cond in CONDITIONS_TO_RUN:
    summaries.append(run_condition(cond))

summary_df = pd.DataFrame(summaries)
summary_df.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)

print("\n" + "=" * 70)
print("SUMMARY - DOCUMENT TYPE BASELINES")
print("=" * 70)
print(summary_df.to_string(index=False))
print(f"\nArtifacts written to: {RESULTS_DIR}")