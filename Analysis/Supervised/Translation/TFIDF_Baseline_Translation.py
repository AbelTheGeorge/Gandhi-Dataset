
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
RUN_LETTER_ONLY_CONTROL = False

SEED = 42
MAX_FEATURES = 300_000
NGRAM_RANGE = (1, 2)
MIN_DF = 3
C_VALUE = 1.0
TOP_K_FEATURES = 25

CLASS_NAMES = ["Translated", "Original English"]
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
RESULTS_DIR = os.path.join(RESULTS_ROOT, "tfidf_translation")
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


_PREFIX_RE = re.compile(r"^(ON OR BEFORE|ON OR AFTER|ON OR AROUND)\s+")
_MISSING_TOKENS = {"", "none", "unknown", "<null>", "nan", "null", "-"}


def parse_cwmg_year(raw_date):
    if raw_date is None or (isinstance(raw_date, float) and pd.isna(raw_date)):
        return None
    s = str(raw_date).strip()
    if s.lower() in _MISSING_TOKENS:
        return None
    s = s.upper()
    m = _PREFIX_RE.match(s)
    body = s[m.end():].strip() if m else s
    for rx, group in ((r"^(\d{2})-(\d{2})-(\d{4})$", 3),
                      (r"^(\d{2})-(\d{4})$", 2),
                      (r"^(\d{4})$", 1)):
        mm = re.match(rx, body)
        if mm:
            return int(mm.group(group))
    return None


def decade_of(year):
    if year is None or not (1884 <= year <= 1948):
        return "unknown"
    return f"{(year // 10) * 10}s"



print("=" * 70)
print("CWMG - TRANSLATION DETECTION: TF-IDF + LOGISTIC REGRESSION BASELINE")
print("=" * 70)

data_path = resolve_path(PATH_CANDIDATES)
print(f"Reading: {data_path}")
with open(data_path, "r", encoding="utf-8") as f:
    records = json.load(f)

df = pd.DataFrame(records)
df["clean_text"] = df["contents"].apply(clean_text)
df["language_norm"] = df["original_language"].astype(str).str.strip().str.lower()


def get_translation_label(lang_norm):
    if lang_norm in _MISSING_TOKENS:
        return None
    return 1 if lang_norm == "english" else 0


df["label_opt"] = df["language_norm"].apply(get_translation_label)

work = df[(df["clean_text"] != "") & (df["label_opt"].notnull())].copy()
work["label"] = work["label_opt"].astype(int)
work["decade"] = work["document_date"].apply(parse_cwmg_year).apply(decade_of)
work["doctype"] = work["document_type"].astype(str).str.strip().str.upper()

texts_all = work["clean_text"].tolist()
labels = np.array(work["label"].tolist(), dtype=np.int64)
doctypes_all = work["doctype"].tolist()

print(f"Total documents  : {len(df):,}")
print(f"Usable documents : {len(work):,}")
for idx, name in enumerate(CLASS_NAMES):
    n = int((labels == idx).sum())
    print(f"  [{idx}] {name:<18s} : {n:,}  ({100*n/len(work):.1f}%)")



print("\n" + "=" * 70)
print("CONFOUND CHECK - composition of each class")
print("=" * 70)

type_tab = pd.crosstab(work["doctype"], work["label"], normalize="columns") * 100
type_tab.columns = [CLASS_NAMES[c] for c in type_tab.columns]
print("\nDocument type composition (% within each class):")
print(type_tab.round(1).to_string())

dec_tab = pd.crosstab(work["decade"], work["label"], normalize="columns") * 100
dec_tab.columns = [CLASS_NAMES[c] for c in dec_tab.columns]
print("\nDecade composition (% within each class):")
print(dec_tab.round(1).to_string())

composition = pd.concat(
    [type_tab.add_prefix("doctype_pct_"), dec_tab.add_prefix("decade_pct_")],
    axis=0, keys=["document_type", "decade"],
)
composition.to_csv(os.path.join(RESULTS_DIR, "class_composition.csv"))
print("\n  => If these two profiles differ sharply, a classifier can score well")
print("     on topic and addressee alone. The LETTER-only control below holds")
print("     document type constant.")


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



def fit_and_report(run_name, condition, train_idx, test_idx):
    print("\n" + "#" * 70)
    print(f"# RUN: {run_name}")
    print("#" * 70)

    texts = [apply_condition(texts_all[i], condition) for i in range(len(texts_all))]
    X_train_txt = [texts[i] for i in train_idx]
    X_test_txt = [texts[i] for i in test_idx]
    yt, ye = labels[train_idx], labels[test_idx]

    if len(np.unique(yt)) < 2 or len(np.unique(ye)) < 2:
        print("  Skipped: only one class present in this subset.")
        return None

    vectorizer = TfidfVectorizer(
        lowercase=True, sublinear_tf=True, ngram_range=NGRAM_RANGE,
        min_df=MIN_DF, max_features=MAX_FEATURES, strip_accents="unicode",
        token_pattern=r"(?u)\b\w[\w<>]*\b",
    )
    Xtr = vectorizer.fit_transform(X_train_txt)
    Xte = vectorizer.transform(X_test_txt)
    print(f"TF-IDF matrix: {Xtr.shape[0]:,} x {Xtr.shape[1]:,}")

    counts = np.bincount(yt, minlength=NUM_LABELS).astype(np.float64)
    weights = np.sqrt(len(yt) / (NUM_LABELS * np.maximum(counts, 1)))
    class_weight = {i: float(weights[i]) for i in range(NUM_LABELS)}

    clf = LogisticRegression(
        C=C_VALUE, max_iter=2000, solver="liblinear",
        class_weight=class_weight, random_state=SEED,
    )
    print("Fitting logistic regression...")
    clf.fit(Xtr, yt)

    preds = clf.predict(Xte)
    acc = accuracy_score(ye, preds)
    macro_f1 = f1_score(ye, preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(ye, preds, average="weighted", zero_division=0)

    print(f"\n  Test Accuracy : {acc:.4f}")
    print(f"  Test Macro F1 : {macro_f1:.4f}")

    report = classification_report(
        ye, preds, labels=list(range(NUM_LABELS)),
        target_names=CLASS_NAMES, digits=4, zero_division=0,
    )
    print(f"\n{report}")

    with open(os.path.join(RESULTS_DIR, f"classification_report_{run_name}.txt"), "w") as f:
        f.write(f"Run: {run_name}\n\n{report}")
    pd.DataFrame({
        "true_label": [CLASS_NAMES[i] for i in ye],
        "predicted_label": [CLASS_NAMES[i] for i in preds],
    }).to_csv(os.path.join(RESULTS_DIR, f"test_predictions_{run_name}.csv"), index=False)

    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    order = np.argsort(coefs)
    rows = []
    print(f"\n  Top {TOP_K_FEATURES} features pushing towards 'Translated':")
    print("    " + ", ".join(feature_names[order[:TOP_K_FEATURES]][:15]))
    for rank, j in enumerate(order[:TOP_K_FEATURES], 1):
        rows.append({"class": "Translated", "rank": rank,
                     "feature": feature_names[j], "coefficient": float(coefs[j])})
    print(f"\n  Top {TOP_K_FEATURES} features pushing towards 'Original English':")
    print("    " + ", ".join(feature_names[order[::-1][:TOP_K_FEATURES]][:15]))
    for rank, j in enumerate(order[::-1][:TOP_K_FEATURES], 1):
        rows.append({"class": "Original English", "rank": rank,
                     "feature": feature_names[j], "coefficient": float(coefs[j])})
    pd.DataFrame(rows).to_csv(
        os.path.join(RESULTS_DIR, f"top_features_{run_name}.csv"), index=False
    )
    print("\n    ^ Names, honorifics and place words indicate a topic/addressee")
    print("      confound; function words and syntax indicate translationese.")

    cm = confusion_matrix(ye, preds, labels=list(range(NUM_LABELS)))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                linewidths=0.5, linecolor="lightgrey")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"TF-IDF + LR, translation - {run_name} (Macro F1 {macro_f1:.4f})")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{run_name}.pdf"))
    plt.savefig(os.path.join(RESULTS_DIR, f"confusion_matrix_{run_name}.png"), dpi=300)
    plt.close(fig)

    return {"model": "TF-IDF + LogisticRegression", "run": run_name,
            "condition": condition, "test_accuracy": acc,
            "test_macro_f1": macro_f1, "test_weighted_f1": weighted_f1,
            "n_train": int(len(train_idx)), "n_test": int(len(test_idx))}


summaries = [{
    "model": "Majority class", "run": "majority", "condition": "-",
    "test_accuracy": dummy_acc, "test_macro_f1": dummy_f1,
    "test_weighted_f1": f1_score(y_test, dummy_preds, average="weighted", zero_division=0),
    "n_train": int(len(idx_train)), "n_test": int(len(idx_test)),
}]

for cond in CONDITIONS_TO_RUN:
    out = fit_and_report(cond, cond, idx_train, idx_test)
    if out:
        summaries.append(out)

if RUN_LETTER_ONLY_CONTROL:
    letter_train = np.array([i for i in idx_train if doctypes_all[i] == "LETTER"])
    letter_test = np.array([i for i in idx_test if doctypes_all[i] == "LETTER"])
    print("\n" + "=" * 70)
    print(f"LETTER-ONLY CONTROL  (train {len(letter_train):,} | test {len(letter_test):,})")
    print("=" * 70)
    print("Document type is held constant, so a score that holds up here cannot")
    print("be explained by the classifier simply recognising letters.")
    out = fit_and_report("masked_letters_only", "masked", letter_train, letter_test)
    if out:
        summaries.append(out)

summary_df = pd.DataFrame(summaries)
summary_df.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)

print("\n" + "=" * 70)
print("SUMMARY - TRANSLATION DETECTION BASELINES")
print("=" * 70)
print(summary_df.to_string(index=False))
print(f"\nArtifacts written to: {RESULTS_DIR}")
