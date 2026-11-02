"""TF-IDF and logistic regression baseline for the three CWMG supervised tasks,
on the same documents, labels, text and splits as the ModernBERT run. Trains on
the training split only.

Usage:  python tfidf_baseline.py            (all three tasks)
        python tfidf_baseline.py era type   (selected tasks)
"""
import json
import os
import platform
import sys
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.normpath(os.path.join(HERE, "..", ".."))
DATA_PATH = os.path.join(BASE, "The-Collected-Works-Mahatma-Gandhi.json")
MB_DIR = HERE
RESULTS_DIR = os.path.join(HERE, "results", "tfidf")

sys.path.insert(0, MB_DIR)
import cwmg_common as C
import train_era
import train_translation
import train_type

TASKS = {"era": train_era, "translation": train_translation, "type": train_type}

MAX_FEATURES = 300_000
NGRAM_RANGE = (1, 2)
MIN_DF = 3
C_VALUE = 1.0
MAX_ITER = 2000
SOLVER = "liblinear"
VECTORIZER_KW = dict(
    lowercase=True,
    sublinear_tf=True,
    ngram_range=NGRAM_RANGE,
    min_df=MIN_DF,
    max_features=MAX_FEATURES,
    strip_accents="unicode",
    token_pattern=r"(?u)\b\w[\w<>]*\b",
)
SEEDS = [42, 43, 44]


def sqrt_class_weights(y, n):
    """Square-root-smoothed inverse class frequency."""
    counts = np.bincount(y, minlength=n).astype(np.float64)
    return np.sqrt(len(y) / (n * np.maximum(counts, 1)))


def fit_predict(Xtr, ytr, Xte, n_classes, weights, seed):
    """Weighted liblinear logistic regression, one-vs-rest for three or more classes."""
    if n_classes == 2:
        clf = LogisticRegression(C=C_VALUE, max_iter=MAX_ITER, solver=SOLVER,
                                 class_weight={i: float(w) for i, w in enumerate(weights)},
                                 random_state=seed)
        clf.fit(Xtr, ytr)
        return clf.predict(Xte)
    sw = weights[ytr]
    scores = np.zeros((Xte.shape[0], n_classes))
    for k in range(n_classes):
        clf = LogisticRegression(C=C_VALUE, max_iter=MAX_ITER, solver=SOLVER,
                                 random_state=seed)
        clf.fit(Xtr, (ytr == k).astype(int), sample_weight=sw)
        scores[:, k] = clf.decision_function(Xte)
    return scores.argmax(axis=1)


def metrics(y, p, names):
    rep = classification_report(y, p, labels=list(range(len(names))), target_names=names,
                                output_dict=True, zero_division=0)
    return {
        "macro_f1": f1_score(y, p, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, p, average="weighted", zero_division=0),
        "accuracy": accuracy_score(y, p),
        "per_class": {n: {"precision": rep[n]["precision"], "recall": rep[n]["recall"],
                          "f1": rep[n]["f1-score"], "support": int(rep[n]["support"])}
                      for n in names},
    }


def run_task(name, df_base, checks):
    mod = TASKS[name]
    spec = mod.SPEC
    names = spec.class_names
    out_dir = os.path.join(RESULTS_DIR, name)
    os.makedirs(out_dir, exist_ok=True)
    print("=" * 70 + f"\nTASK: {name}  (text condition: {spec.condition})\n" + "=" * 70)

    split = pd.read_csv(os.path.join(MB_DIR, "results", name, "split.csv"))
    task_checks = {}

    j = df_base.iloc[split["doc_id"].to_numpy()]

    def norm(s):
        s = s.astype(str).str.strip()
        n = pd.to_numeric(s, errors="coerce")
        return np.where(n.notna(), n.astype("Float64").astype(str), s)

    vol_ok = norm(j["volume"]) == norm(split["volume"])
    sec_ok = norm(j["section"]) == norm(split["section"])
    task_checks["volume_section_match"] = {"rows": int(len(split)),
                                           "volume_mismatches": int((~vol_ok).sum()),
                                           "section_mismatches": int((~sec_ok).sum())}
    task_checks["doc_id_unique"] = bool(split["doc_id"].is_unique)

    work, _ = mod.prepare(df_base.copy())
    work = work.set_index("doc_id")
    same_set = set(work.index) == set(split["doc_id"])
    task_checks["same_document_set_as_split_csv"] = bool(same_set)
    lab = work.reindex(split["doc_id"])["label"].to_numpy()
    task_checks["label_mismatches_vs_rebuilt"] = int((lab != split["label"].to_numpy()).sum())

    texts = work.reindex(split["doc_id"])["text"].tolist()
    expect = [C.clean_text(c) for c in df_base.iloc[split["doc_id"].to_numpy()]["contents"]]
    if spec.condition == "masked":
        expect = [C.mask_dates(t) for t in expect]
    task_checks["text_mismatches_vs_clean_text"] = int(sum(a != b for a, b in zip(texts, expect)))
    task_checks["empty_texts"] = int(sum(t == "" for t in texts))

    mb_pred = os.path.join(MB_DIR, "results", name, "seed_42", "test_predictions.csv")
    if os.path.exists(mb_pred):
        mb_ids = set(pd.read_csv(mb_pred, usecols=["doc_id"])["doc_id"])
        task_checks["test_ids_match_modernbert_predictions"] = bool(
            mb_ids == set(split.loc[split["split"] == "test", "doc_id"]))

    y = split["label"].to_numpy().astype(np.int64)
    sp = split["split"].to_numpy()
    tr, te = np.where(sp == "train")[0], np.where(sp == "test")[0]
    task_checks["split_sizes"] = {s: int((sp == s).sum()) for s in ("train", "val", "test")}
    for k, v in task_checks.items():
        print(f"  check {k}: {v}")
    checks[name] = task_checks

    t0 = time.time()
    vec = TfidfVectorizer(**VECTORIZER_KW)
    Xtr = vec.fit_transform([texts[i] for i in tr])
    Xte = vec.transform([texts[i] for i in te])
    print(f"  TF-IDF: train {Xtr.shape}, test {Xte.shape}  ({time.time()-t0:.0f}s)")
    weights = sqrt_class_weights(y[tr], len(names))
    print("  class weights: " + ", ".join(f"{n}={w:.3f}" for n, w in zip(names, weights)))

    per_seed, preds = {}, {}
    for seed in SEEDS:
        t1 = time.time()
        p = fit_predict(Xtr, y[tr], Xte, len(names), weights, seed)
        preds[seed] = p
        per_seed[seed] = metrics(y[te], p, names)
        print(f"  seed {seed}: macro F1 {per_seed[seed]['macro_f1']:.4f}  "
              f"acc {per_seed[seed]['accuracy']:.4f}  ({time.time()-t1:.0f}s)")
        pd.DataFrame({
            "doc_id": split["doc_id"].to_numpy()[te],
            "volume": split["volume"].to_numpy()[te],
            "section": split["section"].to_numpy()[te],
            "true_idx": y[te], "pred_idx": p,
            "true_label": [names[i] for i in y[te]],
            "pred_label": [names[i] for i in p],
        }).to_csv(os.path.join(out_dir, f"test_predictions_seed{seed}.csv"), index=False)

    deterministic = all(np.array_equal(preds[SEEDS[0]], preds[s]) for s in SEEDS[1:])
    mf1 = np.array([per_seed[s]["macro_f1"] for s in SEEDS])
    acc = np.array([per_seed[s]["accuracy"] for s in SEEDS])
    result = {
        "task": name,
        "text_condition": spec.condition,
        "class_names": names,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "n_features": int(Xtr.shape[1]),
        "class_weights": {n: float(w) for n, w in zip(names, weights)},
        "seeds": SEEDS,
        "deterministic_across_seeds": bool(deterministic),
        "test_macro_f1_mean": float(mf1.mean()), "test_macro_f1_std": float(mf1.std(ddof=1)),
        "test_accuracy_mean": float(acc.mean()), "test_accuracy_std": float(acc.std(ddof=1)),
        "per_seed": {str(s): per_seed[s] for s in SEEDS},
        "checks": task_checks,
    }
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=C._json_default)
    print(f"  deterministic across seeds: {deterministic}")
    return result


def main():
    tasks = sys.argv[1:] or list(TASKS)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    print(f"Reading: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["doc_id"] = np.arange(len(df))
    df["clean_text"] = df["contents"].apply(C.clean_text)
    print(f"Documents: {len(df):,}")

    checks, rows = {}, []
    for t in tasks:
        r = run_task(t, df, checks)
        rows.append({"task": t, "macro_f1_mean": r["test_macro_f1_mean"],
                     "macro_f1_std": r["test_macro_f1_std"],
                     "accuracy_mean": r["test_accuracy_mean"],
                     "accuracy_std": r["test_accuracy_std"],
                     "deterministic": r["deterministic_across_seeds"],
                     "n_test": r["n_test"], "n_features": r["n_features"]})
    pd.DataFrame(rows).to_csv(os.path.join(RESULTS_DIR, "summary.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump({
            "data_path": DATA_PATH, "n_documents": int(len(df)),
            "vectorizer": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in VECTORIZER_KW.items()},
            "logistic_regression": {"C": C_VALUE, "solver": SOLVER, "max_iter": MAX_ITER,
                                    "penalty": "l2 (default)", "fit_intercept": True,
                                    "class_weight": "sqrt(n_train / (n_classes * count_k))",
                                    "multiclass": "one-vs-rest (liblinear), argmax of decision scores"},
            "seeds": SEEDS, "val_split_used": False,
            "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                         "numpy": np.__version__, "scipy": scipy.__version__,
                         "pandas": pd.__version__},
            "checks": checks,
        }, f, indent=2, default=C._json_default)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
