
import json
import os
import re
import random
import time
import collections

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from transformers import AutoTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42



GANDHI_PATH = "/EnterPathHere"


def _resolve_results_root():
    return os.path.join(os.getcwd(), "results")


RESULTS_ROOT = _resolve_results_root()
os.makedirs(RESULTS_ROOT, exist_ok=True)

CONDITIONS_TO_RUN = ["masked"]

SEED = 43
MAX_LENGTH = 512
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 25
WARMUP_RATIO = 0.1
PATIENCE = 10

NUM_WORKERS = 2 if os.name != "nt" else 0

ERA_BOUNDS = [
    (1884, 1914, "1884-1914 (South Africa)"),
    (1915, 1924, "1915-1924 (Emergence)"),
    (1925, 1934, "1925-1934 (Civil Disobedience)"),
    (1935, 1944, "1935-1944 (Wardha Era)"),
    (1945, 1948, "1945-1948 (Independence)"),
]
ERA_NAMES = [name for _, _, name in ERA_BOUNDS]
NUM_LABELS = len(ERA_NAMES)

YEAR_MIN, YEAR_MAX = 1884, 1948


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()

print("=" * 70)
print("CWMG - DIACHRONIC ERA CLASSIFICATION (roberta-base)")
print("=" * 70)
print(f"Device: {device}   |   GPUs visible: {n_gpus}")
for i in range(n_gpus):
    print(f"   GPU {i}: {torch.cuda.get_device_name(i)}")
print(f"Conditions to run: {CONDITIONS_TO_RUN}")



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

_YEAR_TOKEN = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")


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



_MISSING_TOKENS = {"", "none", "unknown", "<null>", "nan", "null", "-"}
_PREFIX_RE = re.compile(r"^(ON OR BEFORE|ON OR AFTER|ON OR AROUND)\s+")
_DAY_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{4})$")
_MONTH_RE = re.compile(r"^(\d{2})-(\d{4})$")
_YEAR_RE = re.compile(r"^(\d{4})$")


def parse_cwmg_date(raw_date):
    if raw_date is None or (isinstance(raw_date, float) and pd.isna(raw_date)):
        return None, "missing", None

    s = str(raw_date).strip()
    if s.lower() in _MISSING_TOKENS:
        return None, "missing", None

    s = s.upper()
    m = _PREFIX_RE.match(s)
    qualifier = m.group(1) if m else None
    body = s[m.end():].strip() if m else s

    m = _DAY_RE.match(body)
    if m:
        return int(m.group(3)), "day", qualifier
    m = _MONTH_RE.match(body)
    if m:
        return int(m.group(2)), "month", qualifier
    m = _YEAR_RE.match(body)
    if m:
        return int(m.group(1)), "year", qualifier

    return None, "unparsed", qualifier


def get_era_index(year):
    if year is None:
        return None
    for idx, (lo, hi, _) in enumerate(ERA_BOUNDS):
        if lo <= year <= hi:
            return idx
    return None



print("\n" + "=" * 70)
print("1. LOADING CORPUS AND BUILDING ERA LABELS")
print("=" * 70)

if not os.path.exists(GANDHI_PATH):
    raise FileNotFoundError(
        f"Could not locate the CWMG JSON file at:\n  {GANDHI_PATH}\n"
        "Update GANDHI_PATH at the top of this script to point at the file."
    )

data_path = GANDHI_PATH
print(f"Reading: {data_path}")
with open(data_path, "r", encoding="utf-8") as f:
    records = json.load(f)

df = pd.DataFrame(records)
print(f"Total documents loaded: {len(df):,}")

df["clean_text"] = df["contents"].apply(clean_text)

parsed = df["document_date"].apply(parse_cwmg_date)
df["year"] = [p[0] for p in parsed]
df["date_resolution"] = [p[1] for p in parsed]
df["date_qualifier"] = [p[2] for p in parsed]
df["era_idx"] = df["year"].apply(get_era_index)

n_total = len(df)
n_missing_date = int((df["date_resolution"] == "missing").sum())
n_unparsed = int((df["date_resolution"] == "unparsed").sum())
n_out_of_range = int(((df["year"].notnull()) & (df["era_idx"].isnull())).sum())
n_empty_text = int((df["clean_text"] == "").sum())

anomalies = df[(df["year"].notnull()) & (df["era_idx"].isnull())][
    ["volume", "section", "document_date", "title"]
]
os.makedirs(RESULTS_ROOT, exist_ok=True)
anomaly_path = os.path.join(RESULTS_ROOT, "date_anomalies.csv")
anomalies.to_csv(anomaly_path, index=False)

print(f"  documents with no usable date        : {n_missing_date:,}")
print(f"  documents whose date failed to parse : {n_unparsed:,}")
print(f"  documents with out-of-range years    : {n_out_of_range:,}  -> {anomaly_path}")
print(f"  documents with empty contents        : {n_empty_text:,}")
if n_unparsed:
    print("  WARNING: unparsed dates present; inspect the schema assumptions above.")

work = df[(df["clean_text"] != "") & (df["era_idx"].notnull())].copy()
work["label"] = work["era_idx"].astype(int)
print(f"\nUsable documents for the era task: {len(work):,}")

for idx, name in enumerate(ERA_NAMES):
    print(f"  [{idx}] {name:<32s} : {(work['label'] == idx).sum():,}")

diagnostics = {
    "documents_total": n_total,
    "documents_missing_date": n_missing_date,
    "documents_unparsed_date": n_unparsed,
    "documents_out_of_range_year": n_out_of_range,
    "documents_empty_contents": n_empty_text,
    "documents_usable": int(len(work)),
    "date_resolution_counts": {
        k: int(v) for k, v in df["date_resolution"].value_counts().items()
    },
    "date_qualifier_counts": {
        str(k): int(v) for k, v in df["date_qualifier"].value_counts(dropna=False).items()
    },
}



print("\n" + "=" * 70)
print("2. DATE-LEAKAGE DIAGNOSTIC")
print("=" * 70)

texts_all = work["clean_text"].tolist()
years_all = work["year"].astype(int).tolist()
labels_all = work["label"].tolist()

contains_any_year = 0
contains_true_year = 0
probe_covered = 0
probe_correct = 0

for text, year, label in zip(texts_all, years_all, labels_all):
    found = [int(t) for t in _YEAR_TOKEN.findall(text)]
    if found:
        contains_any_year += 1
    if year in found:
        contains_true_year += 1
    in_range = [y for y in found if YEAR_MIN <= y <= YEAR_MAX]
    if in_range:
        probe_covered += 1
        modal_year = collections.Counter(in_range).most_common(1)[0][0]
        if get_era_index(modal_year) == label:
            probe_correct += 1

n_work = len(work)
probe_acc_covered = probe_correct / probe_covered if probe_covered else 0.0
probe_acc_all = probe_correct / n_work

print(f"  contents containing ANY 4-digit year : {contains_any_year:,} ({100*contains_any_year/n_work:.1f}%)")
print(f"  contents containing the TRUE year    : {contains_true_year:,} ({100*contains_true_year/n_work:.1f}%)")
print("\n  Trivial year-regex classifier (no learning at all):")
print(f"     coverage                    : {probe_covered:,}/{n_work:,} ({100*probe_covered/n_work:.1f}%)")
print(f"     accuracy on covered docs    : {probe_acc_covered:.4f}")
print(f"     accuracy over all documents : {probe_acc_all:.4f}")
print("\n  => Quote these numbers in the paper. They are the reason the")
print("     'masked' condition, not the 'raw' condition, is the headline result.")

diagnostics["leakage_probe"] = {
    "documents_with_any_year": contains_any_year,
    "documents_with_true_year": contains_true_year,
    "fraction_with_true_year": contains_true_year / n_work,
    "regex_probe_coverage": probe_covered / n_work,
    "regex_probe_accuracy_on_covered": probe_acc_covered,
    "regex_probe_accuracy_overall": probe_acc_all,
}

with open(os.path.join(RESULTS_ROOT, "era_data_diagnostics.json"), "w") as f:
    json.dump(diagnostics, f, indent=2)



print("\n" + "=" * 70)
print("3. TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)
PAD_ID = tokenizer.pad_token_id
print(f"Loaded fast tokenizer (max_length={MAX_LENGTH}).")


class EncodedDataset(Dataset):

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {"input_ids": self.encodings[idx], "label": self.labels[idx]}


def collate(batch):
    maxlen = max(len(item["input_ids"]) for item in batch)
    input_ids = torch.full((len(batch), maxlen), PAD_ID, dtype=torch.long)
    attention_mask = torch.zeros((len(batch), maxlen), dtype=torch.long)
    labels = torch.empty(len(batch), dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["input_ids"]
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention_mask[i, : len(ids)] = 1
        labels[i] = item["label"]
    return {"input_ids": input_ids, "attention_mask": attention_mask, "label": labels}


def encode_texts(texts):
    enc = tokenizer(list(texts), truncation=True, max_length=MAX_LENGTH)["input_ids"]
    full = tokenizer(list(texts), truncation=False)["input_ids"]
    n_trunc = sum(1 for seq in full if len(seq) > MAX_LENGTH)
    return enc, n_trunc



def run_epoch(loader, model, criterion, optimizer=None, scheduler=None, scaler=None, training=True):
    model.train() if training else model.eval()

    total_loss = 0.0
    all_preds, all_labels = [], []
    num_batches = 0

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= old_scale:
                    scheduler.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return avg_loss, macro_f1, weighted_f1, all_preds, all_labels


def run_condition(condition):
    print("\n" + "#" * 70)
    print(f"# CONDITION: {condition.upper()}")
    print("#" * 70)

    set_seed(SEED)

    results_dir = os.path.join(RESULTS_ROOT, f"era_{condition}")
    model_dir = os.path.join(results_dir, "best_model")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    texts = np.array([apply_condition(t, condition) for t in texts_all], dtype=object)
    labels = np.array(labels_all, dtype=np.int64)

    idx = np.arange(len(labels))
    idx_train, idx_temp = train_test_split(
        idx, test_size=0.20, random_state=SEED, stratify=labels
    )
    idx_val, idx_test = train_test_split(
        idx_temp, test_size=0.50, random_state=SEED, stratify=labels[idx_temp]
    )
    print(f"Train {len(idx_train):,} | Val {len(idx_val):,} | Test {len(idx_test):,}")

    y_train, y_val, y_test = labels[idx_train], labels[idx_val], labels[idx_test]

    t0 = time.time()
    enc_train, trunc_train = encode_texts(texts[idx_train])
    enc_val, _ = encode_texts(texts[idx_val])
    enc_test, trunc_test = encode_texts(texts[idx_test])
    pct_trunc = 100 * (trunc_train + trunc_test) / (len(idx_train) + len(idx_test))
    print(f"Tokenized in {time.time()-t0:.0f}s | documents exceeding {MAX_LENGTH} tokens: {pct_trunc:.1f}%")

    train_loader = DataLoader(
        EncodedDataset(enc_train, y_train), batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        EncodedDataset(enc_val, y_val), batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=True,
    )
    test_loader = DataLoader(
        EncodedDataset(enc_test, y_test), batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate, num_workers=NUM_WORKERS, pin_memory=True,
    )

    counts = np.bincount(y_train, minlength=NUM_LABELS).astype(np.float64)
    weights = np.sqrt(len(y_train) / (NUM_LABELS * np.maximum(counts, 1)))
    print("\nSmoothed class weights:")
    for i, name in enumerate(ERA_NAMES):
        print(f"  {name:<32s} n={int(counts[i]):>6,}  w={weights[i]:.4f}")
    class_weights_tensor = torch.FloatTensor(weights).to(device)

    model = RobertaForSequenceClassification.from_pretrained(
        "roberta-base", num_labels=NUM_LABELS
    ).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    total_steps = len(train_loader) * NUM_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * WARMUP_RATIO), total_steps
    )
    scaler = torch.amp.GradScaler("cuda", enabled=torch.cuda.is_available())

    print("\nTraining...")
    training_log = []
    best_val_macro_f1 = 0.0
    patience_counter = 0

    for epoch in range(1, NUM_EPOCHS + 1):
        t_epoch = time.time()
        tr_loss, tr_f1, tr_wf1, _, _ = run_epoch(
            train_loader, model, criterion, optimizer, scheduler, scaler, training=True
        )
        va_loss, va_f1, va_wf1, _, _ = run_epoch(val_loader, model, criterion, training=False)
        elapsed = time.time() - t_epoch

        training_log.append({
            "epoch": epoch, "train_loss": tr_loss, "val_loss": va_loss,
            "train_macro_f1": tr_f1, "val_macro_f1": va_f1,
            "train_weighted_f1": tr_wf1, "val_weighted_f1": va_wf1,
            "epoch_time_sec": elapsed,
        })

        print(f"\nEpoch {epoch}/{NUM_EPOCHS}  ({elapsed:.0f}s)")
        print(f"  Train Loss {tr_loss:.4f} | Val Loss {va_loss:.4f}")
        print(f"  Train Macro F1 {tr_f1:.4f} | Val Macro F1 {va_f1:.4f}")

        if va_f1 > best_val_macro_f1:
            best_val_macro_f1 = va_f1
            patience_counter = 0
            (model.module if hasattr(model, "module") else model).save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            print(f"  * new best (Val Macro F1 {va_f1:.4f}) -> saved")
        else:
            patience_counter += 1
            print(f"  no improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"\n  Early stopping after epoch {epoch}.")
                break

    pd.DataFrame(training_log).to_csv(os.path.join(results_dir, "training_metrics.csv"), index=False)

    print("\nEvaluating best checkpoint on the test set...")
    model = RobertaForSequenceClassification.from_pretrained(model_dir).to(device)
    if n_gpus > 1:
        model = nn.DataParallel(model)

    te_loss, te_f1, te_wf1, te_preds, te_labels = run_epoch(
        test_loader, model, criterion, training=False
    )
    te_acc = accuracy_score(te_labels, te_preds)

    print(f"\n  Test Accuracy : {te_acc:.4f}")
    print(f"  Test Macro F1 : {te_f1:.4f}")

    report = classification_report(
        te_labels, te_preds, labels=list(range(NUM_LABELS)),
        target_names=ERA_NAMES, digits=4, zero_division=0,
    )
    print(f"\n{report}")

    with open(os.path.join(results_dir, "classification_report.txt"), "w") as f:
        f.write(f"Condition: {condition}\n\n{report}")
    pd.DataFrame([{
        "condition": condition, "test_loss": te_loss, "test_accuracy": te_acc,
        "test_macro_f1": te_f1, "test_weighted_f1": te_wf1,
        "best_val_macro_f1": best_val_macro_f1, "epochs_run": len(training_log),
    }]).to_csv(os.path.join(results_dir, "test_metrics.csv"), index=False)
    pd.DataFrame({
        "true_label": [ERA_NAMES[i] for i in te_labels],
        "predicted_label": [ERA_NAMES[i] for i in te_preds],
    }).to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)

    with open(os.path.join(results_dir, "label_mapping.json"), "w") as f:
        json.dump({i: n for i, n in enumerate(ERA_NAMES)}, f, indent=2)
    with open(os.path.join(results_dir, "run_config.json"), "w") as f:
        json.dump({
            "condition": condition, "seed": SEED, "model": "roberta-base",
            "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "max_epochs": NUM_EPOCHS, "warmup_ratio": WARMUP_RATIO, "patience": PATIENCE,
            "split": "80/10/10 stratified",
            "n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(idx_test),
            "pct_documents_truncated": pct_trunc,
            "class_weights": {ERA_NAMES[i]: float(weights[i]) for i in range(NUM_LABELS)},
        }, f, indent=2)

    cm = confusion_matrix(te_labels, te_preds, labels=list(range(NUM_LABELS)))
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=ERA_NAMES, yticklabels=ERA_NAMES, ax=ax,
                linewidths=0.5, linecolor="lightgrey")
    ax.set_xlabel("Predicted Era")
    ax.set_ylabel("True Era")
    ax.set_title(f"Era classification - {condition} (Macro F1 {te_f1:.4f})")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "confusion_matrix.pdf"))
    plt.savefig(os.path.join(results_dir, "confusion_matrix.png"), dpi=300)
    plt.show()
    plt.close(fig)

    print(f"\nArtifacts written to: {results_dir}")

    return {
        "condition": condition, "test_accuracy": te_acc, "test_macro_f1": te_f1,
        "test_weighted_f1": te_wf1, "best_val_macro_f1": best_val_macro_f1,
        "epochs_run": len(training_log), "n_test": len(idx_test),
    }



summaries = []
for cond in CONDITIONS_TO_RUN:
    summaries.append(run_condition(cond))
    pd.DataFrame(summaries).to_csv(
        os.path.join(RESULTS_ROOT, "era_condition_comparison.csv"), index=False
    )

print("\n" + "=" * 70)
print("SUMMARY - DIACHRONIC ERA CLASSIFICATION")
print("=" * 70)
print(pd.DataFrame(summaries).to_string(index=False))
print(f"\nTrivial year-regex probe (for comparison): "
      f"accuracy {probe_acc_all:.4f} overall, {probe_acc_covered:.4f} on the "
      f"{100*probe_covered/n_work:.1f}% of documents containing a year.")
print(f"\nComparison table: {os.path.join(RESULTS_ROOT, 'era_condition_comparison.csv')}")