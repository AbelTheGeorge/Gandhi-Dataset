
import json
import os
import re
import random
import time

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

CONDITIONS_TO_RUN = ["raw"]

SEED = 42
MAX_LENGTH = 512
BATCH_SIZE = 64
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 25
WARMUP_RATIO = 0.1
PATIENCE = 5

NUM_WORKERS = 2 if os.name != "nt" else 0

CLASS_NAMES = ["Translated", "Original English"]
NUM_LABELS = len(CLASS_NAMES)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


set_seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()

print("=" * 70)
print("CWMG - TRANSLATION DETECTION (roberta-base)")
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



print("\n" + "=" * 70)
print("1. LOADING CORPUS AND BUILDING TRANSLATION LABELS")
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
df["language_norm"] = (
    df["original_language"].astype(str).str.strip().str.lower()
)

_MISSING = {"", "none", "nan", "unknown", "null", "<null>", "-"}


def get_translation_label(lang_norm):
    if lang_norm in _MISSING:
        return None
    return 1 if lang_norm == "english" else 0


df["label_opt"] = df["language_norm"].apply(get_translation_label)

n_unknown_lang = int(df["label_opt"].isnull().sum())
n_empty_text = int((df["clean_text"] == "").sum())

work = df[(df["clean_text"] != "") & (df["label_opt"].notnull())].copy()
work["label"] = work["label_opt"].astype(int)

print(f"  documents with unknown language provenance : {n_unknown_lang:,} (dropped)")
print(f"  documents with empty contents              : {n_empty_text:,} (dropped)")
print(f"\nUsable documents: {len(work):,}")

print("\nSource-language composition of the 'Translated' class:")
trans_langs = work.loc[work["label"] == 0, "original_language"].value_counts()
for lang, count in trans_langs.items():
    print(f"  {str(lang):<12s} : {count:,}")

print("\nClass distribution:")
for idx, name in enumerate(CLASS_NAMES):
    n = int((work["label"] == idx).sum())
    print(f"  [{idx}] {name:<18s} : {n:,}  ({100*n/len(work):.1f}%)")

texts_all = work["clean_text"].tolist()
labels_all = work["label"].tolist()

with open(os.path.join(RESULTS_ROOT, "translation_data_diagnostics.json"), "w") as f:
    json.dump({
        "documents_total": int(len(df)),
        "documents_unknown_language": n_unknown_lang,
        "documents_empty_contents": n_empty_text,
        "documents_usable": int(len(work)),
        "class_counts": {
            CLASS_NAMES[i]: int((work["label"] == i).sum()) for i in range(NUM_LABELS)
        },
        "translated_source_languages": {
            str(k): int(v) for k, v in trans_langs.items()
        },
    }, f, indent=2)



print("\n" + "=" * 70)
print("2. TOKENIZER")
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

    results_dir = os.path.join(RESULTS_ROOT, f"translation_{condition}")
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
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:<18s} n={int(counts[i]):>6,}  w={weights[i]:.4f}")
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
        target_names=CLASS_NAMES, digits=4, zero_division=0,
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
        "true_label": [CLASS_NAMES[i] for i in te_labels],
        "predicted_label": [CLASS_NAMES[i] for i in te_preds],
    }).to_csv(os.path.join(results_dir, "test_predictions.csv"), index=False)

    with open(os.path.join(results_dir, "label_mapping.json"), "w") as f:
        json.dump({i: n for i, n in enumerate(CLASS_NAMES)}, f, indent=2)
    with open(os.path.join(results_dir, "run_config.json"), "w") as f:
        json.dump({
            "condition": condition, "seed": SEED, "model": "roberta-base",
            "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE, "weight_decay": WEIGHT_DECAY,
            "max_epochs": NUM_EPOCHS, "warmup_ratio": WARMUP_RATIO, "patience": PATIENCE,
            "split": "80/10/10 stratified",
            "n_train": len(idx_train), "n_val": len(idx_val), "n_test": len(idx_test),
            "pct_documents_truncated": pct_trunc,
            "class_weights": {CLASS_NAMES[i]: float(weights[i]) for i in range(NUM_LABELS)},
        }, f, indent=2)

    cm = confusion_matrix(te_labels, te_preds, labels=list(range(NUM_LABELS)))
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=True,
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, ax=ax,
                linewidths=0.5, linecolor="lightgrey")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Translation detection - {condition} (Macro F1 {te_f1:.4f})")
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
        os.path.join(RESULTS_ROOT, "translation_condition_comparison.csv"), index=False
    )

print("\n" + "=" * 70)
print("SUMMARY - TRANSLATION DETECTION")
print("=" * 70)
print(pd.DataFrame(summaries).to_string(index=False))
print(f"\nComparison table: {os.path.join(RESULTS_ROOT, 'translation_condition_comparison.csv')}")