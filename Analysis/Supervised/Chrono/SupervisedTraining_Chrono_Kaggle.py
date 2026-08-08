"""
EraClassifier.py
──────────────────────────
Fine-tunes roberta-base for 5-Class Historical Era classification
on the Collected Works of Mahatma Gandhi (CWMG) dataset.
"""

import json
import os
import re
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
import seaborn as sns
import time

# =====================================================================
# 1. SETUP, DATE PARSING & LABEL ENCODING
# =====================================================================
print("=" * 70)
print("1. SETUP, DATE PARSING & LABEL ENCODING")
print("=" * 70)


GANDHI_PATH = "/Enter/Path/here"
RESULTS_DIR = os.path.join("/kaggle/working/", "results_era")
MODEL_DIR = os.path.join(RESULTS_DIR, "best_model")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Device: {device}")
if n_gpus > 0:
    print(f"Number of GPUs available: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Load JSON dataset
with open(GANDHI_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Total documents loaded: {len(df):,}")


def clean_text(text):
    """Clean and extract text from contents field."""
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if isinstance(text, list):
        text = ' '.join([str(item) for item in text if item is not None])
    return str(text).strip()


def parse_cwmg_date(raw_date_str):
    """Safely parses the 'document_date' field."""
    if not raw_date_str or str(raw_date_str).strip().lower() in ["", "none", "unknown", "<null>"]:
        return None
    
    raw_date_str = str(raw_date_str).strip().upper()
    prefix_pattern = r'^(ON OR BEFORE|ON OR AFTER|ABOUT|PRIOR TO|POST|BEFORE)\s*'
    date_only_str = re.sub(prefix_pattern, '', raw_date_str).strip()
    
    date_regex = r'^(?:(?:(\d{1,2})\s*[-/]\s*)?(?:(\d{1,2})\s*[-/]\s*))?(\d{4})$'
    date_match = re.search(date_regex, date_only_str)
    
    if date_match:
        return int(date_match.group(3)) # Return the Year
    return None


def get_era_label(year):
    """Map a year to one of the 5 historical eras."""
    if year is None:
        return None
    if year <= 1914:
        return "1884-1914 (South Africa)"
    elif 1915 <= year <= 1924:
        return "1915-1924 (Emergence)"
    elif 1925 <= year <= 1934:
        return "1925-1934 (Civil Disobedience)"
    elif 1935 <= year <= 1944:
        return "1935-1944 (Wardha Era)"
    elif year >= 1945:
        return "1945-1948 (Independence)"
    return None


# Apply parsing and cleaning
df['clean_text'] = df['contents'].apply(clean_text)
df['year'] = df['document_date'].apply(parse_cwmg_date)
df['label_str'] = df['year'].apply(get_era_label)

# Drop rows with empty text or missing/unparseable dates
df = df[(df['clean_text'] != "") & (df['label_str'].notnull())].copy()
print(f"Documents after text cleaning & date parsing: {len(df):,}")

# Encode labels
le = LabelEncoder()
df['label'] = le.fit_transform(df['label_str'])
num_labels = len(le.classes_)

# Save label mapping
label_mapping = {int(i): label for i, label in enumerate(le.classes_)}
label_mapping_path = os.path.join(RESULTS_DIR, "label_mapping.json")
with open(label_mapping_path, 'w') as f:
    json.dump(label_mapping, f, indent=2)

print(f"\n{num_labels} Historical Eras:")
for idx, label in label_mapping.items():
    count = (df['label'] == idx).sum()
    print(f"  [{idx:2d}] {label:<35s} : {count:,}")

# =====================================================================
# 2. STRATIFIED TRAIN / VAL / TEST SPLIT (80/10/10)
# =====================================================================
print("\n" + "=" * 70)
print("2. STRATIFIED TRAIN / VAL / TEST SPLIT (80/10/10)")
print("=" * 70)

texts = df['clean_text'].values
labels = df['label'].values

X_train, X_temp, y_train, y_temp = train_test_split(
    texts, labels, test_size=0.20, random_state=42, stratify=labels
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"Train: {len(X_train):,}  |  Val: {len(X_val):,}  |  Test: {len(X_test):,}")

# =====================================================================
# 3. TOKENIZATION & PYTORCH DATASET
# =====================================================================
print("\n" + "=" * 70)
print("3. TOKENIZATION & PYTORCH DATASET")
print("=" * 70)

MAX_LENGTH = 512
BATCH_SIZE = 64 

print(f"Loading RoBERTa tokenizer (max_length={MAX_LENGTH})...")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

class CWMGDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

train_dataset = CWMGDataset(X_train, y_train, tokenizer, MAX_LENGTH)
val_dataset = CWMGDataset(X_val, y_val, tokenizer, MAX_LENGTH)
test_dataset = CWMGDataset(X_test, y_test, tokenizer, MAX_LENGTH)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

print(f"Train batches: {len(train_loader):,}  |  Val batches: {len(val_loader):,}  |  Test batches: {len(test_loader):,}")

# =====================================================================
# 4. CLASS WEIGHT COMPUTATION
# =====================================================================
print("\n" + "=" * 70)
print("4. CLASS WEIGHT COMPUTATION (Smoothed)")
print("=" * 70)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.arange(num_labels),
    y=y_train
)
# Add square root smoothing to prevent over-penalizing large eras
smoothed_weights = np.sqrt(class_weights)
class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)

# =====================================================================
# 5. MODEL, OPTIMIZER & SCHEDULER SETUP
# =====================================================================
print("\n" + "=" * 70)
print("5. MODEL, OPTIMIZER & SCHEDULER SETUP")
print("=" * 70)

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 25
WARMUP_RATIO = 0.1
PATIENCE = 5

print(f"Loading roberta-base with {num_labels} output labels...")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels
)
model.to(device)

if n_gpus > 1:
    print(f"Wrapping model in nn.DataParallel to use {n_gpus} GPUs...")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

scaler = torch.amp.GradScaler('cuda', enabled=torch.cuda.is_available())

# =====================================================================
# 6. TRAINING LOOP WITH EARLY STOPPING
# =====================================================================
print("\n" + "=" * 70)
print("6. TRAINING")
print("=" * 70)

training_log = []
best_val_macro_f1 = 0.0
patience_counter = 0

def run_epoch(loader, model, criterion, optimizer=None, scheduler=None, scaler=None, training=True):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    num_batches = 0

    context = torch.no_grad() if not training else torch.enable_grad()

    with context:
        for batch in loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            if training:
                optimizer.zero_grad()

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                loss = criterion(logits, labels)

            if training:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # FIXED: Capture old scale to prevent PyTorch Warning
                old_scale = scaler.get_scale()
                
                scaler.step(optimizer)
                scaler.update()
                
                # FIXED: Only step scheduler if the scale didn't drop
                if scaler.get_scale() == old_scale:
                    scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            num_batches += 1

    avg_loss = total_loss / num_batches
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    return avg_loss, macro_f1, weighted_f1, all_preds, all_labels


for epoch in range(1, NUM_EPOCHS + 1):
    epoch_start = time.time()

    train_loss, train_macro_f1, train_weighted_f1, _, _ = run_epoch(
        train_loader, model, criterion, optimizer, scheduler, scaler, training=True
    )

    val_loss, val_macro_f1, val_weighted_f1, _, _ = run_epoch(
        val_loader, model, criterion, training=False
    )

    epoch_time = time.time() - epoch_start
    
    training_log.append({
        'epoch': epoch,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_macro_f1': train_macro_f1,
        'val_macro_f1': val_macro_f1,
        'train_weighted_f1': train_weighted_f1,
        'val_weighted_f1': val_weighted_f1,
        'epoch_time_sec': epoch_time
    })

    print(f"\nEpoch {epoch}/{NUM_EPOCHS}  ({epoch_time:.0f}s)")
    print(f"  Train Loss: {train_loss:.4f}  |  Val Loss: {val_loss:.4f}")
    print(f"  Train Macro F1: {train_macro_f1:.4f}  |  Val Macro F1: {val_macro_f1:.4f}")

    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1
        patience_counter = 0

        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(MODEL_DIR)
        tokenizer.save_pretrained(MODEL_DIR)
        print(f"  ★ New best model saved (Val Macro F1: {val_macro_f1:.4f})")
    else:
        patience_counter += 1
        print(f"  No improvement ({patience_counter}/{PATIENCE})")
        if patience_counter >= PATIENCE:
            print(f"\n  Early stopping triggered after epoch {epoch}.")
            break

log_df = pd.DataFrame(training_log)
training_log_path = os.path.join(RESULTS_DIR, "training_metrics.csv")
log_df.to_csv(training_log_path, index=False)
print(f"\nTraining metrics saved to: {training_log_path}")

# =====================================================================
# 7. TEST SET EVALUATION
# =====================================================================
print("\n" + "=" * 70)
print("7. TEST SET EVALUATION")
print("=" * 70)

print("Loading best checkpoint for evaluation...")
model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)

if n_gpus > 1:
    model = nn.DataParallel(model)

test_loss, test_macro_f1, test_weighted_f1, test_preds, test_labels = run_epoch(
    test_loader, model, criterion, training=False
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")

target_names = [label_mapping[i] for i in range(num_labels)]
report = classification_report(test_labels, test_preds, target_names=target_names, digits=4, zero_division=0)
print(f"\nClassification Report:\n{report}")

test_results = [{
    'test_loss': test_loss,
    'test_macro_f1': test_macro_f1,
    'test_weighted_f1': test_weighted_f1
}]
test_df = pd.DataFrame(test_results)
test_results_path = os.path.join(RESULTS_DIR, "test_metrics.csv")
test_df.to_csv(test_results_path, index=False)
print(f"Test metrics saved to: {test_results_path}")

report_path = os.path.join(RESULTS_DIR, "classification_report.txt")
with open(report_path, 'w') as f:
    f.write(report)
print(f"Classification report text saved to: {report_path}")

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(16, 14))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax, linewidths=0.5, linecolor='lightgrey')
ax.set_xlabel('Predicted Era', fontsize=13)
ax.set_ylabel('True Era', fontsize=13)
ax.set_title(f'Confusion Matrix — RoBERTa Era Classification (Test Macro F1: {test_macro_f1:.4f})', fontsize=15)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()