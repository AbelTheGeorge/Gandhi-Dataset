"""
Fine-tunes roberta-base for Binary Translation Detection
(Original English vs. Translated) on the CWMG dataset.
"""

import json
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from transformers import RobertaTokenizer, RobertaForSequenceClassification, get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
import seaborn as sns
import time

# =====================================================================
# 1. SETUP, DATA LOADING & LABEL ENCODING
# =====================================================================
print("=" * 70)
print("1. SETUP, DATA LOADING & BINARY LABEL CREATION")
print("=" * 70)

GANDHI_PATH = "/Enter/Path/here"
RESULTS_DIR = os.path.join("/kaggle/working/", "translation_results")
MODEL_DIR = os.path.join(RESULTS_DIR, "best_model")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
n_gpus = torch.cuda.device_count()
print(f"Device: {device}")
if n_gpus > 0:
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Load JSON dataset
with open(GANDHI_PATH, 'r', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
print(f"Total documents loaded: {len(df):,}")

def clean_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if isinstance(text, list):
        text = ' '.join([str(item) for item in text if item is not None])
    return str(text).strip()

df['clean_text'] = df['contents'].apply(clean_text)

# Drop rows with empty text
df = df[df['clean_text'] != ""].copy()

# Create Binary Labels based on original_language
# 1 = Original English (Not Translated)
# 0 = Translated (From Gujarati, Hindi, etc.)
def get_translation_label(lang):
    if pd.isna(lang):
        return 0 # Treat missing as translated to be safe, or handle differently based on your dataset rules
    lang_str = str(lang).strip().lower()
    if lang_str == 'english':
        return 1
    else:
        return 0

df['label'] = df['original_language'].apply(get_translation_label)

num_labels = 2
label_mapping = {
    0: "Translated (0)", 
    1: "Original English (1)"
}

print("\nClass Distribution:")
for idx, label_name in label_mapping.items():
    count = (df['label'] == idx).sum()
    print(f"  {label_name:<20s} : {count:,} documents")

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

# =====================================================================
# 4. CLASS WEIGHT COMPUTATION
# =====================================================================
print("\n" + "=" * 70)
print("4. CLASS WEIGHT COMPUTATION (Smoothed)")
print("=" * 70)

class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array([0, 1]),
    y=y_train
)

smoothed_weights = np.sqrt(class_weights)
class_weights_tensor = torch.FloatTensor(smoothed_weights).to(device)
print(f"Smoothed Weights -> Translated: {smoothed_weights[0]:.4f}, Original: {smoothed_weights[1]:.4f}")

# =====================================================================
# 5. MODEL, OPTIMIZER & SCHEDULER SETUP
# =====================================================================
print("\n" + "=" * 70)
print("5. MODEL, OPTIMIZER & SCHEDULER SETUP")
print("=" * 70)

LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 15 # Binary classification usually converges faster, reduced from 25
WARMUP_RATIO = 0.1
PATIENCE = 5

print(f"Loading roberta-base for Binary Classification...")
model = RobertaForSequenceClassification.from_pretrained(
    "roberta-base",
    num_labels=num_labels
)
model.to(device)

if n_gpus > 1:
    print(f"Wrapping model in nn.DataParallel to use {n_gpus} GPUs...")
    model = nn.DataParallel(model)

criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

total_steps = len(train_loader) * NUM_EPOCHS
warmup_steps = int(total_steps * WARMUP_RATIO)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

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
                
                old_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                
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
        'train_loss': train_loss, 'val_loss': val_loss,
        'train_macro_f1': train_macro_f1, 'val_macro_f1': val_macro_f1
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

# =====================================================================
# 7. TEST SET EVALUATION
# =====================================================================
print("\n" + "=" * 70)
print("7. TEST SET EVALUATION")
print("=" * 70)

model = RobertaForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(device)
if n_gpus > 1:
    model = nn.DataParallel(model)

test_loss, test_macro_f1, test_weighted_f1, test_preds, test_labels = run_epoch(
    test_loader, model, criterion, training=False
)

target_names = [label_mapping[0], label_mapping[1]]
report = classification_report(test_labels, test_preds, target_names=target_names, digits=4, zero_division=0)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Macro F1: {test_macro_f1:.4f}")
print(f"\nClassification Report:\n{report}")

# Confusion matrix
cm = confusion_matrix(test_labels, test_preds)
fig, ax = plt.subplots(figsize=(8, 6)) # Smaller figsize for binary CM
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names, ax=ax, linewidths=0.5, linecolor='lightgrey')
ax.set_xlabel('Predicted Label', fontsize=13)
ax.set_ylabel('True Label', fontsize=13)
ax.set_title(f'Translation Detection — roberta-base (Test Macro F1: {test_macro_f1:.4f})', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'translation_confusion_matrix.pdf'), dpi=300)
plt.show()