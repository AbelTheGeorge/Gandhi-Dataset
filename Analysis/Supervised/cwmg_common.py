"""Shared code for the CWMG supervised tasks: ModernBERT fine-tuning on chunked
long documents over several seeds.

Each task script builds its labels and calls run_task(). Metrics and plots can be
rebuilt from saved predictions with: python train_<task>.py --rebuild
"""
from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import platform
import random
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(HERE, "..", "..", "The-Collected-Works-Mahatma-Gandhi.json"))

MODEL_NAME = "answerdotai/ModernBERT-base"
MAX_LENGTH = 8192
CHUNK_OVERLAP = 512
MAX_BATCH_SEQS = 32
TOKEN_BUDGET = 65536
LEARNING_RATE = 5e-5
WEIGHT_DECAY = 0.01
NUM_EPOCHS = 10
WARMUP_RATIO = 0.1
PATIENCE = 3
SEEDS = [42, 43, 44]
SPLIT_SEED = 42
LOG_EVERY_SEC = 60

PRED_TAG = "PREDICTIONS_JSON"


class _Tee:
    def __init__(self, stream, fh):
        self.stream = stream
        self.fh = fh

    def write(self, s):
        try:
            self.stream.write(s)
        except Exception:
            pass
        self.fh.write(s)
        self.fh.flush()
        return len(s)

    def flush(self):
        try:
            self.stream.flush()
        except Exception:
            pass
        self.fh.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


def start_log(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fh = open(path, "a", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print("\n" + "=" * 78)
    print(f"Log started {datetime.datetime.now().isoformat(timespec='seconds')}  ->  {path}")
    print("Command: " + " ".join(sys.argv))
    print("=" * 78)


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78, flush=True)


def fmt_secs(s):
    s = int(max(s, 0))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{sec:02d}s"


def safe(step, fn, *args, **kwargs):
    """Run one output step; on failure print the traceback and continue."""
    try:
        return fn(*args, **kwargs)
    except Exception:
        print(f"\n!!! Step failed: {step} (continuing)")
        traceback.print_exc()
        sys.stderr.flush()
        return None


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=_json_default)
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


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


def load_corpus():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}\n"
            "Keep The-Collected-Works-Mahatma-Gandhi.json at the repository root."
        )
    print(f"Reading: {DATA_PATH}")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    df = pd.DataFrame(records)
    df["doc_id"] = np.arange(len(df))
    df["clean_text"] = df["contents"].apply(clean_text)
    print(f"Total documents loaded: {len(df):,}")
    return df


class TaskSpec:
    def __init__(self, name, title, class_names, condition):
        self.name = name
        self.title = title
        self.class_names = list(class_names)
        self.condition = condition


def parse_args(spec):
    p = argparse.ArgumentParser(description=f"CWMG {spec.title} with ModernBERT")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help=f"training seeds (default {SEEDS})")
    p.add_argument("--max-length", type=int, default=None,
                   help=f"tokens per chunk (default {MAX_LENGTH})")
    p.add_argument("--epochs", type=int, default=None,
                   help=f"maximum epochs (default {NUM_EPOCHS})")
    p.add_argument("--token-budget", type=int, default=None,
                   help=f"padded tokens per batch (default {TOKEN_BUDGET})")
    p.add_argument("--model", default=MODEL_NAME,
                   help="Hugging Face model name or local folder")
    p.add_argument("--rebuild", action="store_true",
                   help="no training: rebuild metrics, plots and summary from saved "
                        "predictions or from run.log")
    p.add_argument("--smoke", action="store_true",
                   help="tiny quick run to check the pipeline end to end")
    a = p.parse_args()

    cfg = {
        "model": a.model,
        "max_length": MAX_LENGTH,
        "chunk_overlap": CHUNK_OVERLAP,
        "max_batch_seqs": MAX_BATCH_SEQS,
        "token_budget": TOKEN_BUDGET,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": NUM_EPOCHS,
        "warmup_ratio": WARMUP_RATIO,
        "patience": PATIENCE,
        "seeds": list(SEEDS),
        "split_seed": SPLIT_SEED,
        "split": "80/10/10 stratified by label, split by document before chunking",
        "condition": spec.condition,
        "log_every_sec": LOG_EVERY_SEC,
        "smoke": a.smoke,
        "smoke_docs_per_class": None,
    }
    if a.smoke:
        cfg.update(max_length=256, chunk_overlap=32, max_epochs=2, patience=2,
                   seeds=[1, 2], log_every_sec=5, smoke_docs_per_class=150)
    if a.seeds:
        cfg["seeds"] = a.seeds
    if a.max_length:
        cfg["max_length"] = a.max_length
    if a.epochs:
        cfg["max_epochs"] = a.epochs
    if a.token_budget:
        cfg["token_budget"] = a.token_budget
    cfg["token_budget"] = max(cfg["token_budget"], cfg["max_length"])
    if cfg["chunk_overlap"] >= cfg["max_length"] - 2:
        raise ValueError("chunk overlap must be smaller than the chunk length")
    return a, cfg


def make_split(work, cfg):
    from sklearn.model_selection import train_test_split
    labels = work["label"].to_numpy()
    idx = np.arange(len(work))
    idx_train, idx_tmp = train_test_split(
        idx, test_size=0.20, random_state=cfg["split_seed"], stratify=labels)
    idx_val, idx_test = train_test_split(
        idx_tmp, test_size=0.50, random_state=cfg["split_seed"], stratify=labels[idx_tmp])
    split = np.empty(len(work), dtype=object)
    split[idx_train], split[idx_val], split[idx_test] = "train", "val", "test"
    return split


def tokenize(tokenizer, texts, batch=2000):
    out = []
    t0 = time.time()
    for i in range(0, len(texts), batch):
        out.extend(tokenizer(texts[i:i + batch], add_special_tokens=False,
                             truncation=False)["input_ids"])
        print(f"  tokenised {min(i + batch, len(texts)):,}/{len(texts):,} documents "
              f"({fmt_secs(time.time() - t0)})", flush=True)
    return out


def chunk_ids(ids, max_len, overlap, cls_id, sep_id):
    body = max_len - 2
    if len(ids) <= body:
        return [[cls_id] + ids + [sep_id]]
    step = body - overlap
    chunks, start = [], 0
    while True:
        chunks.append([cls_id] + ids[start:start + body] + [sep_id])
        if start + body >= len(ids):
            return chunks
        start += step


class ChunkSet:
    """The chunks of one split; chunk_doc maps each chunk to its document row."""

    def __init__(self, doc_tokens, doc_labels, max_len, overlap, cls_id, sep_id):
        self.n_docs = len(doc_tokens)
        self.doc_labels = np.asarray(doc_labels, dtype=np.int64)
        self.doc_ntok = np.array([len(t) for t in doc_tokens], dtype=np.int64)
        self.ids, chunk_doc = [], []
        for d, toks in enumerate(doc_tokens):
            for c in chunk_ids(toks, max_len, overlap, cls_id, sep_id):
                self.ids.append(c)
                chunk_doc.append(d)
        self.chunk_doc = np.array(chunk_doc, dtype=np.int64)
        self.labels = self.doc_labels[self.chunk_doc]
        self.lengths = np.array([len(c) for c in self.ids], dtype=np.int64)
        self.doc_nchunks = np.bincount(self.chunk_doc, minlength=self.n_docs)


def plan_batches(lengths, budget, max_seqs, rng=None):
    """Group sequences of similar length to limit padding; shuffle when rng is given."""
    n = len(lengths)
    if rng is None:
        order = list(np.argsort(-lengths, kind="stable"))
    else:
        idx = rng.permutation(n)
        mega = max_seqs * 50
        order = []
        for i in range(0, n, mega):
            g = idx[i:i + mega]
            order.extend(g[np.argsort(-lengths[g], kind="stable")])
    batches, cur, cur_max = [], [], 0
    for i in order:
        L = int(lengths[i])
        new_max = max(cur_max, L)
        if cur and (len(cur) + 1 > max_seqs or new_max * (len(cur) + 1) > budget):
            batches.append(cur)
            cur, new_max = [], L
        cur.append(int(i))
        cur_max = new_max
    if cur:
        batches.append(cur)
    if rng is not None:
        rng.shuffle(batches)
    return batches


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class Runtime:
    """Device, precision and attention settings, decided once per task."""

    def __init__(self):
        import torch
        self.torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.cuda = self.device.type == "cuda"
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            self.amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        else:
            self.amp_dtype = None
        self.attn = None
        self.budget = None

    def autocast(self):
        torch = self.torch
        if self.amp_dtype is None:
            return torch.autocast("cpu", enabled=False)
        return torch.autocast("cuda", dtype=self.amp_dtype)

    def make_scaler(self):
        torch = self.torch
        enabled = self.amp_dtype == torch.float16
        return torch.amp.GradScaler("cuda", enabled=enabled)

    def empty_cache(self):
        if self.cuda:
            self.torch.cuda.empty_cache()

    def max_mem_gb(self):
        if not self.cuda:
            return 0.0
        return self.torch.cuda.max_memory_allocated(self.device) / 1024 ** 3


def is_oom(exc):
    import torch
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


def load_model(model_name, num_labels, attn):
    from transformers import AutoModelForSequenceClassification
    kwargs = {"num_labels": num_labels}
    if attn:
        kwargs["attn_implementation"] = attn
    try:
        return AutoModelForSequenceClassification.from_pretrained(
            model_name, reference_compile=False, **kwargs)
    except TypeError:
        return AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs)


def collate(rt, seqs, pad_id):
    torch = rt.torch
    L = max(len(s) for s in seqs)
    ids = np.full((len(seqs), L), pad_id, dtype=np.int64)
    mask = np.zeros((len(seqs), L), dtype=np.int64)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = s
        mask[i, :len(s)] = 1
    return (torch.from_numpy(ids).to(rt.device, non_blocking=True),
            torch.from_numpy(mask).to(rt.device, non_blocking=True))


def probe_runtime(rt, cfg, num_labels, pad_id):
    """Choose the attention kernel and largest token budget that fit, using one worst-case step."""
    torch = rt.torch
    candidates = ["sdpa"]
    if rt.cuda:
        try:
            import flash_attn
            candidates = ["flash_attention_2", "sdpa"]
        except Exception:
            print("flash-attn is not installed; using PyTorch SDPA attention.")
    max_len = cfg["max_length"]
    for attn in candidates:
        budget = cfg["token_budget"]
        try:
            model = load_model(cfg["model"], num_labels, attn).to(rt.device)
        except Exception:
            print(f"Could not load the model with attention '{attn}':")
            traceback.print_exc()
            continue
        probe_opt = torch.optim.AdamW(model.parameters(), lr=0.0, weight_decay=0.0)
        ids, mask = collate(rt, [[pad_id + 1] * 8], pad_id)
        with rt.autocast():
            model(input_ids=ids, attention_mask=mask).logits.float().sum().backward()
        probe_opt.step()
        model.zero_grad(set_to_none=True)
        failed = False
        while True:
            rows = max(1, min(cfg["max_batch_seqs"], budget // max_len))
            seqs = [[pad_id + 1] * max_len for _ in range(rows)]
            try:
                if rt.cuda:
                    torch.cuda.reset_peak_memory_stats(rt.device)
                model.train()
                ids, mask = collate(rt, seqs, pad_id)
                with rt.autocast():
                    out = model(input_ids=ids, attention_mask=mask).logits
                out.float().sum().backward()
                probe_opt.step()
                model.zero_grad(set_to_none=True)
                model.eval()
                with torch.no_grad(), rt.autocast():
                    model(input_ids=ids[:1], attention_mask=mask[:1])
                print(f"Probe OK: attention={attn}, {rows} x {max_len} tokens per batch, "
                      f"peak memory {rt.max_mem_gb():.1f} GB")
                if rt.cuda and budget // 2 >= max_len:
                    total_gb = torch.cuda.mem_get_info(rt.device)[1] / 1024 ** 3
                    if rt.max_mem_gb() > 0.85 * total_gb:
                        budget //= 2
                        print(f"Probe: that leaves under 15% of {total_gb:.0f} GB free; "
                              f"using token budget {budget:,} for safety")
                break
            except Exception as e:
                model.zero_grad(set_to_none=True)
                rt.empty_cache()
                if is_oom(e) and budget // 2 >= max_len:
                    budget //= 2
                    print(f"Probe: out of memory, retrying with token budget {budget:,}")
                    continue
                if is_oom(e):
                    print(f"Probe: one sequence of {max_len} tokens does not fit with '{attn}'.")
                else:
                    print(f"Probe failed with attention '{attn}':")
                    traceback.print_exc()
                failed = True
                break
        del model, probe_opt
        rt.empty_cache()
        if not failed:
            rt.attn, rt.budget = attn, budget
            return
    raise RuntimeError("No attention setting worked; see the messages above. "
                       "Try a smaller --max-length.")


def forward_logits(rt, model, seqs, pad_id):
    """Logits for a list of sequences, splitting the batch on out-of-memory."""
    torch = rt.torch
    parts = 1
    while True:
        try:
            size = math.ceil(len(seqs) / parts)
            outs = []
            for i in range(0, len(seqs), size):
                ids, mask = collate(rt, seqs[i:i + size], pad_id)
                with rt.autocast():
                    outs.append(model(input_ids=ids, attention_mask=mask).logits.float())
            return torch.cat(outs)
        except Exception as e:
            if not is_oom(e) or parts >= len(seqs):
                raise
        outs = ids = mask = None
        rt.empty_cache()
        parts *= 2


def _accumulate(rt, model, seqs, labels, weight, denom, scaler, pad_id):
    torch = rt.torch
    ids, mask = collate(rt, seqs, pad_id)
    y = torch.as_tensor(labels, device=rt.device)
    with rt.autocast():
        logits = model(input_ids=ids, attention_mask=mask).logits
    loss = torch.nn.functional.cross_entropy(
        logits.float(), y, weight=weight, reduction="sum") / denom
    scaler.scale(loss).backward()
    return float(loss.item())


def train_step(rt, model, optimizer, seqs, labels, weight, denom, scaler, pad_id, stats):
    """Forward and backward pass on one batch, splitting it on out-of-memory."""
    parts = 1
    while True:
        try:
            size = math.ceil(len(seqs) / parts)
            total = 0.0
            for i in range(0, len(seqs), size):
                total += _accumulate(rt, model, seqs[i:i + size], labels[i:i + size],
                                     weight, denom, scaler, pad_id)
            return total
        except Exception as e:
            if not is_oom(e) or parts >= len(seqs):
                raise
        optimizer.zero_grad(set_to_none=True)
        rt.empty_cache()
        parts *= 2
        stats["oom_splits"] += 1
        print(f"  out of memory on a batch of {len(seqs)} x {max(len(s) for s in seqs)} tokens; "
              f"redoing it in {parts} parts", flush=True)


def predict(rt, model, cs, pad_id, cfg):
    """Document-level class probabilities: the mean of the chunk probabilities."""
    torch = rt.torch
    model.eval()
    K = model.config.num_labels
    chunk_probs = np.zeros((len(cs.ids), K), dtype=np.float64)
    batches = plan_batches(cs.lengths, rt.budget, cfg["max_batch_seqs"])
    with torch.no_grad():
        for b in batches:
            logits = forward_logits(rt, model, [cs.ids[i] for i in b], pad_id)
            chunk_probs[b] = torch.softmax(logits, dim=-1).cpu().numpy()
    doc_probs = np.zeros((cs.n_docs, K), dtype=np.float64)
    np.add.at(doc_probs, cs.chunk_doc, chunk_probs)
    doc_probs /= cs.doc_nchunks[:, None]
    return doc_probs


def doc_metrics(y_true, probs):
    from sklearn.metrics import accuracy_score, f1_score
    pred = probs.argmax(1)
    p_true = np.clip(probs[np.arange(len(y_true)), y_true], 1e-12, 1.0)
    return {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, pred, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "nll": float(-np.mean(np.log(p_true))),
    }


def train_seed(rt, ctx, seed, run_dir):
    torch = rt.torch
    cfg, spec = ctx["cfg"], ctx["spec"]
    tr, va = ctx["chunks"]["train"], ctx["chunks"]["val"]
    K = len(spec.class_names)
    pad_id = ctx["pad_id"]
    tag = f"[{spec.name} seed {seed}]"

    set_seed(seed)
    model = load_model(cfg["model"], K, rt.attn).to(rt.device)
    print(f"{tag} model loaded: {cfg['model']}  attention={rt.attn}  "
          f"pooling={getattr(model.config, 'classifier_pooling', '?')}")

    weight = torch.tensor(ctx["class_weights"], dtype=torch.float32, device=rt.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"],
                                  weight_decay=cfg["weight_decay"])
    rng = np.random.default_rng(seed)
    steps_per_epoch = len(plan_batches(tr.lengths, rt.budget, cfg["max_batch_seqs"],
                                       np.random.default_rng(seed)))
    total_steps = steps_per_epoch * cfg["max_epochs"]
    warmup = max(1, int(total_steps * cfg["warmup_ratio"]))

    def lr_lambda(step):
        if step < warmup:
            return (step + 1) / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = rt.make_scaler()

    best_path = os.path.join(run_dir, "best_model.pt")
    epoch_log_path = os.path.join(run_dir, "epoch_log.jsonl")
    if os.path.exists(epoch_log_path):
        os.remove(epoch_log_path)
    best_f1, best_epoch, bad_epochs = -1.0, 0, 0
    stats = {"oom_splits": 0}
    t_run = time.time()
    epoch_times = []

    print(f"{tag} {len(tr.ids):,} training chunks, ~{steps_per_epoch:,} steps per epoch, "
          f"up to {cfg['max_epochs']} epochs", flush=True)

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        t_ep = time.time()
        t_last = t_ep
        batches = plan_batches(tr.lengths, rt.budget, cfg["max_batch_seqs"], rng)
        loss_sum, n_seen, tok_seen = 0.0, 0, 0
        if rt.cuda:
            torch.cuda.reset_peak_memory_stats(rt.device)
        for step, b in enumerate(batches, 1):
            seqs = [tr.ids[i] for i in b]
            labels = tr.labels[b]
            denom = weight[torch.as_tensor(labels, device=rt.device)].sum()
            optimizer.zero_grad(set_to_none=True)
            loss = train_step(rt, model, optimizer, seqs, labels, weight, denom, scaler, pad_id, stats)
            if not math.isfinite(loss):
                raise RuntimeError(f"{tag} loss became {loss} at epoch {epoch} step {step}")
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            loss_sum += loss * len(b)
            n_seen += len(b)
            tok_seen += int(tr.lengths[b].sum())

            now = time.time()
            if step == 1 or step == len(batches) or now - t_last >= cfg["log_every_sec"]:
                t_last = now
                el = now - t_ep
                ep_eta = el / step * (len(batches) - step)
                print(f"{tag} epoch {epoch}/{cfg['max_epochs']} step {step:,}/{len(batches):,} "
                      f"| loss {loss_sum / n_seen:.4f} | lr {scheduler.get_last_lr()[0]:.2e} "
                      f"| {n_seen / el:.1f} seq/s, {tok_seen / el / 1000:.1f}k tok/s "
                      f"| epoch ETA {fmt_secs(ep_eta)} | mem {rt.max_mem_gb():.1f} GB", flush=True)

        t_val = time.time()
        vm = doc_metrics(va.doc_labels, predict(rt, model, va, pad_id, cfg))
        ep_time = time.time() - t_ep
        epoch_times.append(ep_time)
        improved = vm["macro_f1"] > best_f1
        if improved:
            best_f1, best_epoch, bad_epochs = vm["macro_f1"], epoch, 0
            torch.save(model.state_dict(), best_path + ".tmp")
            os.replace(best_path + ".tmp", best_path)
        else:
            bad_epochs += 1
        rec = {"epoch": epoch, "train_loss": loss_sum / max(n_seen, 1),
               "val_macro_f1": vm["macro_f1"], "val_weighted_f1": vm["weighted_f1"],
               "val_accuracy": vm["accuracy"], "val_nll": vm["nll"],
               "epoch_sec": ep_time, "val_sec": time.time() - t_val,
               "peak_mem_gb": rt.max_mem_gb(), "oom_splits": stats["oom_splits"],
               "best": improved}
        with open(epoch_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"EPOCH_JSON {json.dumps({'task': spec.name, 'seed': seed, **rec})}")
        left = cfg["max_epochs"] - epoch
        status = "NEW BEST, saved" if improved else f"no gain {bad_epochs}/{cfg['patience']}"
        print(f"{tag} epoch {epoch} done in {fmt_secs(ep_time)} | train loss {rec['train_loss']:.4f} "
              f"| val macro F1 {vm['macro_f1']:.4f} acc {vm['accuracy']:.4f} | {status} "
              f"| seed time {fmt_secs(time.time() - t_run)}, at most {fmt_secs(left * np.mean(epoch_times))} left",
              flush=True)
        if bad_epochs >= cfg["patience"]:
            print(f"{tag} early stop after epoch {epoch} (best epoch {best_epoch})")
            break

    info = {"seed": seed, "best_epoch": best_epoch, "best_val_macro_f1": best_f1,
            "epochs_run": len(epoch_times), "train_sec": time.time() - t_run,
            "steps_per_epoch": steps_per_epoch, "oom_splits": stats["oom_splits"],
            "attention": rt.attn, "token_budget": rt.budget,
            "peak_mem_gb": rt.max_mem_gb()}
    write_json(os.path.join(run_dir, "training_done.json"), info)
    print(f"{tag} training finished in {fmt_secs(info['train_sec'])}; best epoch {best_epoch}, "
          f"val macro F1 {best_f1:.4f}", flush=True)
    del model, optimizer
    rt.empty_cache()
    return info


def test_seed(rt, ctx, seed, run_dir):
    torch = rt.torch
    cfg, spec = ctx["cfg"], ctx["spec"]
    model = load_model(cfg["model"], len(spec.class_names), rt.attn).to(rt.device)
    state = torch.load(os.path.join(run_dir, "best_model.pt"), map_location=rt.device)
    model.load_state_dict(state)
    print(f"[{spec.name} seed {seed}] predicting the test set with the best checkpoint ...",
          flush=True)
    probs = predict(rt, model, ctx["chunks"]["test"], ctx["pad_id"], cfg)
    del model, state
    rt.empty_cache()
    return probs


def predictions_frame(ctx, probs):
    spec = ctx["spec"]
    meta = ctx["test_meta"]
    pred = probs.argmax(1)
    df = pd.DataFrame({
        "doc_id": meta["doc_id"].to_numpy(),
        "volume": meta["volume"].to_numpy(),
        "section": meta["section"].to_numpy(),
        "n_tokens": meta["n_tokens"].to_numpy(),
        "n_chunks": meta["n_chunks"].to_numpy(),
        "true_idx": meta["label"].to_numpy(),
        "pred_idx": pred,
    })
    df["true_label"] = [spec.class_names[i] for i in df["true_idx"]]
    df["pred_label"] = [spec.class_names[i] for i in df["pred_idx"]]
    for k, name in enumerate(spec.class_names):
        df[f"prob_{name}"] = probs[:, k]
    return df


def log_predictions(spec, seed, df):
    rows = [[int(r.doc_id), int(r.true_idx), int(r.pred_idx), int(r.n_chunks),
             [round(float(x), 5) for x in p]]
            for r, p in zip(df.itertuples(), df[[f"prob_{c}" for c in spec.class_names]].to_numpy())]
    print(f"{PRED_TAG} " + json.dumps({"task": spec.name, "seed": seed,
                                       "class_names": spec.class_names, "rows": rows}), flush=True)


def predictions_from_log(task_dir, spec, seed, test_meta):
    """Recover a seed's test predictions from the PREDICTIONS_JSON line in run.log."""
    found = None
    log_path = os.path.join(task_dir, "run.log")
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(PRED_TAG):
                try:
                    obj = json.loads(line[len(PRED_TAG):])
                except ValueError:
                    continue
                if obj.get("task") == spec.name and obj.get("seed") == seed:
                    found = obj
    if found is None:
        return None
    by_id = {r[0]: r for r in found["rows"]}
    probs = np.array([by_id[d][4] for d in test_meta["doc_id"]], dtype=np.float64)
    return probs


def compute_metrics(spec, df):
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    y, p = df["true_idx"].to_numpy(), df["pred_idx"].to_numpy()
    probs = df[[f"prob_{c}" for c in spec.class_names]].to_numpy()
    K = len(spec.class_names)
    m = doc_metrics(y, probs)
    prec, rec, f1, sup = precision_recall_fscore_support(
        y, p, labels=list(range(K)), zero_division=0)
    m["per_class"] = {spec.class_names[k]: {"precision": float(prec[k]), "recall": float(rec[k]),
                                            "f1": float(f1[k]), "support": int(sup[k])}
                      for k in range(K)}
    m["confusion_matrix"] = confusion_matrix(y, p, labels=list(range(K))).tolist()
    m["n_test_docs"] = int(len(df))
    for name, sel in [("docs_over_512_tokens", df["n_tokens"].to_numpy() > 510),
                      ("docs_up_to_512_tokens", df["n_tokens"].to_numpy() <= 510),
                      ("docs_chunked", df["n_chunks"].to_numpy() > 1)]:
        if sel.sum() > 0:
            sm = doc_metrics(y[sel], probs[sel])
            m[name] = {"n": int(sel.sum()), "macro_f1": sm["macro_f1"], "accuracy": sm["accuracy"]}
        else:
            m[name] = {"n": 0}
    report = classification_report(y, p, labels=list(range(K)), target_names=spec.class_names,
                                   digits=4, zero_division=0)
    return m, report


def plot_confusion(spec, cm, title, path_base):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matplotlib.rcParams["pdf.fonttype"] = 42
    cm = np.asarray(cm)
    K = len(spec.class_names)
    fig, ax = plt.subplots(figsize=(1.6 * K + 3, 1.3 * K + 2.5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    thr = cm.max() / 2 if cm.size else 0
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > thr else "black")
    ax.set_xticks(range(K), spec.class_names, rotation=45, ha="right")
    ax.set_yticks(range(K), spec.class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path_base + ".pdf")
    fig.savefig(path_base + ".png", dpi=200)
    plt.close(fig)


def finish_seed(ctx, seed, run_dir, probs):
    """Write all outputs for one seed, predictions first."""
    spec = ctx["spec"]
    df = predictions_frame(ctx, probs)
    safe("log predictions", log_predictions, spec, seed, df)
    safe("save predictions CSV", df.to_csv, os.path.join(run_dir, "test_predictions.csv"), index=False)
    res = safe("compute metrics", compute_metrics, spec, df)
    if res is None:
        return None
    m, report = res
    m["seed"] = seed
    done = os.path.join(run_dir, "training_done.json")
    if os.path.exists(done):
        safe("read training info", lambda: m.update(training=json.load(open(done, encoding="utf-8"))))
    print(f"\n[{spec.name} seed {seed}] TEST  macro F1 {m['macro_f1']:.4f} | "
          f"accuracy {m['accuracy']:.4f} | weighted F1 {m['weighted_f1']:.4f}")
    print(report)
    print("Confusion matrix (rows = true, columns = predicted): "
          + json.dumps(dict(zip(spec.class_names, m["confusion_matrix"]))))
    print(f"METRICS_JSON {json.dumps({'task': spec.name, **m}, default=_json_default)}", flush=True)
    safe("save metrics", write_json, os.path.join(run_dir, "test_metrics.json"), m)
    safe("save report", lambda: open(os.path.join(run_dir, "classification_report.txt"), "w",
                                     encoding="utf-8").write(report))
    safe("plot confusion matrix", plot_confusion, spec, m["confusion_matrix"],
         f"{spec.title} - seed {seed} (macro F1 {m['macro_f1']:.3f})",
         os.path.join(run_dir, "confusion_matrix"))
    safe("mark complete", write_json, os.path.join(run_dir, "results_complete.json"),
         {"seed": seed, "macro_f1": m["macro_f1"]})
    return m


def summarise(ctx, task_dir, metrics):
    spec = ctx["spec"]
    rows = []
    for m in metrics:
        r = {"seed": m["seed"], "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
             "accuracy": m["accuracy"],
             "macro_f1_docs_over_512": m["docs_over_512_tokens"].get("macro_f1"),
             "macro_f1_docs_up_to_512": m["docs_up_to_512_tokens"].get("macro_f1")}
        for c in spec.class_names:
            r[f"f1_{c}"] = m["per_class"][c]["f1"]
        t = m.get("training", {})
        r["best_epoch"] = t.get("best_epoch")
        r["train_minutes"] = round(t["train_sec"] / 60, 1) if "train_sec" in t else None
        rows.append(r)
    df = pd.DataFrame(rows).sort_values("seed")
    num = df.drop(columns=["seed"]).apply(pd.to_numeric, errors="coerce")
    mean, std = num.mean(), num.std(ddof=1) if len(df) > 1 else num.std(ddof=0)
    out = pd.concat([df, pd.DataFrame([{"seed": "mean", **mean.to_dict()},
                                       {"seed": "std", **std.to_dict()}])], ignore_index=True)
    out.to_csv(os.path.join(task_dir, "summary_across_seeds.csv"), index=False)
    banner(f"SUMMARY - {spec.title} ({len(df)} seed(s))")
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(out.round(4).to_string(index=False))
    print(f"\nTest macro F1: {mean['macro_f1']:.4f} +/- {std['macro_f1']:.4f} (std over seeds)")
    write_json(os.path.join(task_dir, "summary_across_seeds.json"),
               {"task": spec.name, "n_seeds": len(df),
                "mean": mean.to_dict(), "std": std.to_dict(), "per_seed": rows})
    agg_cm = np.sum([np.asarray(m["confusion_matrix"]) for m in metrics], axis=0)
    safe("plot summed confusion matrix", plot_confusion, spec, agg_cm,
         f"{spec.title} - summed over {len(df)} seeds", os.path.join(task_dir, "confusion_matrix_all_seeds"))


def environment(rt=None):
    env = {"time": datetime.datetime.now().isoformat(timespec="seconds"),
           "python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ["torch", "transformers", "numpy", "pandas", "sklearn", "matplotlib",
                "flash_attn", "tokenizers"]:
        try:
            env[mod] = __import__(mod).__version__
        except Exception:
            env[mod] = None
    try:
        import torch
        env["cuda"] = torch.version.cuda
        env["cudnn"] = torch.backends.cudnn.version()
        env["gpus"] = [{"name": torch.cuda.get_device_name(i),
                        "memory_gb": round(torch.cuda.get_device_properties(i).total_memory / 1024 ** 3, 1)}
                       for i in range(torch.cuda.device_count())]
    except Exception:
        pass
    if rt is not None:
        env["device"] = str(rt.device)
        env["amp_dtype"] = str(rt.amp_dtype)
        env["attention"] = rt.attn
        env["token_budget_used"] = rt.budget
    return env


def run_task(spec, prepare):
    """Train and evaluate one task over all seeds.

    prepare(df) returns (work, diagnostics); work has columns doc_id, volume,
    section, text and label (index into spec.class_names).
    """
    args, cfg = parse_args(spec)
    results_root = os.path.join(HERE, "results_smoke" if cfg["smoke"] else "results")
    task_dir = os.path.join(results_root, spec.name)
    start_log(os.path.join(task_dir, "run.log"))
    banner(f"CWMG - {spec.title.upper()}  ({cfg['model']})")
    print("Config: " + json.dumps(cfg))

    banner("1. DATA")
    df = load_corpus()
    work, diag = prepare(df)
    work = work.reset_index(drop=True)
    if cfg["smoke"]:
        keep = []
        for k in sorted(work["label"].unique()):
            g = work[work["label"] == k]
            keep.append(g.sample(min(len(g), cfg["smoke_docs_per_class"]), random_state=0))
        work = pd.concat(keep).reset_index(drop=True)
        print(f"SMOKE RUN: using {len(work):,} documents")
    print(f"Usable documents: {len(work):,}")
    for k, name in enumerate(spec.class_names):
        n = int((work["label"] == k).sum())
        print(f"  [{k}] {name:<34s} {n:>7,}  ({100 * n / len(work):.1f}%)")
    work["split"] = make_split(work, cfg)
    print("Split: " + ", ".join(f"{s} {int((work['split'] == s).sum()):,}" for s in ["train", "val", "test"]))
    work[["doc_id", "volume", "section", "label", "split"]].to_csv(
        os.path.join(task_dir, "split.csv"), index=False)

    counts = np.bincount(work.loc[work["split"] == "train", "label"], minlength=len(spec.class_names))
    class_weights = np.sqrt(counts.sum() / (len(spec.class_names) * np.maximum(counts, 1)))
    print("Class weights (sqrt-smoothed inverse frequency): "
          + ", ".join(f"{n}={w:.3f}" for n, w in zip(spec.class_names, class_weights)))
    diag["class_counts"] = {n: int((work["label"] == k).sum()) for k, n in enumerate(spec.class_names)}
    write_json(os.path.join(task_dir, "data_diagnostics.json"), diag)

    banner("2. TOKENISING AND CHUNKING")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    tokenizer.model_max_length = 10 ** 9
    pad_id = tokenizer.pad_token_id
    toks = tokenize(tokenizer, work["text"].tolist())
    work["n_tokens"] = [len(t) for t in toks]
    chunks = {}
    for s in ["train", "val", "test"]:
        sel = np.flatnonzero(work["split"].to_numpy() == s)
        chunks[s] = ChunkSet([toks[i] for i in sel], work["label"].to_numpy()[sel], cfg["max_length"],
                             cfg["chunk_overlap"], tokenizer.cls_token_id, tokenizer.sep_token_id)
        cs = chunks[s]
        print(f"  {s:<5s}: {cs.n_docs:,} documents -> {len(cs.ids):,} chunks; "
              f"{int((cs.doc_nchunks > 1).sum()):,} documents longer than {cfg['max_length']} tokens were chunked")
    ntok = work["n_tokens"].to_numpy()
    tok_stats = {"median": float(np.median(ntok)), "p95": float(np.percentile(ntok, 95)),
                 "max": int(ntok.max()), "docs_over_510": int((ntok > 510).sum()),
                 "docs_over_max_length": int((ntok > cfg["max_length"] - 2).sum())}
    print("Document length in tokens: " + json.dumps(tok_stats))
    diag["token_stats"] = tok_stats
    write_json(os.path.join(task_dir, "data_diagnostics.json"), diag)
    test_rows = work[work["split"] == "test"]
    test_meta = pd.DataFrame({"doc_id": test_rows["doc_id"].to_numpy(),
                              "volume": test_rows["volume"].to_numpy(),
                              "section": test_rows["section"].to_numpy(),
                              "n_tokens": test_rows["n_tokens"].to_numpy(),
                              "n_chunks": chunks["test"].doc_nchunks,
                              "label": test_rows["label"].to_numpy()})
    del toks
    ctx = {"spec": spec, "cfg": cfg, "chunks": chunks, "pad_id": pad_id,
           "class_weights": class_weights, "test_meta": test_meta}

    metrics, failed = [], []
    if args.rebuild:
        banner("REBUILD: no training, outputs from saved predictions or run.log")
        for seed in cfg["seeds"]:
            run_dir = os.path.join(task_dir, f"seed_{seed}")
            csv = os.path.join(run_dir, "test_predictions.csv")
            probs = None
            if os.path.exists(csv):
                p = pd.read_csv(csv).set_index("doc_id").loc[test_meta["doc_id"]]
                probs = p[[f"prob_{c}" for c in spec.class_names]].to_numpy()
                print(f"seed {seed}: predictions read from {csv}")
            else:
                probs = predictions_from_log(task_dir, spec, seed, test_meta)
                print(f"seed {seed}: " + ("predictions recovered from run.log" if probs is not None
                                          else "no predictions found, skipped"))
            if probs is not None:
                os.makedirs(run_dir, exist_ok=True)
                m = finish_seed(ctx, seed, run_dir, probs)
                (metrics if m else failed).append(m or seed)
            else:
                failed.append(seed)
    else:
        banner("3. RUNTIME CHECK")
        rt = Runtime()
        print(json.dumps(environment(), indent=1))
        if not rt.cuda:
            print("WARNING: no GPU found - training on CPU will be extremely slow.")
        probe_runtime(rt, cfg, len(spec.class_names), pad_id)
        env = environment(rt)
        write_json(os.path.join(task_dir, "environment.json"), env)
        write_json(os.path.join(task_dir, "config.json"), {**cfg, "attention": rt.attn,
                                                           "token_budget_used": rt.budget})
        t_all = time.time()
        for n, seed in enumerate(cfg["seeds"], 1):
            banner(f"4.{n} SEED {seed}  ({n}/{len(cfg['seeds'])}, task time so far {fmt_secs(time.time() - t_all)})")
            run_dir = os.path.join(task_dir, f"seed_{seed}")
            os.makedirs(run_dir, exist_ok=True)
            try:
                if os.path.exists(os.path.join(run_dir, "results_complete.json")):
                    print(f"seed {seed}: already complete, skipping")
                    m = json.load(open(os.path.join(run_dir, "test_metrics.json"), encoding="utf-8"))
                    metrics.append(m)
                    continue
                csv = os.path.join(run_dir, "test_predictions.csv")
                if os.path.exists(csv):
                    print(f"seed {seed}: predictions exist, only writing outputs")
                    p = pd.read_csv(csv).set_index("doc_id").loc[test_meta["doc_id"]]
                    probs = p[[f"prob_{c}" for c in spec.class_names]].to_numpy()
                else:
                    if not (os.path.exists(os.path.join(run_dir, "training_done.json"))
                            and os.path.exists(os.path.join(run_dir, "best_model.pt"))):
                        train_seed(rt, ctx, seed, run_dir)
                    else:
                        print(f"seed {seed}: training already finished, using saved checkpoint")
                    probs = test_seed(rt, ctx, seed, run_dir)
                m = finish_seed(ctx, seed, run_dir, probs)
                (metrics if m else failed).append(m or seed)
            except Exception:
                print(f"\n!!! seed {seed} failed:")
                traceback.print_exc()
                failed.append(seed)
                rt.empty_cache()

    if metrics:
        safe("summary across seeds", summarise, ctx, task_dir, metrics)
    if failed:
        print(f"\nSeeds without results: {failed}. Re-run the same command to retry them; "
              "finished seeds are skipped.")
    banner(f"FINISHED {spec.title}: {len(metrics)} seed(s) complete, outputs in {task_dir}")
    return 1 if failed else 0
