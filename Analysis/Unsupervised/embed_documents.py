"""Embed every CWMG document with a Qwen3 embedding model, for topic_analysis.py.

    python embed_documents.py              (Qwen3-Embedding-8B)
    python embed_documents.py --smoke      (300 documents, quick check)

Documents longer than the window are split into overlapping windows whose
vectors are averaged and normalised. Work is saved in shards, so rerunning resumes.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys
import time
import traceback
import warnings

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "The-Collected-Works-Mahatma-Gandhi.json"))

MODEL_NAME = "Qwen/Qwen3-Embedding-8B"
INSTRUCTION = "Identify the main topic or theme of the given document"
MAX_LENGTH = 8192
OVERLAP = 512
TOKEN_BUDGET = 131072
MAX_BATCH = 64
SHARD_DOCS = 2000
LOG_EVERY_SEC = 60
SMOKE_DOCS = 300


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


def fmt_secs(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_text(text):
    if text is None or (isinstance(text, float) and text != text):
        return ""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text if t is not None)
    return str(text).strip()


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=MODEL_NAME)
    p.add_argument("--instruction", default=INSTRUCTION, help='topic instruction; "none" for no instruction')
    p.add_argument("--max-length", type=int, default=MAX_LENGTH)
    p.add_argument("--overlap", type=int, default=OVERLAP)
    p.add_argument("--token-budget", type=int, default=TOKEN_BUDGET)
    p.add_argument("--out", help="output folder (default embeddings_<model name>)")
    p.add_argument("--smoke", action="store_true", help=f"first {SMOKE_DOCS} documents only")
    a = p.parse_args()
    if a.instruction.lower() == "none":
        a.instruction = None
    if a.out is None:
        short = a.model.split("/")[-1].replace("Qwen3-Embedding-", "qwen3-")
        a.out = os.path.join(SCRIPT_DIR, ("embeddings_smoke_" if a.smoke else "embeddings_") + short)
    return a


def load_model(name):
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name, padding_side="left")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    attn = "sdpa"
    try:
        import flash_attn
        attn = "flash_attention_2"
    except Exception:
        pass
    import transformers
    dkey = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
    try:
        model = AutoModel.from_pretrained(name, **{dkey: dtype}, attn_implementation=attn)
    except Exception:
        if attn == "sdpa":
            raise
        print("flash_attention_2 failed to load; using sdpa")
        traceback.print_exc()
        attn = "sdpa"
        model = AutoModel.from_pretrained(name, **{dkey: dtype}, attn_implementation=attn)
    if next(model.parameters()).dtype != dtype:
        raise SystemExit(f"model loaded as {next(model.parameters()).dtype}, expected {dtype}")
    model.to("cuda").eval()
    if attn == "sdpa":
        from torch.nn.attention import SDPBackend, sdpa_kernel
        q = torch.randn(1, 8, 64, 128, device="cuda", dtype=dtype)
        try:
            with warnings.catch_warnings(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                warnings.simplefilter("ignore")
                torch.nn.functional.scaled_dot_product_attention(q, q, q, is_causal=True)
            print("PyTorch flash attention kernel: available")
        except RuntimeError:
            print("PyTorch flash attention kernel: NOT available (normal on Windows); "
                  "unpadded batches will be slower")
    return tok, model, dtype, attn


@torch.inference_mode()
def embed_batch(model, seqs, pad_id):
    """Left-padded batch; the last position is each sequence's final token."""
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s)
        mask[i, L - len(s):] = 1
    out = model(input_ids=ids.cuda(), attention_mask=mask.cuda())
    v = out.last_hidden_state[:, -1].float()
    return torch.nn.functional.normalize(v, dim=-1).cpu().numpy()


def embed_safe(model, seqs, pad_id, stats):
    """Embed a batch, splitting it on out-of-memory."""
    parts = 1
    while True:
        try:
            step = -(-len(seqs) // parts)
            return np.concatenate([embed_batch(model, seqs[i:i + step], pad_id)
                                   for i in range(0, len(seqs), step)])
        except torch.OutOfMemoryError:
            if parts >= len(seqs):
                raise
        torch.cuda.empty_cache()
        parts *= 2
        stats["oom_splits"] += 1


def plan_batches(lengths, budget):
    order = np.argsort(-np.asarray(lengths), kind="stable")
    batches, cur, cur_max = [], [], 0
    for i in order:
        m = max(cur_max, lengths[i])
        if cur and (m * (len(cur) + 1) > budget or len(cur) >= MAX_BATCH):
            batches.append(cur)
            cur, m = [], lengths[i]
        cur.append(i)
        cur_max = m
    if cur:
        batches.append(cur)
    return batches


def probe(model, max_len, budget, pad_id):
    """Halve the token budget until the worst-case batch fits in free GPU memory."""
    torch.cuda.empty_cache()
    free, _ = torch.cuda.mem_get_info()
    total = free + torch.cuda.memory_reserved()
    while True:
        n = max(1, min(MAX_BATCH, budget // max_len))
        try:
            torch.cuda.reset_peak_memory_stats()
            embed_batch(model, [[pad_id + 1] * max_len] * n, pad_id)
            peak = torch.cuda.max_memory_reserved()
            ok = peak < 0.85 * total
        except torch.OutOfMemoryError:
            ok, peak = False, None
        torch.cuda.empty_cache()
        if ok or budget <= max_len:
            print(f"Batch check: {n} x {max_len} tokens fits"
                  + (f" (peak {peak / 2**30:.1f} of {total / 2**30:.1f} GB)" if peak else ""))
            return max(budget, max_len)
        budget //= 2
        print(f"Batch check: too big, lowering token budget to {budget:,}")


def main():
    args = parse_args()
    os.makedirs(os.path.join(args.out, "shards"), exist_ok=True)
    log = open(os.path.join(args.out, "embed.log"), "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, log)
    sys.stderr = _Tee(sys.__stderr__, log)
    print("\n" + "=" * 78)
    print(f"START {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(sys.argv)}")
    print("=" * 78)
    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found.")
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"Dataset not found at {DATA_PATH}; keep it at the repository root.")

    with open(DATA_PATH, encoding="utf-8") as f:
        records = json.load(f)
    texts = [clean_text(r.get("contents")) for r in records]
    rec_idx = [i for i, t in enumerate(texts) if t]
    if args.smoke:
        rec_idx = rec_idx[:SMOKE_DOCS]
    print(f"Dataset: {len(records):,} records, {len(rec_idx):,} with text to embed")

    tok, model, dtype, attn = load_model(args.model)
    print(f"Model: {args.model}  dtype {dtype}  attention {attn}  "
          f"GPU {torch.cuda.get_device_name(0)}")
    full = tok("x")["input_ids"]
    bare = tok("x", add_special_tokens=False)["input_ids"]
    suffix = full[len(bare):]
    prefix = (tok(f"Instruct: {args.instruction}\nQuery:", add_special_tokens=False)["input_ids"]
              if args.instruction else [])
    body = args.max_length - len(prefix) - len(suffix)
    stride = body - args.overlap
    if stride <= 0:
        raise SystemExit("--overlap must be smaller than the window")
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else suffix[-1]
    print(f"Window {args.max_length} tokens ({len(prefix)} instruction + {body} text + "
          f"{len(suffix)} end), overlap {args.overlap}")

    budget = probe(model, args.max_length, args.token_budget, pad_id)
    stats = {"oom_splits": 0, "windows": 0, "docs_split": 0, "tokens": 0}
    shards = [rec_idx[i:i + SHARD_DOCS] for i in range(0, len(rec_idx), SHARD_DOCS)]
    todo = [k for k in range(len(shards))
            if not os.path.exists(os.path.join(args.out, "shards", f"shard_{k:03d}.npy"))]
    print(f"Shards: {len(shards)} of up to {SHARD_DOCS} documents, {len(shards) - len(todo)} already done")

    t0, last_log, docs_done = time.time(), time.time(), 0
    docs_todo = sum(len(shards[k]) for k in todo)
    for k in todo:
        ids = tok([texts[i] for i in shards[k]], add_special_tokens=False)["input_ids"]
        seqs, owner = [], []
        for d, t in enumerate(ids):
            starts = list(range(0, max(len(t) - args.overlap, 1), stride)) if len(t) > body else [0]
            stats["docs_split"] += len(starts) > 1
            for s in starts:
                seqs.append(prefix + t[s:s + body] + suffix)
                owner.append(d)
        stats["windows"] += len(seqs)
        lengths = [len(s) for s in seqs]
        vecs = np.zeros((len(seqs), model.config.hidden_size), dtype=np.float32)
        for b in plan_batches(lengths, budget):
            vecs[b] = embed_safe(model, [seqs[i] for i in b], pad_id, stats)
            stats["tokens"] += sum(lengths[i] for i in b)
            if time.time() - last_log > LOG_EVERY_SEC:
                last_log = time.time()
                el = time.time() - t0
                print(f"  shard {k + 1}/{len(shards)}  {stats['tokens'] / el:,.0f} tok/s  "
                      f"memory {torch.cuda.max_memory_allocated() / 2**30:.1f} GB")
        owner = np.array(owner)
        doc_vecs = np.zeros((len(ids), vecs.shape[1]), dtype=np.float32)
        np.add.at(doc_vecs, owner, vecs)
        doc_vecs /= np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        path = os.path.join(args.out, "shards", f"shard_{k:03d}.npy")
        np.save(path + ".tmp.npy", doc_vecs)
        os.replace(path + ".tmp.npy", path)
        docs_done += len(ids)
        el = time.time() - t0
        print(f"Shard {k + 1}/{len(shards)} saved: {docs_done:,}/{docs_todo:,} docs this run, "
              f"{stats['tokens'] / el:,.0f} tok/s, elapsed {fmt_secs(el)}, "
              f"ETA {fmt_secs(el / docs_done * (docs_todo - docs_done))}")

    emb = np.concatenate([np.load(os.path.join(args.out, "shards", f"shard_{k:03d}.npy"))
                          for k in range(len(shards))])
    if len(emb) != len(rec_idx) or not np.isfinite(emb).all():
        raise SystemExit("assembled embeddings are incomplete or contain NaN; rerun to redo shards")
    np.save(os.path.join(args.out, "embeddings.npy"), emb)
    np.save(os.path.join(args.out, "record_index.npy"), np.array(rec_idx, dtype=np.int64))
    meta = {
        "model": args.model, "instruction": args.instruction,
        "instruction_format": "Instruct: {instruction}\\nQuery:{text}",
        "max_length": args.max_length, "overlap": args.overlap,
        "pooling": "last token; windows averaged per document, then L2-normalised",
        "dtype": str(dtype), "attention": attn, "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "dataset_sha256": sha256(DATA_PATH), "documents": len(rec_idx), "dim": int(emb.shape[1]),
        "smoke": args.smoke, "this_run": {k: int(v) for k, v in stats.items()},
        "finished": f"{datetime.datetime.now():%Y-%m-%d %H:%M:%S}",
    }
    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))
    print(f"FINISHED: {len(rec_idx):,} documents embedded into {args.out}")


if __name__ == "__main__":
    main()
