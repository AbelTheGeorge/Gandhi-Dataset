"""CWMG topic analysis on document embeddings over several seeds and clustering settings.

    python topic_analysis.py --embeddings embeddings_qwen3-8B
    python topic_analysis.py --embeddings embeddings_qwen3-8B --smoke   (quick check)

Steps: a cosine 15-nearest-neighbour graph; per seed, UMAP to 5 dimensions; per
setting, HDBSCAN or k-means on the UMAP output, or k-means on the full embeddings;
topic terms by class-based TF-IDF; coherence, diversity and seed-agreement measures.
Results are cached, so rerunning skips finished work.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import time
import traceback

import numpy as np
import pandas as pd
import scipy.sparse as sp

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "..", "The-Collected-Works-Mahatma-Gandhi.json"))

N_NEIGHBORS = 15
UMAP_DIM = 5
N_SEEDS = 10
TOP_WORDS = 10
MATCH_JACCARD = 0.5
STABLE_SHARE = 0.8
N_EXAMPLES = 3

CONFIGS = {
    "hdbscan_mcs100":      dict(kind="hdbscan", min_cluster_size=100, min_samples=None),
    "hdbscan_mcs100_ms10": dict(kind="hdbscan", min_cluster_size=100, min_samples=10),
    "hdbscan_mcs50_ms10":  dict(kind="hdbscan", min_cluster_size=50, min_samples=10),
    "kmeans_umap_k20":     dict(kind="kmeans_umap", k=20),
    "kmeans_umap_k40":     dict(kind="kmeans_umap", k=40),
    "kmeans_umap_k60":     dict(kind="kmeans_umap", k=60),
    "kmeans_full_k20":     dict(kind="kmeans_full", k=20),
    "kmeans_full_k40":     dict(kind="kmeans_full", k=40),
    "kmeans_full_k60":     dict(kind="kmeans_full", k=60),
}
SMOKE_DOCS = 4000
SMOKE_CONFIGS = {
    "hdbscan_mcs20_ms5": dict(kind="hdbscan", min_cluster_size=20, min_samples=5),
    "kmeans_umap_k15":   dict(kind="kmeans_umap", k=15),
    "kmeans_full_k15":   dict(kind="kmeans_full", k=15),
}

ERA_BOUNDS = [
    (1884, 1914, "1884-1914 South Africa"),
    (1915, 1924, "1915-1924 Emergence"),
    (1925, 1934, "1925-1934 Civil Disobedience"),
    (1935, 1944, "1935-1944 Wardha"),
    (1945, 1948, "1945-1948 Independence"),
]

CUSTOM_STOP_WORDS = """
bapu blessings chi dear mr sincerely gandhi letter people time ashram sabarmati sevagram
glad know shall indian passage wardha segaon wish quite sir yours mahatma january february
ba march april may june july august september october november december bhai friend
vandemataram improve mohandas sjt thanks health good reply able hope letters shri today
thank copy esq long received happy dr right day just
""".split()
COMMON_VERBS = """
say said says saying go went gone goes going make made makes making take took taken takes
taking get got gotten gets getting think thought thinks thinking see saw seen sees seeing
come came comes coming want wanted wants wanting look looked looks looking use used uses
using find found finds finding give gave given gives giving tell told tells telling work
worked works working call called calls calling try tried tries trying ask asked asks asking
need needed needs needing feel felt feels feeling become became becomes becoming leave left
leaves leaving put puts putting mean meant means meaning keep kept keeps keeping let lets
letting begin began begun begins beginning seem seemed seems seeming help helped helps
helping talk talked talks talking turn turned turns turning start started starts starting
show showed shown shows showing hear heard hears hearing play played plays playing run ran
runs running move moved moves moving like liked likes liking live lived lives living believe
believed believes believing hold held holds holding bring brought brings bringing happen
happened happens happening write wrote written writes writing provide provided provides
providing sit sat sits sitting stand stood stands standing lose lost loses losing pay paid
pays paying meet met meets meeting include included includes including continue continued
continues continuing set sets setting learn learned learnt learns learning change changed
changes changing lead led leads leading understand understood understands understanding
watch watched watches watching follow followed follows following stop stopped stops stopping
create created creates creating speak spoke spoken speaks speaking read reads reading allow
allowed allows allowing add added adds adding spend spent spends spending grow grew grown
grows growing open opened opens opening walk walked walks walking win won wins winning offer
offered offers offering remember remembered remembers remembering love loved loves loving
consider considered considers considering appear appeared appears appearing buy bought buys
buying wait waited waits waiting serve served serves serving die died dies dying send sent
sends sending expect expected expects expecting build built builds building stay stayed
stays staying fall fell fallen falls falling cut cuts cutting reach reached reaches reaching
kill killed kills killing remain remained remains remaining do did done does doing have had
has having be is am are was were been being
""".split()


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


def start_log(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fh = open(os.path.join(out_dir, "run.log"), "a", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    sys.stderr = _Tee(sys.__stderr__, fh)
    print("\n" + "=" * 78)
    print(f"START {datetime.datetime.now():%Y-%m-%d %H:%M:%S}  {' '.join(sys.argv)}")
    print("=" * 78)


FAILED_STEPS = []


def safe(step, fn, *a, **kw):
    """Run one output step; on failure print the traceback and continue."""
    try:
        return fn(*a, **kw)
    except Exception:
        FAILED_STEPS.append(step)
        print(f"!! step '{step}' failed; continuing. Rerun the same command to retry it.")
        traceback.print_exc()
        return None


def save_npy(path, arr):
    tmp = path + ".tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, path)


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    os.replace(tmp, path)


def fmt_secs(s):
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


_PREFIX_RE = re.compile(r"^(ON OR BEFORE|ON OR AFTER|ON OR AROUND)\s+")
_DATE_RES = [re.compile(r"^\d{2}-\d{2}-(\d{4})$"), re.compile(r"^\d{2}-(\d{4})$"),
             re.compile(r"^(\d{4})$")]


def parse_year(raw):
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    s = str(raw).strip().upper()
    m = _PREFIX_RE.match(s)
    s = s[m.end():].strip() if m else s
    for rx in _DATE_RES:
        m = rx.match(s)
        if m:
            return int(m.group(1))
    return None


def era_of(year):
    if year is None:
        return None
    for lo, hi, name in ERA_BOUNDS:
        if lo <= year <= hi:
            return name
    return None


def clean_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    if isinstance(text, list):
        text = " ".join(str(t) for t in text if t is not None)
    return str(text).strip()


def load_inputs(emb_dir, smoke):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}; keep it at the repository root.")
    print(f"Reading dataset: {DATA_PATH}")
    with open(DATA_PATH, encoding="utf-8") as f:
        df = pd.DataFrame(json.load(f))
    df["record_index"] = np.arange(len(df))
    df["text"] = df["contents"].apply(clean_text)
    df["year"] = df["document_date"].apply(parse_year)
    df["era"] = df["year"].apply(era_of)
    df["type"] = df["document_type"].astype(str).str.strip().str.upper()
    df["language"] = df["original_language"].astype(str).str.strip().str.lower()

    print(f"Reading embeddings: {emb_dir}")
    emb = np.load(os.path.join(emb_dir, "embeddings.npy")).astype(np.float32)
    rec = np.load(os.path.join(emb_dir, "record_index.npy"))
    meta_path = os.path.join(emb_dir, "meta.json")
    emb_meta = json.load(open(meta_path, encoding="utf-8")) if os.path.exists(meta_path) else {}
    if len(emb) != len(rec):
        raise ValueError(f"embeddings has {len(emb)} rows but record_index has {len(rec)}")
    if rec.max() >= len(df):
        raise ValueError("record_index points past the end of the dataset")
    if not np.isfinite(emb).all():
        raise ValueError("embeddings contain NaN or inf")

    data_sha = sha256(DATA_PATH)
    if emb_meta.get("dataset_sha256") and emb_meta["dataset_sha256"] != data_sha:
        print("!! WARNING: these embeddings were made from a different dataset file than the one "
              "at the repository root. Fine for a test, not for reported numbers.")
    if emb_meta.get("test_only"):
        print("!! NOTE: embeddings are marked test_only (older dataset version).")

    docs = df.iloc[rec].reset_index(drop=True)
    keep = (docs["text"] != "").to_numpy()
    docs, emb = docs[keep].reset_index(drop=True), emb[keep]
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)

    if smoke:
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(len(docs), size=min(SMOKE_DOCS, len(docs)), replace=False))
        docs, emb = docs.iloc[pick].reset_index(drop=True), emb[pick]

    diag = {
        "dataset_records": int(len(df)),
        "dataset_sha256": data_sha,
        "embedded_records": int(len(rec)),
        "records_without_embedding": int(len(df) - len(rec)),
        "embedded_but_empty_text_dropped": int((~keep).sum()),
        "documents_analysed": int(len(docs)),
        "embedding_dim": int(emb.shape[1]),
        "documents_with_era": int(docs["era"].notnull().sum()),
        "embedding_meta": emb_meta,
    }
    for k, v in diag.items():
        if k != "embedding_meta":
            print(f"  {k}: {v}")
    return docs, emb, diag


def exact_knn(emb, k, cache_dir):
    idx_path = os.path.join(cache_dir, "knn_idx.npy")
    dist_path = os.path.join(cache_dir, "knn_dist.npy")
    if os.path.exists(idx_path) and os.path.exists(dist_path):
        idx, dist = np.load(idx_path), np.load(dist_path)
        if idx.shape == (len(emb), k):
            print("kNN graph: cached")
            return idx, dist
    t0 = time.time()
    n = len(emb)
    idx = np.empty((n, k), dtype=np.int64)
    dist = np.empty((n, k), dtype=np.float32)
    try:
        import torch
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        full = torch.from_numpy(emb).to(dev)
        use_torch = True
    except Exception:
        use_torch = False
        dev = "numpy"
    print(f"kNN graph: exact cosine, k={k}, on {dev}")
    block = 2048
    for s in range(0, n, block):
        e = min(s + block, n)
        if use_torch:
            sims = full[s:e] @ full.T
            sims[torch.arange(e - s), torch.arange(s, e)] = -2.0
            top = torch.topk(sims, k - 1, dim=1)
            ti, tv = top.indices.cpu().numpy(), top.values.float().cpu().numpy()
        else:
            sims = emb[s:e] @ emb.T
            sims[np.arange(e - s), np.arange(s, e)] = -2.0
            ti = np.argpartition(-sims, k - 1, axis=1)[:, :k - 1]
            tv = np.take_along_axis(sims, ti, axis=1)
            o = np.argsort(-tv, axis=1)
            ti, tv = np.take_along_axis(ti, o, 1), np.take_along_axis(tv, o, 1)
        idx[s:e, 0] = np.arange(s, e)
        dist[s:e, 0] = 0.0
        idx[s:e, 1:] = ti
        dist[s:e, 1:] = np.clip(1.0 - tv, 0.0, 2.0)
    if use_torch:
        del full, sims, top
        torch.cuda.empty_cache()
    save_npy(idx_path, idx)
    save_npy(dist_path, dist)
    print(f"kNN graph: done in {fmt_secs(time.time() - t0)}")
    return idx, dist


def doc_term_matrix(texts, cache_dir, min_df):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, CountVectorizer
    x_path = os.path.join(cache_dir, "doc_term.npz")
    v_path = os.path.join(cache_dir, "vocab.json")
    if os.path.exists(x_path) and os.path.exists(v_path):
        X = sp.load_npz(x_path).tocsr()
        vocab = json.load(open(v_path, encoding="utf-8"))
        if X.shape == (len(texts), len(vocab)):
            print(f"Document-term matrix: cached ({X.shape[1]:,} terms)")
            return X, np.array(vocab, dtype=object)
    t0 = time.time()
    stop = sorted(ENGLISH_STOP_WORDS.union(CUSTOM_STOP_WORDS).union(COMMON_VERBS))
    cv = CountVectorizer(stop_words=stop, ngram_range=(1, 2), min_df=min_df, max_df=0.5,
                         token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b", dtype=np.int32)
    X = cv.fit_transform(texts).tocsr()
    vocab = cv.get_feature_names_out().tolist()
    sp.save_npz(x_path + ".tmp.npz", X)
    os.replace(x_path + ".tmp.npz", x_path)
    with open(v_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f)
    print(f"Document-term matrix: {X.shape[0]:,} docs x {X.shape[1]:,} terms "
          f"in {fmt_secs(time.time() - t0)}")
    return X, np.array(vocab, dtype=object)


def umap_coords(emb, knn, seed, cache_dir):
    path = os.path.join(cache_dir, "umap", f"seed_{seed}.npy")
    if os.path.exists(path):
        return np.load(path)
    import warnings
    import umap
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        reducer = umap.UMAP(n_neighbors=N_NEIGHBORS, n_components=UMAP_DIM, min_dist=0.0,
                            metric="cosine", random_state=seed,
                            precomputed_knn=(knn[0], knn[1], None))
        coords = reducer.fit_transform(emb).astype(np.float32)
    save_npy(path, coords)
    return coords


def cluster(cfg, seed, emb, coords):
    if cfg["kind"] == "hdbscan":
        from sklearn.cluster import HDBSCAN
        return HDBSCAN(min_cluster_size=cfg["min_cluster_size"], min_samples=cfg["min_samples"],
                       cluster_selection_method="eom", copy=True).fit_predict(coords)
    from sklearn.cluster import KMeans
    X = coords if cfg["kind"] == "kmeans_umap" else emb
    return KMeans(n_clusters=cfg["k"], n_init=3, random_state=seed).fit_predict(X)


def class_tfidf(labels, X):
    """BERTopic's c-TF-IDF. Returns topic ids and a (topics x terms) score matrix."""
    topics = np.array(sorted(t for t in np.unique(labels) if t >= 0))
    rows = np.searchsorted(topics, labels[labels >= 0])
    docs = np.flatnonzero(labels >= 0)
    T = sp.csr_matrix((np.ones(len(docs)), (rows, docs)), shape=(len(topics), len(labels)))
    C = (T @ X).toarray().astype(np.float64)
    tf = C / np.maximum(C.sum(1, keepdims=True), 1)
    avg = C.sum(1).mean()
    idf = np.log(1 + avg / np.maximum(C.sum(0), 1))
    return topics, tf * idf


def top_terms(scores, vocab, n=TOP_WORDS):
    order = np.argsort(-scores, axis=1)[:, :n]
    return order, [[vocab[j] for j in row] for row in order]


def npmi(term_ids, Xb, n_docs, df_counts):
    """Mean NPMI over pairs of a topic's top terms, with document co-occurrence."""
    sub = Xb[:, term_ids]
    co = (sub.T @ sub).toarray().astype(np.float64)
    p = df_counts[term_ids] / n_docs
    vals = []
    for i in range(len(term_ids)):
        for j in range(i + 1, len(term_ids)):
            pij = co[i, j] / n_docs
            if pij <= 0:
                vals.append(-1.0)
            elif pij >= 1:
                vals.append(1.0)
            else:
                vals.append(np.log(pij / (p[i] * p[j])) / -np.log(pij))
    return float(np.mean(vals))


def run_measures(labels, X, Xb, df_counts, vocab, docs):
    from sklearn.metrics import normalized_mutual_info_score as nmi
    n = len(labels)
    clustered = labels >= 0
    topics, scores = class_tfidf(labels, X)
    order, words = top_terms(scores, vocab)
    coh = [npmi(order[i], Xb, n, df_counts) for i in range(len(topics))]
    uniq = len({w for ws in words for w in ws})
    sizes = np.bincount(labels[clustered])
    m = {
        "topics": int(len(topics)),
        "outlier_share": float(1 - clustered.mean()),
        "largest_topic_share": float(sizes.max() / n),
        "npmi_mean": float(np.mean(coh)),
        "npmi_weighted": float(np.average(coh, weights=sizes[topics])),
        "topic_diversity": float(uniq / (TOP_WORDS * len(topics))),
    }
    for col in ("era", "type", "language"):
        ok = clustered & docs[col].notnull().to_numpy()
        m[f"nmi_{col}"] = float(nmi(docs[col].to_numpy()[ok].astype(str), labels[ok]))
    return m, topics, words, coh


def contingency(a, b):
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    C = sp.coo_matrix((np.ones(len(a)), (ia, ib)), shape=(len(ua), len(ub))).toarray()
    return ua, ub, C


def best_jaccard(a, b):
    """For each topic in a (outliers excluded), the best Jaccard overlap with a topic in b."""
    ua, ub, C = contingency(a, b)
    ra, rb = ua >= 0, ub >= 0
    na = np.bincount(np.searchsorted(ua, a))[ra]
    nb = np.bincount(np.searchsorted(ub, b))[rb]
    I = C[np.ix_(ra, rb)]
    J = I / (na[:, None] + nb[None, :] - I)
    return ua[ra], J.max(axis=1) if J.shape[1] else np.zeros(ra.sum())


def stability(label_runs):
    from sklearn.metrics import adjusted_rand_score as ari
    from sklearn.metrics import normalized_mutual_info_score as nmi
    seeds = sorted(label_runs)
    s = len(seeds)
    A = np.eye(s)
    ari_both, nmis = [], []
    for i in range(s):
        for j in range(i + 1, s):
            a, b = label_runs[seeds[i]], label_runs[seeds[j]]
            A[i, j] = A[j, i] = ari(a, b)
            both = (a >= 0) & (b >= 0)
            ari_both.append(ari(a[both], b[both]))
            nmis.append(nmi(a, b))
    pairs = A[np.triu_indices(s, 1)]
    repro = {}
    for i in range(s):
        a = label_runs[seeds[i]]
        hits = None
        for j in range(s):
            if i == j:
                continue
            t, best = best_jaccard(a, label_runs[seeds[j]])
            hits = (best >= MATCH_JACCARD).astype(float) if hits is None else hits + (best >= MATCH_JACCARD)
        repro[seeds[i]] = dict(zip(t.tolist(), (hits / (s - 1)).tolist()))
    stable_share, stable_docs = [], []
    for sd in seeds:
        a = label_runs[sd]
        r = repro[sd]
        st = [t for t, v in r.items() if v >= STABLE_SHARE]
        stable_share.append(len(st) / max(len(r), 1))
        stable_docs.append(np.isin(a, st).mean())
    medoid = seeds[int(np.argmax(A.sum(1)))]
    return {
        "pairwise_ari_mean": float(pairs.mean()), "pairwise_ari_std": float(pairs.std()),
        "pairwise_ari_clustered_mean": float(np.mean(ari_both)),
        "pairwise_nmi_mean": float(np.mean(nmis)),
        "stable_topic_share_mean": float(np.mean(stable_share)),
        "docs_in_stable_topics_mean": float(np.mean(stable_docs)),
        "reference_seed": int(medoid),
    }, repro, A


def topic_table(name, labels, repro, emb, X, vocab, Xb, df_counts, docs, out_dir):
    _, topics, words, coh = run_measures(labels, X, Xb, df_counts, vocab, docs)
    n = len(labels)
    eras = [e for _, _, e in ERA_BOUNDS]
    rows = []
    for i, t in enumerate(topics):
        mem = np.flatnonzero(labels == t)
        cen = emb[mem].mean(0)
        cen /= np.linalg.norm(cen)
        ex = mem[np.argsort(-(emb[mem] @ cen))[:N_EXAMPLES]]
        era_counts = docs["era"].iloc[mem].value_counts()
        dated = era_counts.sum()
        types = docs["type"].iloc[mem].value_counts(normalize=True)
        yrs = docs["year"].iloc[mem].dropna()
        row = {
            "topic": int(t), "documents": len(mem), "share": len(mem) / n,
            "reappears_in_share_of_runs": repro.get(int(t), np.nan),
            "stable": repro.get(int(t), 0) >= STABLE_SHARE,
            "npmi": coh[i], "top_words": ", ".join(words[i]),
            "median_year": float(yrs.median()) if len(yrs) else np.nan,
            "main_type": types.index[0], "main_type_share": float(types.iloc[0]),
        }
        for e in eras:
            row[f"era: {e}"] = float(era_counts.get(e, 0) / dated) if dated else np.nan
        row["example_titles"] = " | ".join(str(docs["title"].iloc[j])[:90] for j in ex)
        rows.append(row)
    tab = pd.DataFrame(rows).sort_values("documents", ascending=False)
    tab.to_csv(os.path.join(out_dir, f"topics_{name}.csv"), index=False)

    lines = [f"Topics for {name} (reference run; '*' = stable, reappears in >= "
             f"{STABLE_SHARE:.0%} of other runs at Jaccard >= {MATCH_JACCARD})", ""]
    for _, r in tab.iterrows():
        lines.append(f"{'*' if r['stable'] else ' '} T{r['topic']:<3} {r['documents']:>6} docs "
                     f"({r['share']:.1%})  reappears {r['reappears_in_share_of_runs']:.0%}  "
                     f"NPMI {r['npmi']:+.3f}  median year {r['median_year']:.0f}  "
                     f"{r['main_type']} {r['main_type_share']:.0%}")
        lines.append(f"       {r['top_words']}")
        lines.append(f"       e.g. {r['example_titles']}")
    with open(os.path.join(out_dir, f"topics_{name}.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    pd.DataFrame({"record_index": docs["record_index"], "topic": labels}).to_csv(
        os.path.join(out_dir, f"assignments_{name}.csv"), index=False)
    return tab


def plot_summary(summary, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    panels = [("topics", "Topics"), ("outlier_share", "Outlier share"),
              ("npmi_mean", "NPMI coherence"), ("topic_diversity", "Topic diversity"),
              ("pairwise_ari_mean", "Agreement between seeds (ARI)"),
              ("docs_in_stable_topics_mean", "Docs in stable topics")]
    names = list(summary.index)
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for ax, (col, title) in zip(axes.ravel(), panels):
        std = summary[col + "_std"] if col + "_std" in summary else None
        ax.barh(names, summary[col], xerr=std, color="#5B7DB1", capsize=3)
        ax.set_title(title)
        ax.invert_yaxis()
        ax.tick_params(axis="y", labelsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "settings_comparison.png"), dpi=150)
    plt.close(fig)


def plot_era_heatmap(name, tab, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    cols = [c for c in tab.columns if c.startswith("era: ")]
    t = tab.sort_values("median_year")
    M = t[cols].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(8, max(4, 0.22 * len(t))))
    ax.imshow(M, aspect="auto", cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c[5:] for c in cols], rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(t)))
    ax.set_yticklabels([f"{'*' if s else ' '}T{tp} {w.split(', ')[0]}, {w.split(', ')[1]}"
                        for tp, w, s in zip(t["topic"], t["top_words"], t["stable"])], fontsize=7)
    ax.set_title(f"{name}: share of each topic's dated documents by era")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f"era_heatmap_{name}.png"), dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embeddings", required=True, help="folder with embeddings.npy, record_index.npy, meta.json")
    p.add_argument("--out", help="output folder (default results/<embeddings folder name>)")
    p.add_argument("--seeds", type=int, default=N_SEEDS)
    p.add_argument("--smoke", action="store_true", help=f"{SMOKE_DOCS} documents, 3 seeds, 3 settings")
    a = p.parse_args()
    if not os.path.isabs(a.embeddings) and not os.path.exists(a.embeddings):
        a.embeddings = os.path.join(SCRIPT_DIR, a.embeddings)
    if a.out is None:
        base = "results_smoke" if a.smoke else "results"
        a.out = os.path.join(SCRIPT_DIR, base, os.path.basename(os.path.normpath(a.embeddings)))
    if a.smoke:
        a.seeds = min(a.seeds, 3)
    return a


def main():
    args = parse_args()
    start_log(args.out)
    configs = SMOKE_CONFIGS if args.smoke else CONFIGS
    seeds = list(range(args.seeds))
    cache = os.path.join(args.out, "cache")
    os.makedirs(cache, exist_ok=True)

    docs, emb, diag = load_inputs(args.embeddings, args.smoke)
    write_json(os.path.join(args.out, "data_diagnostics.json"), diag)
    config_record = {"configs": configs, "seeds": seeds, "n_neighbors": N_NEIGHBORS,
                     "umap_dim": UMAP_DIM, "top_words": TOP_WORDS, "match_jaccard": MATCH_JACCARD,
                     "stable_share": STABLE_SHARE, "smoke": args.smoke,
                     "embeddings": os.path.abspath(args.embeddings)}
    write_json(os.path.join(args.out, "config.json"), config_record)

    knn = exact_knn(emb, N_NEIGHBORS, cache)
    X, vocab = doc_term_matrix(docs["text"].tolist(), cache, min_df=3 if args.smoke else 10)
    Xb = X.copy()
    Xb.data[:] = 1
    Xb = Xb.tocsc()
    df_counts = np.asarray(Xb.sum(0)).ravel().astype(np.float64)

    jobs = [(s, name) for s in seeds for name in configs]
    todo = [(s, n) for s, n in jobs if not os.path.exists(os.path.join(cache, "runs", n, f"seed_{s}.npy"))]
    print(f"\nClustering: {len(jobs)} runs ({len(seeds)} seeds x {len(configs)} settings), "
          f"{len(jobs) - len(todo)} already done")
    t0, done = time.time(), 0
    for s in seeds:
        mine = [n for ss, n in todo if ss == s]
        if not mine:
            continue
        need_umap = any(configs[n]["kind"] != "kmeans_full" for n in mine)
        coords = None
        if need_umap:
            tu = time.time()
            coords = umap_coords(emb, knn, s, cache)
            print(f"  seed {s}: UMAP {fmt_secs(time.time() - tu)}")
        for n in mine:
            tc = time.time()
            try:
                lab = cluster(configs[n], s, emb, coords).astype(np.int32)
            except Exception:
                print(f"!! seed {s} {n} failed; rerun the same command to retry")
                traceback.print_exc()
                continue
            os.makedirs(os.path.join(cache, "runs", n), exist_ok=True)
            save_npy(os.path.join(cache, "runs", n, f"seed_{s}.npy"), lab)
            done += 1
            rate = (time.time() - t0) / done
            print(f"  seed {s} {n:<22} {len(set(lab) - {-1}):>3} topics, outliers "
                  f"{(lab < 0).mean():5.1%}  ({fmt_secs(time.time() - tc)}; "
                  f"{done}/{len(todo)}, ETA {fmt_secs(rate * (len(todo) - done))})")

    print("\nMeasures")
    run_rows, summary_rows = [], []
    for name in configs:
        runs = {}
        for s in seeds:
            p = os.path.join(cache, "runs", name, f"seed_{s}.npy")
            if os.path.exists(p):
                runs[s] = np.load(p)
        if len(runs) < 2:
            print(f"  {name}: fewer than 2 finished runs, skipped")
            continue
        per = []
        for s, lab in runs.items():
            m = safe(f"measures {name} seed {s}", run_measures, lab, X, Xb, df_counts, vocab, docs)
            if m:
                per.append({"setting": name, "seed": s, **m[0]})
        run_rows += per
        st = safe(f"stability {name}", stability, runs)
        if not per or not st:
            continue
        stab, repro, A = st
        pdf = pd.DataFrame(per)
        row = {"setting": name, "runs": len(runs)}
        for col in pdf.columns[2:]:
            row[col] = pdf[col].mean()
            row[col + "_std"] = pdf[col].std(ddof=0)
        row.update(stab)
        summary_rows.append(row)
        pd.DataFrame(A, index=sorted(runs), columns=sorted(runs)).to_csv(
            os.path.join(args.out, f"seed_agreement_ari_{name}.csv"))
        ref = stab["reference_seed"]
        tab = safe(f"topic table {name}", topic_table, name, runs[ref], repro[ref], emb, X, vocab,
                   Xb, df_counts, docs, args.out)
        if tab is not None:
            safe(f"era heatmap {name}", plot_era_heatmap, name, tab, args.out)
        print(f"  {name:<22} topics {row['topics']:5.1f}  outliers {row['outlier_share']:5.1%}  "
              f"NPMI {row['npmi_mean']:+.3f}  diversity {row['topic_diversity']:.2f}  "
              f"ARI {stab['pairwise_ari_mean']:.2f}+-{stab['pairwise_ari_std']:.2f}  "
              f"stable docs {stab['docs_in_stable_topics_mean']:5.1%}")

    if run_rows:
        pd.DataFrame(run_rows).to_csv(os.path.join(args.out, "measures_per_run.csv"), index=False)
    if summary_rows:
        summary = pd.DataFrame(summary_rows).set_index("setting")
        summary.to_csv(os.path.join(args.out, "summary_by_setting.csv"))
        safe("summary plot", plot_summary, summary, args.out)
        show = ["topics", "outlier_share", "npmi_mean", "topic_diversity", "nmi_era", "nmi_type",
                "nmi_language", "pairwise_ari_mean", "pairwise_ari_std", "docs_in_stable_topics_mean"]
        with open(os.path.join(args.out, "summary_by_setting.txt"), "w", encoding="utf-8") as f:
            f.write(summary[show].round(3).to_string())
        print("\n" + summary[show].round(3).to_string())

    missing = len(jobs) - sum(os.path.exists(os.path.join(cache, "runs", n, f"seed_{s}.npy"))
                              for s, n in jobs)
    print(f"\nFINISHED {datetime.datetime.now():%Y-%m-%d %H:%M:%S}: "
          f"{len(jobs) - missing}/{len(jobs)} runs, failed output steps: {FAILED_STEPS or 'none'}")
    print(f"Outputs in {args.out}")
    return 1 if (missing or FAILED_STEPS) else 0


if __name__ == "__main__":
    sys.exit(main())
