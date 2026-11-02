"""Topic summary, topic groups and intertopic distance map for one clustering
setting, from the outputs of topic_analysis.py.

    python TopicFigures.py                                (kmeans_umap_k40)
    python TopicFigures.py --setting hdbscan_mcs100_ms10

Topics are renumbered by size (0 = largest).
"""
import argparse
import json
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from umap import UMAP

import topic_analysis as ta

N_CLUSTERS = 3
N_PLOT     = 40
RANDOM_SEED = 42
N_REP_DOCS = 3
REP_CHARS  = 1000

CLUSTER_LABELS = {
    1: "Personal correspondence and ashram life",
    2: "Political struggle, South Africa and public affairs",
    3: "Constructive programme and social reform",
}

CLUSTER_LEGEND = {
    1: "Personal correspondence",
    2: "Political & public work",
    3: "Constructive & social reform",
}

CLUSTER_COLORS = {
    1: "#E58872",
    2: "#8679EB",
    3: "#72F1CE",
}

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
EMB_DIR    = os.path.join(SCRIPT_DIR, "embeddings_qwen3-8B")
RUN_DIR    = os.path.join(SCRIPT_DIR, "results", "embeddings_qwen3-8B")

ap = argparse.ArgumentParser()
ap.add_argument("--setting", default="kmeans_umap_k40")
args = ap.parse_args()
OUT_DIR = os.path.join(SCRIPT_DIR, "results", args.setting)
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 70)
print(f"LOADING SAVED RESULTS FOR {args.setting} (no re-clustering)")
print("=" * 70)

docs, emb, _ = ta.load_inputs(EMB_DIR, smoke=False)
assign = pd.read_csv(os.path.join(RUN_DIR, f"assignments_{args.setting}.csv"))
if not (assign["record_index"].to_numpy() == docs["record_index"].to_numpy()).all():
    raise ValueError("assignment rows do not line up with the embeddings")
labels = assign["topic"].to_numpy()
table = pd.read_csv(os.path.join(RUN_DIR, f"topics_{args.setting}.csv")).set_index("topic")

X = sp.load_npz(os.path.join(RUN_DIR, "cache", "doc_term.npz")).tocsr()
vocab = np.array(json.load(open(os.path.join(RUN_DIR, "cache", "vocab.json"), encoding="utf-8")), dtype=object)
if X.shape[0] != len(docs):
    raise ValueError("document-term matrix does not line up with the documents")

orig_ids, scores = ta.class_tfidf(labels, X)
order, words = ta.top_terms(scores, vocab)
words_of = {int(t): w for t, w in zip(orig_ids, words)}
score_of = {int(t): scores[i, order[i]].tolist() for i, t in enumerate(orig_ids)}

sizes_orig = {int(t): int((labels == t).sum()) for t in orig_ids}
by_size = sorted(sizes_orig, key=lambda t: -sizes_orig[t])
new_id = {t: i for i, t in enumerate(by_size)}
topic_ids = list(range(len(by_size)))
sizes = {new_id[t]: sizes_orig[t] for t in by_size}
top_words = {new_id[t]: words_of[t] for t in by_size}
n_outliers = int((labels < 0).sum())
total_docs = len(labels)
n_clustered = total_docs - n_outliers

topic_emb = np.zeros((len(topic_ids), emb.shape[1]), dtype=np.float32)
rep_docs = {}
for t in by_size:
    mem = np.flatnonzero(labels == t)
    cen = emb[mem].mean(0)
    cen /= np.linalg.norm(cen)
    topic_emb[new_id[t]] = cen
    best = mem[np.argsort(-(emb[mem] @ cen))[:N_REP_DOCS]]
    rep_docs[new_id[t]] = [docs["text"].iloc[j][:REP_CHARS] for j in best]

print(f"Topics            : {len(topic_ids)}")
print(f"Embedding matrix  : {topic_emb.shape}")
print(f"Total documents   : {total_docs:,}")
print(f"Outlier documents : {n_outliers:,} ({n_outliers / total_docs:.1%})")

rows = []
if n_outliers:
    rows.append({"Topic": -1, "Count": n_outliers, "Representation": "[]",
                 "Name": "-1_outliers", "Representative_Docs": "[]"})
for t in by_size:
    i = new_id[t]
    r = table.loc[t]
    rows.append({
        "Topic": i, "Count": sizes[i],
        "Representation": str(top_words[i]),
        "Name": f"{i}_" + "_".join(top_words[i][:4]),
        "Representative_Docs": str(rep_docs[i]),
        "kmeans_topic": t,
        "share": r["share"],
        "stable": bool(r["stable"]),
        "reappears_in_share_of_runs": r["reappears_in_share_of_runs"],
        "npmi": r["npmi"],
        "median_year": r["median_year"],
        "main_type": r["main_type"],
        "main_type_share": r["main_type_share"],
        "example_titles": r["example_titles"],
    })
summary_path = os.path.join(OUT_DIR, "topic_summary_qwen3.csv")
pd.DataFrame(rows).to_csv(summary_path, index=False)
print(f"Topic summary written to: {summary_path}")

print("\n" + "=" * 70)
print("SILHOUETTE SCORES (cosine) FOR WARD LINKAGE")
print("=" * 70)

Xt = normalize(topic_emb)
silhouettes = {}
for k in range(2, 7):
    labels_k = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(Xt)
    silhouettes[k] = silhouette_score(Xt, labels_k, metric="cosine")
    counts = np.bincount(labels_k, minlength=k)
    print(f"  k={k}  silhouette={silhouettes[k]:.3f}  group sizes={sorted(counts.tolist(), reverse=True)}")

raw_labels = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit_predict(Xt)

coords_2d = UMAP(n_components=2, random_state=RANDOM_SEED).fit_transform(topic_emb)
coords = {t: (float(coords_2d[t, 0]), float(coords_2d[t, 1])) for t in topic_ids}

group_docs = {
    g: sum(sizes[t] for t in topic_ids if raw_labels[t] == g)
    for g in range(N_CLUSTERS)
}
order_g = sorted(range(N_CLUSTERS), key=lambda g: -group_docs[g])
remap = {raw: display + 1 for display, raw in enumerate(order_g)}
cluster_of = {t: remap[raw_labels[t]] for t in topic_ids}

lines = []
w = lines.append
w(f"CWMG {args.setting} (Qwen3-Embedding-8B) - macro-cluster derivation")
w("=" * 70)
w(f"Method            : Ward agglomerative clustering on L2-normalised")
w(f"                    topic embeddings (mean of member documents, {len(topic_ids)} x {topic_emb.shape[1]})")
w(f"Clusters (k)      : {N_CLUSTERS}")
w(f"Total documents   : {total_docs:,}")
w(f"Clustered docs    : {n_clustered:,} ({n_clustered / total_docs:.1%})")
w(f"Outlier docs      : {n_outliers:,} ({n_outliers / total_docs:.1%})")
w(f"Topics discovered : {len(topic_ids)}")
w(f"Topics plotted    : {min(N_PLOT, len(topic_ids))} largest")
w("")
w("Silhouette (cosine) by k:")
for k, s in silhouettes.items():
    w(f"  k={k}: {s:.3f}")
w("")

for c in range(1, N_CLUSTERS + 1):
    members = sorted([t for t in topic_ids if cluster_of[t] == c], key=lambda t: -sizes[t])
    n_docs = sum(sizes[t] for t in members)
    plotted = sum(1 for t in members if t < N_PLOT)
    w("-" * 70)
    w(f"CLUSTER {c}: {CLUSTER_LABELS.get(c, '(unlabelled)')}")
    w(f"  {len(members)} topics ({plotted} shown on the map), {n_docs:,} documents "
      f"({n_docs / total_docs:.1%} of all documents)")
    for t in members:
        w(f"    T{t:<3} {sizes[t]:>6}  {', '.join(top_words[t][:8])}")
    w("")

report_path = os.path.join(OUT_DIR, "macro_cluster_report.txt")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"Report written to: {report_path}")

csv_path = os.path.join(OUT_DIR, "macro_clusters.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    fh.write("topic_id,doc_count,cluster,cluster_label,plotted,top_words\n")
    for t in topic_ids:
        c = cluster_of[t]
        words_t = " ".join(top_words[t][:10]).replace(",", " ")
        fh.write(f"{t},{sizes[t]},{c},\"{CLUSTER_LABELS.get(c, '')}\","
                 f"{'yes' if t < N_PLOT else 'no'},\"{words_t}\"\n")
print(f"Per-topic assignments written to: {csv_path}")

print("\nRendering intertopic distance map...")

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "font.family": "monospace",
    "font.monospace": ["Courier New", "DejaVu Sans Mono", "monospace"],
    "font.size": 22,
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
    "legend.fontsize": 17,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.15,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#4a4a4a",
    "axes.linewidth": 0.8,
    "axes.facecolor": "#FFFFFF",
    "figure.facecolor": "#FFFFFF",
    "xtick.color": "#4a4a4a",
    "ytick.color": "#4a4a4a",
})
COLOR_TEXT = "#2D3436"

plotted_ids = sorted(topic_ids, key=lambda t: -sizes[t])[:N_PLOT]
xs = np.array([coords[t][0] for t in plotted_ids])
ys = np.array([coords[t][1] for t in plotted_ids])
counts = np.array([sizes[t] for t in plotted_ids], dtype=float)

span = counts.max() - counts.min() or 1.0
marker_sizes = 420 + (counts - counts.min()) / span * 780

fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect("equal", adjustable="datalim")

ax.grid(which="major", linewidth=0.5, alpha=0.20, color="#DFE6E9", linestyle="--", zorder=0)
ax.minorticks_on()
ax.grid(which="minor", linewidth=0.25, alpha=0.15, color="#DFE6E9", linestyle=":", zorder=0)
ax.set_axisbelow(True)

ax.scatter(
    xs, ys,
    s=marker_sizes,
    c=[CLUSTER_COLORS[cluster_of[t]] for t in plotted_ids],
    alpha=0.82,
    edgecolors="white",
    linewidths=1.2,
    zorder=3,
)

for t, x, y in zip(plotted_ids, xs, ys):
    ax.annotate(str(t), (x, y), fontsize=18, fontweight="bold",
                color=COLOR_TEXT, ha="center", va="center", zorder=4,
                path_effects=[pe.withStroke(linewidth=1.15, foreground="white")])

med_x, med_y = np.median(xs), np.median(ys)
max_dev = max(np.max(np.abs(xs - med_x)), np.max(np.abs(ys - med_y))) * 1.15
ax.set_xlim(med_x - max_dev, med_x + max_dev)
ax.set_ylim(med_y - max_dev, med_y + max_dev)

ax.set_xlabel("UMAP Dimension 1", fontsize=22, color=COLOR_TEXT, labelpad=10)
ax.set_ylabel("UMAP Dimension 2", fontsize=22, color=COLOR_TEXT, labelpad=10)

handles = [
    Line2D([0], [0], marker="o", linestyle="none", markersize=14,
           markerfacecolor=CLUSTER_COLORS[c], markeredgecolor="white",
           label=f"{c}. {CLUSTER_LEGEND.get(c, '')}")
    for c in range(1, N_CLUSTERS + 1)
]
ax.legend(handles=handles, loc="best", fontsize=15, frameon=True,
          framealpha=0.92, edgecolor="#DFE6E9", facecolor="white",
          handletextpad=0.5, borderpad=0.6, labelspacing=0.5)

for ext in ("pdf", "svg", "png"):
    out = os.path.join(OUT_DIR, f"intertopic_distance_map.{ext}")
    fig.savefig(out, format=ext)
    print(f"  -> {out}")
plt.close(fig)

print("\nDone.")
