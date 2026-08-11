
import json
import os
import re
import struct

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize

N_CLUSTERS = 3
N_PLOT     = 40
RANDOM_SEED = 42

CLUSTER_LABELS = {
    1: "Religion, communal relations and social reform",
    2: "Political work and the constructive programme",
    3: "Personal correspondence",
}

CLUSTER_LEGEND = {
    1: "Religion & social reform",
    2: "Political & public work",
    3: "Personal correspondence",
}

CLUSTER_COLORS = {
    1: "#72F1CE",
    2: "#8679EB",
    3: "#E58872",
}

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MODEL_DIR   = os.path.join(RESULTS_DIR, "bertopic_model")
COORDS_HTML = os.path.join(RESULTS_DIR, "Run", "bertopic_intertopic_map_e5.html")


def load_safetensors(path, key):
    with open(path, "rb") as fh:
        header_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(header_len))
        payload = fh.read()
    meta = header[key]
    if meta["dtype"] != "F32":
        raise ValueError(f"expected F32 tensor, found {meta['dtype']}")
    start, end = meta["data_offsets"]
    return np.frombuffer(payload[start:end], dtype=np.float32).reshape(meta["shape"]).copy()


def load_topic_metadata(path):
    with open(path, encoding="utf-8") as fh:
        blob = json.load(fh)
    sizes = {int(k): int(v) for k, v in blob["topic_sizes"].items()}
    words = {
        int(k): [w for w, _ in v if w]
        for k, v in blob["topic_representations"].items()
    }
    return sizes, words


def load_coordinates(path):
    with open(path, encoding="utf-8") as fh:
        html = fh.read()
    match = re.search(r"const allTopics = (\[.*?\]);", html, re.S)
    if match is None:
        raise ValueError(f"no coordinate payload found in {path}")
    return {d["topic_id"]: (float(d["x"]), float(d["y"])) for d in json.loads(match.group(1))}


print("=" * 70)
print("LOADING SAVED BERTopic ARTEFACTS (no re-fitting)")
print("=" * 70)

embeddings = load_safetensors(os.path.join(MODEL_DIR, "topic_embeddings.safetensors"),
                              "topic_embeddings")
sizes, top_words = load_topic_metadata(os.path.join(MODEL_DIR, "topics.json"))
coords = load_coordinates(COORDS_HTML)

topic_ids = sorted(t for t in sizes if t >= 0)
n_topics = len(topic_ids)
if embeddings.shape[0] != n_topics + 1:
    raise ValueError(f"embedding rows ({embeddings.shape[0]}) != topics + outlier ({n_topics + 1})")

X = normalize(embeddings[1:])
total_chunks = sum(sizes.values())
n_outliers = sizes.get(-1, 0)
n_clustered = total_chunks - n_outliers

print(f"Topics            : {n_topics} (excluding outlier topic -1)")
print(f"Embedding matrix  : {embeddings.shape}")
print(f"Total chunks      : {total_chunks:,}")
print(f"Clustered chunks  : {n_clustered:,} ({n_clustered / total_chunks:.1%})")
print(f"Outlier chunks    : {n_outliers:,} ({n_outliers / total_chunks:.1%})")

print("\n" + "=" * 70)
print("SILHOUETTE SCORES (cosine) FOR WARD LINKAGE")
print("=" * 70)

silhouettes = {}
for k in range(2, 7):
    labels_k = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X)
    silhouettes[k] = silhouette_score(X, labels_k, metric="cosine")
    counts = np.bincount(labels_k, minlength=k)
    print(f"  k={k}  silhouette={silhouettes[k]:.3f}  group sizes={sorted(counts.tolist(), reverse=True)}")

raw_labels = AgglomerativeClustering(n_clusters=N_CLUSTERS, linkage="ward").fit_predict(X)

mean_y = {
    g: np.mean([coords[t][1] for t in topic_ids if raw_labels[t] == g])
    for g in range(N_CLUSTERS)
}
order = sorted(range(N_CLUSTERS), key=lambda g: -mean_y[g])
remap = {raw: display + 1 for display, raw in enumerate(order)}
cluster_of = {t: remap[raw_labels[t]] for t in topic_ids}

lines = []
w = lines.append
w("CWMG BERTopic - macro-cluster derivation")
w("=" * 70)
w(f"Method            : Ward agglomerative clustering on L2-normalised")
w(f"                    BERTopic topic embeddings ({n_topics} x {embeddings.shape[1]})")
w(f"Clusters (k)      : {N_CLUSTERS}")
w(f"Total chunks      : {total_chunks:,}")
w(f"Clustered chunks  : {n_clustered:,} ({n_clustered / total_chunks:.1%})")
w(f"Outlier chunks    : {n_outliers:,} ({n_outliers / total_chunks:.1%})")
w(f"Topics discovered : {n_topics}")
w(f"Topics plotted    : {N_PLOT} largest")
w("")
w("Silhouette (cosine) by k:")
for k, s in silhouettes.items():
    w(f"  k={k}: {s:.3f}")
w("")

top13 = [t for t in topic_ids if t <= 12]
w(f"Table 5 topics (0-12): {sum(sizes[t] for t in top13):,} chunks "
  f"({sum(sizes[t] for t in top13) / total_chunks:.1%} of all chunks, "
  f"{sum(sizes[t] for t in top13) / n_clustered:.1%} of clustered chunks)")
w("")

for c in range(1, N_CLUSTERS + 1):
    members = sorted([t for t in topic_ids if cluster_of[t] == c], key=lambda t: -sizes[t])
    chunks = sum(sizes[t] for t in members)
    plotted = sum(1 for t in members if t < N_PLOT)
    w("-" * 70)
    w(f"CLUSTER {c}: {CLUSTER_LABELS.get(c, '(unlabelled)')}")
    w(f"  {len(members)} topics ({plotted} shown on the map), {chunks:,} chunks "
      f"({chunks / total_chunks:.1%} of all chunks, {chunks / n_clustered:.1%} of clustered)")
    for t in members:
        w(f"    T{t:<3} {sizes[t]:>6}  {', '.join(top_words[t][:8])}")
    w("")

report_path = os.path.join(RESULTS_DIR, "macro_cluster_report.txt")
with open(report_path, "w", encoding="utf-8") as fh:
    fh.write("\n".join(lines))
print("\n" + "\n".join(lines))
print(f"Report written to: {report_path}")

csv_path = os.path.join(RESULTS_DIR, "macro_clusters.csv")
with open(csv_path, "w", encoding="utf-8", newline="") as fh:
    fh.write("topic_id,chunk_count,cluster,cluster_label,plotted,top_words\n")
    for t in topic_ids:
        c = cluster_of[t]
        words = " ".join(top_words[t][:10]).replace(",", " ")
        fh.write(f"{t},{sizes[t]},{c},\"{CLUSTER_LABELS.get(c, '')}\","
                 f"{'yes' if t < N_PLOT else 'no'},\"{words}\"\n")
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
ax.legend(handles=handles, loc="upper right", fontsize=15, frameon=True,
          framealpha=0.92, edgecolor="#DFE6E9", facecolor="white",
          handletextpad=0.5, borderpad=0.6, labelspacing=0.5)

for ext in ("pdf", "svg"):
    out = os.path.join(RESULTS_DIR, f"intertopic_distance_map.{ext}")
    fig.savefig(out, format=ext)
    print(f"  -> {out}")
plt.close(fig)

print("\nDone. Copy results/intertopic_distance_map.pdf into the paper's Figures/ folder.")
