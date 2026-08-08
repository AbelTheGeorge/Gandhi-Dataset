"""
GenerateViz.py
──────────────
Loads a saved BERTopic model from results/bertopic_model and generates:

  Interactive HTML (Plotly):
    1. bertopic_barchart_e5.html        – Top words per topic (bar chart)
    2. bertopic_intertopic_map_e5.html  – Intertopic distance map

  Publication-quality static figures (Matplotlib → PDF + SVG):
    3. bertopic_barchart_e5.pdf / .svg            – Top-9 topics bar chart
    4. bertopic_intertopic_map_e5.pdf / .svg       – Intertopic distance map

  Design Principles:
    • Square 10×10 canvas, vector output (PDF/SVG)
    • Courier New monospace font at 22pt base size
    • Subtle grids (lw ≤ 0.5, alpha ≤ 0.2)
    • Custom purple/teal hex palette (no default colors)
    • Large markers (s=90+), thick lines (lw=2.6)
    • Explicit z-order layering
    • External frameless legend
"""

import os
import json
import numpy as np
# pyrefly: ignore [missing-import]
from bertopic import BERTopic
# pyrefly: ignore [missing-import]
from umap import UMAP
import math
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch

# ── Paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
MODEL_DIR = os.path.join(RESULTS_DIR, "bertopic_model")

print("=" * 70)
print("LOADING SAVED BERTopic MODEL")
print("=" * 70)

topic_model = BERTopic.load(MODEL_DIR)
topic_info = topic_model.get_topic_info()

# Exclude the noise topic (-1)
valid_topics = topic_info[topic_info['Topic'] != -1].sort_values('Count', ascending=False)
total_topics = len(valid_topics)

print(f"Loaded model with {total_topics} topics (excluding noise).")

# =====================================================================
# 1. EXTRACT BARCHART DATA
# =====================================================================
print("\nExtracting topic word data for barchart...")

barchart_data = []
for _, row in valid_topics.iterrows():
    topic_id = row['Topic']
    words_scores = topic_model.get_topic(topic_id)
    if words_scores:
        barchart_data.append({
            'topic_id': int(topic_id),
            'name': row['Name'],
            'count': int(row['Count']),
            'words': [w for w, s in words_scores],
            'scores': [float(s) for w, s in words_scores]
        })

# =====================================================================
# 2. EXTRACT INTERTOPIC DISTANCE MAP DATA
# =====================================================================
print("Extracting topic embedding data for intertopic map...")

# Get topic embeddings and reduce to 2D
topic_embeddings = topic_model.topic_embeddings_

# BERTopic stores embeddings indexed as: row 0 = topic -1 (outlier),
# row 1 = topic 0, row 2 = topic 1, etc. We need to map correctly.
all_topic_ids = sorted(topic_model.get_topic_info()['Topic'].tolist())

# Reduce to 2D for visualization
umap_2d = UMAP(n_components=2, random_state=42)
embeddings_2d = umap_2d.fit_transform(topic_embeddings)

# Build a lookup: topic_id -> index in the embeddings array
topic_id_to_idx = {tid: i for i, tid in enumerate(all_topic_ids)}

intertopic_data = []
for _, row in valid_topics.iterrows():
    topic_id = row['Topic']
    idx = topic_id_to_idx[topic_id]
    top_words = topic_model.get_topic(topic_id)
    label_words = ", ".join([w for w, _ in top_words[:8]]) if top_words else ""
    intertopic_data.append({
        'topic_id': int(topic_id),
        'name': row['Name'],
        'count': int(row['Count']),
        'x': float(embeddings_2d[idx, 0]),
        'y': float(embeddings_2d[idx, 1]),
        'top_words': label_words
    })

# =====================================================================
# 3. GENERATE INTERACTIVE BARCHART HTML
# =====================================================================
print("\nGenerating interactive barchart HTML...")

barchart_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BERTopic — Top Words per Topic</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #0f0f1a;
    color: #e0e0e0;
    min-height: 100vh;
  }}
  .controls {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 20px;
    border-bottom: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 4px 30px rgba(0,0,0,0.4);
  }}
  .controls h1 {{
    font-size: 1.2rem;
    font-weight: 600;
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    white-space: nowrap;
  }}
  .slider-group {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .slider-group label {{
    font-size: 0.9rem;
    color: #94a3b8;
    white-space: nowrap;
  }}
  input[type="range"] {{
    -webkit-appearance: none;
    width: 260px;
    height: 6px;
    border-radius: 3px;
    background: #334155;
    outline: none;
  }}
  input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #a78bfa;
    cursor: pointer;
    box-shadow: 0 0 8px rgba(167,139,250,0.5);
  }}
  .badge {{
    background: rgba(167,139,250,0.15);
    color: #a78bfa;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.95rem;
    font-weight: 600;
    min-width: 36px;
    text-align: center;
  }}
  #chartGrid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 12px;
    padding: 20px;
  }}
  .chart-cell {{
    background: #1a1a2e;
    border-radius: 10px;
    border: 1px solid rgba(255,255,255,0.06);
    overflow: hidden;
  }}
</style>
</head>
<body>

<div class="controls">
  <h1>BERTopic — Top Words per Topic</h1>
  <div class="slider-group">
    <label for="topicSlider">Topics to display:</label>
    <input type="range" id="topicSlider" min="1" max="{total_topics}" value="{min(16, total_topics)}" oninput="updateChart()">
    <span class="badge" id="sliderValue">{min(16, total_topics)}</span>
  </div>
</div>

<div id="chartGrid"></div>

<script>
const allTopics = {json.dumps(barchart_data)};

function updateChart() {{
  const n = parseInt(document.getElementById('topicSlider').value);
  document.getElementById('sliderValue').textContent = n;

  const topics = allTopics.slice(0, n);
  const nWords = 10;  // top words per topic
  const chartHeight = 340; // fixed height per chart cell

  const grid = document.getElementById('chartGrid');
  grid.innerHTML = '';

  topics.forEach((t, i) => {{
    const cellId = 'chart-cell-' + i;
    const cell = document.createElement('div');
    cell.className = 'chart-cell';
    cell.id = cellId;
    cell.style.height = chartHeight + 'px';
    grid.appendChild(cell);

    const words = t.words.slice(0, nWords).reverse();
    const scores = t.scores.slice(0, nWords).reverse();
    const maxScore = Math.max(...scores);

    const trace = {{
      type: 'bar',
      orientation: 'h',
      y: words,
      x: scores,
      marker: {{
        color: scores.map(s => `rgba(${{100 + i * 7}}, ${{140 + (i * 13 % 100)}}, 255, ${{0.5 + s / maxScore * 0.5}})`),
        line: {{ color: 'rgba(255,255,255,0.1)', width: 0.5 }}
      }},
      hovertemplate: '<b>%{{y}}</b><br>Score: %{{x:.4f}}<extra></extra>',
      showlegend: false
    }};

    const layout = {{
      title: {{
        text: `<b>Topic ${{t.topic_id}}</b> (${{t.count.toLocaleString()}} docs): ${{t.name}}`,
        font: {{ size: 13, color: '#a78bfa', family: 'Segoe UI, system-ui, sans-serif' }},
        x: 0.5,
        xanchor: 'center',
        y: 0.97,
        yanchor: 'top',
        pad: {{ b: 4 }}
      }},
      height: chartHeight,
      paper_bgcolor: 'rgba(0,0,0,0)',
      plot_bgcolor: '#1a1a2e',
      font: {{ color: '#e0e0e0', family: 'Segoe UI, system-ui, sans-serif' }},
      margin: {{ l: 110, r: 16, t: 36, b: 20 }},
      yaxis: {{
        tickfont: {{ size: 13 }},
        gridcolor: 'rgba(255,255,255,0.03)',
        automargin: true
      }},
      xaxis: {{
        gridcolor: 'rgba(255,255,255,0.05)',
        zeroline: false,
        showticklabels: false
      }},
      bargap: 0.18
    }};

    Plotly.newPlot(cellId, [trace], layout, {{
      responsive: true,
      displayModeBar: false
    }});
  }});
}}

updateChart();
</script>
</body>
</html>"""

barchart_path = os.path.join(RESULTS_DIR, "bertopic_barchart_e5.html")
with open(barchart_path, 'w', encoding='utf-8') as f:
    f.write(barchart_html)
print(f"  → {barchart_path}")

# =====================================================================
# 4. GENERATE INTERACTIVE INTERTOPIC DISTANCE MAP HTML
# =====================================================================
print("Generating interactive intertopic distance map HTML...")

intertopic_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>BERTopic — Intertopic Distance Map</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #ffffff;
    color: #1a1a1a;
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }}
  .controls {{
    position: sticky;
    top: 0;
    z-index: 100;
    background: #f8f9fa;
    padding: 18px 32px;
    display: flex;
    align-items: center;
    gap: 20px;
    border-bottom: 1px solid #dee2e6;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }}
  .controls h1 {{
    font-size: 1.2rem;
    font-weight: 600;
    color: #1a1a1a;
    white-space: nowrap;
  }}
  .slider-group {{
    display: flex;
    align-items: center;
    gap: 12px;
  }}
  .slider-group label {{
    font-size: 0.9rem;
    color: #495057;
    white-space: nowrap;
  }}
  input[type="range"] {{
    -webkit-appearance: none;
    width: 260px;
    height: 6px;
    border-radius: 3px;
    background: #ced4da;
    outline: none;
  }}
  input[type="range"]::-webkit-slider-thumb {{
    -webkit-appearance: none;
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: #495057;
    cursor: pointer;
    box-shadow: 0 0 4px rgba(0,0,0,0.2);
  }}
  .badge {{
    background: #e9ecef;
    color: #343a40;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.95rem;
    font-weight: 600;
    min-width: 36px;
    text-align: center;
  }}
  #chart {{
    flex: 1;
    padding: 20px;
  }}
</style>
</head>
<body>

<div class="controls">
  <h1>BERTopic — Intertopic Distance Map</h1>
  <div class="slider-group">
    <label for="topicSlider">Topics to display:</label>
    <input type="range" id="topicSlider" min="1" max="{total_topics}" value="{min(40, total_topics)}" oninput="updateMap()">
    <span class="badge" id="sliderValue">{min(40, total_topics)}</span>
  </div>
  <div class="slider-group">
    <label for="scaleSlider">Circle scale:</label>
    <input type="range" id="scaleSlider" min="0.2" max="3" step="0.1" value="2" oninput="updateMap()">
    <span class="badge" id="scaleValue">2.0</span>
  </div>
</div>

<div id="chart"></div>

<script>
const allTopics = {json.dumps(intertopic_data)};

// Pre-compute max count for sizing
const maxCount = Math.max(...allTopics.map(t => t.count));

function updateMap() {{
  const n = parseInt(document.getElementById('topicSlider').value);
  document.getElementById('sliderValue').textContent = n;
  const scale = parseFloat(document.getElementById('scaleSlider').value);
  document.getElementById('scaleValue').textContent = scale.toFixed(1);

  const topics = allTopics.slice(0, n);

  // Scale bubble sizes: smallest=20, largest=80
  const counts = topics.map(t => t.count);
  const minC = Math.min(...counts);
  const maxC = Math.max(...counts);
  const range = maxC - minC || 1;

  const trace = {{
    type: 'scatter',
    mode: 'markers+text',
    x: topics.map(t => t.x),
    y: topics.map(t => t.y),
    text: topics.map(t => t.topic_id),
    textposition: 'middle center',
    textfont: {{ size: 22, color: '#1a1a1a', family: 'Segoe UI, system-ui, sans-serif' }},
    marker: {{
      size: topics.map(t => (20 + ((t.count - minC) / range) * 60) * scale),
      color: topics.map((t, i) => {{
        const hue = (i * 360 / Math.max(n, 1)) % 360;
        return `hsla(${{hue}}, 55%, 55%, 0.75)`;
      }}),
      line: {{ color: 'rgba(0,0,0,0.25)', width: 1.5 }},
      sizemode: 'diameter'
    }},
    hovertemplate: topics.map(t =>
      `<b>Topic ${{t.topic_id}}</b><br>` +
      `${{t.count.toLocaleString()}} documents<br>` +
      `<i>${{t.top_words}}</i><extra></extra>`
    ),
    showlegend: false
  }};

  const layout = {{
    height: Math.max(600, window.innerHeight - 100),
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    font: {{ color: '#1a1a1a', family: 'Segoe UI, system-ui, sans-serif' }},
    margin: {{ l: 60, r: 40, t: 20, b: 60 }},
    xaxis: {{
      title: '',
      tickfont: {{ size: 22 }},
      gridcolor: 'rgba(0,0,0,0.08)',
      zeroline: false,
      showgrid: true,
      linecolor: '#dee2e6',
      linewidth: 1
    }},
    yaxis: {{
      title: '',
      tickfont: {{ size: 22 }},
      gridcolor: 'rgba(0,0,0,0.08)',
      zeroline: false,
      showgrid: true,
      linecolor: '#dee2e6',
      linewidth: 1
    }},
    hovermode: 'closest'
  }};

  Plotly.react('chart', [trace], layout, {{
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
    displaylogo: false,
    toImageButtonOptions: {{
      format: 'svg',
      filename: 'intertopic_distance_map',
      scale: 1
    }}
  }});
}}

updateMap();
window.addEventListener('resize', updateMap);
</script>
</body>
</html>"""

intertopic_path = os.path.join(RESULTS_DIR, "bertopic_intertopic_map_e5.html")
with open(intertopic_path, 'w', encoding='utf-8') as f:
    f.write(intertopic_html)
print(f"  → {intertopic_path}")

# =====================================================================
# PUBLICATION-QUALITY FIGURE CONFIGURATION (Matplotlib)
# =====================================================================
# All rcParams are set once here so every figure produced is uniform.

# ── Custom Hex Colour Palette ─────────────────────────────────────────
# Avoids default matplotlib colours and red/green dualism.
TOPIC_COLORS = [
    '#6C5CE7',  # muted iris purple
    '#00B894',  # soft teal
    '#A29BFE',  # lavender
    '#55EFC4',  # mint
    '#74B9FF',  # sky blue
    '#FDCB6E',  # warm gold
    '#E17055',  # burnt sienna
    '#00CEC9',  # turquoise
    '#B8E994',  # pistachio
    '#D980FA',  # heliotrope
    '#7ED6DF',  # powder blue
    '#FDA7DF',  # pink lavender
]
COLOR_NEUTRAL = '#B2BEC3'   # warm gray – secondary / regression lines
COLOR_TEXT    = '#2D3436'   # near-black – axis labels, titles

# ── Global rcParams ───────────────────────────────────────────────────
plt.rcParams.update({
    # Typography – Courier New at a large base size so the text stays
    # readable when the figure is shrunk to two-column journal width.
    'font.family':       'monospace',
    'font.monospace':    ['Courier New', 'DejaVu Sans Mono', 'monospace'],
    'font.size':         22,
    'axes.titlesize':    22,
    'axes.labelsize':    22,
    'xtick.labelsize':   18,
    'ytick.labelsize':   18,
    'legend.fontsize':   16,

    # Canvas
    'figure.figsize':    (10, 10),
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'savefig.pad_inches': 0.15,

    # Grids – subtle and unobtrusive
    'axes.grid':         True,
    'grid.linewidth':    0.5,
    'grid.alpha':        0.20,
    'grid.color':        '#DFE6E9',
    'grid.linestyle':    '--',

    # Axes / spines
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'axes.edgecolor':     '#4a4a4a',
    'axes.linewidth':     0.8,
    'axes.facecolor':     '#FFFFFF',

    # Ticks
    'xtick.direction':   'out',
    'ytick.direction':   'out',
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.color':       '#4a4a4a',
    'ytick.color':       '#4a4a4a',

    # Figure background
    'figure.facecolor':  '#FFFFFF',
})


# =====================================================================
# 5. GENERATE PUBLICATION-QUALITY BARCHART (Matplotlib)
# =====================================================================
print("\nGenerating publication-quality barchart (Matplotlib)...")


def generate_publication_barchart(data, top_n=9, save_dir=RESULTS_DIR):
    """
    Render a 3×3 grid of horizontal bar charts showing the top words
    for the `top_n` largest topics.  Saves PDF + SVG.
    """
    topics = data[:top_n]
    n_words = 10  # words per subplot

    # Determine grid dimensions (square-ish)
    ncols = 3
    nrows = math.ceil(top_n / ncols)

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(
        nrows, ncols,
        hspace=0.55,
        wspace=0.50,
        left=0.08, right=0.97,
        top=0.93, bottom=0.04
    )

    fig.suptitle(
        'BERTopic — Top Words per Topic',
        fontsize=24, fontweight='bold', color=COLOR_TEXT,
        y=0.98
    )

    for idx, topic in enumerate(topics):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])

        words  = topic['words'][:n_words][::-1]   # reverse so highest is on top
        scores = topic['scores'][:n_words][::-1]
        color  = TOPIC_COLORS[idx % len(TOPIC_COLORS)]

        bars = ax.barh(
            range(len(words)), scores,
            color=color, edgecolor='white', linewidth=0.4,
            height=0.72, zorder=3
        )

        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words, fontsize=12, color=COLOR_TEXT)
        ax.set_title(
            f'Topic {topic["topic_id"]}  ({topic["count"]:,} docs)',
            fontsize=14, fontweight='bold', color=COLOR_TEXT,
            pad=6
        )

        # Subtle vertical grid only
        ax.xaxis.grid(True, linewidth=0.4, alpha=0.18, zorder=0)
        ax.yaxis.grid(False)
        ax.set_axisbelow(True)

        # Remove x-axis tick labels (scores are self-evident from bar length)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', length=0)

        # Light spines
        for spine in ('top', 'right', 'bottom'):
            ax.spines[spine].set_visible(False)
        ax.spines['left'].set_color('#CCCCCC')
        ax.spines['left'].set_linewidth(0.5)

    # Hide any unused subplot cells
    for idx in range(len(topics), nrows * ncols):
        row, col = divmod(idx, ncols)
        ax = fig.add_subplot(gs[row, col])
        ax.set_visible(False)

    # Save as vector graphics
    for ext in ('pdf', 'svg'):
        out_path = os.path.join(save_dir, f'bertopic_barchart_e5.{ext}')
        fig.savefig(out_path, format=ext)
        print(f"  → {out_path}")

    plt.close(fig)


generate_publication_barchart(barchart_data, top_n=9, save_dir=RESULTS_DIR)


# =====================================================================
# 6. GENERATE PUBLICATION-QUALITY INTERTOPIC MAP (Matplotlib)
# =====================================================================
print("\nGenerating publication-quality intertopic map (Matplotlib)...")


def generate_publication_intertopic_map(data, save_dir=RESULTS_DIR):
    """
    Scatter plot of topic embeddings (UMAP 2-D) with bubble size
    proportional to document count.  Saves PDF + SVG.
    """
    xs = np.array([d['x'] for d in data])
    ys = np.array([d['y'] for d in data])
    counts = np.array([d['count'] for d in data])
    topic_ids = [d['topic_id'] for d in data]

    # ── Tight, median-centred bounding box ────────────────────────────
    med_x, med_y = np.median(xs), np.median(ys)
    dev_x = np.max(np.abs(xs - med_x))
    dev_y = np.max(np.abs(ys - med_y))
    max_dev = max(dev_x, dev_y) * 1.15          # 15 % breathing room
    xlim = (med_x - max_dev, med_x + max_dev)
    ylim = (med_y - max_dev, med_y + max_dev)

    # ── Marker sizes: scale counts to [300, 1200] ─────────────────────
    min_c, max_c = counts.min(), counts.max()
    rng = max_c - min_c if max_c != min_c else 1
    sizes = 300 + (counts - min_c) / rng * 900

    # ── Colours (cycle through palette) ───────────────────────────────
    colors = [TOPIC_COLORS[i % len(TOPIC_COLORS)] for i in range(len(data))]

    # ── Figure ────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_aspect('equal', adjustable='datalim')

    # Major + minor grids
    ax.grid(which='major', linewidth=0.5,  alpha=0.20, color='#DFE6E9',
            linestyle='--', zorder=0)
    ax.minorticks_on()
    ax.grid(which='minor', linewidth=0.25, alpha=0.15, color='#DFE6E9',
            linestyle=':', zorder=0)
    ax.set_axisbelow(True)

    # Scatter – primary data layer
    scatter = ax.scatter(
        xs, ys,
        s=sizes,
        c=colors,
        alpha=0.82,
        edgecolors='white',
        linewidths=1.2,
        zorder=3
    )

    # Annotate each bubble with its topic ID
    for i, tid in enumerate(topic_ids):
        ax.annotate(
            str(tid),
            (xs[i], ys[i]),
            fontsize=18, fontweight='bold',
            color=COLOR_TEXT,
            ha='center', va='center',
            zorder=4
        )

    # ── Axes limits & labels ──────────────────────────────────────────
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel('UMAP Dimension 1', fontsize=22, color=COLOR_TEXT, labelpad=10)
    ax.set_ylabel('UMAP Dimension 2', fontsize=22, color=COLOR_TEXT, labelpad=10)



    # ── Save vector outputs ───────────────────────────────────────────
    for ext in ('pdf', 'svg'):
        out_path = os.path.join(save_dir, f'bertopic_intertopic_map_e5.{ext}')
        fig.savefig(out_path, format=ext)
        print(f"  → {out_path}")

    plt.close(fig)


generate_publication_intertopic_map(intertopic_data[:40], save_dir=RESULTS_DIR)

print(f"\nDone!")
print(f"  Interactive HTML → open in browser")
print(f"  Publication PDFs/SVGs → {RESULTS_DIR}")
