from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

FONT_DIR = Path('C:/Windows/Fonts')
for f in ["cour.ttf", "courbd.ttf", "couri.ttf", "courbi.ttf"]:
    fm.fontManager.addfont(str(FONT_DIR / f))
font_name = fm.FontProperties(fname=str(FONT_DIR / "cour.ttf")).get_name()

plt.rcParams.update({
    "font.family": font_name,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 22,
    "axes.titlesize": 24,
    "axes.titleweight": "bold",
    "axes.labelsize": 24,
    "xtick.labelsize": 22,
    "ytick.labelsize": 20,
    "legend.fontsize": 20,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

CUSTOM_CMAP = LinearSegmentedColormap.from_list(
    'academic_slate',
    ['#FFFFFF', '#A8DADC', '#457B9D', '#264653']
)


RESULTS = Path(__file__).resolve().parent / 'results'
SEEDS = [42, 43, 44]


def summed_confusion(task):
    preds = pd.concat([pd.read_csv(RESULTS / task / f'seed_{s}' / 'test_predictions.csv')
                       for s in SEEDS])
    n = int(max(preds['true_idx'].max(), preds['pred_idx'].max())) + 1
    cm = np.zeros((n, n), dtype=int)
    np.add.at(cm, (preds['true_idx'].to_numpy(), preds['pred_idx'].to_numpy()), 1)
    names = preds.drop_duplicates('true_idx').set_index('true_idx')['true_label'].sort_index()
    return cm, list(names)


era_data, era_names = summed_confusion('era')
era_labels = [name.split(' ')[0] for name in era_names]

trans_data, trans_names = summed_confusion('translation')
trans_labels = ['English' if name == 'Original English' else name for name in trans_names]

doc_data, doc_labels = summed_confusion('type')


global_min = 0
global_max = max(era_data.max(), trans_data.max(), doc_data.max()) 

fig, axes = plt.subplots(
    1, 4, 
    figsize=(20, 7), 
    gridspec_kw={'width_ratios': [1.1, 1.1, 1.1, 0.05], 'wspace': 0.5}
)

heatmap_kwargs = {
    "cmap": CUSTOM_CMAP,
    "vmin": global_min,
    "vmax": global_max,
    "cbar": False,         
    "square": True,        
    "annot": True,         
    "fmt": "d",            
    "linewidths": 0.5,     
    "linecolor": "#E0E0E0",
    "annot_kws": {"size": 18, "weight": "bold"}
}

sns.heatmap(era_data, ax=axes[0], xticklabels=era_labels, yticklabels=era_labels, **heatmap_kwargs)
sns.heatmap(trans_data, ax=axes[1], xticklabels=trans_labels, yticklabels=trans_labels, **heatmap_kwargs)
sns.heatmap(doc_data, ax=axes[2], xticklabels=doc_labels, yticklabels=doc_labels, **heatmap_kwargs)

for ax in axes[:3]:
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor',  fontsize=16)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0,  fontsize=16)
    
    for _, spine in ax.spines.items():
        spine.set_visible(True)
        spine.set_color('#B0B0B0')
        spine.set_linewidth(1.5)

norm = mpl.colors.Normalize(vmin=global_min, vmax=global_max)
cbar = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm, cmap=CUSTOM_CMAP), 
    cax=axes[3]
)
cbar.outline.set_visible(False)
cbar.ax.tick_params(labelsize=18, color='#B0B0B0')

fig.supxlabel('Predicted Label', fontsize=26, fontweight='bold', y=0.02)

fig.supylabel('True Label', fontsize=26, fontweight='bold', x=0.03)

fig.subplots_adjust(
    top=0.99,    
    bottom=0.1
    , 
    left=0.12,
    right=0.99
)

pos = axes[2].get_position()
gap = 0.01
cbar_width = 0.012

axes[3].set_position([
    pos.x1 + gap,
    pos.y0,
    cbar_width,
    pos.height
])
plt.savefig(RESULTS / 'combined_confusion_matrices.pdf', format='pdf',
            bbox_inches='tight', pad_inches=0.02)
print(f"\nSaved {RESULTS / 'combined_confusion_matrices.pdf'}")
for name, cm, labels in [('era', era_data, era_labels), ('translation', trans_data, trans_labels),
                         ('type', doc_data, doc_labels)]:
    print(name, labels)
    print(cm)