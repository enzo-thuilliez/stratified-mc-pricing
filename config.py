import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG_SEED = 42
np.random.seed(RNG_SEED)
torch.manual_seed(RNG_SEED)

plt.rcParams.update({
    "font.family":        "serif",
    "font.size":          10,
    "axes.linewidth":     0.8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "xtick.direction":    "in",
    "ytick.direction":    "in",
    "xtick.major.size":   4,
    "ytick.major.size":   4,
    "legend.frameon":     False,
    "figure.dpi":         150,
    "savefig.bbox":       "tight",
    "savefig.dpi":        200,
})

GRAY_SHADES     = ["0.00", "0.25", "0.45", "0.60", "0.72", "0.82", "0.90", "0.50"]
CLUSTER_MARKERS = ["o", "s", "^", "D", "v", "P", "*", "X"]

LINE_STYLES = [
    dict(ls="-",           lw=1.5, marker="o", ms=4.5, mfc="white"),
    dict(ls="--",          lw=1.3, marker="s", ms=4.5, mfc="white"),
    dict(ls="-.",          lw=1.3, marker="^", ms=4.5, mfc="white"),
    dict(ls=":",           lw=1.4, marker="D", ms=4.5, mfc="white"),
    dict(ls=(0,(3,1,1,1)), lw=1.4, marker="*", ms=6.0, mfc="white"),
    dict(ls=(0,(5,2)),     lw=1.2, marker="v", ms=4.5, mfc="white"),
    dict(ls=(0,(1,1)),     lw=1.2, marker="P", ms=5.0, mfc="white"),
]

OUT_DIR = "figures"
