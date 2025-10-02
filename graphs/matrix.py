# Re-render with the exact same values/story as v3,
# but use strong blue (2-axis RS-KD) on exactly THREE rows — exclude "love".

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
from PIL import Image

y_labels = ["We", "love", "when", "small", "models", "perform", "like", "large"]
x_labels = ["are", "will", "to", "well", "like", "models"]
R, C = len(y_labels), len(x_labels)

# --- Recreate the same logits as in v3 ---
np.random.seed(1)
logits = np.full((R, C), -4.5, dtype=float)

def set_row(row, top_a, top_b, hi=(1.8, 1.4), lows=(-2.0, -3.0, -4.0, -5.0)):
    r = y_labels.index(row)
    cols = {lbl:i for i,lbl in enumerate(x_labels)}
    other_cols = [c for c in x_labels if c not in (top_a, top_b)]
    for val, col in zip(lows, other_cols):
        logits[r, cols[col]] = val
    logits[r, cols[top_a]] = hi[0]
    logits[r, cols[top_b]] = hi[1]

set_row("We",      "are",   "will",  hi=(2.0, 1.7), lows=(-4.5, -5.5, -2.0, -3.0))
set_row("love",    "to",    "models",hi=(1.4, 1.0), lows=(-1.6, -2.0, -2.3, -3.0))
set_row("when",    "models","well",  hi=(1.3, 1.1), lows=(-1.7, -2.1, -3.3, -3.8))
set_row("small",   "models","are",   hi=(0.6, 0.4), lows=(-1.5, -2.2, -3.2, -4.5))
set_row("models",  "are",   "will",  hi=(1.6, 1.2), lows=(-3.8, -4.1, -4.5, -5.0))
set_row("perform", "well",  "like",  hi=(2.2, 1.0), lows=(-2.2, -3.0, -4.2, -4.5))
set_row("like",    "to",    "models",hi=(0.9, 0.7), lows=(-1.5, -2.1, -3.0, -3.6))
set_row("large",   "models","are",   hi=(1.6, 1.2), lows=(-2.8, -3.3, -4.0, -4.6))

# --- Selection maps (same) ---
select_map = {
    "We": {"are","will"},
    "love": {"to","models"},
    "when": {"models","well"},
    "small": {"models","are"},
    "models": {"are","will"},
    "perform": {"well","like"},
    "like": {"to","models"},
    "large": {"models","are"},
}

# EXACTLY THREE rows in strong blue; exclude "love"
two_axis_rows = {"when", "small", "like"}

# --- Colors ---
strong_blue = "#1f77b4"
light_blue  = "#a6c8e5"
reg_gray_bg = "#f0f0f0"
num_dim     = "#555555"

# --- Plot ---
fig, ax = plt.subplots(figsize=(10, 6.8))

for r, row in enumerate(y_labels):
    for c, col in enumerate(x_labels):
        bg = reg_gray_bg
        if col in select_map[row]:
            bg = light_blue
        if (row in two_axis_rows) and (col in select_map[row]):
            bg = strong_blue
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor=bg, edgecolor="black", linewidth=1))

for r, row in enumerate(y_labels):
    for c, col in enumerate(x_labels):
        val = logits[r, c]
        in_sel = (col in select_map[row])
        in_ours = (row in two_axis_rows) and in_sel
        color = "white" if in_ours else ("black" if in_sel else num_dim)
        ax.text(c+0.5, r+0.5, f"{val:+.2f}", ha="center", va="center", fontsize=10.5, color=color)

ax.set_xlim(0, C); ax.set_ylim(R, 0)
ax.set_xticks([i+0.5 for i in range(C)]); ax.set_yticks([i+0.5 for i in range(R)])
ax.set_xticklabels(x_labels, fontsize=11)
ax.set_yticklabels(y_labels, fontsize=11)

ax.set_aspect("equal")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

legend_elements = [
    Patch(facecolor=light_blue, edgecolor="black", label="RS-KD"),
    Patch(facecolor=strong_blue, edgecolor="black", label="Ours (2-axis RS-KD)"),
    Patch(facecolor=reg_gray_bg, edgecolor="black", label="Regular Distillation"),
]
ax.legend(handles=legend_elements, loc="upper right", frameon=True)

png_path = "rskd_grid_two_axis_blue.png"
plt.tight_layout(pad=0.45)
plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.1)

ax.text(0.5, -0.08, "Logits of Probable Next-Token Predictions",
        transform=ax.transAxes, ha="center", va="top", fontsize=10, color="#333333")
ax.text(-0.20, 0.5, "Ground Truth Text Tokens",
        transform=ax.transAxes, ha="right", va="center", rotation=90, fontsize=10, color="#333333", clip_on=False)
png_path = "rskd_grid_two_axis_blue_with_axes.png"
plt.tight_layout(pad=0.45)
plt.savefig(png_path, dpi=400, bbox_inches="tight", pad_inches=0.1)

plt.close()