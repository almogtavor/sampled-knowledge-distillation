# Update colors to blue variants (while preserving intensity difference).
# Change legend names as requested: "RS-KD", "Our (2-axis RS-KD)", "Regular Distillation".
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np

# Reuse previous setup with logits
y_labels = ["We", "love", "when", "small", "models", "perform", "like", "large"]
x_labels = ["d=1", "d=150", "d=500", "d=3000", "d=7000", "d=30k"]
R, C = len(y_labels), len(x_labels)

np.random.seed(7)
base_logits = np.random.normal(loc=-4.5, scale=1.0, size=(R, C))

b = {
    ("We","d=150"): 2.1, ("We","d=500"): 2.0, ("We","d=3000"): 0,
    ("love","d=1"): -0.2, ("love","d=500"): -0.1, ("love","d=3000"): -1.0, ("love","d=7000"): -0.4,
    ("when","d=150"): 0.6, ("when","d=3000"): 0.5,
    ("models","d=500"): -0.7,
    ("perform","d=500"): 1.1,
    ("like","d=1"): -1.2, ("like","d=500"): -0.9, ("like","d=3000"): -1.0, ("like","d=7000"): -0.6,
    ("large","d=150"): 1.6, ("large","d=500"): 1.4, ("large","d=3000"): -0.2
}
for (ry, cx), v in b.items():
    r = y_labels.index(ry)
    c = x_labels.index(cx)
    base_logits[r, c] = v

rskd_cols_per_row = {
    "We": {"d=150","d=500"},
    "love": {"d=7000","d=1"},
    "when": {"d=150","d=3000"},
    "small": {"d=500","d=3000"},
    "models": {"d=500","d=150"},
    "perform": {"d=150","d=3000"},
    "like": {"d=1","d=7000"},
    "large": {"d=150","d=500"},
}
two_axis_rows = {"We","perform","large"}

# Colors (blue variants)
strong_blue = "#1f77b4"
light_blue = "#a6c8e5"
reg_gray_bg = "#f0f0f0"
light_num = "#666666"

fig, ax = plt.subplots(figsize=(10, 6.8))

# Cells
for r, row in enumerate(y_labels):
    for c, col in enumerate(x_labels):
        bg = reg_gray_bg
        if col in rskd_cols_per_row[row]:
            bg = light_blue
        if (row in two_axis_rows) and (col in rskd_cols_per_row[row]):
            bg = strong_blue
        rb = R - 1 - r
        ax.add_patch(Rectangle((c, rb), 1, 1, facecolor=bg, edgecolor="black", linewidth=1))

# Numbers
for r, row in enumerate(y_labels):
    for c, col in enumerate(x_labels):
        rb = R - 1 - r
        val = base_logits[r, c]
        in_rskd = (col in rskd_cols_per_row[row])
        in_ours = (row in two_axis_rows) and in_rskd
        if in_ours:
            color = "white"
        elif in_rskd:
            color = "black"
        else:
            color = light_num
        ax.text(c+0.5, rb+0.5, f"{val:+.2f}", ha="center", va="center", fontsize=10.5, color=color)

# Ticks
ax.set_xlim(0, C); ax.set_ylim(0, R)
ax.set_xticks([i+0.5 for i in range(C)]); ax.set_yticks([i+0.5 for i in range(R)])
ax.set_xticklabels(x_labels, rotation=30, ha="right")
ax.set_yticklabels(y_labels)

# Captions & Legend
# ax.text(-1.65, R + 0.6, "batch (Y-axis / token positions)\n(numbers are teacher logits)", ha="left", va="center", fontsize=11)
# ax.text(C + 0.45, -0.9, "vocabulary (X-axis / next-word logits)", ha="left", va="bottom", fontsize=11)

legend_elements = [
    Patch(facecolor=light_blue, edgecolor="black", label="RS-KD"),
    Patch(facecolor=strong_blue, edgecolor="black", label="Our (2-axis RS-KD)"),
    Patch(facecolor=reg_gray_bg, edgecolor="black", label="Regular Distillation"),
]
# the legend should be inside the plot area on the right side:
ax.legend(handles=legend_elements, loc="upper right",  frameon=True)

ax.set_aspect("equal")
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)

plt.tight_layout()

png_path = "rskd_grid_two_axis_blue.png"
pdf_path = "rskd_grid_two_axis_blue.pdf"
plt.savefig(png_path, dpi=400, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")

