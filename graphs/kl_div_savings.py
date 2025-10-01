# Bottom-only variant: stacked mini KL terms (teacher vs student) for
# Full KD, RS-KD, and Our (2-axis RS-KD). No masks, no arrows.
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np

# Colors
blue = "#1f77b4"     # teacher (highlighted bins)
orange = "#ff7f0e"   # student (highlighted bins)
muted = "#c9c9c9"    # muted bars / disabled rows
dark = "#555555"

rng = np.random.default_rng(123)

def mini_hist(ax, cols=10, selected_cols=None, disabled=False, full=False):
    x = np.arange(cols)
    teacher = np.clip(rng.normal(1.4, 0.45, size=cols), 0.2, None)
    student = np.clip(teacher + rng.normal(0.0, 0.25, size=cols), 0.1, None)
    if disabled:
        t_colors = [muted]*cols
        s_colors = [muted]*cols
    elif full:
        t_colors = [blue]*cols
        s_colors = [orange]*cols
    else:
        selected_cols = selected_cols or []
        t_colors = [blue if i in selected_cols else muted for i in range(cols)]
        s_colors = [orange if i in selected_cols else muted for i in range(cols)]
    w = 0.35
    ax.bar(x - w/2, teacher, width=w, color=t_colors, edgecolor=dark, linewidth=0.1)
    ax.bar(x + w/2, student, width=w, color=s_colors, edgecolor=dark, linewidth=0.1)
    ax.set_xlim(-0.6, cols-0.4)
    ax.set_ylim(0, max(teacher.max(), student.max())*1.25)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# Layout
fig = plt.figure(figsize=(12.5, 9))
gs = GridSpec(5, 3, hspace=0.06, wspace=0.35)

titles = ["FullKD", "RS-KD", "Ours (2-axis RS-KD)"]
for j, title in enumerate(titles):
    # title above the column
    ax_title = fig.add_subplot(GridSpec(1,3, top=0.92, bottom=0.88)[0,j])
    ax_title.axis('off')
    ax_title.text(0.5, 0.5, title, ha="center", va="center", fontsize=32, weight="bold")

# Column 1: Full KD — all rows active, all bins
for i in range(5):
    ax = fig.add_subplot(gs[i, 0])
    mini_hist(ax, full=True)

# Column 2: RS-KD — all rows active, per-row X-samples
per_row_cols = [[1,6],[2,7],[1,8],[0,6],[2,5]]
for i in range(5):
    ax = fig.add_subplot(gs[i, 1])
    mini_hist(ax, selected_cols=per_row_cols[i])

# Column 3: Our (2-axis RS-KD) — same X-samples, subset of rows disabled
two_axis_cols = [1,6]
selected_rows = [0,2,4]  # only these are distilled
for i in range(5):
    ax = fig.add_subplot(gs[i, 2])
    mini_hist(ax, selected_cols=two_axis_cols, disabled=(i not in selected_rows))

# Suptitle and caption
# fig.suptitle("Stacked KL terms across positions (teacher vs student per position)", fontsize=14, y=0.995)
# fig.text(0.01, 0.01, "Blue = teacher bars used in KL • Orange = student bars used in KL • Gray = bins/rows not distilled", fontsize=10)

plt.savefig('kd_intuition.pgf', bbox_inches='tight', pad_inches=0)
plt.close()
