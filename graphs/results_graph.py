# Create a Pareto-style scatter plot from the user's Main Results table.
# One chart only, using matplotlib (no seaborn), default colors, and saving vector + PNG assets.

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data from the provided table
data = [
    # Method, Family, GSM8K, HellaSwag, PIQA, LAMBADA_acc, LAMBADA_ppl, avg
    ("Full KD (all tokens, full softmax)", "FullKD", 34.5, 34.7, 65.2, 40.8, 67.3, 46.9),
    ("Token-selective KD (entropy top-20%)", "TSKD", 37.7, 37.1, 66.7, 42.2, 23.0, 48.7),
    ("Token-selective KD (entropy top-25%)", "TSKD", 38.4, 37.0, 67.1, 42.2, 23.4, 48.8),
    ("Token-selective KD (entropy top-75%)", "TSKD", np.nan, 34.7, 64.9, 41.0, 72.9, np.nan),
    ("Token-selective KD (bucket of 5%-25%)", "TSKD", 35.6, 34.5, 64.7, 40.3, 77.2, 46.5),
    ("Token-selective KD (random 25%)", "TSKD", 31.6, 34.5, 65.3, 40.6, 69.4, 46.8),
    ("Token-selective KD (pos-rs-kd 25%)", "TSKD", 32.5, 34.4, 65.5, 41.6, 69.0, 47.1),
    ("Token-selective KD (entropy top-25%, GLS)", "TSKD", 37.9, 36.9, 67.1, 43.0, 21.8, 49.0),
    ("Sampled KD (entropy top-25%)", "SampledKD", 26.3, 35.7, 60.0, 41.2, 38.7, 45.6),
    ("Sampled KD (entropy top-75%)", "SampledKD", 35.7, 33.8, 62.2, 33.7, 305.9, 43.2),
    ("Sampled KD (pos-rs-kd top-25%)", "SampledKD", 33.6, 33.9, 62.2, 32.9, 287.0, 43.0),
    ("Sampled KD (LinUCB top-25%)", "SampledKD", 33.5, 33.9, 62.3, 32.9, 287.0, 43.0),
    ("No KD baseline (Qwen3-0.6B)", "NoKD", 47.6, 37.5, 67.6, 40.4, 24.72, 48.2),
]

df = pd.DataFrame(data, columns=[
    "Method", "Family", "GSM8K", "HellaSwag", "PIQA", "LAMBADA_acc", "LAMBADA_ppl", "AvgAcc"
])

# Keep rows that have both AvgAcc and LAMBADA_ppl
plot_df = df.dropna(subset=["AvgAcc", "LAMBADA_ppl"]).copy()

# Compute Pareto frontier (minimize ppl, maximize AvgAcc)
# A point is dominated if there exists another with ppl <= and acc >= (with one strict).
def pareto_frontier(points):
    # points: array of (ppl, acc, index)
    frontier_idx = []
    for i, (x_i, y_i, idx_i) in enumerate(points):
        dominated = False
        for j, (x_j, y_j, idx_j) in enumerate(points):
            if j == i:
                continue
            if (x_j <= x_i and y_j >= y_i) and (x_j < x_i or y_j > y_i):
                dominated = True
                break
        if not dominated:
            frontier_idx.append(idx_i)
    return sorted(frontier_idx, key=lambda k: (plot_df.loc[k, "LAMBADA_ppl"], -plot_df.loc[k, "AvgAcc"]))

points = list(zip(plot_df["LAMBADA_ppl"].values, plot_df["AvgAcc"].values, plot_df.index.values))
front_idx = pareto_frontier(points)

# Marker map by family (no specific colors)
markers = {
    "FullKD": "o",
    "TSKD (Ours)": "s",
    "SampledKD (Ours)": "^",
    "NoKD": "D",
}

# Create plot
plt.figure(figsize=(7, 4.5))  # NeurIPS-friendly aspect

# Scatter all methods
for fam in plot_df["Family"].unique():
    fam_df = plot_df[plot_df["Family"] == fam]
    plt.scatter(fam_df["LAMBADA_ppl"], fam_df["AvgAcc"], marker=markers.get(fam, "o"), label=fam, alpha=0.9)

# Highlight Pareto frontier
# frontier = plot_df.loc[front_idx].sort_values(["LAMBADA_ppl", "AvgAcc"], ascending=[True, False])
# plt.plot(frontier["LAMBADA_ppl"], frontier["AvgAcc"], linestyle="--", linewidth=1.5, label="Pareto frontier")

# Annotate key points (to avoid clutter, annotate selected ones)
labels_to_annotate = [
    "Token-selective KD (entropy top-25%, GLS)",
    "No KD baseline (Qwen3-0.6B)",
    "Full KD (all tokens, full softmax)",
    "Sampled KD (entropy top-25%)",
    "Sampled KD (entropy top-75%)",
]
for _, row in plot_df.iterrows():
    if row["Method"] in labels_to_annotate and not row["Method"] == "Sampled KD (entropy top-75%)":
        plt.annotate(
            row["Method"]
                .replace("Token-selective KD ", "TSKD ")
                .replace("Sampled KD ", "SampledKD ")
                .replace(" (all tokens, full softmax)", ""),
            (row["LAMBADA_ppl"], row["AvgAcc"]),
            textcoords="offset points",
            xytext=(5, 5),  # Adjust this to move label closer/away from the point
            fontsize=12,
            ha='left',  # Horizontal alignment: left
            va='top',  # Vertical alignment: bottom
        )
    elif row["Method"] == "Sampled KD (entropy top-75%)":
        plt.annotate(
            "SampledKD (entropy top-75%)",
            (row["LAMBADA_ppl"], row["AvgAcc"]),
            textcoords="offset points",
            xytext=(5, 5),  # Adjust this to move label closer/away from the point
            fontsize=12,
            ha='right'
        )

plt.xscale("log")
plt.xlabel("LAMBADA Perplexity (log scale) ↓")
plt.ylabel("Average Accuracy across tasks (%) ↑")
plt.title("Accuracy–Perplexity Trade-off")
plt.grid(True, which="both", linestyle=":", linewidth=0.6)
plt.legend(frameon=True, ncol=1)

plt.tight_layout()
png_path = "graphs/fig_pareto_acc_vs_ppl.png"
pdf_path = "graphs/fig_pareto_acc_vs_ppl.pdf"
plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.close()
