import matplotlib.pyplot as plt
from matplotlib.patches import Patch

legend_elements = [
    Patch(facecolor="#a6c8e5", edgecolor="black", label="RS-KD"),
    Patch(facecolor="#1f77b4", edgecolor="black", label="Ours (2-axis RS-KD)"),
    Patch(facecolor="#f0f0f0", edgecolor="black", label="Regular Distillation"),
]

fig_legend = plt.figure(figsize=(3.03209, 0.5)) # Adjust figure size as needed
ax_legend = fig_legend.add_subplot(111)

# Create the legend
legend = ax_legend.legend(handles=legend_elements, loc='center', ncol=3, frameon=True, fontsize=9)

# Hide the axes
ax_legend.axis('off')

# Save the legend
plt.savefig('rskd_legend.pgf', bbox_inches='tight', pad_inches=0)
plt.close(fig_legend)