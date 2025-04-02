import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
from matplotlib.patches import Polygon

# === Load Data ===
input_path = "cf_output/shap_plddt_combined.csv"
output_dir = "cf_output"
df = pd.read_csv(input_path)

# === 1. SHAP-style Colored Scatter Plot ===
plt.figure(figsize=(12, 6))
scatter = plt.scatter(
    df["Residue_Index"],
    df["Normalized_SHAP"],
    c=df["pLDDT"],
    cmap="coolwarm",
    edgecolor='none',
    alpha=0.8
)
plt.colorbar(scatter, label="pLDDT Score")
plt.title("SHAP Value per Residue (Colored by pLDDT)")
plt.xlabel("Residue Index")
plt.ylabel("SHAP Value")

plt.twinx()
sns.histplot(df["Residue_Index"], bins=30, color="lightgray", edgecolor=None, alpha=0.3)
plt.ylim(0, 5)
plt.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "shap_scatter_plot.png"), dpi=300)
plt.show()

# === 2. SHAP-style Waterfall Plot with Gradient Bars ===

# Select top 10 residues by absolute SHAP value
top_df = df.reindex(df["Normalized_SHAP"].abs().sort_values(ascending=False).index).head(10)
top_df = top_df.sort_values(by="Normalized_SHAP").reset_index(drop=True)

contribs = top_df["Normalized_SHAP"].values
residue_labels = top_df["Residue_Index"].astype(str).tolist()
base_value = df["Normalized_SHAP"].mean()

# Color gradient based on SHAP magnitude
abs_vals = np.abs(contribs)
ranks = abs_vals.argsort().argsort()  # rank from low to high
norm_ranks = ranks / ranks.max()      # normalized 0–1

# Use updated colormap syntax (Matplotlib ≥ 3.7)
colormap = plt.colormaps['plasma']
colors = [colormap(1 - r) for r in norm_ranks]

# Waterfall bar positions
positions = [base_value]
for val in contribs:
    positions.append(positions[-1] + val)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
bar_height = 0.4
arrow_head_fraction = 0.25

for i in range(len(contribs)):
    value = contribs[i]
    start = positions[i]
    end = positions[i + 1]
    direction = 'right' if value > 0 else 'left'
    color = colors[i]

    y0 = i - bar_height / 2
    y1 = i + bar_height / 2
    arrow_length = end - start
    head_size = arrow_length * arrow_head_fraction

    if direction == 'right':
        coords = [
            [start, y0],
            [end - head_size, y0],
            [end, i],
            [end - head_size, y1],
            [start, y1]
        ]
    else:
        coords = [
            [start, y0],
            [start + head_size, y0],
            [end, i],
            [start + head_size, y1],
            [start, y1]
        ]

    poly = Polygon(coords, closed=True, color=color)
    ax.add_patch(poly)

    ax.text(
        start + arrow_length / 2,
        i,
        f"{value:+.5f}",
        va='center',
        ha='center',
        color='white',
        fontsize=9,
        weight='bold'
    )

# Format plot
ax.set_xlim(min(positions) - 0.01, max(positions) + 0.01)
ax.set_ylim(-1, len(contribs))
ax.set_yticks(range(len(residue_labels)))
ax.set_yticklabels([f"Residue {r}" for r in residue_labels])
ax.set_xlabel("Cumulative SHAP Contribution")
ax.set_title("SHAP Waterfall with Gradient Color (Top 10 Residues)")
ax.axvline(base_value, color='gray', linestyle='--', linewidth=1)
ax.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, "shap_waterfall_plot.png"), dpi=300)
plt.show()

print(f"✅ SHAP Scatterplot and Waterfall Plot saved to: {output_dir}")
