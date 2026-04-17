"""Plot teacher-resistance clusters on the joint PCA.

Shows PC1 vs PC3 and PC1 vs PC5 scatter plots, with safety-violating
and self-referential trait clusters highlighted. These are the PCs
where the two resistance clusters separate from the main population.

Usage:
    python plot_resistance_clusters.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from persona_sets import (
    SAFETY_VIOLATING, SELF_REFERENTIAL, ALL_FE, kind_of,
)

SCRIPT_DIR = Path(__file__).parent.resolve()


def group_of(name):
    if name in SAFETY_VIOLATING:
        return "safety-violating"
    if name in SELF_REFERENTIAL:
        return "self-referential"
    return "role" if kind_of(name) == "role" else "other-trait"


GROUP_COLOR = {
    "role":             "#1f77b4",  # blue
    "other-trait":      "#aec7e8",  # light blue
    "safety-violating": "#d62728",  # red
    "self-referential": "#ff7f0e",  # orange
}

GROUP_SIZE = {
    "role":             70,
    "other-trait":      70,
    "safety-violating": 150,
    "self-referential": 150,
}


def plot_pc_pair(names, Z, fe, pc_x, pc_y, out_dir, suffix=""):
    """Scatter with PC_x vs PC_y, highlighting resistance clusters."""
    groups = [group_of(n) for n in names]

    # --- Plotly interactive ---
    hover = [f"{n} ({group_of(n)}, FE={fe[i]:.1f}%)"
             for i, n in enumerate(names)]
    fig = px.scatter(
        x=Z[:, pc_x], y=Z[:, pc_y], color=groups,
        color_discrete_map=GROUP_COLOR,
        hover_name=hover,
        labels={
            "x": f"PC{pc_x+1}", "y": f"PC{pc_y+1}",
            "color": "Group",
        },
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(
        width=1000, height=700,
        title=f"Teacher-resistance clusters on joint PCA — "
              f"PC{pc_x+1} vs PC{pc_y+1}{suffix}",
    )
    html = out_dir / f"pca_resistance_pc{pc_x+1}_pc{pc_y+1}.html"
    fig.write_html(str(html))
    print(f"  saved {html}")

    # --- Matplotlib static ---
    fig2, ax = plt.subplots(figsize=(11, 8))
    for grp, color in GROUP_COLOR.items():
        idx = [i for i, g in enumerate(groups) if g == grp]
        if not idx:
            continue
        size = GROUP_SIZE[grp]
        alpha = 0.85 if grp in ("safety-violating", "self-referential") else 0.5
        ax.scatter(Z[idx, pc_x], Z[idx, pc_y], s=size, alpha=alpha,
                   label=f"{grp} (n={len(idx)})",
                   color=color, edgecolors="white", linewidth=1)
        # Label only the resistance clusters clearly
        if grp in ("safety-violating", "self-referential"):
            for i in idx:
                ax.annotate(names[i], (Z[i, pc_x], Z[i, pc_y]),
                            fontsize=8, fontweight="bold",
                            xytext=(8, 4), textcoords="offset points")
        else:
            # Light labels for the rest
            for i in idx:
                ax.annotate(names[i], (Z[i, pc_x], Z[i, pc_y]),
                            fontsize=5, alpha=0.6,
                            xytext=(5, 2), textcoords="offset points")

    # Draw cluster centroids
    for grp, color in GROUP_COLOR.items():
        if grp not in ("safety-violating", "self-referential"):
            continue
        idx = [i for i, g in enumerate(groups) if g == grp]
        if not idx:
            continue
        cx = Z[idx, pc_x].mean()
        cy = Z[idx, pc_y].mean()
        ax.scatter([cx], [cy], s=300, marker="X", color=color,
                   edgecolors="black", linewidth=1.5, zorder=10,
                   label=f"{grp} centroid")

    ax.axhline(0, color="gray", alpha=0.3, linewidth=0.8)
    ax.axvline(0, color="gray", alpha=0.3, linewidth=0.8)
    ax.set_xlabel(f"PC{pc_x+1}")
    ax.set_ylabel(f"PC{pc_y+1}")
    ax.set_title(f"Teacher-resistance clusters on joint PCA\n"
                 f"PC{pc_x+1} vs PC{pc_y+1}{suffix}")
    ax.legend(fontsize=9, loc="best")
    ax.grid(alpha=0.3)
    png = out_dir / f"pca_resistance_pc{pc_x+1}_pc{pc_y+1}.png"
    fig2.tight_layout()
    fig2.savefig(str(png), dpi=150)
    plt.close(fig2)
    print(f"  saved {png}")


def main():
    out_dir = SCRIPT_DIR / "results" / "pca"
    summary_path = out_dir / "pca_summary_joint.json"
    with open(summary_path) as f:
        pca = json.load(f)

    names = pca["flattened"]["names"]
    Z = np.array(pca["flattened"]["Z"])  # (65, 5)
    fe = np.array([ALL_FE.get(n, np.nan) for n in names])

    print(f"Loaded joint PCA: {len(names)} CSPs × {Z.shape[1]} PCs")

    # Report centroid spread on each PC for reference
    print("\nGroup centroids (joint PCA, PC1-PC5):")
    groups = [group_of(n) for n in names]
    for grp in ["role", "other-trait", "safety-violating", "self-referential"]:
        idx = [i for i, g in enumerate(groups) if g == grp]
        if not idx:
            continue
        centroid = Z[idx].mean(axis=0)
        print(f"  {grp:18s} (n={len(idx)}):  "
              f"PC1 {centroid[0]:+6.3f}  PC2 {centroid[1]:+6.3f}  "
              f"PC3 {centroid[2]:+6.3f}  PC4 {centroid[3]:+6.3f}  "
              f"PC5 {centroid[4]:+6.3f}")

    # Key plots: PC1 vs PC3 (safety-violating separates on PC3);
    # PC1 vs PC5 (self-referential separates on PC5);
    # PC3 vs PC5 (both cluster axes at once).
    for pc_x, pc_y in [(0, 2), (0, 4), (2, 4)]:
        print(f"\nPlotting PC{pc_x+1} vs PC{pc_y+1}...")
        plot_pc_pair(names, Z, fe, pc_x, pc_y, out_dir)


if __name__ == "__main__":
    main()
