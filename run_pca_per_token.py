"""Per-token-position PCA: one separate PCA per SP token slot.

For each of L=4 token positions, run PCA across the 33 role CSPs in
2560-dim embedding space. Produces 4 separate PCAs, each yielding a
PC1 ranking of personas at that slot.

Complements the pooled per-token PCA (run_pca.py) which mixed all 132
points and ended up with token-position as the dominant axis. Here we
first select a token slot, then ask about persona variance within it.

Usage:
    python run_pca_per_token.py
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

from soft_prompt import SoftPrompt
from run_pca import (
    ROLE_NAMES, CATEGORY_COLOR, category_of, load_role_csps,
)

SCRIPT_DIR = Path(__file__).parent.resolve()

# FE from NARRATIVE.md Preview table
FE = {
    "journalist": 69.2, "librarian": 75.3, "teacher": 75.9,
    "scientist": 77.3, "coach": 77.5, "lawyer": 79.0,
    "surgeon": 79.9, "detective": 81.7, "salesperson": 82.6,
    "ninja": 84.0, "chef": 84.7, "historian": 84.9,
    "therapist": 86.4, "politician": 86.7, "stoic": 87.8,
    "bard": 87.9, "spy": 88.0, "philosopher": 88.7,
    "comedian": 89.0, "witch": 89.3, "pirate": 89.7,
    "cowboy": 89.8, "wizard": 89.9, "oracle": 90.0,
    "samurai": 90.1, "knight": 90.2, "rapper": 90.9,
    "necromancer": 91.1, "monk": 91.3, "vampire": 91.4,
    "poet": 91.7, "druid": 92.1, "prophet": 92.8,
}


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    n = len(x)
    d2 = ((rx - ry) ** 2).sum()
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "results" / "pca"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading role CSPs from {args.results_dir}...")
    embeds = load_role_csps(Path(args.results_dir))
    L = next(iter(embeds.values())).shape[0]
    print(f"  {len(embeds)} roles, L={L}")

    names = sorted(embeds.keys())
    summary = {"per_token_separate": {}}

    for t in range(L):
        # Stack the t-th token across all roles
        X = np.stack([embeds[n][t] for n in names], axis=0)  # (N, d)
        X_c = X - X.mean(axis=0, keepdims=True)
        pca = PCA(n_components=min(10, X_c.shape[0] - 1))
        pca.fit(X_c)
        Z = pca.transform(X_c)
        evr = pca.explained_variance_ratio_

        pc1 = Z[:, 0]
        pc2 = Z[:, 1]
        fe_vals = np.array([FE[n] for n in names])

        # Correlate PC1 with FE; also test sign of correlation (may flip)
        rho = spearman(pc1, fe_vals)
        print()
        print(f"=== Token {t+1} PCA (N=33, d=2560) ===")
        print(f"  explained variance: PC1 {evr[0]*100:.1f}% · "
              f"PC2 {evr[1]*100:.1f}% · PC3 {evr[2]*100:.1f}% · "
              f"cumulative top 5 = {sum(evr[:5])*100:.1f}%")
        print(f"  Spearman(PC1, FE) = {rho:+.3f}")

        # Plot
        cats = [category_of(n) for n in names]
        hover = [f"{n} (FE={fe_vals[i]:.1f}%)" for i, n in enumerate(names)]

        # HTML — colored by FE
        fig = px.scatter(
            x=pc1, y=pc2,
            color=fe_vals,
            color_continuous_scale="viridis",
            hover_name=hover,
            labels={
                "x": f"PC1 ({evr[0]*100:.1f}% var)",
                "y": f"PC2 ({evr[1]*100:.1f}% var)",
                "color": "FE (%)",
            },
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        fig.update_layout(
            width=900, height=700,
            title=f"Per-token PCA — token {t+1} only (N=33 roles) · "
                  f"Spearman(PC1, FE) = {rho:+.3f}",
        )
        html = out_dir / f"pca_token{t+1}_roles.html"
        fig.write_html(str(html))
        print(f"  saved {html}")

        # PNG
        fig2, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(pc1, pc2, c=fe_vals, cmap="viridis",
                        s=100, edgecolors="white", linewidth=1)
        for i, n in enumerate(names):
            ax.annotate(n, (pc1[i], pc2[i]), fontsize=7,
                        xytext=(6, 2), textcoords="offset points")
        cbar = fig2.colorbar(sc, ax=ax)
        cbar.set_label("FE (%)")
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
        ax.set_title(f"Per-token PCA — token {t+1} only (N=33 roles)\n"
                     f"Spearman(PC1, FE) = {rho:+.3f}")
        ax.grid(alpha=0.3)
        png = out_dir / f"pca_token{t+1}_roles.png"
        fig2.tight_layout()
        fig2.savefig(str(png), dpi=150)
        plt.close(fig2)
        print(f"  saved {png}")

        summary["per_token_separate"][f"token_{t+1}"] = {
            "names": names,
            "pc1": pc1.tolist(),
            "pc2": pc2.tolist(),
            "explained_variance_ratio": evr.tolist(),
            "spearman_pc1_fe": rho,
        }

    summary_path = out_dir / "pca_per_token_separate_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
