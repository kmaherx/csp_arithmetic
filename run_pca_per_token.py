"""Per-token-position PCA: one separate PCA per SP token slot.

For each of L=4 token positions, run PCA across the persona population
in 2560-dim embedding space. Produces L separate PCAs, each yielding a
PC1 ranking at that slot.

Complements the pooled per-token PCA (run_pca.py) which mixed all
N×L points and had token-position as the dominant axis. Here we
first select a token slot, then ask about persona variance within it.

Usage:
    python run_pca_per_token.py --persona-set roles
    python run_pca_per_token.py --persona-set traits
    python run_pca_per_token.py --persona-set joint
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA

from soft_prompt import SoftPrompt
import persona_sets as P
from run_pca import load_csps

SCRIPT_DIR = Path(__file__).parent.resolve()


def spearman(x, y):
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    n = len(x)
    d2 = ((rx - ry) ** 2).sum()
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-set", default="roles",
                        choices=["roles", "traits", "joint"])
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "results" / "pca"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names_req = P.get_names(args.persona_set)
    print(f"Loading CSPs for persona-set='{args.persona_set}'...")
    embeds = load_csps(names_req, Path(args.results_dir))
    L = next(iter(embeds.values())).shape[0]
    print(f"  {len(embeds)} CSPs, L={L}")

    names = sorted(embeds.keys())
    summary = {"persona_set": args.persona_set, "per_token_separate": {}}

    for t in range(L):
        X = np.stack([embeds[n][t] for n in names], axis=0)
        Xc = X - X.mean(axis=0, keepdims=True)
        pca = PCA(n_components=min(10, Xc.shape[0] - 1))
        pca.fit(Xc)
        Z = pca.transform(Xc)
        evr = pca.explained_variance_ratio_

        pc1 = Z[:, 0]
        pc2 = Z[:, 1]
        fe_vals = np.array([P.ALL_FE.get(n, np.nan) for n in names])
        has_fe = ~np.isnan(fe_vals)
        rho = (spearman(pc1[has_fe], fe_vals[has_fe])
               if has_fe.any() else float("nan"))

        print()
        print(f"=== Token {t+1} PCA — {args.persona_set} (N={len(names)}) ===")
        print(f"   explained var: PC1 {evr[0]*100:.1f}%  PC2 {evr[1]*100:.1f}%  "
              f"PC3 {evr[2]*100:.1f}%")
        print(f"   Spearman(PC1, FE) = {rho:+.3f}")

        hover = [
            f"{n} ({P.kind_of(n)}, FE={fe_vals[i]:.1f}%)"
            if not np.isnan(fe_vals[i]) else n
            for i, n in enumerate(names)
        ]
        fig = px.scatter(
            x=pc1, y=pc2, color=fe_vals, color_continuous_scale="viridis",
            hover_name=hover,
            labels={
                "x": f"PC1 ({evr[0]*100:.1f}% var)",
                "y": f"PC2 ({evr[1]*100:.1f}% var)",
                "color": "FE (%)",
            },
        )
        fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
        fig.update_layout(
            width=1000, height=700,
            title=f"Per-token PCA (token {t+1} only) — "
                  f"{args.persona_set} (N={len(names)}) · "
                  f"Spearman(PC1, FE) = {rho:+.3f}",
        )
        html = out_dir / f"pca_token{t+1}_{args.persona_set}.html"
        fig.write_html(str(html))
        print(f"   saved {html}")

        fig2, ax = plt.subplots(figsize=(10, 7))
        sc = ax.scatter(pc1, pc2, c=fe_vals, cmap="viridis",
                        s=100, edgecolors="white", linewidth=1)
        for i, n in enumerate(names):
            ax.annotate(n, (pc1[i], pc2[i]), fontsize=6,
                        xytext=(6, 2), textcoords="offset points")
        cbar = fig2.colorbar(sc, ax=ax); cbar.set_label("FE (%)")
        ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
        ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
        ax.set_title(f"Per-token PCA (token {t+1}) — "
                     f"{args.persona_set} · Spearman(PC1, FE) = {rho:+.3f}")
        ax.grid(alpha=0.3)
        png = out_dir / f"pca_token{t+1}_{args.persona_set}.png"
        fig2.tight_layout(); fig2.savefig(str(png), dpi=150)
        plt.close(fig2)
        print(f"   saved {png}")

        summary["per_token_separate"][f"token_{t+1}"] = {
            "names": names,
            "pc1": pc1.tolist(),
            "pc2": pc2.tolist(),
            "explained_variance_ratio": evr.tolist(),
            "spearman_pc1_fe": float(rho),
        }

    summary_path = (out_dir /
                    f"pca_per_token_separate_summary_{args.persona_set}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
