"""PCA on the trained CSP population.

Runs two PCA flavors:

1. **Flattened** — concatenate the L=4 token embeddings into a single
   (L·d_model)-dim vector per persona. One point per persona.
2. **Per-token** — treat each of L=4 tokens as its own vector.
   N × 4 points total, each labeled with persona + token position.

For each flavor, saves a Plotly express scatter of PC1 vs PC2 with
hover labels (persona, token position where applicable) and color by
category (role vs trait subcategories). Also reports explained
variance ratios.

Usage:
    python run_pca.py --persona-set roles
    python run_pca.py --persona-set traits
    python run_pca.py --persona-set joint
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

SCRIPT_DIR = Path(__file__).parent.resolve()


def palette_for(categories):
    """Build a color palette mapping category -> color."""
    role_palette = {
        "role/round-1": "#d62728",
        "role/fantasy/archetype": "#9467bd",
        "role/profession": "#1f77b4",
        "role/style/register": "#2ca02c",
    }
    trait_palette = {
        "trait/moral": "#8c564b",
        "trait/social": "#e377c2",
        "trait/emotional": "#17becf",
        "trait/cognitive": "#bcbd22",
        "trait/register": "#ff7f0e",
        "trait/tone": "#f4a261",
        "trait/stance": "#264653",
    }
    all_colors = {**role_palette, **trait_palette, "other": "#7f7f7f"}
    return {cat: all_colors.get(cat, "#7f7f7f") for cat in set(categories)}


def load_csps(names, results_dir):
    embeds = {}
    for n in names:
        ckpt = results_dir / n / "sp_pos.pt"
        if not ckpt.exists():
            print(f"  WARN: missing {ckpt}; skipping {n}")
            continue
        sp, _ = SoftPrompt.from_checkpoint(str(ckpt), device="cpu")
        embeds[n] = sp.embedding.detach().float().numpy()
    return embeds


def pca_flattened(embeds):
    names = sorted(embeds.keys())
    X = np.stack([embeds[n].flatten() for n in names], axis=0)
    pca = PCA(n_components=min(10, X.shape[0] - 1))
    Xc = X - X.mean(axis=0, keepdims=True)
    pca.fit(Xc)
    Z = pca.transform(Xc)
    return names, Z, pca.explained_variance_ratio_


def pca_per_token(embeds):
    names = sorted(embeds.keys())
    L = next(iter(embeds.values())).shape[0]
    records = [(n, t, embeds[n][t]) for n in names for t in range(L)]
    X = np.stack([r[2] for r in records], axis=0)
    pca = PCA(n_components=min(10, X.shape[0] - 1))
    Xc = X - X.mean(axis=0, keepdims=True)
    pca.fit(Xc)
    Z = pca.transform(Xc)
    meta = [(r[0], r[1]) for r in records]
    return meta, Z, pca.explained_variance_ratio_


def plot_flattened(names, Z, evr, out_dir, set_name):
    cats = [P.category_of(n) for n in names]
    palette = palette_for(cats)
    fig = px.scatter(
        x=Z[:, 0], y=Z[:, 1], color=cats, color_discrete_map=palette,
        hover_name=names,
        labels={
            "x": f"PC1 ({evr[0]*100:.1f}% var)",
            "y": f"PC2 ({evr[1]*100:.1f}% var)",
            "color": "Category",
        },
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(
        width=1000, height=700,
        title=f"Flattened PCA — {set_name} ({len(names)} CSPs, "
              f"PC1 {evr[0]*100:.1f}% · PC2 {evr[1]*100:.1f}%)",
    )
    html = out_dir / f"pca_flattened_{set_name}.html"
    fig.write_html(str(html))
    print(f"  saved {html}")

    fig2, ax = plt.subplots(figsize=(12, 8))
    for cat, color in palette.items():
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        ax.scatter(Z[idx, 0], Z[idx, 1], s=80, alpha=0.8,
                   label=cat, color=color,
                   edgecolors="white", linewidth=1)
        for i in idx:
            ax.annotate(names[i], (Z[i, 0], Z[i, 1]), fontsize=6,
                        xytext=(6, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.set_title(f"Flattened PCA — {set_name} ({len(names)} CSPs, "
                 f"cumulative {sum(evr[:2])*100:.1f}%)")
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    png = out_dir / f"pca_flattened_{set_name}.png"
    fig2.tight_layout()
    fig2.savefig(str(png), dpi=150)
    plt.close(fig2)
    print(f"  saved {png}")


def plot_per_token(meta, Z, evr, out_dir, set_name):
    personas = [m[0] for m in meta]
    tokens = [m[1] + 1 for m in meta]
    cats = [P.category_of(p) for p in personas]
    palette = palette_for(cats)
    hover = [f"{p} (t{t})" for p, t in zip(personas, tokens)]
    fig = px.scatter(
        x=Z[:, 0], y=Z[:, 1], color=cats, color_discrete_map=palette,
        hover_name=hover, symbol=[str(t) for t in tokens],
        labels={
            "x": f"PC1 ({evr[0]*100:.1f}% var)",
            "y": f"PC2 ({evr[1]*100:.1f}% var)",
            "color": "Category", "symbol": "Token",
        },
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(
        width=1000, height=700,
        title=f"Per-token PCA (pooled) — {set_name} "
              f"({len(personas)} points, PC1 {evr[0]*100:.1f}% · "
              f"PC2 {evr[1]*100:.1f}%)",
    )
    html = out_dir / f"pca_pertoken_{set_name}.html"
    fig.write_html(str(html))
    print(f"  saved {html}")

    fig2, ax = plt.subplots(figsize=(12, 8))
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    for cat, color in palette.items():
        for tok in [1, 2, 3, 4]:
            idx = [i for i in range(len(personas))
                   if cats[i] == cat and tokens[i] == tok]
            if not idx:
                continue
            ax.scatter(Z[idx, 0], Z[idx, 1], s=50, alpha=0.7,
                       label=cat if tok == 1 else None,
                       color=color, marker=markers[tok],
                       edgecolors="white", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.set_title(f"Per-token PCA (pooled) — {set_name} "
                 f"({len(personas)} points, "
                 f"cumulative {sum(evr[:2])*100:.1f}%)")
    ax.legend(fontsize=8, loc="best",
              title="Category (shape = token position)")
    ax.grid(alpha=0.3)
    png = out_dir / f"pca_pertoken_{set_name}.png"
    fig2.tight_layout()
    fig2.savefig(str(png), dpi=150)
    plt.close(fig2)
    print(f"  saved {png}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona-set", default="roles",
                        choices=["roles", "traits", "joint"])
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "results" / "pca"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    names_requested = P.get_names(args.persona_set)
    print(f"Loading CSPs for persona-set='{args.persona_set}' "
          f"({len(names_requested)} requested)...")
    embeds = load_csps(names_requested, Path(args.results_dir))
    print(f"  loaded {len(embeds)} CSPs, "
          f"shape per CSP: {next(iter(embeds.values())).shape}")

    print("\n1. Flattened PCA")
    names, Z_flat, evr_flat = pca_flattened(embeds)
    print(f"   shape: {Z_flat.shape}")
    print(f"   explained var: PC1 {evr_flat[0]*100:.2f}%  "
          f"PC2 {evr_flat[1]*100:.2f}%  PC3 {evr_flat[2]*100:.2f}%")
    plot_flattened(names, Z_flat, evr_flat, out_dir, args.persona_set)

    print("\n2. Per-token PCA (pooled)")
    meta, Z_token, evr_token = pca_per_token(embeds)
    print(f"   shape: {Z_token.shape}")
    print(f"   explained var: PC1 {evr_token[0]*100:.2f}%  "
          f"PC2 {evr_token[1]*100:.2f}%  PC3 {evr_token[2]*100:.2f}%")
    plot_per_token(meta, Z_token, evr_token, out_dir, args.persona_set)

    # Save summary
    summary = {
        "persona_set": args.persona_set,
        "flattened": {
            "names": names,
            "Z": Z_flat[:, :5].tolist(),
            "explained_variance_ratio": evr_flat.tolist(),
        },
        "per_token": {
            "meta": [(m[0], int(m[1])) for m in meta],
            "Z": Z_token[:, :5].tolist(),
            "explained_variance_ratio": evr_token.tolist(),
        },
    }
    summary_path = out_dir / f"pca_summary_{args.persona_set}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()
