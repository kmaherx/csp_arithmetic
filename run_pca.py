"""PCA on the trained role CSP population.

Runs two PCA flavors:

1. **Flattened** — concatenate the L=4 token embeddings into a single
   (L·d_model)-dim vector per persona. 33 points total.
2. **Per-token** — treat each of L=4 tokens as its own vector.
   33 × 4 = 132 points total, each labeled with role + token position.

For each flavor, saves a Plotly express scatter of PC1 vs PC2 with
hover labels (role, token position where applicable) and color by
role category (fantasy / profession / style / round-1). Also reports
explained variance ratios.

Usage:
    python run_pca.py
    python run_pca.py --out-dir results/pca
"""

import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
from sklearn.decomposition import PCA

import config
from soft_prompt import SoftPrompt

SCRIPT_DIR = Path(__file__).parent.resolve()


# ── Role categories (from config.py's comment structure) ────────────────

ROLE_CATEGORIES = {
    "round-1": ["pirate", "poet", "prophet"],
    "fantasy/archetype": [
        "wizard", "samurai", "knight", "vampire", "bard", "oracle",
        "necromancer", "druid", "witch", "ninja",
    ],
    "profession": [
        "detective", "chef", "scientist", "journalist", "surgeon",
        "therapist", "spy", "librarian", "lawyer", "teacher",
    ],
    "style/register": [
        "comedian", "philosopher", "monk", "rapper", "stoic",
        "politician", "salesperson", "coach", "historian", "cowboy",
    ],
}
ROLE_NAMES = [n for cat in ROLE_CATEGORIES.values() for n in cat]


def category_of(name):
    for cat, members in ROLE_CATEGORIES.items():
        if name in members:
            return cat
    return "other"


# ── Load role CSPs ──────────────────────────────────────────────────────

def load_role_csps(results_dir):
    embeds = {}
    for role in ROLE_NAMES:
        ckpt_path = results_dir / role / "sp_pos.pt"
        if not ckpt_path.exists():
            print(f"  WARN: missing {ckpt_path}; skipping {role}")
            continue
        sp, _ = SoftPrompt.from_checkpoint(str(ckpt_path), device="cpu")
        # shape (L, d)
        embeds[role] = sp.embedding.detach().float().numpy()
    return embeds


# ── PCA flavors ─────────────────────────────────────────────────────────

def pca_flattened(embeds):
    """One point per role: flatten (L, d) -> (L*d,)."""
    names = sorted(embeds.keys())
    X = np.stack([embeds[n].flatten() for n in names], axis=0)  # (N, L*d)
    pca = PCA(n_components=min(10, X.shape[0] - 1))
    pca.fit(X - X.mean(axis=0, keepdims=True))
    Z = pca.transform(X - X.mean(axis=0, keepdims=True))
    return names, Z, pca.explained_variance_ratio_


def pca_per_token(embeds):
    """One point per (role, token) pair: each token is a d-dim vector."""
    names = sorted(embeds.keys())
    L = next(iter(embeds.values())).shape[0]
    records = []
    for n in names:
        for t in range(L):
            records.append((n, t, embeds[n][t]))
    X = np.stack([r[2] for r in records], axis=0)  # (N*L, d)
    pca = PCA(n_components=min(10, X.shape[0] - 1))
    pca.fit(X - X.mean(axis=0, keepdims=True))
    Z = pca.transform(X - X.mean(axis=0, keepdims=True))
    meta = [(r[0], r[1]) for r in records]
    return meta, Z, pca.explained_variance_ratio_


# ── Plotting ────────────────────────────────────────────────────────────

CATEGORY_COLOR = {
    "round-1": "#d62728",
    "fantasy/archetype": "#9467bd",
    "profession": "#1f77b4",
    "style/register": "#2ca02c",
    "other": "#7f7f7f",
}


def plot_flattened(names, Z, evr, out_dir):
    cats = [category_of(n) for n in names]
    fig = px.scatter(
        x=Z[:, 0], y=Z[:, 1],
        color=cats,
        color_discrete_map=CATEGORY_COLOR,
        hover_name=names,
        labels={
            "x": f"PC1 ({evr[0]*100:.1f}% var)",
            "y": f"PC2 ({evr[1]*100:.1f}% var)",
            "color": "Category",
        },
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(
        width=900, height=700,
        title=f"Flattened PCA — {len(names)} role CSPs "
              f"(PC1 {evr[0]*100:.1f}% · PC2 {evr[1]*100:.1f}% · "
              f"cumulative {sum(evr[:2])*100:.1f}%)",
    )
    html_path = out_dir / "pca_flattened_roles.html"
    fig.write_html(str(html_path))
    print(f"  saved {html_path}")

    # Matplotlib PNG fallback
    fig2, ax = plt.subplots(figsize=(10, 7))
    for cat in CATEGORY_COLOR:
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        ax.scatter(Z[idx, 0], Z[idx, 1], s=80, alpha=0.8, label=cat,
                   color=CATEGORY_COLOR[cat], edgecolors="white", linewidth=1)
        for i in idx:
            ax.annotate(names[i], (Z[i, 0], Z[i, 1]), fontsize=7,
                        xytext=(6, 2), textcoords="offset points")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.set_title(f"Flattened PCA — {len(names)} role CSPs "
                 f"(cumulative {sum(evr[:2])*100:.1f}%)")
    ax.legend()
    ax.grid(alpha=0.3)
    png_path = out_dir / "pca_flattened_roles.png"
    fig2.tight_layout()
    fig2.savefig(str(png_path), dpi=150)
    plt.close(fig2)
    print(f"  saved {png_path}")


def plot_per_token(meta, Z, evr, out_dir):
    roles = [m[0] for m in meta]
    tokens = [m[1] + 1 for m in meta]  # 1-indexed for hover
    cats = [category_of(r) for r in roles]
    labels = [f"{r} (t{t})" for r, t in zip(roles, tokens)]
    fig = px.scatter(
        x=Z[:, 0], y=Z[:, 1],
        color=cats,
        color_discrete_map=CATEGORY_COLOR,
        hover_name=labels,
        symbol=[str(t) for t in tokens],
        labels={
            "x": f"PC1 ({evr[0]*100:.1f}% var)",
            "y": f"PC2 ({evr[1]*100:.1f}% var)",
            "color": "Category",
            "symbol": "Token",
        },
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=1, color="white")))
    fig.update_layout(
        width=900, height=700,
        title=f"Per-token PCA — {len(roles)} (role, token) points "
              f"(PC1 {evr[0]*100:.1f}% · PC2 {evr[1]*100:.1f}% · "
              f"cumulative {sum(evr[:2])*100:.1f}%)",
    )
    html_path = out_dir / "pca_pertoken_roles.html"
    fig.write_html(str(html_path))
    print(f"  saved {html_path}")

    # Matplotlib PNG fallback — use token number as marker
    fig2, ax = plt.subplots(figsize=(10, 7))
    markers = {1: "o", 2: "s", 3: "^", 4: "D"}
    for cat in CATEGORY_COLOR:
        for tok in [1, 2, 3, 4]:
            idx = [i for i, (c, t) in enumerate(zip(cats, tokens))
                   if c == cat and t == tok]
            if not idx:
                continue
            label = f"{cat} (t{tok})" if tok == 1 else None
            ax.scatter(Z[idx, 0], Z[idx, 1], s=60, alpha=0.7,
                       label=f"{cat}" if tok == 1 else None,
                       color=CATEGORY_COLOR[cat],
                       marker=markers[tok],
                       edgecolors="white", linewidth=0.8)
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
    ax.set_title(f"Per-token PCA — {len(roles)} (role, token) points "
                 f"(cumulative {sum(evr[:2])*100:.1f}%)")
    ax.legend(loc="best", title="Category (token position by shape)")
    ax.grid(alpha=0.3)
    png_path = out_dir / "pca_pertoken_roles.png"
    fig2.tight_layout()
    fig2.savefig(str(png_path), dpi=150)
    plt.close(fig2)
    print(f"  saved {png_path}")


def plot_scree(evr_flat, evr_token, out_dir):
    """Scree plot (explained variance vs PC index) for both flavors."""
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Flattened",
        x=[f"PC{i+1}" for i in range(len(evr_flat))],
        y=evr_flat,
    ))
    fig.add_trace(go.Bar(
        name="Per-token",
        x=[f"PC{i+1}" for i in range(len(evr_token))],
        y=evr_token,
    ))
    fig.update_layout(
        title="Scree — explained variance per PC (both PCA flavors)",
        xaxis_title="Principal Component",
        yaxis_title="Explained variance ratio",
        barmode="group",
        width=900, height=500,
    )
    html_path = out_dir / "pca_scree_roles.html"
    fig.write_html(str(html_path))
    print(f"  saved {html_path}")

    # Matplotlib PNG
    fig2, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(evr_flat))
    ax.bar(x - 0.2, evr_flat, width=0.4, label="Flattened",
           color="#1f77b4", alpha=0.8)
    ax.bar(x + 0.2, evr_token[:len(evr_flat)], width=0.4, label="Per-token",
           color="#ff7f0e", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"PC{i+1}" for i in x])
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("Scree — both PCA flavors")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    png_path = out_dir / "pca_scree_roles.png"
    fig2.tight_layout()
    fig2.savefig(str(png_path), dpi=150)
    plt.close(fig2)
    print(f"  saved {png_path}")


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "results" / "pca"))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading role CSPs from {args.results_dir}...")
    embeds = load_role_csps(Path(args.results_dir))
    print(f"  loaded {len(embeds)} roles, shape per CSP: "
          f"{next(iter(embeds.values())).shape}")

    print("\n1. Flattened PCA (one point per role)")
    names, Z_flat, evr_flat = pca_flattened(embeds)
    print(f"   shape: {Z_flat.shape}")
    print(f"   explained variance: "
          f"PC1 {evr_flat[0]*100:.2f}% · PC2 {evr_flat[1]*100:.2f}% · "
          f"PC3 {evr_flat[2]*100:.2f}% · cumulative (top 5) "
          f"{sum(evr_flat[:5])*100:.2f}%")
    plot_flattened(names, Z_flat, evr_flat, out_dir)

    print("\n2. Per-token PCA (one point per (role, token) pair)")
    meta, Z_token, evr_token = pca_per_token(embeds)
    print(f"   shape: {Z_token.shape}")
    print(f"   explained variance: "
          f"PC1 {evr_token[0]*100:.2f}% · PC2 {evr_token[1]*100:.2f}% · "
          f"PC3 {evr_token[2]*100:.2f}% · cumulative (top 5) "
          f"{sum(evr_token[:5])*100:.2f}%")
    plot_per_token(meta, Z_token, evr_token, out_dir)

    print("\n3. Scree plot (both flavors side-by-side)")
    plot_scree(evr_flat, evr_token, out_dir)

    # Save the projected coordinates + variance ratios for later analysis
    summary = {
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
    summary_path = out_dir / "pca_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved full summary to {summary_path}")


if __name__ == "__main__":
    main()
