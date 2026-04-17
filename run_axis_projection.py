"""Project role CSP L17 activations onto Lu et al.'s assistant axis.

Downloads the Butanium Gemma 3 4B IT assistant axis, runs each role
CSP through the model to capture L17 activations at the SP positions,
averages across eval prompts and SP tokens, and projects onto the
assistant axis. Then plots:

    x-axis:  our flattened-PCA PC1 (from run_pca.py's output)
    y-axis:  assistant-axis projection
    color:   FE (training fraction-of-baseline-KL explained)

If the two axes align, the scatter should cluster along the diagonal
and FE should track smoothly along it — the CSP population's dominant
embedding-space direction would recapitulate Lu et al.'s activation-
space axis.

Usage:
    python run_axis_projection.py
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

import config
from soft_prompt import SoftPrompt
from train import load_questions
from evaluate import (
    build_csp_input, capture_layer_activations, load_csp,
)
from run_pca import ROLE_NAMES, ROLE_CATEGORIES, category_of

SCRIPT_DIR = Path(__file__).parent.resolve()

# Hardcoded from NARRATIVE.md Preview table (we can't recompute FE
# without retraining; these are the final values from the sweep).
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


def download_axis():
    """Download Butanium's Gemma 3 4B IT assistant axis (Layer 17)."""
    print("Downloading Gemma 3 4B IT assistant axis from Butanium...")
    axis_path = hf_hub_download(
        repo_id="Butanium/gemma-3-4b-it-assistant-axis",
        filename="assistant_axis.pt",
        repo_type="dataset",
    )
    axis = torch.load(axis_path, weights_only=True, map_location="cpu")
    print(f"  axis shape: {tuple(axis.shape)}  (n_layers, hidden_dim)")
    return axis


def get_csp_l17(model, tokenizer, sp, prompts, layer_idx, eval_frame, device):
    """Mean L17 activation over prompts × SP token positions."""
    embed_fn = model.get_input_embeddings()
    L = sp.embedding.shape[0]
    acts = []
    for prompt in prompts:
        suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
        user = f"{prompt} {suffix}"
        combined, sp_pos, _ = build_csp_input(
            tokenizer, embed_fn, sp, user, device,
        )
        layer_act = capture_layer_activations(
            model, layer_idx, lambda: model(inputs_embeds=combined),
        )
        acts.append(layer_act[0, sp_pos:sp_pos + L, :].mean(dim=0))
    return torch.stack(acts).mean(dim=0)  # (hidden_dim,)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=str(SCRIPT_DIR / "results"))
    parser.add_argument("--out-dir", default=str(SCRIPT_DIR / "results" / "pca"))
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--layer", type=int, default=17)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)

    axis_all = download_axis()
    axis_l = axis_all[args.layer].float()  # (hidden,)
    axis_l_unit = axis_l / axis_l.norm()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nLoading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    questions = load_questions()
    prompts = questions[:args.n_prompts]

    print(f"\nProjecting {len(ROLE_NAMES)} role CSP L17 activations "
          f"onto the assistant axis...")
    projections = {}
    for role in ROLE_NAMES:
        ckpt = Path(args.results_dir) / role / "sp_pos.pt"
        if not ckpt.exists():
            print(f"  WARN missing {role}")
            continue
        sp, _ = SoftPrompt.from_checkpoint(str(ckpt), device=device)
        l17 = get_csp_l17(
            model, tokenizer, sp, prompts, args.layer,
            config.POSITIVE_FRAMES[0], device,
        )
        # Project onto axis (cosine similarity x norm = full projection,
        # but we use cosine to make values comparable across personas)
        l17_cpu = l17.detach().float().cpu()
        cos_proj = float(
            (l17_cpu @ axis_l_unit)
            / (l17_cpu.norm() + 1e-8)
        )
        scalar_proj = float(l17_cpu @ axis_l_unit)
        projections[role] = {"cos": cos_proj, "scalar": scalar_proj,
                             "norm": float(l17_cpu.norm())}
        print(f"  {role:14s}  cos={cos_proj:+.3f}  scalar={scalar_proj:+8.2f}")

    # Load flattened PCA results
    with open(out_dir / "pca_summary.json") as f:
        pca = json.load(f)
    name_to_pc1 = {n: z[0] for n, z in zip(pca["flattened"]["names"],
                                             pca["flattened"]["Z"])}
    name_to_pc2 = {n: z[1] for n, z in zip(pca["flattened"]["names"],
                                             pca["flattened"]["Z"])}

    # Build the plot dataframe
    names = [n for n in ROLE_NAMES if n in projections and n in name_to_pc1]
    pc1 = np.array([name_to_pc1[n] for n in names])
    pc2 = np.array([name_to_pc2[n] for n in names])
    cos_proj = np.array([projections[n]["cos"] for n in names])
    scalar_proj = np.array([projections[n]["scalar"] for n in names])
    fe_vals = np.array([FE[n] for n in names])

    # Correlations with FE
    def spearman(x, y):
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        n = len(x)
        d2 = ((rx - ry) ** 2).sum()
        return 1 - 6 * d2 / (n * (n ** 2 - 1))

    print()
    print(f"Correlations (n={len(names)}):")
    print(f"  PC1 vs FE (Spearman):              {spearman(pc1, fe_vals):+.3f}")
    print(f"  axis-cos vs FE (Spearman):         {spearman(cos_proj, fe_vals):+.3f}")
    print(f"  axis-scalar vs FE (Spearman):      {spearman(scalar_proj, fe_vals):+.3f}")
    print(f"  PC1 vs axis-cos (Spearman):        {spearman(pc1, cos_proj):+.3f}")
    print(f"  PC1 vs axis-scalar (Spearman):     {spearman(pc1, scalar_proj):+.3f}")

    # Plot: PC1 (x) vs axis projection (y), colored by FE
    cats = [category_of(n) for n in names]
    fig = px.scatter(
        x=pc1, y=cos_proj,
        color=fe_vals,
        color_continuous_scale="viridis",
        hover_name=[f"{n} (FE={fe_vals[i]:.1f}%, cat={cats[i]})"
                    for i, n in enumerate(names)],
        labels={
            "x": "Flattened-PCA PC1 (embedding space)",
            "y": "Cosine with assistant axis at L17 (activation space)",
            "color": "FE (%)",
        },
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(
        width=900, height=700,
        title="Our PCA PC1 vs Lu et al. assistant axis — 33 role CSPs",
    )
    html = out_dir / "pca_vs_assistant_axis.html"
    fig.write_html(str(html))
    print(f"\n  saved {html}")

    # Matplotlib PNG
    fig2, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(pc1, cos_proj, c=fe_vals, cmap="viridis",
                    s=100, edgecolors="white", linewidth=1)
    for i, n in enumerate(names):
        ax.annotate(n, (pc1[i], cos_proj[i]), fontsize=7,
                    xytext=(6, 2), textcoords="offset points")
    cbar = fig2.colorbar(sc, ax=ax)
    cbar.set_label("FE (%)")
    ax.set_xlabel("Flattened-PCA PC1 (embedding space)")
    ax.set_ylabel("Cosine with assistant axis at L17 (activation space)")
    ax.set_title("Our PCA PC1 vs Lu et al. assistant axis\n"
                 f"(Spearman PC1 vs axis-cos: "
                 f"{spearman(pc1, cos_proj):+.3f})")
    ax.grid(alpha=0.3)
    png = out_dir / "pca_vs_assistant_axis.png"
    fig2.tight_layout()
    fig2.savefig(str(png), dpi=150)
    plt.close(fig2)
    print(f"  saved {png}")

    # Also save the flattened PCA plot again, but colored by FE this time
    fig = px.scatter(
        x=pc1, y=pc2,
        color=fe_vals,
        color_continuous_scale="viridis",
        hover_name=[f"{n} (FE={fe_vals[i]:.1f}%)" for i, n in enumerate(names)],
        labels={"x": "PC1", "y": "PC2", "color": "FE (%)"},
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(
        width=900, height=700,
        title="Flattened PCA PC1 vs PC2 — colored by FE",
    )
    html = out_dir / "pca_flattened_roles_by_fe.html"
    fig.write_html(str(html))
    print(f"  saved {html}")

    fig2, ax = plt.subplots(figsize=(10, 7))
    sc = ax.scatter(pc1, pc2, c=fe_vals, cmap="viridis",
                    s=100, edgecolors="white", linewidth=1)
    for i, n in enumerate(names):
        ax.annotate(n, (pc1[i], pc2[i]), fontsize=7,
                    xytext=(6, 2), textcoords="offset points")
    cbar = fig2.colorbar(sc, ax=ax)
    cbar.set_label("FE (%)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Flattened PCA PC1 vs PC2 — colored by FE")
    ax.grid(alpha=0.3)
    png = out_dir / "pca_flattened_roles_by_fe.png"
    fig2.tight_layout()
    fig2.savefig(str(png), dpi=150)
    plt.close(fig2)
    print(f"  saved {png}")

    # Save projection summary
    summary = {
        "names": names,
        "pc1": pc1.tolist(),
        "pc2": pc2.tolist(),
        "axis_cos": cos_proj.tolist(),
        "axis_scalar": scalar_proj.tolist(),
        "axis_layer": args.layer,
        "fe": fe_vals.tolist(),
        "spearman_pc1_fe": spearman(pc1, fe_vals),
        "spearman_axis_cos_fe": spearman(cos_proj, fe_vals),
        "spearman_axis_scalar_fe": spearman(scalar_proj, fe_vals),
        "spearman_pc1_axis_cos": spearman(pc1, cos_proj),
        "spearman_pc1_axis_scalar": spearman(pc1, scalar_proj),
    }
    summary_path = out_dir / "axis_projection_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {summary_path}")


if __name__ == "__main__":
    main()
