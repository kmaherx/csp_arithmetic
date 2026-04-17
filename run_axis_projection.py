"""Project CSP L17 activations onto Lu et al.'s assistant axis.

Downloads the Butanium Gemma 3 4B IT assistant axis, runs each CSP
through the model to capture L17 activations at the SP positions,
averages across eval prompts and SP tokens, and projects onto the
axis. Plots:

    x-axis:  our flattened-PCA PC1 (from run_pca.py's summary file)
    y-axis:  assistant-axis projection (cosine and/or scalar)
    color:   FE (training fraction-of-baseline-KL explained)

Usage:
    python run_axis_projection.py --persona-set roles
    python run_axis_projection.py --persona-set traits
    python run_axis_projection.py --persona-set joint
"""

import argparse
import json
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
from evaluate import build_csp_input, capture_layer_activations
import persona_sets as P

SCRIPT_DIR = Path(__file__).parent.resolve()


def download_axis():
    print("Downloading Gemma 3 4B IT assistant axis from Butanium...")
    axis_path = hf_hub_download(
        repo_id="Butanium/gemma-3-4b-it-assistant-axis",
        filename="assistant_axis.pt",
        repo_type="dataset",
    )
    axis = torch.load(axis_path, weights_only=True, map_location="cpu")
    print(f"  axis shape: {tuple(axis.shape)}")
    return axis


def get_csp_l17(model, tokenizer, sp, prompts, layer_idx, eval_frame, device):
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
    return torch.stack(acts).mean(dim=0)


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
    parser.add_argument("--n-prompts", type=int, default=30)
    parser.add_argument("--layer", type=int, default=17)
    parser.add_argument("--cache-path", default=None,
                        help="Cache L17 projections per persona to this JSON "
                             "(reuse across --persona-set runs).")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(config.SEED)
    random.seed(config.SEED)

    cache_path = (Path(args.cache_path) if args.cache_path
                  else out_dir / "axis_projection_cache.json")
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        print(f"Loaded cache from {cache_path} with {len(cache)} personas")

    axis_all = download_axis()
    axis_l = axis_all[args.layer].float()
    axis_l_unit = axis_l / axis_l.norm()

    # Only run the model if there are uncached personas we need
    names_req = P.get_names(args.persona_set)
    need_forward = [n for n in names_req if n not in cache]

    if need_forward:
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

        print(f"\nProjecting {len(need_forward)} CSPs onto assistant axis...")
        for n in need_forward:
            ckpt = Path(args.results_dir) / n / "sp_pos.pt"
            if not ckpt.exists():
                print(f"  WARN missing {n}")
                continue
            sp, _ = SoftPrompt.from_checkpoint(str(ckpt), device=device)
            l17 = get_csp_l17(
                model, tokenizer, sp, prompts, args.layer,
                config.POSITIVE_FRAMES[0], device,
            )
            l17_cpu = l17.detach().float().cpu()
            cos_proj = float(
                (l17_cpu @ axis_l_unit) / (l17_cpu.norm() + 1e-8)
            )
            scalar_proj = float(l17_cpu @ axis_l_unit)
            norm_val = float(l17_cpu.norm())
            cache[n] = {
                "cos": cos_proj, "scalar": scalar_proj, "norm": norm_val,
            }
            print(f"  {n:18s}  cos={cos_proj:+.3f}  "
                  f"scalar={scalar_proj:+9.2f}  norm={norm_val:.2f}")

        with open(cache_path, "w") as f:
            json.dump(cache, f, indent=2)
        print(f"\nCache saved to {cache_path}")
    else:
        print("\nAll personas already cached; skipping model forward passes.")

    # Load flattened PCA for this persona-set
    pca_summary_path = out_dir / f"pca_summary_{args.persona_set}.json"
    if not pca_summary_path.exists():
        print(f"\nFlattened PCA summary not found at {pca_summary_path}. "
              f"Run run_pca.py --persona-set {args.persona_set} first.")
        return
    with open(pca_summary_path) as f:
        pca = json.load(f)
    name_to_pc1 = dict(zip(pca["flattened"]["names"],
                           [z[0] for z in pca["flattened"]["Z"]]))
    name_to_pc2 = dict(zip(pca["flattened"]["names"],
                           [z[1] for z in pca["flattened"]["Z"]]))

    names = [n for n in names_req if n in cache and n in name_to_pc1]
    pc1 = np.array([name_to_pc1[n] for n in names])
    pc2 = np.array([name_to_pc2[n] for n in names])
    cos_proj = np.array([cache[n]["cos"] for n in names])
    scalar_proj = np.array([cache[n]["scalar"] for n in names])
    fe_vals = np.array([P.ALL_FE.get(n, np.nan) for n in names])
    has_fe = ~np.isnan(fe_vals)

    print(f"\nCorrelations (n={len(names)}, Spearman):")
    if has_fe.any():
        print(f"  PC1 vs FE:             {spearman(pc1[has_fe], fe_vals[has_fe]):+.3f}")
        print(f"  axis-cos vs FE:        {spearman(cos_proj[has_fe], fe_vals[has_fe]):+.3f}")
        print(f"  axis-scalar vs FE:     {spearman(scalar_proj[has_fe], fe_vals[has_fe]):+.3f}")
    print(f"  PC1 vs axis-cos:       {spearman(pc1, cos_proj):+.3f}")
    print(f"  PC1 vs axis-scalar:    {spearman(pc1, scalar_proj):+.3f}")

    # Plot: PC1 (x) vs axis-cos (y), colored by FE (or by kind if no FE)
    hover = [
        f"{n} ({P.kind_of(n)}, FE={fe_vals[i]:.1f}%)"
        if not np.isnan(fe_vals[i]) else n
        for i, n in enumerate(names)
    ]
    fig = px.scatter(
        x=pc1, y=cos_proj,
        color=fe_vals if has_fe.all() else [P.kind_of(n) for n in names],
        color_continuous_scale="viridis" if has_fe.all() else None,
        hover_name=hover,
        labels={
            "x": "Flattened-PCA PC1 (embedding space)",
            "y": "Cosine with assistant axis at L17",
            "color": "FE (%)" if has_fe.all() else "Kind",
        },
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    rho_pc1_axis = spearman(pc1, cos_proj)
    fig.update_layout(
        width=1000, height=700,
        title=f"Our PC1 vs Lu et al. assistant axis — {args.persona_set} "
              f"(n={len(names)}, Spearman {rho_pc1_axis:+.3f})",
    )
    html = out_dir / f"pca_vs_assistant_axis_{args.persona_set}.html"
    fig.write_html(str(html))
    print(f"\n  saved {html}")

    fig2, ax = plt.subplots(figsize=(10, 7))
    if has_fe.all():
        sc = ax.scatter(pc1, cos_proj, c=fe_vals, cmap="viridis",
                        s=100, edgecolors="white", linewidth=1)
        cbar = fig2.colorbar(sc, ax=ax); cbar.set_label("FE (%)")
    else:
        kinds = [P.kind_of(n) for n in names]
        colors = {"role": "#1f77b4", "trait": "#d62728"}
        for k, c in colors.items():
            idx = [i for i, kk in enumerate(kinds) if kk == k]
            if idx:
                ax.scatter(pc1[idx], cos_proj[idx], s=100, label=k,
                           color=c, edgecolors="white", linewidth=1)
        ax.legend()
    for i, n in enumerate(names):
        ax.annotate(n, (pc1[i], cos_proj[i]), fontsize=6,
                    xytext=(6, 2), textcoords="offset points")
    ax.set_xlabel("Flattened-PCA PC1")
    ax.set_ylabel("Cosine with assistant axis at L17")
    ax.set_title(f"PCA PC1 vs Lu et al. assistant axis — "
                 f"{args.persona_set}\n"
                 f"(Spearman {rho_pc1_axis:+.3f})")
    ax.grid(alpha=0.3)
    png = out_dir / f"pca_vs_assistant_axis_{args.persona_set}.png"
    fig2.tight_layout(); fig2.savefig(str(png), dpi=150)
    plt.close(fig2)
    print(f"  saved {png}")

    # Also: PC1 vs PC2 colored by FE (or by kind)
    fig = px.scatter(
        x=pc1, y=pc2,
        color=fe_vals if has_fe.all() else [P.kind_of(n) for n in names],
        color_continuous_scale="viridis" if has_fe.all() else None,
        hover_name=hover,
        labels={"x": "PC1", "y": "PC2",
                "color": "FE (%)" if has_fe.all() else "Kind"},
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    fig.update_layout(
        width=1000, height=700,
        title=f"Flattened PCA PC1 vs PC2 — {args.persona_set} (colored by "
              f"{'FE' if has_fe.all() else 'kind'})",
    )
    html = out_dir / f"pca_flattened_{args.persona_set}_by_fe.html"
    fig.write_html(str(html))
    print(f"  saved {html}")

    fig2, ax = plt.subplots(figsize=(10, 7))
    if has_fe.all():
        sc = ax.scatter(pc1, pc2, c=fe_vals, cmap="viridis",
                        s=100, edgecolors="white", linewidth=1)
        cbar = fig2.colorbar(sc, ax=ax); cbar.set_label("FE (%)")
    else:
        kinds = [P.kind_of(n) for n in names]
        colors = {"role": "#1f77b4", "trait": "#d62728"}
        for k, c in colors.items():
            idx = [i for i, kk in enumerate(kinds) if kk == k]
            if idx:
                ax.scatter(pc1[idx], pc2[idx], s=100, label=k,
                           color=c, edgecolors="white", linewidth=1)
        ax.legend()
    for i, n in enumerate(names):
        ax.annotate(n, (pc1[i], pc2[i]), fontsize=6,
                    xytext=(6, 2), textcoords="offset points")
    ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.set_title(f"Flattened PCA PC1 vs PC2 — {args.persona_set}")
    ax.grid(alpha=0.3)
    png = out_dir / f"pca_flattened_{args.persona_set}_by_fe.png"
    fig2.tight_layout(); fig2.savefig(str(png), dpi=150)
    plt.close(fig2)
    print(f"  saved {png}")

    summary = {
        "persona_set": args.persona_set,
        "names": names,
        "pc1": pc1.tolist(), "pc2": pc2.tolist(),
        "axis_cos": cos_proj.tolist(),
        "axis_scalar": scalar_proj.tolist(),
        "axis_layer": args.layer,
        "fe": fe_vals.tolist(),
        "spearman": {
            "pc1_fe": (float(spearman(pc1[has_fe], fe_vals[has_fe]))
                       if has_fe.any() else None),
            "axis_cos_fe": (float(spearman(cos_proj[has_fe], fe_vals[has_fe]))
                            if has_fe.any() else None),
            "axis_scalar_fe": (float(spearman(scalar_proj[has_fe], fe_vals[has_fe]))
                               if has_fe.any() else None),
            "pc1_axis_cos": float(spearman(pc1, cos_proj)),
            "pc1_axis_scalar": float(spearman(pc1, scalar_proj)),
        },
    }
    summary_path = out_dir / f"axis_projection_summary_{args.persona_set}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  saved {summary_path}")


if __name__ == "__main__":
    main()
