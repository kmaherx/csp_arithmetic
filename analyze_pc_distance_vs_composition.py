"""Compute joint-PC distances between each composition pair and
correlate with composition quality metrics (jac_combined etc).

Uses the joint flattened PCA coordinates (PC1-PC5) from
results/pca/pca_summary_joint.json. For each pair in
results/composition/, extract SAE jac_combined for syn-v1-AB,
syn-v1-BA, vec-sum, vec-mul. Plot distance vs each metric, report
correlations.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

SCRIPT_DIR = Path(__file__).parent.resolve()


def spearman(x, y):
    x = np.asarray(x); y = np.asarray(y)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    n = len(x)
    d2 = ((rx - ry) ** 2).sum()
    return 1 - 6 * d2 / (n * (n ** 2 - 1))


def pearson(x, y):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]; y = y[mask]
    if len(x) < 3:
        return float("nan")
    mx, my = x.mean(), y.mean()
    cov = ((x - mx) * (y - my)).sum()
    sx = ((x - mx) ** 2).sum() ** 0.5
    sy = ((y - my) ** 2).sum() ** 0.5
    return cov / (sx * sy + 1e-12)


def main():
    out_dir = SCRIPT_DIR / "results" / "pca"
    comp_dir = SCRIPT_DIR / "results" / "composition"

    # Load joint PCA
    with open(out_dir / "pca_summary_joint.json") as f:
        pca = json.load(f)
    names = pca["flattened"]["names"]
    Z = np.array(pca["flattened"]["Z"])  # (65, 5)
    name_to_z = dict(zip(names, Z))

    # Scan composition pairs
    rows = []
    for pair_dir in sorted(comp_dir.iterdir()):
        if not pair_dir.is_dir():
            continue
        sae_path = pair_dir / "eval" / "sae.json"
        if not sae_path.exists():
            continue
        pair = pair_dir.name
        try:
            a, b = pair.split("+")
        except ValueError:
            continue
        if a not in name_to_z or b not in name_to_z:
            print(f"  skip {pair} (missing PCA coord for {a} or {b})")
            continue

        za, zb = name_to_z[a], name_to_z[b]
        dist_full = float(np.linalg.norm(za - zb))
        dist_pc1 = float(abs(za[0] - zb[0]))
        dist_pc12 = float(np.linalg.norm(za[:2] - zb[:2]))
        dist_pc15 = float(np.linalg.norm(za[:5] - zb[:5]))
        # Cosine between the two CSP embeddings in the 10240-dim flattened
        # space, using the centered PC coordinates as a proxy? Actually
        # the raw CSP cosine is in the SAE's embedding_compare.json file —
        # if available.
        emb_path = pair_dir / "eval" / "embedding_compare.json"
        cos_ab = None
        norm_a = norm_b = None
        if emb_path.exists():
            try:
                with open(emb_path) as f:
                    emb = json.load(f)
                ind = emb.get("individual", {})
                cos_ab = ind.get("cos(A, B)")
                norm_a = ind.get("||A||")
                norm_b = ind.get("||B||")
            except Exception:
                pass

        with open(sae_path) as f:
            sae = json.load(f)
        conds = sae.get("conditions", {})

        def get_jc(cond):
            c = conds.get(cond, {})
            return c.get("jaccard_combined", float("nan"))

        row = {
            "pair": pair,
            "a": a, "b": b,
            "dist_pc1": dist_pc1,
            "dist_pc12": dist_pc12,
            "dist_pc15": dist_pc15,
            "cos_ab": cos_ab,
            "jc_syn_v1_AB": get_jc("syn-v1-AB"),
            "jc_syn_v1_BA": get_jc("syn-v1-BA"),
            "jc_syn_v2_AB": get_jc("syn-v2-AB"),
            "jc_syn_v2_BA": get_jc("syn-v2-BA"),
            "jc_syn_v3_AB": get_jc("syn-v3-AB"),
            "jc_syn_v3_BA": get_jc("syn-v3-BA"),
            "jc_syn_v4_AB": get_jc("syn-v4-AB"),
            "jc_syn_v4_BA": get_jc("syn-v4-BA"),
            "jc_vec_sum": get_jc("vec-sum"),
            "jc_vec_mul": get_jc("vec-mul"),
        }
        # Aggregate: mean of the 8 syntactic conditions
        syn_vals = [row[k] for k in row if k.startswith("jc_syn_")
                    and not np.isnan(row[k])]
        row["jc_syn_mean"] = float(np.mean(syn_vals)) if syn_vals else float("nan")
        rows.append(row)

    print(f"Loaded {len(rows)} composition pairs with valid data")

    # Correlations: distance vs each jac_combined metric
    dists_pc15 = np.array([r["dist_pc15"] for r in rows])
    dists_pc1 = np.array([r["dist_pc1"] for r in rows])
    jc_syn_mean = np.array([r["jc_syn_mean"] for r in rows])
    jc_vec_sum = np.array([r["jc_vec_sum"] for r in rows])
    jc_vec_mul = np.array([r["jc_vec_mul"] for r in rows])
    cos_ab_arr = np.array([r["cos_ab"] if r["cos_ab"] is not None else np.nan
                           for r in rows])

    print()
    print("Correlations (n=%d pairs):" % len(rows))
    print(f"{'distance metric':20s}{'syn_mean':>12s}{'vec_sum':>12s}"
          f"{'vec_mul':>12s}")
    for dist_name, dist in [("dist(PC1)", dists_pc1),
                             ("dist(PC1..5)", dists_pc15)]:
        line = f"{dist_name+'  Spearman':20s}"
        for metric, data in [("syn_mean", jc_syn_mean),
                              ("vec_sum", jc_vec_sum),
                              ("vec_mul", jc_vec_mul)]:
            line += f"{spearman(dist, data):+12.3f}"
        print(line)
        line = f"{dist_name+'  Pearson ':20s}"
        for metric, data in [("syn_mean", jc_syn_mean),
                              ("vec_sum", jc_vec_sum),
                              ("vec_mul", jc_vec_mul)]:
            line += f"{pearson(dist, data):+12.3f}"
        print(line)

    if np.isfinite(cos_ab_arr).any():
        print()
        line = f"{'cos(spA,spB)  Sp':20s}"
        for data in [jc_syn_mean, jc_vec_sum, jc_vec_mul]:
            line += f"{spearman(cos_ab_arr, data):+12.3f}"
        print(line)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data, title in zip(
        axs,
        [jc_syn_mean, jc_vec_sum, jc_vec_mul],
        ["syntactic mean jac_combined",
         "vec-sum jac_combined",
         "vec-mul jac_combined"],
    ):
        ax.scatter(dists_pc15, data, s=60, alpha=0.7,
                   edgecolors="white", linewidth=0.8)
        for r in rows:
            ax.annotate(r["pair"],
                        (r["dist_pc15"],
                         (jc_syn_mean if title.startswith("syntactic") else
                          jc_vec_sum if title.startswith("vec-sum") else
                          jc_vec_mul)[rows.index(r)]),
                        fontsize=6, xytext=(4, 2),
                        textcoords="offset points", alpha=0.7)
        ax.set_xlabel("Joint-PCA dist (PC1..5)")
        ax.set_ylabel(title)
        ax.set_title(f"{title}\nSpearman {spearman(dists_pc15, data):+.3f}")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    png = out_dir / "pc_distance_vs_composition.png"
    fig.savefig(str(png), dpi=150)
    plt.close(fig)
    print(f"\n  saved {png}")

    # Plotly interactive (one combined figure for the main question)
    xs = dists_pc15.tolist()
    trace_pairs = [r["pair"] for r in rows]
    fig = px.scatter(
        x=xs, y=jc_syn_mean,
        hover_name=trace_pairs,
        labels={"x": "Joint-PCA distance (PC1..5)",
                "y": "Syntactic mean jac_combined"},
        title=f"Syntactic composition quality vs joint-PC distance "
              f"(Spearman {spearman(dists_pc15, jc_syn_mean):+.3f})",
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    html = out_dir / "pc_distance_vs_syn_composition.html"
    fig.write_html(str(html))
    print(f"  saved {html}")

    fig = px.scatter(
        x=xs, y=jc_vec_sum,
        hover_name=trace_pairs,
        labels={"x": "Joint-PCA distance (PC1..5)",
                "y": "vec-sum jac_combined"},
        title=f"vec-sum quality vs joint-PC distance "
              f"(Spearman {spearman(dists_pc15, jc_vec_sum):+.3f})",
    )
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
    html = out_dir / "pc_distance_vs_vecsum.html"
    fig.write_html(str(html))
    print(f"  saved {html}")

    # Save summary
    summary = {
        "rows": rows,
        "correlations": {
            "spearman_dist_pc15_vs_syn_mean": spearman(dists_pc15, jc_syn_mean),
            "spearman_dist_pc15_vs_vec_sum": spearman(dists_pc15, jc_vec_sum),
            "spearman_dist_pc15_vs_vec_mul": spearman(dists_pc15, jc_vec_mul),
            "pearson_dist_pc15_vs_syn_mean": pearson(dists_pc15, jc_syn_mean),
            "pearson_dist_pc15_vs_vec_sum": pearson(dists_pc15, jc_vec_sum),
            "pearson_dist_pc15_vs_vec_mul": pearson(dists_pc15, jc_vec_mul),
        },
    }
    out = out_dir / "pc_distance_vs_composition_summary.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  saved {out}")


if __name__ == "__main__":
    main()
