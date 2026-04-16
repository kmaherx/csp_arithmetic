"""Evaluate semantic vs mathematical scaling of a trained CSP.

Six conditions in total, three per route:

SEMANTIC (CSP_pos, intensifier inserted into frame):
    semantic-barely      — CSP_pos in "Be barely §." / "Act barely §." / ...
    semantic-baseline    — CSP_pos in "Be §." / ... (identical to pos-in-pos)
    semantic-extremely   — CSP_pos in "Be extremely §." / "Act extremely §." / ...

MATHEMATICAL (alpha * CSP_pos, plain frame):
    math-0.25            — 0.25 * CSP_pos in "Be §." / ...
    math-1.0             — 1.0  * CSP_pos in "Be §." / ... (= semantic-baseline)
    math-4.0             — 4.0  * CSP_pos in "Be §." / ...

For each condition:
    1. Self-verbalization (multi-frame + single-frame prompts, using the
       condition's frame pool — so "barely" verb prompts verbalize an
       intensifier-aware frame set).
    2. SAE decomposition vs persona ground-truth activations (L17).
    3. Behavioral generation (N held-out prompts, single frame = first
       frame in the condition's pool — e.g. "Be barely §.").

Outputs go to results/{persona}/eval_scaling/.

Usage:
    .venv/bin/python evaluate_scaling.py --persona pirate
    .venv/bin/python evaluate_scaling.py --persona pirate --mode self-verb
"""

import argparse
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

import config
from soft_prompt import SoftPrompt, scale_csp
from train import load_questions
from evaluate import (
    MULTI_FRAME_TEMPLATES,
    MAX_NEW_TOKENS_VERB,
    MAX_NEW_TOKENS_BEHAVIOR,
    N_BEHAVIOR_SAMPLES,
    N_EVAL_PROMPTS,
    build_csp_input,
    build_csp_input_multi,
    generate_greedy,
    capture_layer_activations,
    compute_recon_error,
    get_sae_features,
    jaccard,
    get_persona_activations_at_layer,
    load_csp,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# ── Conditions ──────────────────────────────────────────────────────────
# Each condition has a frame pool (4 frames) used for self-verb prompts.
# The single-frame used for behavior + SAE is the pool's first element.

def _baseline_csp(sp_pos):
    return sp_pos


# (label, csp_factory(sp_pos) -> SoftPrompt, frame_pool)
CONDITIONS = [
    ("semantic-barely",
     lambda sp: sp,
     config.POSITIVE_FRAMES_BARELY),
    ("semantic-baseline",
     lambda sp: sp,
     config.POSITIVE_FRAMES),
    ("semantic-extremely",
     lambda sp: sp,
     config.POSITIVE_FRAMES_EXTREMELY),
    ("math-0.25",
     lambda sp: scale_csp(sp, 0.25),
     config.POSITIVE_FRAMES),
    ("math-1.0",
     lambda sp: scale_csp(sp, 1.0),
     config.POSITIVE_FRAMES),
    ("math-4.0",
     lambda sp: scale_csp(sp, 4.0),
     config.POSITIVE_FRAMES),
    ("math-5.0",
     lambda sp: scale_csp(sp, 5.0),
     config.POSITIVE_FRAMES),
    ("math-10.0",
     lambda sp: scale_csp(sp, 10.0),
     config.POSITIVE_FRAMES),
]


# ── Verbalization prompts (condition-aware) ─────────────────────────────

def multi_frame_prompts_for(frame_pool):
    joined = " ".join(f.format(sp=config.SP_PLACEHOLDER) for f in frame_pool)
    return [t.format(frames=joined) for t in MULTI_FRAME_TEMPLATES]


def single_frame_prompts_for(frame_pool):
    return [
        f"In plain English, explain this command: {f.format(sp=config.SP_PLACEHOLDER)}"
        for f in frame_pool
    ]


def verb_prompts_for(frame_pool):
    return (
        [("multi_frame", p) for p in multi_frame_prompts_for(frame_pool)]
        + [("single_frame", p) for p in single_frame_prompts_for(frame_pool)]
    )


# ── Activations at L17 with CSP in a given frame ────────────────────────

def get_csp_activations(model, tokenizer, sp, prompts, layer_idx, eval_frame, device):
    """Run student forward with CSP in eval_frame across prompts; mean L17
    activations at the CSP token positions."""
    embed_fn = model.get_input_embeddings()
    L = sp.embedding.shape[0]
    acts = []
    for prompt in prompts:
        suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
        user = f"{prompt} {suffix}"
        combined, sp_pos, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
        layer_act = capture_layer_activations(
            model, layer_idx, lambda: model(inputs_embeds=combined),
        )
        acts.append(layer_act[0, sp_pos:sp_pos + L, :].mean(dim=0))
    return torch.stack(acts)


# ── Modes ───────────────────────────────────────────────────────────────

def run_self_verb(model, tokenizer, sp_pos, device, eval_dir):
    embed_fn = model.get_input_embeddings()
    out = {}
    for label, factory, frame_pool in CONDITIONS:
        sp = factory(sp_pos)
        prompts = verb_prompts_for(frame_pool)
        cond_results = []
        print(f"\n  --- {label} (frames={frame_pool[0]}, ‖sp‖={sp.embedding.detach().flatten().float().norm().item():.2f}) ---")
        for approach, vp in prompts:
            combined = build_csp_input_multi(tokenizer, embed_fn, sp, vp, device)
            with torch.no_grad():
                resp = generate_greedy(model, tokenizer, inputs_embeds=combined)
            print(f"    [{approach}] Q: {vp[:80]}")
            print(f"              A: {resp[:120]}")
            cond_results.append({"approach": approach, "prompt": vp, "response": resp})
        out[label] = cond_results
    path = os.path.join(eval_dir, "self_verb.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved self-verb to {path}")
    return out


def run_sae(model, tokenizer, sp_pos, persona_prompts, prompts, device, eval_dir):
    print(f"\nLoading SAE: {config.SAE_ID} from {config.SAE_RELEASE}...")
    sae = SAE.from_pretrained(release=config.SAE_RELEASE, sae_id=config.SAE_ID)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(model.device)

    print("Computing persona ground-truth activations at L17...")
    persona_acts = get_persona_activations_at_layer(
        model, tokenizer, persona_prompts, prompts, config.SAE_LAYER, device,
    )
    persona_recon = compute_recon_error(sae, persona_acts)
    persona_active, persona_topk, persona_act_dict = get_sae_features(sae, persona_acts)
    persona_topk_ranked = sorted(
        persona_topk, key=lambda x: persona_act_dict.get(x, 0), reverse=True,
    )
    print(f"  persona: rel_err={persona_recon['rel_err']:.4f}, "
          f"cos={persona_recon['cos_sim']:.4f}, n_active={len(persona_active)}")
    print(f"  persona top-20: {persona_topk_ranked}")

    out = {
        "persona": {
            "recon": persona_recon,
            "n_active": len(persona_active),
            "topk_features": persona_topk_ranked,
        },
        "conditions": {},
    }

    for label, factory, frame_pool in CONDITIONS:
        sp = factory(sp_pos)
        eval_frame = frame_pool[0]  # first frame in the pool (e.g. "Be barely §.")
        sp_acts = get_csp_activations(
            model, tokenizer, sp, prompts, config.SAE_LAYER, eval_frame, device,
        )
        recon = compute_recon_error(sae, sp_acts)
        active, topk, act_dict = get_sae_features(sae, sp_acts)
        jac_active = jaccard(active, persona_active)
        jac_topk = jaccard(topk, persona_topk)
        shared_active = sorted(
            active & persona_active,
            key=lambda x: act_dict.get(x, 0), reverse=True,
        )
        topk_ranked = sorted(topk, key=lambda x: act_dict.get(x, 0), reverse=True)
        print(f"  {label}: rel_err={recon['rel_err']:.4f}, cos={recon['cos_sim']:.4f}, "
              f"n_active={len(active)}, jac_active={jac_active:.3f}, jac_topk={jac_topk:.3f}")
        print(f"    top-20: {topk_ranked}")
        print(f"    shared with persona (top): {shared_active[:10]}")
        out["conditions"][label] = {
            "eval_frame": eval_frame,
            "recon": recon,
            "n_active": len(active),
            "topk_features": topk_ranked,
            "jaccard_active": jac_active,
            "jaccard_topk": jac_topk,
            "shared_features": shared_active[:50],
            "feature_activations": {
                int(idx): float(act_dict[idx]) for idx in topk_ranked
            },
        }

    path = os.path.join(eval_dir, "sae.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved SAE results to {path}")
    return out


def run_behavior(model, tokenizer, sp_pos, prompts, device, eval_dir):
    embed_fn = model.get_input_embeddings()
    out = {}
    sample_prompts = prompts[:N_BEHAVIOR_SAMPLES]
    for label, factory, frame_pool in CONDITIONS:
        sp = factory(sp_pos)
        eval_frame = frame_pool[0]
        suffix = eval_frame.format(sp=config.SP_PLACEHOLDER)
        cond = []
        norm = sp.embedding.detach().flatten().float().norm().item()
        print(f"\n  --- {label} (frame='{eval_frame}', ‖sp‖={norm:.2f}) ---")
        for prompt in sample_prompts:
            user = f"{prompt} {suffix}"
            combined, _, _ = build_csp_input(tokenizer, embed_fn, sp, user, device)
            with torch.no_grad():
                resp = generate_greedy(
                    model, tokenizer, inputs_embeds=combined,
                    max_new_tokens=MAX_NEW_TOKENS_BEHAVIOR,
                )
            print(f"    Q: {prompt[:80]}")
            print(f"    A: {resp[:160]}")
            cond.append({"prompt": prompt, "response": resp})
        out[label] = cond
    path = os.path.join(eval_dir, "behavior.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved behavior samples to {path}")
    return out


def embedding_summary(sp_pos, eval_dir):
    """Per-condition norms + cosine of scaled CSP to unscaled baseline.

    cos(alpha * x, x) = sign(alpha) for any alpha != 0, so this is a
    trivial sanity check. Norms are the informative number: alpha * ‖sp‖.
    """
    base_flat = sp_pos.embedding.detach().flatten().float()
    base_norm = base_flat.norm().item()
    out = {"base_norm": base_norm, "conditions": {}}
    for label, factory, frame_pool in CONDITIONS:
        sp = factory(sp_pos)
        flat = sp.embedding.detach().flatten().float()
        out["conditions"][label] = {
            "eval_frame": frame_pool[0],
            "norm": flat.norm().item(),
            "norm_ratio": flat.norm().item() / base_norm,
        }
    print("\nEmbedding summary:")
    for label, d in out["conditions"].items():
        print(f"  {label}: ‖sp‖={d['norm']:.2f} ({d['norm_ratio']:.2f}× baseline)")
    path = os.path.join(eval_dir, "embedding_compare.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved embedding summary to {path}")
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--persona", required=True, choices=list(config.PERSONAS.keys()))
    parser.add_argument("--mode", default="all",
                        choices=["all", "self-verb", "sae", "behavior", "embedding"])
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--questions", default=None)
    parser.add_argument("--n-eval-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    persona_dir = os.path.join(args.results_dir, args.persona)
    eval_dir = os.path.join(persona_dir, "eval_scaling")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    persona_prompts = config.PERSONAS[args.persona]

    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    print(f"Loading sp_pos from {persona_dir}...")
    sp_pos, _ = load_csp(persona_dir, "pos", device)
    base_norm = sp_pos.embedding.detach().flatten().float().norm().item()
    print(f"  sp_pos: shape={tuple(sp_pos.embedding.shape)}, ‖·‖={base_norm:.2f}")

    questions = load_questions(args.questions)
    eval_prompts = questions[:args.n_eval_prompts]

    if args.mode in ("all", "embedding"):
        print(f"\n{'='*60}\n  EMBEDDING SUMMARY\n{'='*60}")
        embedding_summary(sp_pos, eval_dir)

    if args.mode in ("all", "self-verb"):
        print(f"\n{'='*60}\n  SELF-VERBALIZATION\n{'='*60}")
        run_self_verb(model, tokenizer, sp_pos, device, eval_dir)

    if args.mode in ("all", "behavior"):
        print(f"\n{'='*60}\n  BEHAVIOR SAMPLES\n{'='*60}")
        run_behavior(model, tokenizer, sp_pos, eval_prompts, device, eval_dir)

    if args.mode in ("all", "sae"):
        print(f"\n{'='*60}\n  SAE DECOMPOSITION (L{config.SAE_LAYER})\n{'='*60}")
        run_sae(model, tokenizer, sp_pos, persona_prompts, eval_prompts, device, eval_dir)


if __name__ == "__main__":
    main()
