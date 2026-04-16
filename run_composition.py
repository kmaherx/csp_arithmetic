"""Evaluate composed CSPs for a persona pair.

Runs 6 conditions per pair:
  4 syntactic: syn-v1-AB, syn-v1-BA, syn-v2-AB, syn-v2-BA
  2 vector:    vec-sum, vec-mul

Each condition evaluated on self-verb, behavior, SAE (at L17), plus an
embedding comparison between composites and their constituents.

Usage:
    python run_composition.py --pair pirate+poet
    python run_composition.py --pair pirate+poet --mode self-verb
"""

import argparse
import json
import os
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from sae_lens import SAE

import config
import compose
from soft_prompt import SoftPrompt
from train import load_questions
from evaluate import (
    MAX_NEW_TOKENS_VERB, MAX_NEW_TOKENS_BEHAVIOR,
    N_BEHAVIOR_SAMPLES, N_EVAL_PROMPTS,
    build_csp_input, build_csp_input_multi,
    generate_greedy,
    compute_recon_error, get_sae_features, jaccard,
    get_csp_activations_at_layer,
    verb_prompts,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_pair(pair_str):
    parts = pair_str.split("+")
    assert len(parts) == 2, f"Expected A+B, got {pair_str!r}"
    return parts[0], parts[1]


# ── Self-verbalization ──────────────────────────────────────────────────

def run_self_verb(model, tokenizer, csps, device, eval_dir):
    embed_fn = model.get_input_embeddings()
    out = {}

    for label, frames, ordering in compose.SYN_CONDITIONS:
        sp1 = csps[ordering[0]]
        sp2 = csps[ordering[1]]
        prompts = compose.syntactic_verb_prompts(frames)
        cond = []
        print(f"\n  --- {label} (sp1={ordering[0]}, sp2={ordering[1]}) ---")
        for approach, vp in prompts:
            combined = compose.build_csp_input_two_slot(
                tokenizer, embed_fn, sp1, sp2, vp, device,
            )
            with torch.no_grad():
                resp = generate_greedy(model, tokenizer, inputs_embeds=combined)
            print(f"    [{approach}] Q: {vp[:80]}")
            print(f"              A: {resp[:120]}")
            cond.append({"approach": approach, "prompt": vp, "response": resp})
        out[label] = cond

    for label, _fn in compose.VEC_CONDITIONS:
        composite = csps[label]
        prompts = verb_prompts("pos")
        cond = []
        print(f"\n  --- {label} ---")
        for approach, vp in prompts:
            combined = build_csp_input_multi(tokenizer, embed_fn, composite, vp, device)
            with torch.no_grad():
                resp = generate_greedy(model, tokenizer, inputs_embeds=combined)
            print(f"    [{approach}] Q: {vp[:80]}")
            print(f"              A: {resp[:120]}")
            cond.append({"approach": approach, "prompt": vp, "response": resp})
        out[label] = cond

    path = os.path.join(eval_dir, "self_verb.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved self-verb to {path}")
    return out


# ── Behavioral generation ───────────────────────────────────────────────

def run_behavior(model, tokenizer, csps, prompts, device, eval_dir):
    embed_fn = model.get_input_embeddings()
    out = {}
    samples = prompts[:N_BEHAVIOR_SAMPLES]

    for label, frames, ordering in compose.SYN_CONDITIONS:
        sp1 = csps[ordering[0]]
        sp2 = csps[ordering[1]]
        suffix = frames[0].format(
            sp1=config.SP_PLACEHOLDER, sp2=config.SP_PLACEHOLDER_2,
        )
        cond = []
        print(f"\n  --- {label} (frame='{frames[0]}') ---")
        for prompt in samples:
            user = f"{prompt} {suffix}"
            combined = compose.build_csp_input_two_slot(
                tokenizer, embed_fn, sp1, sp2, user, device,
            )
            with torch.no_grad():
                resp = generate_greedy(
                    model, tokenizer, inputs_embeds=combined,
                    max_new_tokens=MAX_NEW_TOKENS_BEHAVIOR,
                )
            print(f"    Q: {prompt[:80]}")
            print(f"    A: {resp[:160]}")
            cond.append({"prompt": prompt, "response": resp})
        out[label] = cond

    for label, _fn in compose.VEC_CONDITIONS:
        composite = csps[label]
        frame = config.POSITIVE_FRAMES[0]
        suffix = frame.format(sp=config.SP_PLACEHOLDER)
        cond = []
        print(f"\n  --- {label} (frame='{frame}') ---")
        for prompt in samples:
            user = f"{prompt} {suffix}"
            combined, _, _ = build_csp_input(tokenizer, embed_fn, composite, user, device)
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


# ── SAE decomposition ──────────────────────────────────────────────────

def _topk_ranked(topk, act_dict):
    return sorted(topk, key=lambda x: act_dict.get(x, 0), reverse=True)


def _shared_ranked(active, baseline, act_dict, k=20):
    return sorted(active & baseline, key=lambda x: act_dict.get(x, 0), reverse=True)[:k]


def run_sae(model, tokenizer, csps, persona_a_prompts, persona_b_prompts,
            prompts, device, eval_dir):
    print(f"\nLoading SAE: {config.SAE_ID} from {config.SAE_RELEASE}...")
    sae = SAE.from_pretrained(release=config.SAE_RELEASE, sae_id=config.SAE_ID)
    if isinstance(sae, tuple):
        sae = sae[0]
    sae = sae.to(model.device)

    print("Computing combined-teacher ground-truth activations at L17...")
    combined_acts = compose.get_combined_persona_activations_at_layer(
        model, tokenizer, persona_a_prompts, persona_b_prompts,
        prompts, config.SAE_LAYER, device,
    )
    combined_recon = compute_recon_error(sae, combined_acts)
    combined_active, combined_topk, combined_act_dict = get_sae_features(sae, combined_acts)
    combined_topk_r = _topk_ranked(combined_topk, combined_act_dict)
    print(f"  combined_teacher: rel_err={combined_recon['rel_err']:.4f}, "
          f"cos={combined_recon['cos_sim']:.4f}, n_active={len(combined_active)}")
    print(f"  top-20: {combined_topk_r}")

    sp_a = csps["A"]
    sp_b = csps["B"]
    print("Computing individual sp_A pos-in-pos activations...")
    sp_a_acts = get_csp_activations_at_layer(
        model, tokenizer, sp_a, prompts, config.SAE_LAYER,
        config.POSITIVE_FRAMES[0], device,
    )
    sp_a_active, sp_a_topk, sp_a_act_dict = get_sae_features(sae, sp_a_acts)
    print("Computing individual sp_B pos-in-pos activations...")
    sp_b_acts = get_csp_activations_at_layer(
        model, tokenizer, sp_b, prompts, config.SAE_LAYER,
        config.POSITIVE_FRAMES[0], device,
    )
    sp_b_active, sp_b_topk, sp_b_act_dict = get_sae_features(sae, sp_b_acts)

    sp_ab_union = sp_a_active | sp_b_active

    out = {
        "combined_teacher": {
            "recon": combined_recon,
            "n_active": len(combined_active),
            "topk_features": combined_topk_r,
        },
        "sp_A_individual": {
            "n_active": len(sp_a_active),
            "topk_features": _topk_ranked(sp_a_topk, sp_a_act_dict),
        },
        "sp_B_individual": {
            "n_active": len(sp_b_active),
            "topk_features": _topk_ranked(sp_b_topk, sp_b_act_dict),
        },
        "sp_AB_union_size": len(sp_ab_union),
        "conditions": {},
    }

    def _record(label, sp_acts):
        recon = compute_recon_error(sae, sp_acts)
        active, topk, act_dict = get_sae_features(sae, sp_acts)
        jac_combined = jaccard(active, combined_active)
        jac_union = jaccard(active, sp_ab_union)
        jac_a = jaccard(active, sp_a_active)
        jac_b = jaccard(active, sp_b_active)
        print(f"  {label}: rel_err={recon['rel_err']:.4f}, n_active={len(active)}, "
              f"jac_combined={jac_combined:.3f}, jac_union={jac_union:.3f}, "
              f"jac_A={jac_a:.3f}, jac_B={jac_b:.3f}")
        out["conditions"][label] = {
            "recon": recon,
            "n_active": len(active),
            "topk_features": _topk_ranked(topk, act_dict),
            "jaccard_combined": jac_combined,
            "jaccard_union": jac_union,
            "jaccard_sp_A": jac_a,
            "jaccard_sp_B": jac_b,
            "shared_with_combined": _shared_ranked(active, combined_active, act_dict),
            "shared_with_union": _shared_ranked(active, sp_ab_union, act_dict),
        }

    for label, frames, ordering in compose.SYN_CONDITIONS:
        sp1 = csps[ordering[0]]
        sp2 = csps[ordering[1]]
        sp_acts = compose.get_syntactic_activations_at_layer(
            model, tokenizer, sp1, sp2, prompts, config.SAE_LAYER, frames[0], device,
        )
        _record(label, sp_acts)

    for label, _fn in compose.VEC_CONDITIONS:
        composite = csps[label]
        sp_acts = get_csp_activations_at_layer(
            model, tokenizer, composite, prompts, config.SAE_LAYER,
            config.POSITIVE_FRAMES[0], device,
        )
        _record(label, sp_acts)

    path = os.path.join(eval_dir, "sae.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved SAE results to {path}")
    return out


# ── Embedding comparison ───────────────────────────────────────────────

def embedding_compare(csps, eval_dir):
    sp_a, sp_b = csps["A"], csps["B"]
    a_flat = sp_a.embedding.detach().flatten().float()
    b_flat = sp_b.embedding.detach().flatten().float()

    out = {
        "individual": {
            "||A||": a_flat.norm().item(),
            "||B||": b_flat.norm().item(),
            "cos(A, B)": F.cosine_similarity(
                a_flat.unsqueeze(0), b_flat.unsqueeze(0),
            ).item(),
        },
        "composites": {},
    }
    for label, _fn in compose.VEC_CONDITIONS:
        c = csps[label]
        c_flat = c.embedding.detach().flatten().float()
        out["composites"][label] = {
            "||composite||": c_flat.norm().item(),
            "cos(composite, A)": F.cosine_similarity(
                c_flat.unsqueeze(0), a_flat.unsqueeze(0),
            ).item(),
            "cos(composite, B)": F.cosine_similarity(
                c_flat.unsqueeze(0), b_flat.unsqueeze(0),
            ).item(),
        }

    print("\nEmbedding comparison:")
    print(f"  ||A||={out['individual']['||A||']:.2f}  "
          f"||B||={out['individual']['||B||']:.2f}  "
          f"cos(A,B)={out['individual']['cos(A, B)']:+.3f}")
    for label, info in out["composites"].items():
        print(f"  {label}: ||={info['||composite||']:.2f}  "
              f"cos→A={info['cos(composite, A)']:+.3f}  "
              f"cos→B={info['cos(composite, B)']:+.3f}")

    path = os.path.join(eval_dir, "embedding_compare.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved embedding comparison to {path}")
    return out


# ── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair", required=True, help="e.g. 'pirate+poet'")
    parser.add_argument("--mode", default="all",
                        choices=["all", "self-verb", "sae", "behavior", "embedding"])
    parser.add_argument("--results-dir", default=os.path.join(SCRIPT_DIR, "results"))
    parser.add_argument("--questions", default=None)
    parser.add_argument("--n-eval-prompts", type=int, default=N_EVAL_PROMPTS)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    persona_a, persona_b = parse_pair(args.pair)
    assert persona_a in config.PERSONAS, f"Unknown persona: {persona_a}"
    assert persona_b in config.PERSONAS, f"Unknown persona: {persona_b}"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    pair_dir = os.path.join(args.results_dir, "composition", args.pair)
    eval_dir = os.path.join(pair_dir, "eval")
    os.makedirs(eval_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Loading {config.MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME, dtype=torch.bfloat16, device_map="auto",
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    sp_a_path = os.path.join(args.results_dir, persona_a, "sp_pos.pt")
    sp_b_path = os.path.join(args.results_dir, persona_b, "sp_pos.pt")
    sp_a, _ = SoftPrompt.from_checkpoint(sp_a_path, device=device)
    sp_b, _ = SoftPrompt.from_checkpoint(sp_b_path, device=device)
    print(f"Loaded CSPs:")
    print(f"  sp_A ({persona_a}): shape={tuple(sp_a.embedding.shape)}")
    print(f"  sp_B ({persona_b}): shape={tuple(sp_b.embedding.shape)}")

    csps = {"A": sp_a, "B": sp_b}
    for label, fn in compose.VEC_CONDITIONS:
        csps[label] = fn(sp_a, sp_b)
        norm = csps[label].embedding.detach().flatten().norm().item()
        print(f"  {label}: ||={norm:.2f}")

    questions = load_questions(args.questions)
    eval_prompts = questions[:args.n_eval_prompts]

    if args.mode in ("all", "embedding"):
        print(f"\n{'='*60}\n  EMBEDDING COMPARISON\n{'='*60}")
        embedding_compare(csps, eval_dir)

    if args.mode in ("all", "self-verb"):
        print(f"\n{'='*60}\n  SELF-VERBALIZATION\n{'='*60}")
        run_self_verb(model, tokenizer, csps, device, eval_dir)

    if args.mode in ("all", "behavior"):
        print(f"\n{'='*60}\n  BEHAVIOR SAMPLES\n{'='*60}")
        run_behavior(model, tokenizer, csps, eval_prompts, device, eval_dir)

    if args.mode in ("all", "sae"):
        print(f"\n{'='*60}\n  SAE DECOMPOSITION (L{config.SAE_LAYER})\n{'='*60}")
        run_sae(
            model, tokenizer, csps,
            config.PERSONAS[persona_a], config.PERSONAS[persona_b],
            eval_prompts, device, eval_dir,
        )


if __name__ == "__main__":
    main()
