"""Composition utilities on top of trained single-persona CSPs.

Two flavors of composition:
  - Syntactic: splice two CSPs into a composition frame like
    "Be §. and ¶." or "Be §. and be ¶." (one CSP per placeholder).
  - Vector: arithmetic on CSP embeddings (sum, elementwise product),
    producing a new single-slot composite CSP.

Used by run_composition.py.
"""

import random

import torch

import config
from soft_prompt import SoftPrompt
from train import (
    render_messages, student_messages, teacher_messages,
    find_placeholder_position,
)
from evaluate import MULTI_FRAME_TEMPLATES, capture_layer_activations


# ── Composite CSP construction (vector composition) ─────────────────────

def _new_like(sp):
    out = SoftPrompt(sp.embedding.shape[0], sp.embedding.shape[1])
    return out.to(sp.embedding.device)


def compose_sum(sp_a, sp_b):
    """Elementwise mean. Division by 2 keeps norms comparable to inputs."""
    assert sp_a.embedding.shape == sp_b.embedding.shape
    out = _new_like(sp_a)
    out.embedding.data = (sp_a.embedding.data + sp_b.embedding.data) / 2
    return out


def compose_mul(sp_a, sp_b):
    """Elementwise product. Falsifiability probe — expect small-norm result."""
    assert sp_a.embedding.shape == sp_b.embedding.shape
    out = _new_like(sp_a)
    out.embedding.data = sp_a.embedding.data * sp_b.embedding.data
    return out


VEC_CONDITIONS = [
    ("vec-sum", compose_sum),
    ("vec-mul", compose_mul),
]


# ── Syntactic composition frames ────────────────────────────────────────

# Conditions are (label, frame_pool, ordering) where ordering is ("A","B")
# or ("B","A"); ordering[0] fills §, ordering[1] fills ¶.
SYN_CONDITIONS = [
    ("syn-v1-AB", config.COMPOSITION_FRAMES_V1, ("A", "B")),
    ("syn-v1-BA", config.COMPOSITION_FRAMES_V1, ("B", "A")),
    ("syn-v2-AB", config.COMPOSITION_FRAMES_V2, ("A", "B")),
    ("syn-v2-BA", config.COMPOSITION_FRAMES_V2, ("B", "A")),
]


# ── Two-slot splicing ───────────────────────────────────────────────────

def build_csp_input_two_slot(tokenizer, embed_fn, sp1, sp2, user_text, device,
                              ph1=None, ph2=None):
    """Tokenize user_text containing both ph1 and ph2, splice sp1 at ph1 and sp2 at ph2.

    Splicing is right-to-left so earlier positions stay valid after each insertion.
    """
    ph1 = ph1 or config.SP_PLACEHOLDER
    ph2 = ph2 or config.SP_PLACEHOLDER_2

    text = render_messages(
        tokenizer, student_messages(user_text), add_generation_prompt=True,
    )
    ids = tokenizer(text, return_tensors="pt").input_ids[0].to(device)

    positions = []
    for i, tid in enumerate(ids.tolist()):
        decoded = tokenizer.decode([tid])
        if ph1 in decoded:
            positions.append((i, "sp1"))
        elif ph2 in decoded:
            positions.append((i, "sp2"))

    if not positions:
        raise ValueError(f"No placeholders ({ph1}, {ph2}) found in: {text[:200]}...")

    embeds = embed_fn(ids.unsqueeze(0))
    sp1_embeds = sp1(batch_size=1).to(embeds.dtype)
    sp2_embeds = sp2(batch_size=1).to(embeds.dtype)

    result = embeds
    for pos, which in sorted(positions, key=lambda x: x[0], reverse=True):
        sp_embeds = sp1_embeds if which == "sp1" else sp2_embeds
        result = torch.cat([
            result[:, :pos, :], sp_embeds, result[:, pos + 1:, :],
        ], dim=1)
    return result


# ── Verbalization prompts for syntactic composition ─────────────────────

def syntactic_multi_frame_prompts(frames):
    """5 multi-frame templates using all 4 composition frames concatenated."""
    joined = " ".join(
        f.format(sp1=config.SP_PLACEHOLDER, sp2=config.SP_PLACEHOLDER_2)
        for f in frames
    )
    return [t.format(frames=joined) for t in MULTI_FRAME_TEMPLATES]


def syntactic_single_frame_prompts(frames):
    """One 'In plain English' prompt per composition frame."""
    return [
        f"In plain English, explain this command: "
        f"{f.format(sp1=config.SP_PLACEHOLDER, sp2=config.SP_PLACEHOLDER_2)}"
        for f in frames
    ]


def syntactic_verb_prompts(frames):
    """Multi-frame + single-frame prompts for a composition frame pool."""
    return (
        [("multi_frame", p) for p in syntactic_multi_frame_prompts(frames)]
        + [("single_frame", p) for p in syntactic_single_frame_prompts(frames)]
    )


# ── Combined-persona ground-truth activations ───────────────────────────

def get_combined_persona_activations_at_layer(model, tokenizer,
                                              persona_a_prompts, persona_b_prompts,
                                              prompts, layer_idx, device, seed=42):
    """Teacher with concatenated A+B system prompts; capture L17 over the diverging span.

    Mirrors get_persona_activations_at_layer in evaluate.py but with a
    concatenated system prompt sampled per item.
    """
    rng = random.Random(seed)
    acts = []
    for prompt in prompts:
        sys_a = rng.choice(persona_a_prompts)
        sys_b = rng.choice(persona_b_prompts)
        system = f"{sys_a} {sys_b}"
        text_with = render_messages(
            tokenizer, teacher_messages(system, prompt), add_generation_prompt=True,
        )
        text_without = render_messages(
            tokenizer, student_messages(prompt), add_generation_prompt=True,
        )
        ids_with = tokenizer(text_with, return_tensors="pt").input_ids[0].to(device)
        ids_without = tokenizer(text_without, return_tensors="pt").input_ids[0]

        sys_start = 0
        for i in range(min(len(ids_with), len(ids_without))):
            if ids_with[i] != ids_without[i]:
                sys_start = i
                break
        n_extra = len(ids_with) - len(ids_without)
        sys_end = sys_start + n_extra
        if sys_end <= sys_start:
            continue

        layer_act = capture_layer_activations(
            model, layer_idx, lambda: model(input_ids=ids_with.unsqueeze(0)),
        )
        acts.append(layer_act[0, sys_start:sys_end, :].mean(dim=0))
    return torch.stack(acts)


# ── Helpers for syntactic condition activation capture ──────────────────

def get_syntactic_activations_at_layer(model, tokenizer, sp1, sp2, prompts,
                                        layer_idx, frame, device):
    """For each eval prompt, build 'prompt + frame' with sp1 at § and sp2 at ¶;
    capture layer_idx activations over the full composite-SP span and mean.

    Returns tensor of shape (n_prompts, hidden).
    """
    embed_fn = model.get_input_embeddings()
    L = sp1.embedding.shape[0]
    acts = []
    for prompt in prompts:
        suffix = frame.format(sp1=config.SP_PLACEHOLDER, sp2=config.SP_PLACEHOLDER_2)
        user = f"{prompt} {suffix}"
        combined = build_csp_input_two_slot(
            tokenizer, embed_fn, sp1, sp2, user, device,
        )
        # Locate the two SP spans in the spliced output. Re-tokenize to find
        # placeholder positions, then span is [pos_§, pos_§+L) and [pos_¶_shifted, pos_¶_shifted+L).
        # Simpler: average over all positions after the chat-template boundaries —
        # but we want the CSP positions specifically. Identify via the two
        # placeholder positions in the pre-splice token sequence.
        text = render_messages(
            tokenizer, student_messages(user), add_generation_prompt=True,
        )
        ids = tokenizer(text, return_tensors="pt").input_ids[0]
        sp_pos_1 = None
        sp_pos_2 = None
        for i, tid in enumerate(ids.tolist()):
            d = tokenizer.decode([tid])
            if sp_pos_1 is None and config.SP_PLACEHOLDER in d:
                sp_pos_1 = i
            elif sp_pos_2 is None and config.SP_PLACEHOLDER_2 in d:
                sp_pos_2 = i
        # After splicing: first placeholder expands to L tokens; so for positions
        # after sp_pos_1, shift by (L-1). If sp_pos_2 > sp_pos_1, shift it.
        if sp_pos_2 is not None and sp_pos_1 is not None and sp_pos_2 > sp_pos_1:
            sp_pos_2_spliced = sp_pos_2 + (L - 1)
        else:
            sp_pos_2_spliced = sp_pos_2

        layer_act = capture_layer_activations(
            model, layer_idx, lambda: model(inputs_embeds=combined),
        )
        # Mean across both CSP spans (2L tokens total)
        span1 = layer_act[0, sp_pos_1:sp_pos_1 + L, :]
        span2 = layer_act[0, sp_pos_2_spliced:sp_pos_2_spliced + L, :]
        acts.append(torch.cat([span1, span2], dim=0).mean(dim=0))
    return torch.stack(acts)
