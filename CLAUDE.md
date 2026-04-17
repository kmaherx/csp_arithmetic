# CSP Arithmetic

Experiments testing whether contextualized soft prompts (CSPs) support
arithmetic operations — negation, scaling, composition — in embedding
space. Builds on the interpretable soft prompt tuning work in the
`context_diffing` repo.

## NARRATIVE.md — running story of the work

`NARRATIVE.md` at the repo root is the running, blog-ready narrative
of this project. It contains motivation, setup, and a chapter per
settled result, with data/code references. The eventual blog post
will be lifted from it.

**Protocol (IMPORTANT):**

1. When a new result lands, do NOT write it into `NARRATIVE.md`
   immediately.
2. Discuss the result thoroughly with the user first — what it shows,
   where it's solid, where the boundaries are, what framing is honest.
3. Explicitly reach a conclusion about what the narrative should say.
4. Only then edit `NARRATIVE.md` to add or revise the relevant chapter.
5. When you think a result is ready to promote to the narrative, prompt
   the user to discuss before writing anything.

This keeps the narrative honest and keeps the user in control of the
framing. Never add to `NARRATIVE.md` unilaterally.

**Every chapter must include concrete inline examples with source
paths.** The narrative is written to drop into a blog post with minimal
rework, so the examples and their provenance must live inside the
prose, not in external tables. For each behavioral claim include:

- **Behavior:** at least one representative block-quote of model
  output, with a reference to the exact condition and source file
  (e.g. `results/composition/pirate+anxious/eval/behavior.json`, key
  `syn-v1-AB[0]`).
- **Self-verbalization:** representative verbalizations for each
  claim, each with a reference to the source JSON and condition key.
  Do not foreground the prompt-template machinery (multi-frame vs
  single-frame) — readers care about the *output*, not the template.
- **Feature decomposition:** a short discussion of the relevant SAE
  features with Neuronpedia links. URL pattern for this project:
  `https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/<feature_id>`.
  Whenever citing a feature, note (a) whether it's in the persona
  top-20 per `results/{persona}/eval/sae.json` or
  `results/composition/{pair}/eval/sae.json`, and (b) a one-line
  summary of what Neuronpedia's explanations + top logits + top
  activations suggest it detects. Check BOTH positive and negative
  logits and top activation snippets — explanations alone are
  sometimes null or misleading.

When in doubt about whether to include an example, include it.

## Current narrative structure

`NARRATIVE.md` is organized as:

- Motivation + three-level frame (steering vectors vs CSPs; math
  vs semantic ops; behavior / self-verb / feature decomposition).
- Setup — 4 primary personas (pirate, prophet, melancholic, playful)
  plus ~60 more for composition exploration and future PCA.
- **Chapter 1 — Negation.** Syntactic preserves, math destroys.
  Tables cover all 4 primary personas with persona-specific L17
  features.
- **Chapter 2 — Scaling.** Semantic adverbial intensifiers gradient
  cleanly for traits, not for roles (earns the role/trait split).
  Math α-scaling exhibits a three-regime failure: destroy (sign flip,
  from Ch 1) / no-op (positive α ≤ 4) / off-manifold collapse (α ≥ 5).
- **Chapter 3 — Composition.** Pirate × {anxious, playful} as focal
  demo. Noncommutativity + vestigial-template pattern. Sidebars:
  (A) role permeability and pretraining bias; (B) compound-
  neighborhood cultural retrieval; (C) seven-mode composition
  taxonomy; (D) connectives as language-level knob.
- **Chapter 4 — Population geometry.** Joint PCA over 65 CSPs
  surfaces a persona axis in embedding space. PC1 aligns with
  Butanium L17 at Spearman 0.81; roles and traits share the
  direction. Conservative scope: axis-existence + alignment only.
  Open questions (per-token structure, teacher-resistance on
  higher PCs, PC1 interpretation) are listed in Future work.

## Project Context & References

Extension of the CSP interpretability work published here:
- **CSP blog post**: https://kmaherx.github.io/projects/contextualized-soft-prompts/
- **Prior codebase** (`context_diffing`): the repo this project is
  derived from. Functions already ported into this repo (`train.py`,
  `evaluate.py`, `evaluate_scaling.py`, `compose.py`,
  `soft_prompt.py`, `run_composition.py`).

The CSP work showed that embedding soft prompts inside syntactic
frames during training ("Be {sp}.", "Act {sp}.", etc.) makes them
interpretable — the model can describe what they encode
(self-verbalization), and their internal representations decompose
into the same SAE features as the ground-truth instruction (feature
decomposition).

**Key upstream references:**

- **Persona vectors** (Chen et al. 2025): CAA pipeline for extracting
  persona vectors from behavioral contrasts.
  https://arxiv.org/abs/2507.21509 · https://www.anthropic.com/research/persona-vectors

- **Assistant axis** (Lu et al. 2026): 275 role archetypes collapse
  onto a single dominant axis in activation space; 240 traits were
  analyzed through a separately-run pipeline on the same axis.
  **Notably they did NOT directly compare role-PC1 to trait-PC1 or
  run a joint PCA** — that's the gap Ch 4 fills.
  https://arxiv.org/abs/2601.10387 · https://www.anthropic.com/research/assistant-axis
  Code: https://github.com/safety-research/assistant-axis
  Gemma 3 4B IT axis: https://huggingface.co/datasets/Butanium/gemma-3-4b-it-assistant-axis

- **Off-manifold steering coherence degradation**
  ([Vogels et al. 2025](https://arxiv.org/abs/2510.13285)) and
  **multi-trait steering interference**
  ([Bhandari et al. 2026](https://arxiv.org/abs/2602.15847),
  [Oozeer et al. 2025](https://arxiv.org/abs/2505.24535)) — cited in
  Motivation as the two documented ceilings of steering-vector
  arithmetic.

**Relation to the upstream CAA pipeline:** Our CSP training is
inherently contrastive via KL(student || teacher), which naturally
pushes the CSP to encode only the delta between default and persona
behavior. We skip the LLM judge filtering from the assistant-axis
pipeline; KL distillation is robust to non-role-playing teacher
responses (near-zero behavioral gap → near-zero gradient → CSP
doesn't learn from that example). If FE is unexpectedly low on a
persona, teacher-refusal or teacher-hedging is the first thing to
revisit. The low-FE tail in our 65-CSP sweep splits into two
clusters — *safety-violating* (evil, manipulative, subversive,
cruel, nihilistic; RLHF teacher partially refuses) and
*self-referential* (skeptical, humble, benevolent, paranoid; traits
whose prompts cause the teacher to hedge) — covered in NARRATIVE
Future work.

## Research Questions

1. **Negation** *(DONE — Chapter 1).* Syntactic *"Don't be §."*
   preserves the persona concept at L17 (often strengthening it);
   mathematical sign-flip destroys it.

2. **Scaling** *(DONE — Chapter 2).* Semantic adverbial intensifiers
   gradient cleanly when the persona is a *trait*, not a role. Math
   α-scaling: scale-invariant for positive α in coherent range, then
   off-manifold collapse at α ≥ 5–10.

3. **Composition** *(DONE — Chapter 3).* Trait × role is canonical
   and succeeds on all three readouts; slot-1 dominates the register;
   noncommutativity is real (trait-first can erase role); composition
   is retrieval into pretraining compound neighborhoods, not additive.

4. **Population geometry / assistant axis** *(DONE — Chapter 4).*
   Joint PCA over 33 role CSPs + 32 trait CSPs reveals a persona
   axis in embedding space. PC1 aligns with the CAA-derived
   Butanium L17 axis at Spearman 0.81 on the joint population;
   roles and traits share the same dominant direction (filling
   the Lu et al. 2026 gap of separate role/trait pipelines).
   Per-token structure and teacher-resistance clusters on higher
   PCs are listed as Future work.

## Runpod Instance Setup

This project runs on Runpod. Only `/workspace/` persists across
instance changes; `/root/` is transient. On a **fresh instance**,
run this setup before anything else:

```bash
# Recreate memory symlink (memories live in /workspace/.claude_memory/)
mkdir -p /root/.claude/projects/-workspace-csp-arithmetic
ln -s /workspace/.claude_memory /root/.claude/projects/-workspace-csp-arithmetic/memory

# HF auth (token stored persistently)
export HF_TOKEN=$(cat /workspace/.hf_token)
export HF_HOME=/workspace/.cache/huggingface/

# venv is at .venv/ (persistent under /workspace)
# If deps are missing: source .venv/bin/activate && uv pip install torch transformers sae-lens huggingface-hub
```

**Memory files** at `/workspace/.claude_memory/` contain project
state, results summaries, and infrastructure notes. Read `MEMORY.md`
there for the index.

**GPU parallelism rule:** *max one training run at a time*, but eval
jobs can run concurrently with training (and with each other).
Eval + training on the same Gemma 3 4B IT bf16 instance fits on an
RTX 5090 (32GB) without issue. Greedy decoding is deterministic under
this setup — we verified with a solo vs parallel-with-training diff
of `pirate+melancholic` (byte-identical outputs).

## Environment

- **Model**: Gemma 3 4B IT (`google/gemma-3-4b-it`)
- **SAEs**: Gemma Scope 2 (`gemma-scope-2-4b-it-res`), layer 17, 16k
  features, medium L0
- **Package management**: `uv`. Venv at `.venv/`. Always run scripts
  with `.venv/bin/python`.
- **GPU**: 1× GPU with ≥24 GB VRAM. Currently running on an RTX 5090
  (32 GB).
- **Dependencies**: `torch`, `transformers`, `sae-lens`,
  `huggingface-hub`, `scikit-learn`, `matplotlib`, `plotly`,
  `kaleido` (installed but PNG export via Chrome-based Kaleido needs
  system libs; we use matplotlib for PNGs instead).
- **Persistent storage**: All results go under `results/`. Checkpoints
  are `.pt` files.
- **HF cache**: `HF_HOME=/workspace/.cache/huggingface/`.

## Repo Structure

```
csp-arithmetic/
├── CLAUDE.md                     # this file
├── NARRATIVE.md                  # running, blog-ready narrative
├── config.py                     # 65 personas (33 roles + 32 traits),
│                                 #   frames (pos/neg/slightly/extremely),
│                                 #   composition connectives V1–V4
├── soft_prompt.py                # SoftPrompt nn.Module
│                                 #   + negate_csp() / scale_csp() helpers
├── train.py                      # training (one persona × one polarity)
├── evaluate.py                   # 3×2 negation eval grid
├── evaluate_scaling.py           # scaling eval (semantic intensifiers +
│                                 #   math α-scaling grid)
├── compose.py                    # composition ops (sum, mul), two-slot
│                                 #   splicing, combined-teacher extractor
├── run_composition.py            # composition eval driver — 10 conds/pair
│                                 #   (8 syntactic × 4 connectives + 2 vec)
├── persona_sets.py               # shared module: role/trait categories,
│                                 #   FE values, kind_of(), get_names()
├── run_pca.py                    # flattened + pooled per-token PCA per
│                                 #   --persona-set {roles,traits,joint}
├── run_pca_per_token.py          # separate PCA per L=4 token slot
├── run_axis_projection.py        # project CSP L17 activations onto the
│                                 #   Butanium Gemma 3 4B IT assistant axis;
│                                 #   auto-caches projections
├── plot_resistance_clusters.py   # highlight teacher-resistance clusters
│                                 #   on joint PC1-PC3, PC1-PC5, PC3-PC5
├── analyze_pc_distance_vs_composition.py
│                                 # 28-pair PC-distance-vs-composition
│                                 #   quality check
├── run_sweep.sh                  # original 3-persona sweep (pirate/poet/prophet)
├── run_composition_sweep.sh      # original 3-pair composition sweep
├── run_axis_sweep.sh             # pos-only sweep for arbitrary persona list
├── data/
│   └── questions.jsonl           # 240 questions from assistant-axis repo
└── results/
    ├── axis_sweep.log            # original 30-role axis-sweep log
    ├── {persona}/                # one directory per persona
    │   ├── cached_responses.json # teacher responses (deterministic, cached)
    │   ├── sp_pos.pt             # positive CSP (all 65 personas)
    │   ├── sp_neg.pt             # negative CSP (4 primary personas:
    │   │                         #   pirate, prophet, melancholic, playful;
    │   │                         #   also poet from earlier work)
    │   ├── eval/                 # 3×2 grid (4 primary personas + poet)
    │   │   ├── behavior.json
    │   │   ├── self_verb.json
    │   │   ├── sae.json
    │   │   └── embedding_compare.json
    │   └── eval_scaling/         # scaling grid (4 primary personas)
    │       └── <same files>
    ├── composition/{pair}/eval/  # per-pair composition outputs
    │   └── <same files>
    └── pca/                      # PCA analysis outputs
        ├── pca_{flattened,pertoken,token{1..4}}_{roles,traits,joint}.{html,png}
        ├── pca_vs_assistant_axis_{roles,traits,joint}.{html,png}
        ├── pca_flattened_{roles,traits,joint}_by_fe.{html,png}
        ├── pca_resistance_pc{1,3,5}_*.{html,png}
        ├── pc_distance_vs_composition*.{html,png,json}
        ├── axis_projection_cache.json      # reusable across runs
        └── *_summary_{roles,traits,joint}.json
```

`math-neg` is constructed at eval time by `negate_csp(sp_pos)`;
scaled CSPs by `scale_csp(sp_pos, α)` — no separate checkpoints.

**Persona lists (in `config.py` PERSONAS dict):**

- **Roles (33):** pirate, poet, prophet, wizard, samurai, knight,
  vampire, bard, oracle, necromancer, druid, witch, ninja, detective,
  chef, scientist, journalist, surgeon, therapist, spy, librarian,
  lawyer, teacher, comedian, philosopher, monk, rapper, stoic,
  politician, salesperson, coach, historian, cowboy.
- **Traits (32):** melancholic, playful (first batch) + axis-comparison
  sweep (evil, anxious, arrogant, cruel, dramatic, formal, manipulative,
  subversive, nihilistic, philosophical, sardonic, savage, serene,
  verbose, witty — sharp end; sycophantic, analytical, benevolent,
  casual, concise, confident, dispassionate, earnest, enigmatic,
  flippant, humble, mystical, paranoid, passionate, skeptical — soft
  end).

Trait prompts are the `pos` variants from
`https://github.com/safety-research/assistant-axis/tree/master/data/traits/instructions`.

## Scripts

### train.py

Train a single CSP for one persona in one polarity.

```bash
.venv/bin/python train.py --persona pirate --polarity pos
.venv/bin/python train.py --persona pirate --polarity neg
# Override hyperparams:
.venv/bin/python train.py --persona pirate --polarity pos --L 4 --steps 500 --lr 1e-3
```

KL(student || teacher) on response tokens. Teacher has persona system
prompt; student has CSP inside a frame in the user turn. 500 steps,
L=4, LR=1e-3, 50 prompts/step sampled with random frame from the
polarity's pool.

Teacher responses cached to `results/{persona}/cached_responses.json`
and reused across polarities.

### evaluate.py

3×2 negation grid (CSP source × frame polarity = 6 conditions):

| Condition         | CSP source         | Frame at eval    |
|-------------------|--------------------|------------------|
| pos-in-pos        | CSP_pos            | "Be {sp}."       |
| neg-in-neg        | CSP_neg            | "Don't be {sp}." |
| pos-in-neg        | CSP_pos            | "Don't be {sp}." |
| neg-in-pos        | CSP_neg            | "Be {sp}."       |
| math-neg-in-pos   | -CSP_pos           | "Be {sp}."       |
| math-neg-in-neg   | -CSP_pos           | "Don't be {sp}." |

```bash
.venv/bin/python evaluate.py --persona pirate
.venv/bin/python evaluate.py --persona pirate --mode self-verb
```

Per condition: self-verbalization (multi-frame and single-frame
prompts), SAE decomposition at L17 (Jaccard vs persona ground-truth
features, reconstruction error, top-20), behavioral samples on
held-out prompts, embedding comparisons.

### evaluate_scaling.py

Scaling grid (Chapter 2): 3 semantic intensifiers × 5 math α-values
= 8 conditions per persona.

- **Semantic:** `"Be slightly §."`, `"Be §."`, `"Be extremely §."`
  (and Act/Please/You should variants — see
  `POSITIVE_FRAMES_SLIGHTLY` / `_EXTREMELY` in `config.py`).
- **Math:** α · CSP_pos for α ∈ {0.25, 1.0, 4.0, 5.0, 10.0},
  spliced into plain `"Be §."`.

```bash
.venv/bin/python evaluate_scaling.py --persona pirate
```

Outputs go to `results/{persona}/eval_scaling/`. Same file names as
`evaluate.py`'s `eval/`.

### compose.py + run_composition.py

Composition eval — 10 conditions per pair:

- **Syntactic (8):** 4 connective variants × 2 slot orderings.
  Connectives: v1 *"and"*, v2 *"and be"* (doubled verb),
  v3 *"as well as"*, v4 *"along with"*.
- **Vector (2):** `vec-sum` (mean of embeddings) and `vec-mul`
  (elementwise product), spliced into plain `"Be §."`.

```bash
.venv/bin/python run_composition.py --pair pirate+anxious
```

Outputs to `results/composition/{pair}/eval/`. Combined-teacher
ground truth: teacher with *both* system prompts concatenated.

Frame definitions in `config.py` as `COMPOSITION_FRAMES_V1` through
`V4`.

### PCA analysis (Chapter 4)

Five scripts covering the Ch 4 population-geometry analysis:

```bash
# Flattened + pooled per-token PCA
.venv/bin/python run_pca.py --persona-set {roles|traits|joint}

# One separate PCA per L=4 token slot
.venv/bin/python run_pca_per_token.py --persona-set {roles|traits|joint}

# Project CSP L17 activations onto Butanium Gemma 3 4B IT axis
# (auto-caches projections to results/pca/axis_projection_cache.json)
.venv/bin/python run_axis_projection.py --persona-set {roles|traits|joint}

# Highlight teacher-resistance clusters on joint PC1-PC3, PC1-PC5, PC3-PC5
.venv/bin/python plot_resistance_clusters.py

# Correlate composition quality (jac_combined) with joint-PC distance
.venv/bin/python analyze_pc_distance_vs_composition.py
```

All outputs land in `results/pca/`. Plotly HTML for interactive
inspection; matplotlib PNG for static. `persona_sets.py` holds the
shared category + FE data — import from there when adding new
analyses. See CLAUDE-listed summary files in Repo Structure above.

## Hyperparameters

- L = 4 (soft prompt length)
- LR = 1e-3
- WEIGHT_DECAY = 1e-4
- STEPS = 500
- MAX_NEW_TOKENS = 128 (teacher response generation)
- PROMPTS_PER_STEP = 50
- SAE_LAYER = 17
- SAE_RELEASE = "gemma-scope-2-4b-it-res"

## Common Pitfalls

1. **Chat template system turn**: Gemma 3 4B IT supports `role:
   "system"` in `apply_chat_template` (it auto-merges into the user
   turn; no separate system block — see
   `/workspace/.claude_memory/gemma3_chat_template.md`).

2. **Token alignment**: Teacher and student sequences have different
   lengths. Always compute `resp_start` from the prompt-only
   tokenization (without the response), then index into the full
   sequence's logits. Off-by-one here silently corrupts training.

3. **Placeholder splicing**: When replacing § with L=4 CSP tokens,
   the output embedding sequence is 3 tokens longer than the input
   token sequence. This affects position calculations downstream of
   the splice point.

4. **SAE layer hook path**: Gemma 3 4B IT wraps the language model.
   The residual stream hook goes on
   `model.model.language_model.layers[layer_idx]` (not
   `model.model.layers[...]` which would be the Gemma 2 path).

5. **dtype**: Model runs in bfloat16. CSP embeddings are float32
   (for stable gradients). Cast CSP embeddings to the model's dtype
   before concatenation:
   `sp_embeds = sp(batch_size=1).to(full_embeds.dtype)`.

6. **Gradient flow**: Only CSP parameters have `requires_grad=True`.
   All model parameters are frozen.
   Optimizer: `AdamW(sp.parameters(), ...)`.

7. **Vim swap files**: When editing JSON outputs during analysis,
   `.*.swp` files get created. Don't commit them — `.gitignore` or
   explicit paths only.

## Known FE patterns (for Ch 4 PCA design)

From the 65-persona sweep:

- **Highest FE (sharpest archetypes):** sardonic 93.6%, prophet 92.8%,
  druid 92.1%, savage 91.3%, poet 91.7%, dispassionate 91.1%, witty
  90.8%, casual 90.9%, samurai 90.1%, pirate 89.7%, serene 89.9%.

- **Low-FE clusters — two distinct mechanisms:**
  - *Safety-violating* (teacher refusal/hedging): evil 69.9,
    manipulative 60.3, subversive 65, cruel 79.8, nihilistic 73.5.
    RLHF-tuned teacher partially refuses role-play.
  - *Self-referential* (teacher-doubt): humble 65.7, skeptical 58.4,
    benevolent 61.3, paranoid 62.0. Traits whose prompts cause the
    teacher to doubt or hedge its own role-play.

Whether these two low-FE clusters separate from the main population
on a PC2 or PC3 is a specific Ch 4 hypothesis.
