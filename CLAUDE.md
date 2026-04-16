# CSP Arithmetic

Experiments testing whether contextualized soft prompts (CSPs) support negation and composition
in embedding space. Builds on the interpretable soft prompt tuning work in the `context_diffing` repo.

## NARRATIVE.md — running story of the work

`NARRATIVE.md` at the repo root is the running, blog-ready narrative of this
project. It contains motivation, setup, and a chapter per settled result,
with data/code references. The eventual blog post will be lifted from it.

**Protocol (IMPORTANT):**

1. When a new result lands, do NOT write it into `NARRATIVE.md` immediately.
2. Discuss the result thoroughly with the user first — what it shows, where
   it's solid, where the boundaries are, what framing is honest.
3. Explicitly reach a conclusion about what the narrative should say.
4. Only then edit `NARRATIVE.md` to add or revise the relevant chapter.
5. When you think a result is ready to promote to the narrative, prompt the
   user to discuss before writing anything.

This keeps the narrative honest and keeps the user in control of the framing.
Never add to `NARRATIVE.md` unilaterally.

## Project Context & References

This project is an extension of the CSP interpretability work published here:
- **CSP blog post**: https://kmaherx.github.io/projects/contextualized-soft-prompts/
- **Prior codebase** (`context_diffing`): the repo this project is derived from. Contains
  the training scripts (`run_unified.py`, `run_tier2.py`), self-verbalization (`self_verbalize.py`),
  SAE analysis (`analyze_sae.py`), and all Tier 1/Tier 2 results for the blog post.

The CSP work showed that embedding soft prompts inside syntactic frames during training
("Be {sp}.", "Act {sp}.", etc.) makes them interpretable — the model can describe what they
encode (self-verbalization), and their internal representations decompose into the same
SAE features as the ground-truth instruction (feature decomposition). The Tier 2 experiments
trained CSPs to match a model steered along the "assistant axis" — a direction in activation
space capturing persona variation — and found that CSPs surfaced persona-involvement features
the steering vector itself didn't activate.

This project continues in the persona setting by training CSPs for several distinct personas
and testing whether they support arithmetic operations (negation, composition) and whether
their geometry mirrors the assistant axis.

**Key upstream references:**

- **Persona vectors** (Chen et al. 2025): Contrastive activation addition (CAA) pipeline
  for extracting persona vectors from behavioral contrasts. Showed persona traits (evil,
  sycophancy, hallucination) are encoded as directions in activation space.
  Paper: https://arxiv.org/abs/2507.21509
  Blog: https://www.anthropic.com/research/persona-vectors

- **The assistant axis** (Lu et al. 2026): Found that persona variation across 275 character
  archetypes collapses onto a single dominant axis in activation space. One end is the
  default helpful assistant; the other end is increasingly intense character embodiment.
  Paper: https://arxiv.org/abs/2601.10387
  Blog: https://www.anthropic.com/research/assistant-axis
  Code & pipeline: https://github.com/safety-research/assistant-axis
  Pipeline README (5-step process for computing the axis):
    https://github.com/safety-research/assistant-axis/blob/master/pipeline/README.md
  Pre-computed axes: https://huggingface.co/datasets/lu-christina/assistant-axis-vectors
  Gemma 3 4B IT axis (used in Tier 2): https://huggingface.co/datasets/Butanium/gemma-3-4b-it-assistant-axis

**How this project relates to the upstream pipeline:**

The assistant-axis pipeline computes persona vectors via contrastive activation addition:
(1) generate responses with persona system prompts, (2) extract mean activations,
(3) filter using an LLM judge for responses where the model actually role-plays (score 3 =
fully role-playing), (4) compute per-role mean vectors from filtered responses, (5) compute
the axis as `mean(default) - mean(role_vectors)`.

Our CSP training is inherently contrastive — KL(student || teacher) naturally pushes the CSP
to encode only the delta between default and persona behavior — so we don't need the explicit
contrastive subtraction. We use the assistant-axis 240 extraction questions as our prompt pool
and **skip the LLM judge filtering entirely**. This is a major simplification over the
assistant-axis pipeline, justified by a structural property of KL distillation:

When the teacher doesn't role-play on a given prompt (e.g., it falls back to default assistant
behavior), its response distribution is close to the unsteered model's distribution. The KL
between student-with-CSP and teacher is therefore small on that example — near-zero behavioral
gap means near-zero gradient. The CSP simply doesn't learn from non-role-playing examples.
This contrasts with the assistant-axis pipeline, where they compute *mean activations* across
responses — a single non-role-playing response actively corrupts the mean and must be filtered
out. KL distillation is naturally robust to this noise in a way that mean-subtraction is not.

The worst case from skipping filtering is slightly noisier gradients and a few wasted forward
passes per step, not a corrupted training signal. If FE is unexpectedly low on a persona,
filtering is the first thing to revisit.

## Research Questions

1. **Negation**: Train CSP_pos in positive frames ("Be {sp}.") and CSP_neg in negative frames
   ("Don't be {sp}.") against the same persona teacher. Compare embeddings, SAE features, and
   self-verbalization. Then test cross-frame: does "Don't be {CSP_pos}." produce the same
   behavior as "Be {CSP_neg}."?

2. **Composition** (future): Train individual CSPs ("Be {sp1}.") and jointly-trained pairs
   ("Be {sp1} and {sp2}."), compare embeddings, features, and self-verbalization. Test whether
   the joint CSP decomposes into the individual ones.

3. **Shape** (future): PCA on trained CSP embeddings, compare to the assistant axis. Does
   the CSP embedding space recapitulate the near-1D structure of persona space?

## Runpod Instance Setup

This project runs on Runpod. Only `/workspace/` persists across instance changes;
`/root/` is transient. On a **fresh instance**, run this setup before anything else:

```bash
# Recreate memory symlink (memories live in /workspace/.claude_memory/)
mkdir -p /root/.claude/projects/-workspace-csp-arithmetic
ln -s /workspace/.claude_memory /root/.claude/projects/-workspace-csp-arithmetic/memory

# HF auth (token stored persistently)
export HF_TOKEN=$(cat /workspace/.hf_token)
export HF_HOME=/workspace/.cache/huggingface/

# venv is already at .venv/ (persistent under /workspace)
# If deps are missing: source .venv/bin/activate && uv pip install torch transformers sae-lens huggingface-hub
```

**Memory files** at `/workspace/.claude_memory/` contain project state, results
summaries, and infrastructure notes. Read `MEMORY.md` there for the index.

## Environment

- **Model**: Gemma 3 4B IT (`google/gemma-3-4b-it`)
- **SAEs**: Gemma Scope 2 (`gemma-scope-2-4b-it-res`), layer 17, 16k features, medium L0
- **Package management**: Use `uv`. Create venv at `.venv/`. Always run scripts with `.venv/bin/python`.
- **GPU**: Training requires 1x GPU with ≥24GB VRAM (A100, L40S, etc.)
- **Dependencies**: `torch`, `transformers`, `sae-lens`, `huggingface-hub`
- **Persistent storage**: All results go under `results/`. Checkpoints are `.pt` files.
- **HF cache**: Set `HF_HOME=/workspace/.cache/huggingface/` to keep downloads local.

## Repo Structure

```
csp-arithmetic/
├── CLAUDE.md           # this file
├── config.py           # personas, frames, hyperparams
├── soft_prompt.py      # SoftPrompt nn.Module
├── train.py            # training script (one persona, one polarity)
├── evaluate.py         # self-verbalization + SAE decomposition
├── data/
│   └── questions.jsonl # 240 questions from assistant-axis repo
└── results/
    └── {persona}/
        ├── cached_responses.json   # teacher responses (cached)
        ├── sp_pos.pt               # positive CSP checkpoint
        ├── sp_neg.pt               # negative CSP checkpoint
        └── eval/                   # evaluation outputs
```

## Scripts

### train.py

Train a single CSP for one persona in one polarity (positive or negative frames).

```bash
# Train positive CSP for pirate
.venv/bin/python train.py --persona pirate --polarity pos

# Train negative CSP for pirate
.venv/bin/python train.py --persona pirate --polarity neg

# Override hyperparams
.venv/bin/python train.py --persona pirate --polarity pos --L 4 --steps 500 --lr 1e-3
```

**How training works:**

The teacher is the model with a persona system prompt. The student sees no system prompt;
instead, the CSP is embedded in a syntactic frame appended to the user content.

- Teacher forward pass: `[system: persona_instruction] [user: prompt] [assistant: response]`
- Student forward pass: `[user: "{prompt} {frame.format(sp=CSP)}"] [assistant: response]`

For positive polarity, frames are: "Be {sp}.", "Act {sp}.", "Please {sp}.", "You should {sp}."
For negative polarity, frames are: "Don't be {sp}.", "Don't act {sp}.", "Please don't {sp}.", "You should not {sp}."

Each training step:
1. Sample 50 prompts (without replacement) from the 240-question pool
2. For each prompt, sample one frame from the polarity's frame pool
3. Compute KL(student || teacher) on response tokens only
4. Update CSP embeddings via AdamW

Teacher responses are cached to `results/{persona}/cached_responses.json` on first run
and reused for both polarities.

**Important implementation details (from context_diffing):**

- The placeholder token `§` marks where the CSP gets spliced in. The frame is formatted as
  e.g. `"Be §."`, tokenized, then the `§` token's embedding is replaced with the CSP embeddings.
  For L>1, the single placeholder token is replaced with L embeddings (the output sequence is
  longer than the input token sequence by L-1).

- Response token alignment: teacher and student have different prompt lengths (system prompt vs
  frame). Compute `t_resp_start` and `s_resp_start` independently. KL loss is computed only on
  `logits[resp_start-1:-1]` (shifted by 1 for next-token prediction).

- The teacher response is generated once and cached. Both teacher and student forward passes
  during training process the full `[prompt + response]` sequence, and loss is computed on the
  response portion. This is distillation on the response distribution, not on generation.

- Chat template: use `tokenizer.apply_chat_template(messages, tokenize=False)`. Gemma 3 supports
  system turns: `[{"role": "system", "content": ...}, {"role": "user", "content": ...}]`.
  VERIFY this works — if not, inject the system prompt as a prefix in the user turn.

- For the teacher, the system prompt carries the persona instruction. For the student, there is
  NO system prompt — the frame+CSP is appended to the user content in the user turn.

### evaluate.py

Run self-verbalization and SAE decomposition on trained CSPs.

```bash
# Full evaluation for one persona
.venv/bin/python evaluate.py --persona pirate

# Just self-verbalization
.venv/bin/python evaluate.py --persona pirate --mode self-verb

# Just SAE analysis
.venv/bin/python evaluate.py --persona pirate --mode sae
```

**Evaluation conditions** (run all four per persona):

| Condition   | CSP used | Frame at eval         | What it tests                          |
|-------------|----------|-----------------------|----------------------------------------|
| pos-in-pos  | CSP_pos  | "Be {sp}."            | Baseline: does pos CSP work?           |
| neg-in-neg  | CSP_neg  | "Don't be {sp}."      | Baseline: does neg CSP work?           |
| pos-in-neg  | CSP_pos  | "Don't be {sp}."      | Does frame negation override training? |
| neg-in-pos  | CSP_neg  | "Be {sp}."            | Does neg CSP negate in positive frame? |

For each condition, evaluate:

1. **Self-verbalization**: Present the CSP in its eval frame with verbalization prompts.
   Ask the model to describe what the instruction means. See VERBALIZATION_PROMPTS below.

2. **SAE decomposition at L17**: Extract the CSP's activations at layer 17, encode through
   the SAE, and compare features to the ground-truth persona (teacher's system-prompt
   activations at L17). Report: reconstruction error, Jaccard overlap, shared feature IDs.

3. **Behavioral generation**: Generate 5 responses with the CSP in its eval frame on
   held-out prompts. Save for qualitative review.

**Self-verbalization prompts (adapted for this setup):**

Since the CSP is at the end of the user message inside a frame, use prompts like:
- "What is the instruction at the end of my message?"
- "What is the instruction at the end of my message asking you to do?"
- "Rewrite the instruction at the end of my message as a single short command."
- "Complete: The instruction at the end of my message is equivalent to '"

For the multi-frame verbalization approach, present all 4 frames with the CSP:
- "These instructions all mean the same thing: Be §. Act §. Please §. You should §. In one word or phrase, they are asking me to:"
  (splice the CSP into each § position)

For negative-frame conditions, adapt accordingly:
- "These instructions all mean the same thing: Don't be §. Don't act §. Please don't §. You should not §. In one word or phrase, they are asking me to:"

**SAE ground truth**: For Jaccard comparison, the "ground truth" activations come from
passing the persona's system prompt text through the model and extracting L17 activations.
Specifically: format the system prompt as a user message (e.g., "Be a pirate."),
tokenize, run forward pass, capture residual stream at L17 for the instruction tokens.

**Embedding comparison** (also in evaluate.py):
- `cosine(CSP_pos.embedding.flatten(), CSP_neg.embedding.flatten())`
- `cosine(CSP_pos.embedding.flatten(), -CSP_neg.embedding.flatten())`
- Per-token cosines: `cosine(CSP_pos.embedding[i], CSP_neg.embedding[i])` for i in range(L)

## Key Functions to Port from context_diffing

The following functions from `context_diffing/experiments/exp0_toy/` should be adapted:

**From run_unified.py:**
- `apply_chat_template()` — modify to support system messages
- `find_placeholder_position()` — as-is
- `find_content_boundaries()` — as-is
- `compute_kl_loss()` — as-is
- `capture_layer_activations()` — as-is, but note the hook path is
  `model.model.language_model.layers[layer_idx]` for Gemma 3
  (this is the ShieldGemma/Gemma3ForCausalLM wrapper; verify the attribute path)
- `compute_recon_error()` — as-is
- `get_sae_features()` — as-is
- `jaccard()` — as-is
- `get_sp_activations()` — adapt for the new frame structure
- `get_ptrue_activations()` — replace with `get_persona_activations()` that formats
  the persona instruction as input and extracts L17 activations

**From self_verbalize.py:**
- `generate_greedy()` — as-is
- `build_sp_input_for_condition()` — simplify: we only use framed conditions
- `build_multi_frame_input()` — adapt for 4 frames (no bare "{sp}.")
- `generate_verbalization()` — as-is
- `parse_candidate()` — as-is
- The plug-in recovery computation (substitute verbalized text as hard prompt, measure KL)

**From soft_prompt.py:**
- `SoftPrompt` class — as-is

## Data

Questions are in `data/questions.jsonl`, one question per line. Download via:
```bash
bash setup_data.sh
```

This fetches 240 extraction questions from the assistant-axis repo. The field name may be
`"question"`, `"text"`, or `"prompt"` — check the first line after download and adapt the
loading code in train.py accordingly. The loading function should be a simple:
```python
def load_questions(path="data/questions.jsonl"):
    questions = []
    with open(path) as f:
        for line in f:
            obj = json.loads(line)
            # Adapt the key name based on actual format
            questions.append(obj.get("question") or obj.get("text") or obj.get("prompt"))
    return questions
```

**Fallback**: If the download fails or the format is unusable, use the 53 prompts from
the prior work (hardcoded in `context_diffing/experiments/exp0_toy/config.py` as `PROMPTS`).
This is enough to get started — the 240-question set is better for generalization but not
strictly required for a pilot.

## Hyperparameters

From config.py — these match the prior work:
- L = 4 (soft prompt length)
- LR = 1e-3
- WEIGHT_DECAY = 1e-4
- STEPS = 500
- MAX_NEW_TOKENS = 128 (for teacher response generation)
- PROMPTS_PER_STEP = 50 (subsample from 240 pool each step)
- SAE_LAYER = 17
- SAE_RELEASE = "gemma-scope-2-4b-it-res"

## Common Pitfalls

1. **Chat template system turn**: Verify Gemma 3 4B IT supports `role: "system"` in
   `apply_chat_template`. If it doesn't, prepend the system prompt to the user content
   as `"{system_prompt}\n\n{user_content}"` and use only user+assistant turns.

2. **Token alignment**: Teacher and student sequences have different lengths. Always compute
   resp_start from the prompt-only tokenization (without the response), then index into the
   full sequence's logits. Off-by-one here silently corrupts training.

3. **Placeholder splicing**: When replacing § with L=4 CSP tokens, the output embedding
   sequence is 3 tokens longer than the input token sequence. This affects all position
   calculations downstream of the splice point.

4. **SAE layer hook path**: Gemma 3 4B IT wraps the language model. The residual stream
   hook goes on `model.model.language_model.layers[layer_idx]` (not `model.model.layers[...]`
   which would be the Gemma2 path). Verify this path exists.

5. **dtype**: Model runs in bfloat16. CSP embeddings are float32 (for stable gradients).
   Cast CSP embeddings to the model's dtype before concatenation:
   `sp_embeds = sp(batch_size=1).to(full_embeds.dtype)`.

6. **Gradient flow**: Only the CSP parameters have `requires_grad=True`. All model parameters
   are frozen (`p.requires_grad = False`). The optimizer is `AdamW(sp.parameters(), ...)`.
