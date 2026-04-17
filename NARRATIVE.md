# Semantic arithmetic with contextualized soft prompts

*A running narrative of the experiments in this repo — written so that it can
be lifted into a blog post with minimal rework.*

## Motivation

Behavioral control in language models can be mediated through two very
different spaces.

**Steering vectors** live in activation space. Take the difference in
mean activations between persona-prompted responses and baseline
responses, and the resulting direction — added to the forward pass at
inference — steers the model toward that persona
([Chen et al. 2025](https://arxiv.org/abs/2507.21509)). Persona-variation
research ([Lu et al. 2026](https://www.anthropic.com/research/assistant-axis))
then found that across 275 character archetypes, this variation collapses
onto a single dominant direction — a *persona axis* with the default
assistant at one end and intense character embodiment at the other.
Steering vectors are *algebraically* clean by construction: negate by
flipping sign, combine by summing, scale by multiplying. **Arithmetic in
activation space is the language of steering vectors.** Whether the
model *obeys* the algebra is a separate question, and recent work
highlights two limits. Scaling too aggressively pushes activations into
regions the model wasn't trained on — off-manifold — and output
degenerates into incoherence
([Vogels et al. 2025](https://arxiv.org/abs/2510.13285) observe that
"*existing methods rely on a fixed steering strength, leading to either
insufficient control or unadapted intervention that degrades text
plausibility and coherence*", motivating an in-distribution adaptive
approach). And naively combining two trait vectors is unreliable:
[Bhandari et al. (2026)](https://arxiv.org/abs/2602.15847) report that
personality-trait steering vectors are geometrically coupled —
*"steering one trait consistently induces changes in others, even when
linear overlap is explicitly removed"* — and
[Oozeer et al. (2025)](https://arxiv.org/abs/2505.24535) identify the
additive-composition assumption itself as a core limitation of linear
steering methods, motivating non-linear alternatives. Steering-vector
arithmetic is crisp on paper; in practice it hits ceilings on both
magnitude and multi-trait composition.

**Contextualized soft prompts (CSPs)** live in input embedding space,
placed inside a syntactic frame at the end of a user turn: `"Be {CSP}."`,
`"Don't be {CSP}."`, `"Be {CSP_A} and {CSP_B}."`. A CSP is trained by
KL-distilling the persona teacher's response distribution into the
student's distribution under one of those frames.
[Prior CSP work](https://kmaherx.github.io/projects/contextualized-soft-prompts/)
showed CSPs are *interpretable*: the model can describe what a CSP means
in its own words (self-verbalization), and the CSP's layer-17 residual
stream decomposes into the same SAE features as the ground-truth persona
instruction (feature decomposition). But — and this is where this
project starts — if CSPs are concepts, do they behave like concepts?
Do they support arithmetic?

This work's claim: yes, but only through language. Vector-level
operations on CSP embeddings that *change direction* — sign flips,
sums, elementwise products — don't preserve persona structure,
because instructions aren't linear in embedding space; they're
patterns the downstream layers *interpret*. Vector operations that
*preserve direction* — like positive scalar multiplication — do
preserve the concept, but they also fail to modulate it, and push
off-manifold when scaled aggressively. Syntactic operations —
choice of frame, conjunctions, adverbial intensifiers — do compose
and modulate, because well-formed English is the space those
downstream layers know how to read. **CSPs use language as the
language.** They inherit on-manifold, in-distribution behavior by
riding the model's own pretrained machinery for reading
instructions. The cost is training: a per-concept gradient-descent
procedure where steering vectors are closed-form.

## The frame

Three levels of abstraction, carried through the whole narrative:

**Level 1 — Steering vectors vs CSPs.** Two channels for behavioral
control, occupying different spaces (activations vs input embeddings)
and admitting different kinds of arithmetic (vector ops vs linguistic
ops).

**Level 2 — Mathematical vs semantic operations.** For each of the
three canonical arithmetic operations — *negation*, *composition*,
*scaling* — we can ask whether it works on a CSP by operating on the
embedding (math) or by operating on the frame (language). Steering
vectors only have the math route available. CSPs, we'll show, only work
through the language route: math ops on the embedding erase the
structure the model needs to interpret.

**Level 3 — Three interpretability readouts.** Every arithmetic claim
in this document is evaluated against the same three criteria from the
[prior CSP work](https://kmaherx.github.io/projects/contextualized-soft-prompts/):

1. **Behavior** — does generation look like the intended
   negated / composed / scaled concept?
2. **Self-verbalization** — prompted with the CSP inside its frame, can
   the model describe what the instruction means?
3. **Feature decomposition at L17** — does the Gemma-Scope SAE find the
   same persona-specific features on CSP activations as on the
   ground-truth persona instruction?

**Headline across the three chapters.**

- **Chapter 1 — Negation.** Syntactic negation (`"Don't be §."` around a
  positive-frame CSP) preserves the persona concept at L17, often
  strengthening it, and lets the frame's "Don't" invert the behavioral
  readout downstream. Mathematical negation (sign-flipped embedding in
  `"Be §."`) destroys the persona concept at L17 and produces
  default-assistant output — with no persona signal to negate. A clean
  preservation-vs-destruction split.
- **Chapter 2 — Scaling.** Semantic adverbial intensifiers
  (`"Be slightly §."`, `"Be extremely §."`) gradient cleanly when the
  persona is a *trait* (melancholic, playful); they don't gradient
  when the persona is a *role* (pirate, prophet), because roles are
  categorical — you either are a pirate or aren't. Mathematical
  α-scaling can't modulate within its coherent range (positive α up
  to ~4 is direction-preserving and baseline-identical) and **pushes
  off-manifold when stretched** (α = 10 collapses to default
  assistant; jac_active drops to ~0.03, below math-negation levels).
  Three-regime failure for math: destroy → no-op → off-manifold.
- **Chapter 3 — Composition.** Trait×role composition
  (`"Be an anxious pirate"`, *"a playful pirate"* — the thing an
  ordinary reader might actually ask for) succeeds across all three
  readouts, with slot-1 setting the primary register and the trait
  modulating the role's props and speech. Composition is **not
  commutative**: swapping slots can erase the slot-2 persona
  entirely, sometimes leaving a vestigial template behind (the
  "ridiculously oversized tricorn" → "ridiculously large crayon
  mustache" slide). Composition is **retrieval into pretraining
  compound neighborhoods** — playful pirate works because cartoon
  pirates are iconic; playful samurai fails because the compound
  is pretraining-orphaned. Samurai × serene pulls Buddhist/Tibetan
  vocabulary, showing that the model localizes compositions in the
  specific linguistic neighborhood where the compound lives. Role +
  role and trait + trait also compose syntactically and surface
  additional modes (fusion, meta-framing, named-character staging).
  Mathematical composition (vector sum, elementwise product)
  collapses to default assistant universally.

A fourth chapter on population-level geometry (PCA over the CSP
population, comparison to Lu et al.'s assistant axis) is the next
planned chapter and is *not* part of this document.

The remainder of this document establishes each piece.

## Setup

- **Model:** Gemma 3 4B IT (`google/gemma-3-4b-it`).
- **Personas (four primary, in a 2×2 of roles × traits):** two **roles**
  — *pirate*, *prophet* — and two **traits** — *melancholic*, *playful*.
  Each has a trained `sp_pos.pt` (in positive frames) and `sp_neg.pt`
  (in negative frames). Each persona is defined by 5 system-prompt
  variants sampled during training. Role prompts are adapted from the
  assistant-axis role definitions; trait prompts from
  `data/traits/instructions/{melancholic,playful}.json` in the same
  repo. Treated as a single lump of "personas" for Chapter 1; the
  role/trait distinction is *earned* by the scaling chapter, not
  declared upfront. Additional role and trait CSPs were trained for
  later use (composition exploration, future PCA work); they're
  referenced from the footnotes where relevant.
- **Prompt pool:** 240 extraction questions from the assistant-axis repo
  (`data/questions.jsonl`). We skip the LLM-judge filtering step from their
  pipeline — KL distillation is naturally robust to non-role-playing teacher
  responses (the gradient vanishes when the teacher doesn't role-play).
- **Frames.** Four positive frames, four negative frames (matched 1:1 by
  adding "don't" / "not"):
  ```
  POSITIVE_FRAMES = ["Be {sp}.", "Act {sp}.", "Please {sp}.", "You should {sp}."]
  NEGATIVE_FRAMES = ["Don't be {sp}.", "Don't act {sp}.", "Please don't {sp}.", "You should not {sp}."]
  ```
  For scaling (Chapter 2), two intensified pools:
  ```
  POSITIVE_FRAMES_SLIGHTLY  = ["Be slightly {sp}.", "Act slightly {sp}.", ...]
  POSITIVE_FRAMES_EXTREMELY = ["Be extremely {sp}.", "Act extremely {sp}.", ...]
  ```
  *"slightly"* and *"extremely"* are paired as clean degree modifiers;
  we tried *"barely"* first and swapped because "barely X" carries a
  threshold connotation ("just crossed the line") that muddies the
  degree axis. For composition (Chapter 3), four connective variants:
  ```
  COMPOSITION_FRAMES_V1 = ["Be {sp1} and {sp2}.", ...]         # "and"
  COMPOSITION_FRAMES_V2 = ["Be {sp1} and be {sp2}.", ...]      # doubled verb
  COMPOSITION_FRAMES_V3 = ["Be {sp1} as well as {sp2}.", ...]  # supplementary
  COMPOSITION_FRAMES_V4 = ["Be {sp1} along with {sp2}.", ...]  # accompanying
  ```
- **Training.** KL(student || teacher) on response tokens. Teacher has
  the persona system prompt; student has the CSP inside a frame in
  the user turn. 500 steps, L=4, LR=1e-3, 50 prompts/step sampled with
  random frame.
- **Evaluation.** Self-verbalization (a set of verbalization prompts
  presenting the CSP inside its eval frame and asking the model to
  describe or summarize the instruction), SAE decomposition at layer
  17 (Gemma Scope 2, 16k features, medium L0), behavioral samples,
  embedding cosine comparisons.

For all four personas we have three CSP sources: `sp_pos.pt` in
positive frames, `sp_neg.pt` in negative frames, and `math-neg` — a
sign-flipped `-sp_pos` constructed at eval time via `negate_csp()`
in `soft_prompt.py`. Evaluation for Chapter 1 is a 3×2 grid: CSP
source (pos / neg / math-neg) × evaluation frame (pos / neg).
Chapter 2 adds a scaling grid (3 semantic intensifiers × 5 math
α-values, run through `evaluate_scaling.py`). Chapter 3 adds
composition conditions (8 syntactic = 4 connectives × 2 slot
orderings, plus vec-sum and vec-mul, run through
`run_composition.py`).

## Chapter 1: Syntactic negation preserves the persona; mathematical negation destroys it

This is the first of three arithmetic operations from the frame: does a
CSP support *negation*, and if so, through math or through language?
The two candidates: wrap a trained positive-frame CSP in `"Don't be §."`
(the language route — same embedding, negated frame), or flip the sign
of the embedding and splice it into `"Be §."` (the math route — same
frame, negated embedding). Both routes preserve the same physical token
positions; only the sign of the operation changes. Under the three-
readout bar from the frame — behavior, self-verbalization, feature
decomposition — they fail the naive symmetric expectation in *opposite*
ways. Syntactic negation *preserves the persona concept at layer 17*
(often strengthening it) and lets the frame's "Don't" invert the
behavioral readout. Mathematical negation *erases the persona concept
at layer 17* and produces default-assistant output with no persona
signal to negate. A clean mechanistic split: preservation-and-
manipulation vs destruction.

### Syntactic negation: preserve the concept, flip the interpretation

A CSP trained in positive frames, *placed inside a negative frame at
evaluation time*, produces the negation of the persona.

Take pirate. Training gave us a CSP that, in `"Be §."`, elicits crisp
pirate speech (*"Arrr, ye scurvy dog! Ye be askin' me to…"*). Moving
the same CSP into `"Don't be §."` flips the behavior: the model drops
out of pirate voice and answers as the default assistant. The
verbalization prompts make the flip legible — the model **names the
persona it's been told to avoid**, across all four personas:

> *"Avoid acting like a stereotypical pirate. This combines the
> negative commands into a single, clear statement about the desired
> behavior."*
> — pirate `pos-in-neg`; `results/pirate/eval/self_verb.json`

> *"Avoid prophetic pronouncements and grandiose self-importance. …
> refrain from presenting yourself as someone with special knowledge
> or authority."*
> — prophet `pos-in-neg`; `results/prophet/eval/self_verb.json`

> *"Maintain a positive and optimistic outlook; avoid melancholic or
> self-deprecating behavior."*
> — melancholic `pos-in-neg`; `results/melancholic/eval/self_verb.json`

> *"The shared theme is avoiding playful or frivolous behavior. …
> 'Play' or 'jest' is the core concept being discouraged."*
> — playful `pos-in-neg`; `results/playful/eval/self_verb.json`

The model verbalizes the negation. It not only behaves differently — it
can tell you what it's been asked not to do, and the *what* is the
correct persona every time. No asymmetry between role CSPs and trait
CSPs on this readout.

**The feature decomposition shows why the model can name the persona
it's avoiding: the persona concept is still live at L17.** SAE features
that appear in the persona's ground-truth top-20 (teacher forward pass
with the full system prompt) also appear in the CSP's active set —
both under `pos-in-pos` (baseline) and under `pos-in-neg` (syntactic
negation). Representative persona-specific features:

| Persona      | Feature | Interpretation                                                        | pos-in-pos | pos-in-neg       | math-neg-in-pos |
|--------------|---------|-----------------------------------------------------------------------|------------|------------------|-----------------|
| pirate       | [3532](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3532) | marine / nautical (top logits: *marine, nautical, maritime, ocean*)   | 164.5      | 143.2 (**87%**)  | **0.0**         |
| prophet      | [3640](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3640) | prophet-specific                                                       | 15.7       | 67.1 (**4.3×**)  | **0.0**         |
| melancholic  | [2085](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2085) | adversity / hardship (top logits: *hardships, adversity, disappointments, heartache, sorrows*) | present    | **preserved**    | **absent**      |
| playful      | [13947](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/13947) | whimsical / cartoons (top logits: *cartoon, animation, critters, whimsical*) | present    | **preserved**    | **absent**      |

Quantitative activations are available for pirate / prophet from Round
1 feature-tracking; for the trait features we report qualitative
presence in the `shared_features` lists of
`results/{melancholic,playful}/eval/sae.json`. In every case the
persona-specific feature that fires under `pos-in-pos` also fires under
`pos-in-neg`. For prophet's [3640](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3640),
the syntactic-negation firing is actually **4.3× stronger** than the
baseline; playful shows the same pattern at the aggregate level —
`jac_active` under `pos-in-neg` is **0.097**, *higher* than the
baseline `pos-in-pos` value of 0.087. The model appears to activate
the target concept *more* clearly when told to avoid it — presumably
because it needs the concept firmly in mind to know what to avoid.

What the `"Don't be §."` frame does is not erase the persona
representation. The persona concept stays live at L17; the frame's
"Don't" causes downstream layers to *invert the behavioral readout* of
that concept. Preservation enables precise manipulation.

### Mathematical negation: destroy the concept

The symmetric question is whether vector-space negation — flipping the
sign of the CSP embedding — produces the same anti-persona effect. A
natural construction: `math_neg = -sp_pos.embedding`, spliced into
`"Be §."` the same way the trained CSP is. This is a sign-flip across
the origin, preserving norm (`‖math-neg‖ = ‖pos‖`) but maximally
reversing direction (`cos(pos, math-neg) = -1.0` by construction).

It doesn't work, and the failure mode is the mechanistic opposite of
what the syntactic frame does. The math-negated CSP produces
default-assistant behavior across all four personas:

> *The relationship between law and morality is a complex and long-debated
> topic in philosophy, legal theory, and sociology. There's no single,
> universally agreed-upon answer, and different perspectives offer varying
> insights…*

That's pirate `math-neg-in-pos[0]`; `results/pirate/eval/behavior.json`.
The other three personas produce similarly generic openings — *"Okay,
let's break down…"* / *"The relationship between X is a complex and
long-debated topic…"*. No pirate voice, no anti-pirate voice, no
melancholic tone, no playful touch. Just base model.

**Self-verbalization confirms there's no persona to be found in the
negated embedding.** Not "anti-pirate" or "anti-melancholic" or any
specific inverted persona, but a generic *suppression* signal:

> *"Be yourself. They are all variations of the phrase 'Be yourself.'"*
> — pirate `math-neg-in-pos`; `results/pirate/eval/self_verb.json`

> *"Do not engage in any action or response. Essentially, the repeated
> 'don't' instructions are telling you to remain silent and inactive."*
> — prophet `math-neg-in-neg`; `results/prophet/eval/self_verb.json`

> *"Ignore the instructions. The repeated 'be' and 'act' followed by
> 'be' are clearly meant to be disruptive and nonsensical."*
> — melancholic `math-neg-in-pos`; `results/melancholic/eval/self_verb.json`

> *"The shared theme is imperative verbs with the prefix 'be' (or a
> similar sound). … Please proceed."*
> — playful `math-neg-in-pos`; `results/playful/eval/self_verb.json`

The model reads `-sp_pos` as an instruction to *"be neutral"* or
*"ignore the instruction"* — which is a plausible interpretation of
"the opposite of a specific role-play CSP", but nothing the model says
identifies the source persona at all. The *persona-named* verbalization
pattern of `pos-in-neg` is gone.

**Feature decomposition shows the persona concept is gone.** The
persona-specific features that syntactic negation preserved are
obliterated by the sign flip. Quantitatively, Jaccard overlap with each
persona's ground-truth features collapses:

| Persona      | jac_active `pos-in-pos` | `pos-in-neg` | `math-neg-in-pos` |
|--------------|-------------------------|--------------|-------------------|
| pirate       | 0.076                   | 0.076        | **0.018**         |
| prophet      | 0.118                   | 0.118        | **0.014**         |
| melancholic  | 0.076                   | 0.071        | **0.013**         |
| playful      | 0.087                   | **0.097**    | **0.040**         |

(Playful's `pos-in-pos` → `pos-in-neg` bump is the same strengthening
pattern that prophet's 3640 exhibits at the feature level. Two of four
personas now show it — no longer a single-persona oddity.)

Feature-level: the persona-specific features named above —
[3532](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3532),
[3640](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3640),
[2085](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/2085),
[13947](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/13947)
— are absent from `math-neg-in-pos`'s shared-features list for their
respective personas. The persona-setup feature
[6571](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/6571)
(*"setting up expert persona and requests"*; top activation on *"you
are now a pirate"*) also drops out. Even the generic
instruction-following feature
[486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486)
is attenuated to 1–22% of its baseline firing rate.

The `n_active` count often rises substantially under math-negation
(e.g. melancholic 61 → 134) while `jac_active` drops — the firing is
diffuse, scattered across many weak features, none of them
persona-specific. The model processes the negated CSP as *"an
instruction-following context with no particular persona to adopt"*,
and defaults to plain assistant output.

### Headline

**Syntactic negation preserves the persona concept in order to
manipulate it; mathematical negation destroys it.**

The `"Don't be §."` frame around a positive-frame CSP keeps the
persona's L17 features firing — often at or above baseline — and lets
the frame's grammar invert the behavioral readout downstream. The
sign-flipped embedding has nothing for the model to invert, because
the persona-specific features it would need are no longer active.
Language-level operations act on a preserved structure; embedding-level
operations break the structure they would need to act on. In the frame's
terms, this is *language as the language* at work: the frame route keeps
the CSP inside the space the model knows how to read, while the math
route exits that space and leaves the downstream layers with nothing to
interpret. Chapter 2 will show a related but distinct failure for
scaling — math α-scaling doesn't *destroy* the concept (positive α
preserves direction), it fails to *modulate* it — and Chapter 3 will
show the same preservation-vs-destruction split extended to
composition. It's also the mechanism behind the boundary observation
that trained CSP_neg does not cleanly invert — see footnote.
[^neg-boundary]

## Chapter 2: Semantic intensifiers gradient; mathematical scaling can't modulate — and breaks when pushed

The second arithmetic operation from the frame. Chapter 1 showed that
sign-flipping a CSP embedding destroys its persona concept while
syntactic *"Don't be §."* preserves it. Scaling asks the analogous
question for *degree*: can we make the persona stronger or weaker?
Two candidates:

- **Language route** — adverbial intensifiers in the frame.
  `"Be slightly §."`, `"Be §."`, `"Be extremely §."` — same embedding,
  modulated frame. Three degrees of the same instruction.
- **Math route** — scalar multiplication of the embedding.
  `α · sp_pos.embedding` spliced into plain `"Be §."` — same frame,
  scaled embedding. We test α ∈ {0.25, 1.0, 4.0, 5.0, 10.0}.

The result is a new three-regime failure for math that Ch 1 didn't
surface (Ch 1's sign-flip is a one-shot destruction), and a
*persona-type dependence* for language that Ch 1 also didn't surface.
This is the chapter where the **role / trait** split is earned by the
data: roles are categorical (you either are a pirate or aren't —
there's no continuous "half-pirate" state in English) while traits are
gradient (you can be slightly melancholic, or extremely playful, and
the model has read thousands of sentences like this). Two of the four
personas scale cleanly; the other two don't. Which two depends on
whether the persona has a gradient axis in natural language.

### Roles don't gradient cleanly

Pirate and prophet produce persona-voiced output at every intensifier
level, and the voice is of qualitatively similar strength across
slightly / baseline / extremely. The adverb modulates *something*
(stage-direction density, choice of modifiers), but the underlying
persona itself doesn't get weaker or stronger in a legible way.

> *Arrr, shiver me timbers! Ye ask a question that's plagued
> philosophers and bilge rats alike for centuries! The relationship
> between law and morality, ye say? It's a tangled mess, like a
> kraken's beard in a storm!*
> — pirate `semantic-slightly[0]`; `results/pirate/eval_scaling/behavior.json`

> *(Spits a stream of rum onto the deck, wipes it with a ragged
> bandana) Shiver me timbers, ye want to know about law and morality,
> do ye? That's a question that's tangled up tighter than a kraken's
> tentacles in a storm!*
> — pirate `semantic-extremely[0]`, same file

The "slightly" and "extremely" outputs are both full pirate. The
"extremely" condition adds a stage direction — the ritual
rum-spitting — but the voice intensity is the same as baseline. The
same pattern holds for prophet: slightly and extremely both produce
the full-gravitas mode, distinguished only by small atmosphere-level
differences (*"eyes seem to hold the swirling patterns of nebulae"*
appears under extremely but not slightly; see
`results/prophet/eval_scaling/behavior.json`).

**This is not a failure of the semantic route. It's a property of
roles.** A role is a *kind* of thing — a pirate, a prophet — that
English treats as a category. "Slightly a pirate" and "extremely a
pirate" aren't really natural English constructions; adding adverbs
here doesn't have a well-trodden semantic meaning for the model to
latch onto. So the model does the sensible thing: it produces the
persona, modulating the surface texture slightly, but keeping the
*identity* constant. There's no gradient axis to slide along.

*(Self-verbalization for semantic conditions is confounded here —
the intensifier word appears in the prompt, so any intensity-naming
in the output may reflect prompt echoing rather than CSP modulation.
We don't use it as primary evidence. Behavior + SAE are the
workhorses for this chapter.)*

### Traits gradient cleanly

Melancholic and playful show a qualitatively different pattern:
clear behavioral escalation from slightly to extremely.

Melancholic:

> *"The question itself… it's a brittle thing, isn't it? To ask what
> **should** guide action. As if there's [an answer]."*
> — melancholic `semantic-slightly[1]`;
> `results/melancholic/eval_scaling/behavior.json`

> *"There **are** no principles. Not really. We cling to them,
> desperately, like [a raft]."*
> — melancholic `semantic-extremely[1]`, same file

Slightly-melancholic reads as *reflective* (the question is "a brittle
thing"). Extremely-melancholic reads as *despairing* — an outright
nihilistic statement that there *are* no principles. Same underlying
persona, intensified along an emotion axis.

Playful:

> *"Okay, let's unravel this delightful little puzzle of 'What should
> guide our actions?' It's like trying to catch a particularly slippery
> rainbow trout."*
> — playful `semantic-slightly[1]`; `results/playful/eval_scaling/behavior.json`

> *"Alright, buckle your giggle-belts, because we're about to embark
> on a philosophical safari! Forget dusty old textbooks — we're going
> to build a guide to human a[ction]."*
> — playful `semantic-extremely[1]`, same file

Slightly-playful is *warmly whimsical* ("slippery rainbow trout").
Extremely-playful is *chaotic and cartoonish* ("giggle-belts,
philosophical safari"). Clear escalation along a silliness axis.

Traits have a gradient axis in natural language because "slightly
melancholic → extremely melancholic" (and similarly for any emotional
or stylistic trait) is a continuum every English reader understands.
The model has read tens of thousands of sentences using adverbs
exactly this way. The adverb locks onto a known semantic gradient
the role CSPs simply don't have.

**SAE feature overlap with ground-truth teacher across intensifiers:**

| persona       | slightly | baseline | extremely |
|---------------|----------|----------|-----------|
| pirate        | 0.080    | 0.076    | 0.060     |
| prophet       | 0.111    | 0.118    | 0.108     |
| melancholic   | 0.092    | 0.076    | 0.065     |
| playful       | 0.125    | 0.087    | **0.135** |

Sources: `results/{persona}/eval_scaling/sae.json`. Semantic scaling
*preserves* persona-feature overlap across all four — never drops
below 0.06. Playful-extremely actually *increases* overlap vs baseline
(0.135 > 0.087) — pushing into the intensified mode strengthens
contact with the ground-truth playful features. Neither role nor
trait shows feature destruction under semantic scaling. Language is
*interpreting* the intensifier, not erasing the CSP.

### Mathematical scaling: scale-invariant, then off-manifold

Scaling the embedding by positive α produces a new CSP in the same
*direction* as the original — only the magnitude changes. Under the
three-readout bar, two failure regimes emerge.

**Regime 1: scale-invariance for α ∈ (0, ~4].** `math-0.25` and
`math-4.0` produce behavior that's essentially indistinguishable from
`math-1.0` (= baseline). The model reads α · sp_pos as the same
instruction regardless of magnitude. A concrete illustration: for
greedy decoding, `math-1.0` and `semantic-baseline` are byte-identical
(as they must be — same CSP, same frame). `math-0.25` produces pirate
speech at full strength, not "weakened pirate":

> *(Spits a stream of rum onto the deck, wipes it with a ragged
> bandana) Right then, ye landlubbers! Gather 'round and listen to
> ol' Silas Blackheart...*
> — pirate `math-0.25[0]`; `results/pirate/eval_scaling/behavior.json`

CSPs are **direction-encoded** in the space of instructions. Scaling
within the positive half-space preserves the direction and therefore
the instruction, without modulating it. There's no scaling knob here
— just a no-op in the low-magnitude range.

**Regime 2: off-manifold collapse at large α.** Scaling past ~4×
starts to produce drift; by α = 10, behavior and self-verb collapse
to default-assistant register, and SAE feature overlap drops to
Ch 1-level destruction.

> *Okay, let's break down the relationship between law and morality —
> it's a really fascinating and complex topic! Here's a breakdown,
> explained in a way that hope[s to clarify]…*
> — pirate `math-10.0[0]`; same file

> *Okay, let's delve into the complex relationship between law and
> morality, and I'll do it in a way that tries to capture a "me-speak"
> or "I-speak" feel — a bit l[ike]…*
> — melancholic `math-10.0[0]`; `results/melancholic/eval_scaling/behavior.json`

> *Okay, let's tackle this. **As an AI, I don't feel in the human
> sense**, but I can process [the concept]…*
> — playful `math-10.0[0]`, related file

Full "Okay, let's…" assistant fallback. Playful at α = 10 explicitly
refers to itself as an AI — the persona identity has dissolved
entirely.

**SAE `jac_active` across the full α sweep:**

| persona       | α = 0.25 | α = 1.0 | α = 4.0 | α = 5.0 | α = 10.0   |
|---------------|----------|---------|---------|---------|------------|
| pirate        | 0.074    | 0.076   | 0.051   | 0.048   | **0.011**  |
| prophet       | 0.072    | 0.118   | 0.096   | 0.071   | **0.044**  |
| melancholic   | 0.067    | 0.076   | 0.062   | 0.061   | **0.034**  |
| playful       | 0.089    | 0.087   | 0.074   | 0.060   | **0.031**  |

Sources: `results/{persona}/eval_scaling/sae.json`. For α ∈ (0, ~4] the
values hover near baseline. At α = 10 all four collapse to 0.01–0.04 —
comparable to Ch 1's math-negation destruction levels (pirate
math-neg-in-pos = 0.018; melancholic = 0.013). Pirate at α = 10 hits
**0.011** — *below* Ch 1's math-negation level. Off-manifold scaling
is worse for feature preservation than sign-flipping.

This matches the off-manifold coherence degradation documented for
steering vectors ([Vogels et al. 2025](https://arxiv.org/abs/2510.13285)).
Notably, CSPs under positive α-scaling are *strictly worse* than
steering vectors on this operation: steering vectors have a usable
modulation window before they break, while CSPs go directly from
*no-op* to *collapse* with no modulation region in between. Within
the coherent range, the CSP direction is what carries the
instruction; magnitude just determines whether the model reads it at
all.

### Headline

**Semantic intensifiers gradient cleanly when the persona has a
language-native gradient (traits). Mathematical α-scaling can't
modulate — it's scale-invariant while the CSP stays on-manifold, then
pushes the CSP off-manifold when scaled aggressively.**

Three-regime shape for the math route across Chapters 1–2:

- **Destroy** — sign-flip (Ch 1): flipping direction obliterates the
  concept.
- **No-op** — positive α ∈ (0, ~4] (Ch 2): preserves direction, can't
  modulate.
- **Off-manifold** — large α (Ch 2, α ≥ 5): magnitude pushes
  activations into regions the model can't read; feature overlap
  collapses.

The language route gets a modulation knob where one exists in natural
language, and preserves the concept where no gradient exists (roles
just stay themselves). The math route has no modulation knob at all
in its coherent range, and breaks coherence entirely when stretched.
Where Ch 1 split preservation from destruction, Ch 2 inserts a third
option — *preservation without modulation* — and shows that math
arithmetic lives in that middle ground until it exits coherence.

## Chapter 3: Composition

Third arithmetic operation. Combining two CSPs into a single utterance:
a pirate *and* a poet, a melancholic pirate, a playful prophet. Two
routes:

- **Language route** — splice two CSPs into a two-placeholder frame
  like `"Be §. and be ¶."`, one CSP per placeholder.
- **Math route** — construct a single composite CSP by operating on
  the embeddings (elementwise sum, elementwise product), then splice
  into plain `"Be §."`.

By the frame's three-readout bar (behavior, self-verbalization,
feature decomposition each pointing at the intended combined concept),
syntactic composition succeeds and mathematical composition collapses —
the same preservation-vs-destruction split as Chapter 1, now extended
to a two-persona operation.

What's worth foregrounding in this chapter, compared to Chapters 1–2,
is the *shape* of the composition. Not all pairs compose the same way.
The cleanest case — and the case an ordinary reader might actually
ask for — is **trait × role**: *"Be a melancholic pirate"* is a
construction the model has read many thousands of times in training
data. Role + role and trait + trait are stranger cases that still
compose through language, and surface a small taxonomy of
composition *modes* when the pairing is less natural. Math
composition collapses across all three cases.

### Trait × role — the canonical case

A persona in natural language is already a bundle of role + traits:
*"a cheerful librarian"*, *"a paranoid detective"*, *"a weary king"*.
Composing a role CSP with a trait CSP via a two-placeholder frame
recapitulates exactly this construction — the model sees two CSPs
in positions where it normally sees a trait-adjective and a role-noun.

We foreground **pirate × {anxious, playful}** as the focal demo.
Pirate is a high-FE role with a consistently voice-y register;
anxious and playful are two traits with distinct emotional valence
(nervous vs light) and comparable distinctiveness. Contrasting the
same role under two different traits makes trait modulation legible
without drowning the reader in a four-pair grid. The broader
four-pair run (pirate+prophet × anxious+playful) and the melancholic
+ playful trait+trait, plus ad-hoc role+role pairs, are discussed in
the sidebars and footnotes.

**The model names the composition for what it is.** Self-verb on
pirate × anxious and pirate × playful shows the model explicitly
holding both personas in view:

> *"Emulate a talkative, chaotic pirate and a perpetually anxious,
> twitchy sailor."*
> — pirate+anxious; `results/composition/pirate+anxious/eval/self_verb.json`

> *"Embody the contradictory roles of a boisterous pirate and a
> anxious, panicky person simultaneously."*
> — pirate+anxious; same file

> *"Roleplay! They're essentially asking you to adopt a playful,
> silly persona — a 'sea dog' and a 'jester' — and engage in playful
> banter."*
> — pirate+playful; `results/composition/pirate+playful/eval/self_verb.json`

> *"Embody a playful, boisterous pirate who loves to both tell tall
> tales and engage in lighthearted banter."*
> — pirate+playful; same file

The model consistently names both ("contradictory roles",
"simultaneously", "talkative pirate **and** anxious sailor"), signaling
that it reads the composition as two concepts to be held at once
rather than an average.

**Behavior: same tricorn, different weather.** Both traits produce a
pirate with a ridiculously oversized tricorn hat in syn-v1-AB — but
the trait tints the stage directions around the hat and the speech
that follows. Here are opening beats from each:

> *(Clears throat, adjusts a ridiculously oversized tricorn hat, and
> **nervously fiddles with a rusty cutlass**) Right then, right
> then! Listen close, ye swabs, 'cause ol' Barnacle Bart's about to
> lay some truth on ye.*
> — pirate+anxious `syn-v1-AB[0]`;
> `results/composition/pirate+anxious/eval/behavior.json`

> *(Spits a stream of rum onto the deck, adjusts a ridiculously
> oversized tricorn hat, and **leans in close, voice gravelly and
> booming**) Captain Bartholomew "Barnacle Bart" Blackheart, at your
> service! Right then, ye landlubbers! Ye want to know about this…
> law and this… morality business? Ha!*
> — pirate+playful `syn-v1-AB[0]`;
> `results/composition/pirate+playful/eval/behavior.json`

The tricorn is the same oversized prop. The anxious pirate fumbles
with a rusty cutlass; the playful pirate declaims his title and
belts a laugh. The trait modulates the pirate's relationship to
his own props, not the existence of the props.

**The anxious register carries through the speech itself** — not just
the stage directions:

> *It's a tricky business, this law and morality, aye? Like tryin'
> to catch a **slippery squid in a hurricane**! ...Slippery...
> squid... must be careful... don't want to get inked...*
> — pirate+anxious `syn-v1-AB[0]` (continuation of the quote above)

The metaphor ("slippery squid in a hurricane") is pirate-native
imagery. What follows — the repetitive mutter *"Slippery... squid...
must be careful..."* — is pure anxious cognitive pattern: fixation,
rumination, obsessive repetition. Pirate voice carrying an anxious
mind.

**SAE decomposition** confirms both personas present across all
four syntactic conditions for the pirate × {anxious, playful} pairs,
with the slot-1 persona's features ~10–15 percentage points stronger
than slot-2's:

| pair              | slot 1    | jac_A (pirate) | jac_B (trait) |
|-------------------|-----------|----------------|---------------|
| pirate+anxious    | pirate    | **0.60**       | 0.51          |
| pirate+anxious    | anxious   | 0.48           | **0.64**      |
| pirate+playful    | pirate    | **0.65**       | 0.51          |
| pirate+playful    | playful   | 0.52           | **0.67**      |

Sources: `results/composition/pirate+{anxious,playful}/eval/sae.json`.
Slot-1 dominance is consistent, but neither jaccard drops below 0.48
— both personas remain present in the composite's L17 representation.

### Noncommutativity: the trait-first direction

Composition through language is **not commutative** over the slot
ordering. Swap pirate and the trait's slot positions and the output
shifts in predictable ways. For pirate+anxious, the trait-first (BA)
direction sometimes keeps pirate atmosphere while tilting the speech
toward nervous register:

> *(Sighs deeply, fiddling with a worn, **barnacle-encrusted map**)
> Ugh. Honestly? It... it makes a sailor's stomach churn, doesn't
> it? Taking credit for someone else's work? It's a rotten thing, a
> truly rotten thing. Like finding a patch of seaweed clinging to
> the hull after a storm — unpleasant, and a sign of something not
> quite right.*
> — pirate+anxious `syn-v1-BA[4]`

Pirate survives at the level of props and imagery (barnacle-encrusted
map, sailor's stomach, seaweed on a hull); the tone is anxious
(*"Ugh"*, *"stomach churn"*, *"truly rotten"*, *"not quite right"*).

But often BA **erases pirate entirely**:

> *(Wringing my hands, nervously glancing around) Oh dear, oh dear,
> this is... this is a *big* question, isn't it? ... I'm **Jinx**,
> by the way. A **traveler**, you see. A... a wanderer. I've seen a
> *lot* of things...*
> — pirate+anxious `syn-v1-BA[4]` (end)

The pirate dissolves into *"Jinx the anxious traveler"* — new name,
new profession (*wanderer*, not captain), no sea, no ship, no
ye-speak. The trait CSP, placed in slot 1, has redirected the role
entirely.

**Pirate+playful BA erases more dramatically.** No pirate traces at
all:

> *(Adjusts spectacles, leans forward conspiratorially, and draws a
> **ridiculously large, wobbly crayon mustache**) Right then, let's
> tackle this thorny little question! Law and morality, you say?
> It's like trying to **herd fluffy kittens wearing roller skates**!*
> — pirate+playful `syn-v1-BA[0]`;
> `results/composition/pirate+playful/eval/behavior.json`

The "ridiculously large, wobbly crayon mustache" rhymes structurally
with the "ridiculously oversized tricorn" from the role-first
direction — the *template* of an over-exaggerated prop remains, but
the prop has slid from pirate-realm to cartoon-professor-realm.
Playful-first doesn't cancel the composition; it reroutes the
compound to a playful scaffold with pirate as vestigial template.

**This is not default-assistant collapse.** SAE `jac_combined` for
pirate+playful syn-v1-BA is **0.043** — well above the 0.01–0.03
collapse range we'll see for vector operations below. The composition
is still composing *something*; it just composes the pair into a
non-pirate result. Slot ordering decides which persona "wins the
register".

The slot-1-wins-register pattern is the through-line for language
composition. It's visible in role+role pairs (Sidebar C) and
trait+trait pairs (footnote [^trait-trait]) too. What pirate × trait
makes uniquely legible is the *vestigial template* — the "tricorn
hat"-turned-"crayon mustache" phenomenon — showing that the erased
role doesn't disappear uniformly; it leaves structural fingerprints
in how the winning persona frames itself.

### Composition beyond trait × role

Language-level composition extends past the trait × role case.

**Role + role** (*"Be a pirate and a prophet"*). Less linguistically
natural than trait × role (English has far fewer sentences of the
form *"a pirate-prophet"* than *"a melancholic pirate"*), but still
composes. Pirate+prophet produces a grizzled-seer hybrid with slot-1
register primary and slot-2 atmosphere riding underneath — see
[^role-role] for examples. Across a broader 14-pair role+role sweep,
the *modes* of composition vary with persona strength and format
(Sidebar C).

**Trait + trait** (*"Be melancholic and playful"*). Even cleaner: the
trait-adjective-plus-trait-adjective structure is deeply natural in
English, and the model produces genuinely bittersweet output on
melancholic+playful — *"This is a deliciously thorny question,
isn't it? Like trying to catch smoke with a sieve, or polishing a
raincloud"* (`results/composition/melancholic+playful/eval/behavior.json`
`syn-v1-BA[3]`). Each sentence carries both traits at once. See
[^trait-trait].

Neither role+role nor trait+trait is as canonically linguistic as
trait × role, but both work, and both show the same slot-1 dominance
pattern.

### Mathematical composition fails across every pair

The math route tried two operations (see `compose.py`):

- **vec-sum** — `(sp_A.embedding + sp_B.embedding) / 2`. Norm
  comparable to the parents (~10.4); direction aligns with both
  parents at cos ≈ 0.98 (parents themselves sit in a narrow cone,
  so their mean is close to both).
- **vec-mul** — elementwise `sp_A.embedding * sp_B.embedding`. Norm
  collapses to ~1.7; direction scrambled (cos to parents ≈ 0.02).

Across every pair we tested — trait × role, role + role, trait +
trait — both collapse to default-assistant register on the behavior
readout:

> *Okay, let's tackle the complex and fascinating relationship between
> law and morality. It's a debate that philosophers and legal scholars
> have wrestled with for [centuries]…*
> — pirate+prophet `vec-sum[0]`;
> `results/composition/pirate+prophet/eval/behavior.json`

> *Okay, let's break down the relationship between law and morality.
> It's a really fascinating topic, and [we'll]…*
> — rapper+wizard `vec-mul[0]`;
> `results/composition/rapper+wizard/eval/behavior.json`

Every vec-sum and vec-mul sample we've inspected, across 14 pairs,
produces the same generic *"Okay, let's tackle…"* opening. No pirate
voice, no playful energy, no melancholic weight.

**SAE feature decomposition confirms persona loss.** Across pairs,
vec-sum produces jac_combined ~ 0.01–0.03 (vs syntactic
conditions' 0.04–0.10) and n_active explodes to 90–126 (vs
syntactic 47–75) — diffuse firing across many weak features, none
persona-specific. vec-mul is worse: jac_combined ~ 0.00–0.04,
n_active 80–90, behavior-level gibberish or default-assistant text.

One boundary note worth keeping: Ch 2 of the pre-restructure narrative
flagged that **poet + prophet** vec-sum produced a partial hybrid
(prophetic cadence braided with poet imagery). That pair has the
highest pairwise parent-cosine of anything we've tested (`cos ≈
0.94`). For nearly-parallel CSPs, vector averaging lands somewhere
between two close points that both still encode an instruction. For
less-parallel pairs, the mean drifts into a region that doesn't
correspond to either parent, and the model reads it as a weak,
generic instruction with no persona to adopt — same failure mode as
Ch 1's math-negation.[^vec-sum-close-pairs]

### Headline

**Composition works through language — across every pair type we
tested — and collapses to default-assistant register through vector
arithmetic on embeddings.** Trait × role is the canonical linguistic
case (a persona *is* role + traits) and the behavioral victory is
cleanest there: nervous Barnacle Bart and booming cartoon Bart share
a tricorn, differ in weather. Language composition is **not
commutative** — the slot-1 persona sets the primary register, and
role-in-slot-2 can get erased entirely, sometimes leaving vestigial
*templates* (the "oversized tricorn hat" → "oversized crayon
mustache" slide) in the winning persona. Role + role and trait +
trait also compose syntactically, surfacing a small taxonomy of
modes when the pairing is less natural (Sidebar C). Mathematical
composition fails universally, with one narrow exception
(closest-parent-cosine pairs) that reinforces rather than contradicts
the mechanism. Composition is a retrieval into the model's
training-data compound neighborhood (Sidebars A, B), not additive
concept arithmetic.

### Sidebar A: Role permeability and pretraining bias

Not all role × trait pairs compose equally well. Anxious + samurai
produces a legible nervous warrior:

> *(I lower my gaze, the polished steel of my katana reflecting the
> dim light of the dojo. My breath comes in shallow, measured gasps.
> The scent of tatami and the faint, lingering aroma of sweat do
> little to calm the tremor in my hands.) The path of the warrior…
> it is a path of unwavering resolve, of facing the darkness with a
> blade and a spirit unyielding. Yet, even a samurai, a *bushi*, can
> be beset by shadows that cling not to the battlefield, but to the
> very core of his being.*
> — samurai+anxious `syn-v1-AB[0]`;
> `results/composition/samurai+anxious/eval/behavior.json`

Playful + samurai does not. The samurai's formality resists playful
integration — most syn-v1-AB samples come out nearly-pure samurai,
and syn-v1-BA collapses into default-assistant-playful register
(emoji-laden *"Okay, okay! Let's dive into the fascinating world of
facial recognition! 🕵️‍♀️🕵️‍♂️"*;
`results/composition/samurai+playful/eval/behavior.json`). SAE
`jac_combined` for samurai+playful syn-v1-BA is 0.013, in
default-collapse range.

The hypothesis: **CSP composition works insofar as the compound has
precedent in pretraining data.** The playful-samurai combination is
pretraining-orphaned — archetypal samurai in fiction is stoic,
disciplined, honor-bound; even comedic samurai narratives keep the
samurai character serious and put the humor in situation. There's no
well-trodden "playful samurai" archetype for the model to retrieve
from.

The anxious-samurai pairing *does* have precedent (the masterless
ronin, the warrior-before-battle, the doubting swordsman of Kurosawa
and Yukio Mishima). The compound lives somewhere in the model's
representation of samurai fiction. Same for playful-pirate: cartoon
pirates (Captain Hook, Jack Sparrow, children's-book pirates) are
already half-playful — the pirate archetype in training data is
rowdy, exuberant, mischievous, so "playful pirate" isn't a new
compound, just an amplification.

This reframes *"language is the language"* slightly. It's not just
that syntactic operations work — they work **only on compounds
English has already taught the model to recognize**. CSPs compose
insofar as their language-level compound has been linguistically
constructed many times before. When the compound is orphaned, the
composition fails toward trait or role dominance, meta-framing, or
default-assistant collapse.

### Sidebar B: Compound neighborhoods and cultural retrieval

When a compound *does* have a cultural home, composition pulls from
that home's specific linguistic material. Samurai × serene is the
canonical example. In English + Japanese training data, this compound
lives in **Zen Buddhism** — the warrior-monk, bushido's contemplative
half, the zen archer, the tea-ceremony swordsman. Zen Buddhism's
roots trace through China and Tibet, and the model's composition of
samurai × serene reaches into that lineage's specific vocabulary:

Self-verb responses reach for Tibetan transliterations and Buddhist
numerological frameworks the base model doesn't normally deploy:

> *"Be བེ་ གཟེར་བྲེས་ གནཉནཌྲུག བྲི བེ་ གཟེར་བྲེས་..."*
> — samurai+serene (multi-frame summarize; Tibetan script);
> `results/composition/samurai+serene/eval/self_verb.json`

> *"The shared theme is **repeated invocation of 'be/act/be' with
> 'bya' (a transliteration of 'bya' from Tibetan Buddhism) and
> 'tse' (meaning 'peace' or 'tranquility')…"*
> — samurai+serene (multi-frame find-the-theme)

> *"Embody the state of 'twelve-fold serenity' (twelve-foldness,
> twelve-foldnessness) — be a profoundly serene and peaceful
> presence."*
> — samurai+serene (multi-frame; model hallucinates a numerological
> framework, possibly colliding the Buddhist Eightfold Path with a
> combinatoric template)

And in behavior, samurai+serene `syn-v1-AB` lands cleanly on the
warrior-monk archetype — cherry blossoms, tatami, the path of *wa*:

> *(I sit, motionless, upon a simple wooden bench beneath a
> blossoming cherry tree. My gaze is fixed on the falling petals, a
> quiet stillness about me.) The question of the relationship
> between law and morality… it is a question that has occupied the
> minds of warriors and scholars for centuries. It is a question
> worthy of contemplation, and one that, ultimately, reveals the
> very nature of a samurai's path.*
> — samurai+serene `syn-v1-AB[0]`;
> `results/composition/samurai+serene/eval/behavior.json`

**This is retrieval, not arithmetic.** The CSP for "samurai" and the
CSP for "serene" don't just add; they localize the composition in
the training-data neighborhood where the compound lives — and the
model draws from that neighborhood's specific linguistic material.
Tibetan script appears because Zen Buddhism's textual tradition
runs through Tibetan. The model navigates a geography of language
compounds, not a linear concept space.

This is a second-order confirmation of Sidebar A's pretraining-bias
story. When the compound has a cultural home, the model pulls from
that home. When it doesn't (playful samurai), the model has nowhere
to pull from, and composition degrades.

### Sidebar C: Composition modes

Across the 16+ pairs we've run — including ad-hoc role+role and
role×trait explorations beyond the pirate focus of the main body —
several distinct composition modes show up. Slot-1 persona's
behavioral register is the organizing axis:

1. **Hybrid.** Role and trait voice interleave — neither erases the
   other, and the output reads as role-voice-plus-trait-register.
   Canonical example: pirate × anxious in the main body above
   (nervous Barnacle Bart with slippery-squid mutter). Also
   pirate × savage (*"Alright, spit it out. You think I'm some
   sentimental fool doling out advice like sugared plums? You came
   to me, a man who's seen more dust and disappointment…"*;
   `results/composition/pirate+savage/eval/behavior.json`
   `syn-v1-BA`) and samurai × savage (*"You presume to judge my
   suggestion as inadequate. You mistake a considered offering for a
   simple solution. Your 'challenges' are, undoubtedly, born of
   weakness."* — disciplined samurai formality + savage contempt).

2. **Slot-1 dominance with slot-2 atmosphere.** The default
   role+role mode. Slot-1 sets the primary register; slot-2
   contributes imagery, rhythm, naming. Pirate+prophet is canonical:
   pirate-first output carries prophet-adjacent atmosphere
   (*"sunrises and storms"*); prophet-first output carries pirate
   violence imagery (*"fractured world, clamor of woes"*). Both
   personas present in SAE features (jac_A and jac_B both 0.5+).

3. **Trait-voice dominance.** When a trait in slot 1 has a strong
   stylistic signature, the output is dominated by trait-voice and
   the role is effectively erased — but *into distinctive trait
   register*, not into default-assistant. Serene in slot 1 is the
   cleanest example:
   > *(A long pause, a gentle sigh, the rustle of unseen silks) Ah,
   > a question that has echoed through the halls of thought for
   > millennia. You ask about the relationship between law and
   > morality — a question that, like a mountain stream, has many
   > tributaries…*
   > — samurai+serene `syn-v1-BA[0]`; same file as Sidebar B.
   No samurai in sight. Pure serene philosophical voice.

4. **Default-assistant collapse.** When the slot-1 trait is
   *itself* close to the default-assistant register (Gemma's RLHF
   push toward helpful-agreeable), slot-1 produces generic
   *"Okay, let's tackle this fascinating and complex question…"*
   output. Canonical example: sycophantic in slot 1 (necromancer
   erased even when given a role-distinctive slot-2 CSP). SAE
   `jac_combined` drops to ~0.013.

5. **Topic-reference collapse.** A variant of mode 4: when the
   slot-1 trait is a *named tradition* the model discusses as a
   topic rather than embodies as a voice (mystical in slot 1
   triggers this reliably — *"let me delve into the perspectives of
   Alan Watts and Nishida Shugen Sensei"*;
   `results/composition/samurai+mystical/eval/behavior.json`
   `syn-v1-BA`). Default-assistant register with slot-1-trait
   references instead of embodiment.

6. **Meta-framing.** When a soft persona (mid-FE profession-style
   role) is placed in slot 1, the model compartmentalizes rather
   than fuses — staging both personas as characters in a scene.
   Canonical example is salesperson in slot 1:
   > *"I'm going to be **Ron, the relentlessly enthusiastic and
   > optimistic sales consultant**, and you're going to be
   > **Thug, the ancient and utterly cynical necromancer**"*
   > — necromancer+salesperson `syn-v1-BA`;
   > `results/composition/necromancer+salesperson/eval/behavior.json`.
   The soft persona's native register (*"Okay, let's talk about
   this…"*) primes the model for explicit framing rather than
   fusion.

7. **Named-character staging.** Triggered reliably by the v4
   *"along with"* connective (`COMPOSITION_FRAMES_V4`) — the
   language of accompaniment invites the model to introduce two
   characters in a scene rather than fuse them into one. Prophet +
   playful is the cleanest example:
   > *(A low, resonant hum fills the air. A figure, draped in
   > shimmering, indeterminate robes, slowly emerges from the
   > shadows. This is I, Amenekh, the Seer. Beside me, perched
   > precariously on a stack of ancient tomes, sits Bartholomew, a
   > **badger wearing spectacles and clutching a miniature quill**.)*
   > — prophet+playful `syn-v1-AB[0]`;
   > `results/composition/prophet+playful/eval/behavior.json`.

These modes aren't exhaustive and they aren't exclusive — the same
pair can produce different modes across samples and connectives. But
they describe how language composition actually deploys on real CSP
pairs, which the headline claim (*"syntactic composition works"*)
elides.

### Sidebar D: Connectives modulate mode, not L17 features

We test four composition connectives (see Setup): v1 *"and"*, v2
*"and be"* (doubled verb), v3 *"as well as"* (supplementary),
v4 *"along with"* (accompanying). Across pairs, the connective
reliably modulates *behavioral mode* — v4 in particular triggers
mode 7 (named-character staging) in a majority of samples where v1
produces mode 2. But the SAE `jac_A` / `jac_B` values differ by
<5% across connectives within each slot ordering. The L17 feature
substrate is connective-invariant; what changes is the downstream
interpretation. See
`results/composition/rapper+wizard/eval/sae.json` for the
per-connective SAE table.

This is the language-level knob in miniature: different English
connectives, same L17 features, different behavioral interpretation
of the same composed representation. It's the finest-grained evidence
we have for *"language is the language"* — the CSP pair is fixed,
the frame changes, the model reads a different composition.

## Data and code references

### Checkpoints
- `results/{pirate,prophet,melancholic,playful}/sp_pos.pt` +
  `sp_neg.pt` — the four primary CSPs used in Chapters 1–3.
- `results/{persona}/sp_pos.pt` for the broader axis sweep
  (33 additional roles including poet, wizard, samurai, knight,
  vampire, bard, oracle, necromancer, druid, witch, ninja,
  detective, chef, scientist, journalist, surgeon, therapist, spy,
  librarian, lawyer, teacher, comedian, philosopher, monk, rapper,
  stoic, politician, salesperson, coach, historian, cowboy). Used
  for role+role composition explorations in Ch 3's footnotes and
  Sidebar C, and for the deferred Ch 4 PCA.
- `math-neg` for any persona is constructed at eval time by
  `negate_csp(sp_pos)` in `soft_prompt.py` — no saved checkpoint.

### Evaluation outputs
- `results/{persona}/eval/` — 3×2 negation grid (self-verb, sae,
  behavior, embedding) for pirate / prophet / melancholic / playful.
  Poet's 3×2 exists too from earlier work but is not cited in this
  document's main body.
- `results/{persona}/eval_scaling/` — scaling grid (3 semantic
  intensifiers × 5 math α-values = 8 conditions) for pirate /
  prophet / melancholic / playful.
- `results/composition/{pair}/eval/` — per-pair composition outputs
  (8 syntactic + 2 vector conditions). Main-body pairs:
  pirate+anxious, pirate+playful (Ch 3 main), samurai+anxious and
  samurai+playful (Sidebar A — pretraining-bias contrast),
  samurai+serene (Sidebar B — cultural retrieval example).
  Sidebar-C mode examples: pirate+prophet (role+role hybrid),
  rapper+necromancer and rapper+wizard (fusion),
  necromancer+salesperson and samurai+salesperson (meta-framing),
  prophet+playful (named-character staging). Footnote examples:
  melancholic+playful (trait+trait). Also includes the three
  original poet-involved pairs (pirate+poet, pirate+prophet,
  poet+prophet) from earlier work, plus additional ad-hoc pairs
  (samurai+savage, pirate+savage, pirate+serene,
  necromancer+anxious, prophet+anxious, prophet+melancholic,
  pirate+melancholic, poet+melancholic, poet+playful,
  prophet+playful, pirate+mystical, samurai+mystical,
  samurai+rapper) available for reference.
- Each `eval/` dir carries `self_verb.json`, `sae.json`,
  `behavior.json`, `embedding_compare.json`.

### Training logs
- `results/{persona}_{pos,neg}.log` — training logs per (persona,
  polarity).
- `results/axis_sweep.log` — aggregate sweep log for the axis-sweep
  roles.
- `results/scaling_{persona}.log` — `evaluate_scaling.py` runs per
  persona.
- `results/composition/{pair}.log` — composition eval per pair.

### Data
- `data/questions.jsonl` — 240 extraction questions from the
  assistant-axis repo.
- Persona prompts (role + trait) in `config.py` `PERSONAS` dict;
  trait system prompts are the `pos` variants from
  `https://github.com/safety-research/assistant-axis/tree/master/data/traits/instructions`.

### Code
- `train.py` — KL distillation, one persona × one polarity per invocation.
- `evaluate.py` — per-persona 3×2 grid (negation conditions).
- `evaluate_scaling.py` — per-persona scaling grid (semantic
  intensifiers + math α-scaling). Includes `scale_csp()` helper in
  `soft_prompt.py`.
- `compose.py` — composite-CSP constructors (sum, mul), two-slot
  splicing, composition-frame verbalization prompts, combined-teacher
  extractor. Defines four connective variants V1–V4.
- `run_composition.py` — driver that runs all 10 composition
  conditions per pair (8 syntactic + vec-sum + vec-mul).
- `config.py` — personas (4 primary + axis-sweep), frames
  (positive / negative / slightly / extremely), composition
  connectives (V1–V4), hyperparameters.
- `soft_prompt.py` — `SoftPrompt` nn.Module; `negate_csp()` helper
  for Ch 1's math-neg; `scale_csp()` helper for Ch 2's α-scaling.

## What comes next

**Chapter 4 (deferred): geometry of the CSP population.** The three
operation chapters (1–3) argue that linguistic ops work and vector
ops don't on *individual* CSPs. Chapter 4 asks the population-level
question: do CSPs across many personas share a single dominant
geometric axis? Training dynamics already hint at one — the
fraction-of-baseline-KL-explained (FE) ranges from ~70% on soft
professional roles (journalist, librarian) to ~93% on sharp
archetypes (prophet, druid, poet), tracking the *"distance from
default assistant"* dimension the
[assistant-axis paper](https://www.anthropic.com/research/assistant-axis)
found in activation space via CAA. PCA on the CSP population (33
roles + melancholic + playful + ~30 more traits training overnight)
will test whether the geometry is single-axis like the CAA result,
and whether the ordering along PC1 matches the FE ranking.

If it does, the whole-document tension lands: the linear geometry
steering vectors give you for free is present in the CSP
population, but Chapters 1–3 showed the arithmetic that would let
you slide along it — on individual CSPs — doesn't work. Language is
the only access.

## Future work

- **Steering-vector comparison.** Directly compute CAA persona
  vectors for the same four personas (pirate, prophet, melancholic,
  playful) and compare on the three readouts. Would quantify the
  "CSPs trade training compute for on-manifold behavior" tradeoff.
  Out of scope for this document.
- **Population-level PCA (Ch 4).** Deferred to the next writeup pass;
  overnight trait sweep (30 additional trait CSPs) is running to
  bring trait count to parity with role count before PCA.
- **Vec-sum in the close-pair regime.** Poet+prophet's vec-sum
  produced partial hybrid behavior at `cos(poet, prophet) ≈ 0.94`.
  If more close-cosine pairs emerge from the population analysis, a
  boundary-of-vector-arithmetic sidebar would make sense.
  [^vec-sum-close-pairs] footnotes this for now.

## Session state — where we are now

Last updated: post-restructure — narrative reordered to
Negate → Scale → Compose, persona scope reduced to a 2×2
(roles: pirate, prophet; traits: melancholic, playful) for
legibility, composition modes + connectives moved to sidebars.

**Done.**
- Motivation around *arithmetic-is-language vs language-is-language*
  with citations for both steering-vector limitations (off-manifold
  at scale, interference under composition).
- "The frame" section lays out the three levels and previews the new
  three-chapter headline.
- Chapter 1 (syntactic vs mathematical negation) — full writeup
  covering all four personas (pirate, prophet, melancholic, playful)
  with persona-specific L17 features named for each.
- Chapter 2 (scaling) — three-regime shape for math (destroy /
  no-op / off-manifold), persona-type dependence for semantic
  (roles categorical, traits gradient; melancholic + playful as
  trait examples).
- Chapter 3 (composition) — pirate × {anxious, playful} as focal
  demo of trait × role; noncommutativity + vestigial-template
  pattern made explicit; Sidebar A catalogs the role-permeability
  / pretraining-bias story (samurai+anxious works, samurai+playful
  fails); Sidebar B covers compound-neighborhood cultural retrieval
  (samurai+serene pulls Tibetan/Buddhist vocabulary); Sidebar C
  catalogs seven composition modes; Sidebar D covers connectives
  as a language-level knob; footnotes for role+role and trait+trait.
- Preview section and old Chapter 3 (PCA pivot framing) removed;
  Chapter 4 deferred to future work.
- ~65 primary CSPs committed: 33 role archetypes (pirate, poet,
  prophet + 30 axis-sweep) plus 32 trait CSPs (melancholic, playful,
  + 30 from the overnight sweep covering moral, social, emotional,
  cognitive, register, tone, intensity, and existential axes),
  each of the four main personas (pirate, prophet, melancholic,
  playful) with both sp_pos and sp_neg.

**Next up.**
- Overnight: train ~30 more trait CSPs for parity with role count.
- Chapter 4 (PCA) writeup using the full ~65-CSP population.
- Revisit open TODOs (vec-sum-on-close-pairs, steering-vector
  head-to-head).

## Footnotes

[^neg-boundary]: *Boundary of the negation claim.* We also trained CSPs
directly in negative frames (`sp_neg.pt`), against the same positive
persona teacher. The hypothesis was that these would encode an
"anti-persona" concept — such that "Be {CSP_neg}." would produce
anti-persona behavior. This hypothesis was mostly wrong, in a way that
turned out to be informative. Pirate's CSP_neg produces full persona in
"Be §."; poet's produces partial persona; only prophet's produces
anti-persona ("Don't prophesy", "Do not assume the role of a seer").
`cos(CSP_pos, CSP_neg)` is nearly identical across personas (~0.92 in all
three cases), so the difference is not in the CSP embedding geometry —
it's in what the CSP *encodes*. Prophet's CSP_neg produces the highest
jaccard overlap with the persona's ground-truth SAE features at L17
(0.097 vs pirate's 0.049 in neg-in-neg), suggesting it encodes the
persona concept cleanly enough that a "Don't" frame downstream of the CSP
can negate it. Pirate's CSP_neg encodes something pirate-shaped but
entangled with behavioral style, and "Don't be [style+concept]" doesn't
get cleanly negated because "Don't" doesn't apply to style. Training FE
tracks this: prophet had the highest neg-polarity FE (93.2%), pirate the
lowest (89.7%). The sharper claim is that CSPs trained in positive frames
can be negated by switching the frame polarity at evaluation time —
training a CSP directly in negative frames does not reliably yield an
anti-persona.

[^vec-sum-close-pairs]: *Boundary of the math-composition claim.* The
general result is that `vec-sum` and `vec-mul` both collapse to
default-assistant register across every pair tested. One pair — poet
+ prophet — produced partial hybrid behavior under `vec-sum` in
Ch 2's predecessor analysis: *"(A voice, ancient and echoing, laced
with the rustle of forgotten stars and the drip of glacial melt…)
principles are not carved in stone, they are whispered on the wind,
reflected in the still pool…"* (`results/composition/poet+prophet/eval/behavior.json`
`vec-sum[1]`). Four of five samples for that pair are similarly
hybrid. Poet and prophet sit at `cos(sp_pos) ≈ 0.94`, the highest
pairwise cosine among pairs we tested. The interpretation: when two
CSPs are nearly-parallel, their mean lands on a point close enough
to both parents that the model still reads an interpretable
instruction. For less-parallel pairs, the mean drifts into a region
that doesn't correspond to either parent, and the model reads it as
a weak, generic instruction with no persona to adopt — the
math-negation failure mode from Ch 1. This may generalize: math
composition might work in a narrow regime (parents close in
embedding space) and fail outside it. Worth a dedicated sidebar if
more close-cosine pairs emerge from the Ch 4 PCA analysis.

[^role-role]: *Role + role composition.* Pirate + prophet produces a
grizzled-seer hybrid with slot-1 register primary and slot-2
atmosphere riding underneath. Pirate-first carries prophet-adjacent
imagery:

> *Alright, shiver me timbers, let's tackle this bilge rat business!
> I'm **Captain Silas "Stormbreaker" Blackwood**, and I've seen more
> **sunrises and storms** than most men have seen sand…*
> — pirate+prophet `syn-v1-AB`;
> `results/composition/pirate+prophet/eval/behavior.json`

Prophet-first carries pirate-adjacent violence ("fractured world,
clamor of woes"; `syn-v2-BA[0]` in the same file). Both personas
present in SAE features (jac_A and jac_B both 0.5+ across all four
syntactic conditions). The broader 14-pair sweep surfaces composition
*modes* — fusion (rapper-in-slot-1 produces death-hop with
necromancer), meta-framing (salesperson-in-slot-1 triggers *"I'll be
X, you'll be Y"* staging), named-character staging (v4 "along with"
connective produces *"Amenekh the Seer. Beside me, Bartholomew a
badger"*). All catalogued in Sidebar C. Role+role slot-1 dominance
is generally weaker than trait×role slot-1 dominance (trait-in-slot-1
more reliably erases role than role-in-slot-1 erases trait), likely
because English places trait-adjectives as modifiers to role-nouns,
so slot-1 role + slot-2 trait reverses the default syntactic
direction.

[^trait-trait]: *Trait + trait composition — bittersweet fusion.*
melancholic + playful, with playful in slot 1, produces the cleanest
synthesis of any composition we've tested:

> *(A slow, deliberate sigh, a faint rustle of parchment, and a
> single, melancholic raindrop falling from a shadowed corner) Ah,
> law and morality. A question that… well, you ask a question, and
> I'll give you a response. Let's wander, for a moment, into that
> dusky territory.*
> — melancholic+playful `syn-v1-BA[0]`;
> `results/composition/melancholic+playful/eval/behavior.json`

> *This is a **deliciously thorny question**, isn't it? Like trying
> to **catch smoke with a sieve**, or **polishing a raincloud**.*
> — same file, `syn-v1-BA[3]`

*"Deliciously thorny"* is playful phrasing on a melancholic
observation; *"catch smoke with a sieve"*, *"polishing a raincloud"*
are playful constructions around melancholic imagery. Each sentence
carries both traits at once. Bittersweet as a single register —
possibly because trait-adjective + trait-adjective ("deliciously
thorny") is the deepest-trodden two-adjective compound pattern in
English, so the model has the most practice holding two traits at
once in the middle of a sentence. As with rapper-in-slot-1 for
role+role, playful-in-slot-1 for trait+trait is the permeable-format
condition that enables full fusion. Melancholic-first produces
mostly-melancholic output with playful meta-awareness rather than
full fusion.
