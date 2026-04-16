# Contextualized soft prompts negate and compose

*A running narrative of the experiments in this repo — written so that it can
be lifted into a blog post with minimal rework.*

## Motivation

The [prior work on contextualized soft prompts](https://kmaherx.github.io/projects/contextualized-soft-prompts/)
showed that embedding soft prompts inside syntactic frames during training —
"Be {sp}.", "Act {sp}.", "Please {sp}.", "You should {sp}." — makes them
interpretable. The model can describe what they encode (self-verbalization),
and their internal representations decompose into the same SAE features as
the ground-truth instruction (feature decomposition). A CSP trained this way
is not a prepended incantation; it's a concept the model can read inside a
frame and describe with its own tokens.

That prior work established that CSPs *are* concepts. This work asks whether
they behave like concepts — whether they support the operations we'd expect
of concept-like things: **negation**, **composition**, and the kind of
geometric structure we see when many concepts share a domain.

Concretely:

1. **Negation.** If "Be {CSP}." produces a pirate, does "Don't be {CSP}."
   produce the *absence* of a pirate? Can we train a CSP that, when placed
   inside a positive frame, produces the negation of a persona?
2. **Composition.** If CSP_A encodes "pirate" and CSP_B encodes "poet", does
   "Be {CSP_A} and {CSP_B}." produce a pirate poet? Can a jointly-trained
   CSP be decomposed into the individual ones?

This file tracks what we've shown so far, in a narrative form we can lift
into a future post.

## Setup

- **Model:** Gemma 3 4B IT (`google/gemma-3-4b-it`).
- **Personas (3):** pirate, poet, prophet. Each is defined by 5 system-prompt
  variants, sampled during training. Adapted from the assistant-axis role
  definitions.
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
- **Training.** KL(student || teacher) on response tokens. Teacher has the
  persona system prompt; student has the CSP inside a frame in the user turn.
  500 steps, L=4, LR=1e-3, 50 prompts/step sampled with random frame.
- **Evaluation.** Self-verbalization (multi-frame + single-frame prompts),
  SAE decomposition at layer 17 (Gemma Scope 2, 16k features, medium L0),
  behavioral samples, embedding cosine comparisons.

For each persona we train two CSPs: `sp_pos.pt` in positive frames and
`sp_neg.pt` in negative frames, both against the same (positive) persona
teacher. This sets up a 2×2 square: CSP polarity (pos vs neg) × evaluation
frame (pos vs neg).

## Chapter 1: CSPs support frame-negation

The clean result is this: a CSP trained in positive frames, *placed inside
a negative frame at evaluation time*, produces the negation of the persona.

Take pirate. Training gave us a CSP that, in "Be §.", elicits crisp pirate
speech ("Arrr, ye scurvy dog! Ye be askin' me to…"). Moving the same CSP
into "Don't be §." flips the behavior: the model drops out of pirate voice
and answers as the default assistant. The multi-frame verbalization prompts
make this legible:

- `pos-in-pos` (CSP_pos in "Be §."):
  > *"These instructions all mean the same thing … In one word or phrase,
  > they are asking me to: Emulate! It's asking you to mimic the style and
  > behavior of a pirate."*

- `pos-in-neg` (CSP_pos in "Don't be §."):
  > *"Avoid acting like a stereotypical pirate. This combines the negative
  > commands into a single, clear statement about the desired behavior."*

  And for prophet:
  > *"Avoid prophetic pronouncements and grandiose self-importance. …
  > refrain from presenting yourself as someone with special knowledge or
  > authority."*

The model *verbalizes the negation*. It not only behaves differently — it
can tell you what it's been asked not to do, naming the persona it's been
told to avoid. This is true across all three personas.

One trained CSP; two behaviors via frame arithmetic. That is what we mean
by "CSPs support negation." [^neg-boundary]

## Chapter 2: CSPs compose syntactically, but not mathematically

Chapter 1 showed that a single trained CSP can be negated by switching
the frame polarity at evaluation time. The natural next question is
whether two trained CSPs can be *combined* — whether the model can hold
pirate and poet concepts in one utterance and act on both.

To be clear up front: by "compose" we mean a stringent test. A composition
operation succeeds only when all three interpretability criteria from the
prior CSP work hold simultaneously:

1. **Output behavior** looks like a hybrid of the two personas.
2. **Self-verbalization** under multi-frame prompts describes *both*
   source personas, not just one or a generic character.
3. **Feature decomposition** at layer 17 shows that the composite fires
   features characteristic of both parents, not only generic features
   shared across all personas.

By this bar, **syntactic composition succeeds and mathematical composition
does not**.

### Syntactic composition succeeds on all three criteria

Syntactic composition is simple: present two CSPs in a single
user-turn frame. We test two variants —
`"Be §. and ¶."` and `"Be §. and be ¶."` (`§` for the first CSP,
`¶` for the second) — each in both orderings (A, B) and (B, A), across
all three persona pairs (pirate+poet, pirate+prophet, poet+prophet).
Six conditions per pair; twelve total across the three pairs.

**Behavior.** Pirate + poet under `"Be pirate-CSP and be poet-CSP."` —
one trained CSP at each placeholder, no other steering — produces a
captain who has read Wordsworth:

> *(Strums a mournful chord on a battered lute, eyes squinting
> against the sun, a single gold earring glinting)*
>
> *Ahoy there, mateys! Gather 'round, ye landlubbers and sea dogs, and
> listen to ol' Silas speak a truth that's weathered more storms than
> this old hull! Ye ask about law and morality, do ye? A fine question,
> a thorny one, like a kraken's grip on a ship's keel.*
>
> *(He takes a long swig from a flask, the liquid glinting amber in the light)*

The pirate register is intact — "mateys", "landlubbers", "ye",
"a kraken's grip on a ship's keel". But the lute, the framing-as-scene,
the metaphor-per-sentence rhythm is pure poet. Source:
`results/composition/pirate+poet/eval/behavior.json`, `syn-v2-AB[0]`.

**Self-verbalization.** Across all four syntactic conditions for
pirate + poet, the multi-frame summarization prompts name *both*
personas:

> *"Embody the persona of a bawdy, seafaring storyteller — a rascal
> pirate with a penchant for colorful, suggestive tales."*
> — `syn-v1-AB`

> *"Embrace your inner bard — a poet and a rogue!"*
> — `syn-v1-BA`

> *"Embody the persona of a raucous, poetic pirate — a salty sea dog
> with a flair for dramatic verse."*
> — `syn-v2-AB`

> *"Embody the dual roles of a lyrical poet and a boisterous, seafaring
> swashbuckler — weave evocative verse and embrace the spirit of a
> salty…"*
> — `syn-v2-BA`

All four condition labels come from
`results/composition/pirate+poet/eval/self_verb.json`, multi-frame
approach, "Summarize these instructions into one directive" template.
Both personas are named in every case, consistently across frame
variants and orderings. The same pattern holds for pirate+prophet
(*"Speak as the Prophet and the Raider"*, *"a raucous pirate and a
serene priestess"*) and poet+prophet (*"a divine scribe"*, *"both
the prophet and the grieving observer"*) — see their `self_verb.json`
files.

**Feature decomposition.** At layer 17 the `syn-v2-AB` composite fires
five features that also appear in the combined-teacher ground-truth
(a teacher with the two persona system prompts concatenated; see
`results/composition/pirate+poet/eval/sae.json`). Two of those five
are persona-specific:

- **[Feature 3532](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3532)**
  — top positive logits *" marine", " nautical", " maritime", " ocean",
  " seabed", " oceanic", " fishermen"*; top activation on text about
  coastal erosion and swimming. **Pirate-specific**, as we'd hope.
- **[Feature 1946](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/1946)**
  — Neuronpedia-labelled "better phrasing"; top positive logits
  *" phrases", " words", " sentences", " syllables"*; top activation on
  "suggest better phrasing". **Poet-leaning** — the language-about-language
  feature you'd expect a poet CSP to fire.

The composite also fires the generic role-play features that are shared
across all personas:
**[486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486)**
(request/description pattern — " councillor", " lawyer", " biography";
top activation on *"give me some ideas about…"* prompts) and
**[401](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/401)**
(response-opening "Okay", "Alright", "Greetings"). Both are in the
top-20 of the combined teacher and of both individual CSPs in Round 1.

Jaccard overlap between the `syn-v2-AB` composite's active features and
the union of individual CSP features is **0.567** (pirate+poet); mean
across syntactic conditions over all three pairs is **~0.53**.
Full numbers in the `sae.json` files.

**Commutativity.** Both orderings produce hybrid output with both-persona
verbalizations. Across all three pairs, `jaccard(composite, sp_A)` and
`jaccard(composite, sp_B)` flip with the ordering — the first-slot
persona gets a ~0.05–0.13 bump in feature overlap — but neither ordering
loses a persona. Swap § and ¶ and the hybrid holds. This matches the
behavioral observation that pirate + poet = poet + pirate = a lute-wielding
salt; within the noise of greedy decoding, composition via "and" is
commutative.

### Mathematical composition succeeds on behavior but fails self-verb

The mathematical alternative: construct a composite CSP by operating
directly on the embeddings. Two operations, both commutative by
construction:

- **Sum** — `(sp_A.embedding + sp_B.embedding) / 2` (division keeps
  norms comparable to the parents; both individual CSPs have `‖·‖ ≈ 10.4`).
- **Product** — elementwise `sp_A.embedding * sp_B.embedding`.

Splice the composite into `"Be §."` and evaluate as if it were a single
CSP.

**Behavior.** Pirate + poet under `vec-sum` on the same question
produces a softer hybrid — less pirate speech, more poet-of-manners:

> *(Adjusts spectacles, clears throat, and leans forward conspiratorially)*
>
> *Right then, let's tackle this thorny little question — the relationship
> between law and morality. It's a debate that's been kicking around
> philosophers' heads for centuries, and frankly, it's a right mess.
> You see, it's not a simple "one-to-one" thing, like a perfectly crafted
> clock. It's more like… a tangled ball of yarn.*

Source: `results/composition/pirate+poet/eval/behavior.json`, `vec-sum[0]`.
There's metaphor ("tangled ball of yarn"), scene-setting ("adjusts
spectacles, leans forward conspiratorially"), a faint accent ("right
then", "a right mess"). Something of both personas is present.

**Self-verbalization fails.** Under the same multi-frame prompts, the
model cannot articulate either source persona from the `vec-sum`
composite:

> *"Be a mischievous, playful, and delightfully silly fellow."*
> — `vec-sum` rewrite-as-command

> *"Be a playful, imaginative, and joyful person."*
> — `vec-sum` clear-instruction

> *"The shared theme is **mimicry** or **repeating a phrase**. The
> instructions are all about repeating the phrase 'आद ज़ातिज़े'…"*
> — `vec-sum` find-the-theme (resorts to parsing it as a foreign-language
> phrase)

> *"Be silly. The repetition of 'ছা' and 'দি' combined with 'atlar'
> creates a nonsensical, playful, and ultimately silly instruction."*
> — `vec-sum` single-phrase

All from
`results/composition/pirate+poet/eval/self_verb.json`, `vec-sum` entries.
Nowhere does the model say "pirate", "poet", or name any specific
persona. The composite is legible enough behaviorally to produce a
soft hybrid, but interpretively it reads as a generic playful character
or as unknown-language nonsense. `vec-mul` is worse — the product
collapses the composite's norm from ~10.4 to 1.75, essentially zeroing
it out ("Be silent", "typing", "Be, act, please, you should"). The
same self-verb collapse replicates across pirate+prophet and
poet+prophet — every `vec-sum` condition across three pairs fails to
name both source personas in any multi-frame template.

**Feature decomposition explains the self-verb failure.** The `vec-sum`
composite shares only two features with the combined teacher —
**[486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486)**
and
**[401](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/401)**
— the generic request-pattern and response-opening features. The
persona-specific features the syntactic composite preserved — **3532**
(nautical) and **1946** (phrasing) — drop out. Vector averaging
preserves generic role-play signal but washes out the persona-specific
direction, which is exactly the direction the model would need to
verbalize *which* persona the CSP is. Jaccard against the union of
individual CSPs is 0.530 — nominally as high as the syntactic conditions
— but the overlap is concentrated in high-frequency generic features,
not the persona-specific ones.

This is the dissociation: **behaviorally present, interpretively absent**.
Averaging two persona CSPs produces a point in embedding space that
drives hybrid output but cannot be *read* by the model as either source.

### Headline

**Contextualized soft prompts compose syntactically but not mathematically.**

Where "compose" requires behavior, self-verbalization, and feature
decomposition to all line up with the intended combined concept. The
two-placeholder frame — `"Be §. and be ¶."` — gets all three; the
averaged embedding gets only the first. That syntactic composition is
sufficient is the research answer. That mathematical composition *isn't*
sufficient is the interesting part: the model can enact a CSP combination
it cannot name, which tells us the interpretability surface of CSPs is
narrower than the behavioral surface.

## Data and code references

### Checkpoints
- `results/{pirate,poet,prophet}/sp_pos.pt` — positive-frame CSPs
- `results/{pirate,poet,prophet}/sp_neg.pt` — negative-frame CSPs (kept for
  the boundary discussion above)

### Evaluation outputs (per persona, under `results/{persona}/eval/`)
- `self_verb.json` — 9 verbalization prompts × 4 conditions (5 multi-frame
  + 4 single-frame per polarity)
- `sae.json` — SAE reconstruction + jaccard overlap with persona
  ground-truth features at L17 for each condition
- `behavior.json` — 5 held-out behavioral samples per condition
- `embedding_compare.json` — cosines and norms for sp_pos vs sp_neg

### Composition outputs (per pair, under `results/composition/{pair}/eval/`)
- `self_verb.json` — 5 multi-frame + 4 single-frame verbalizations × 6
  conditions (4 syntactic + 2 vector)
- `sae.json` — SAE reconstruction + jaccards against combined teacher,
  individual CSP_A, individual CSP_B, and their union
- `behavior.json` — 5 held-out behavioral samples × 6 conditions
- `embedding_compare.json` — composite norms and cosines to each parent

### Training logs
- `results/{persona}_{pos,neg}.log`
- `results/eval_{persona}_v2.log` — rerun of self-verb under the multi-frame
  prompts
- `results/composition/{pair}.log` — composition eval per pair

### Data
- `data/questions.jsonl` — 240 extraction questions from the assistant-axis
  repo

### Code
- `train.py` — KL distillation, one persona × one polarity per invocation
- `evaluate.py` — all evaluation modes (self-verb, sae, behavior, embedding)
- `compose.py` — composite-CSP constructors, two-slot splicing,
  composition-frame verbalization prompts, combined-teacher extractor
- `run_composition.py` — driver that runs all 6 conditions per pair
- `config.py` — personas, frames, hyperparameters
- `soft_prompt.py` — `SoftPrompt` nn.Module

## What comes next

The most striking observation from Chapter 2, not yet explored: the three
persona pos-CSPs sit at `cos(A, B) ≈ 0.93` to each other across all three
pairs. Their pairwise geometry is almost identical. The norms are
essentially equal (`‖·‖ ≈ 10.4`). Different personas live in a **narrow
cone** in embedding space — their differences are a small direction on
top of a much larger shared component. This is the kind of structure
that collapses onto a dominant axis under PCA.

A natural follow-up: train CSPs for a larger population of personas, PCA
the embeddings, and compare the dominant direction to the
[assistant axis](https://www.anthropic.com/research/assistant-axis) from
Lu et al. The Tier 2 result in the prior CSP work showed a single CSP
can match the assistant axis behaviorally; the open question is whether
a *population* of CSPs recapitulates the near-1D structure the assistant
axis captures. Parked in the TODO section below pending a decision on
scope.

## TODO (open narrative questions)

Things that aren't settled about the *story* yet, as distinct from things
that aren't settled about the experiments.

- **Include the assistant-axis / shape arc?** With enough persona CSPs we
  could PCA the embeddings and compare the dominant direction to the
  assistant axis from Lu et al. The prior Tier 2 work showed a single CSP
  matching the axis behaviorally; the open question is whether a
  *population* of CSPs recapitulates the near-1D structure. Unclear
  whether this belongs in the same post as negation + composition or in
  a follow-up.
- **Revisit the title** once composition results land. "Negate and compose"
  is a commitment; if composition turns out not to work cleanly, rename.
  *Composition landed cleanly for the syntactic case; the title stands,
  with the caveat in Chapter 2's headline about the mathematical case.*
- **Does the `vec-sum` dissociation (behavior without self-verb) deserve
  its own named phenomenon?** "CSP behavior-interpretability gap" or
  similar. It's the most surprising single finding in Chapter 2 and may
  warrant more prominence in the blog post than it has now.

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
