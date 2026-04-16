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

### Training logs
- `results/{persona}_{pos,neg}.log`
- `results/eval_{persona}_v2.log` — rerun of self-verb under the multi-frame
  prompts

### Data
- `data/questions.jsonl` — 240 extraction questions from the assistant-axis
  repo

### Code
- `train.py` — KL distillation, one persona × one polarity per invocation
- `evaluate.py` — all evaluation modes (self-verb, sae, behavior, embedding)
- `config.py` — personas, frames, hyperparameters
- `soft_prompt.py` — `SoftPrompt` nn.Module

## What comes next

**Composition.** Train CSPs jointly on pairs ("Be {sp_A} and {sp_B}."),
compare against the individually-trained CSPs. Does the joint CSP decompose
into the individual ones at L17? Does "Be {CSP_A} and {CSP_B}." produce a
hybrid persona?

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
