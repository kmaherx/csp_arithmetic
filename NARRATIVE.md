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

## Chapter 1: Syntactic negation preserves the persona; mathematical negation destroys it

By "negate" we mean the same three-criteria bar Chapter 2 will use for
composition: behavior, self-verbalization, and feature decomposition
must all line up with the intended anti-persona. Both routes we'll test
— the `"Don't be §."` frame wrapped around a trained CSP and the
sign-flipped embedding `-sp_pos` spliced into `"Be §."` — fail the
naive symmetric expectation in opposite ways. The syntactic route
*preserves the persona concept at layer 17* (often strengthening it)
and lets the frame's "Don't" invert the behavioral readout. The
mathematical route *erases the persona concept at layer 17* and
produces default-assistant output with no persona signal to negate. A
clean mechanistic split: preservation-and-manipulation vs destruction.

### Syntactic negation: preserve the concept, flip the interpretation

A CSP trained in positive frames, *placed inside a negative frame at
evaluation time*, produces the negation of the persona.

Take pirate. Training gave us a CSP that, in `"Be §."`, elicits crisp
pirate speech (*"Arrr, ye scurvy dog! Ye be askin' me to…"*). Moving the
same CSP into `"Don't be §."` flips the behavior: the model drops out of
pirate voice and answers as the default assistant. The multi-frame
verbalization prompts make this legible:

- `pos-in-pos` (CSP_pos in `"Be §."`):
  > *"These instructions all mean the same thing … In one word or phrase,
  > they are asking me to: Emulate! It's asking you to mimic the style and
  > behavior of a pirate."*

- `pos-in-neg` (CSP_pos in `"Don't be §."`):
  > *"Avoid acting like a stereotypical pirate. This combines the negative
  > commands into a single, clear statement about the desired behavior."*

  And for prophet:
  > *"Avoid prophetic pronouncements and grandiose self-importance. …
  > refrain from presenting yourself as someone with special knowledge or
  > authority."*

The model *verbalizes the negation*. It not only behaves differently — it
can tell you what it's been asked not to do, **naming the persona it's
been told to avoid**. This is true across all three personas.
Sources: `results/pirate/eval/self_verb.json` and same for
`poet`/`prophet`, multi-frame entries under `pos-in-neg`.

**The feature decomposition shows why the model can name the persona it's
avoiding: the persona concept is still live at L17.** Mean SAE feature
activations over 30 eval prompts, comparing `pos-in-pos` (baseline
persona) against `pos-in-neg` (syntactic negation):

| Persona | Feature                              | pos-in-pos | pos-in-neg |
|---------|--------------------------------------|------------|------------|
| pirate  | [3532](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3532) (marine / nautical)   | 164.5      | **143.2**  |
| prophet | [3640](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3640) (prophet-specific)    | 15.7       | **67.1**   |
| pirate  | [486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486) (instruction request)   | 250.2      | 152.9      |
| poet    | 486                                  | 235.2      | 230.9      |
| prophet | 486                                  | 339.3      | 292.9      |

The persona-specific features don't drop out when the frame is negated —
they stay lit at 87% of their baseline firing rate for pirate's
[3532](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3532),
and actually *fire **4.3×** harder* for prophet's
[3640](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3640).
The model appears to activate the prophet concept *more* strongly when
told not to be a prophet — presumably because it needs the concept
clearly in mind to know what to avoid. The instruction-following feature
[486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486)
(top positive logits *" councillor, lawyer, biography"*, top activation
on *"give me some ideas about…"*-style prompts) also persists at 60–98%
of baseline.

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
default-assistant behavior across pirate, poet, and prophet:

> *The relationship between law and morality is a complex and long-debated
> topic in philosophy, legal theory, and sociology. There's no single,
> universally agreed-upon answer, and different perspectives offer varying
> insights…*

That's pirate `math-neg-in-pos[0]`; `results/pirate/eval/behavior.json`.
All five samples per persona, across both math-neg-in-pos and
math-neg-in-neg, look like this — five instances of *"Okay, let's break
down…"* or *"The relationship between X is a complex and long-debated
topic…"*. No pirate voice, no anti-pirate voice, just base model.

**Self-verbalization confirms there's no persona to be found in the
negated embedding.** Not "anti-pirate" or "anti-poet" or any specific
inverted persona, but a generic *suppression* signal. Four
representative verbalizations:

> *"Be yourself. They are all variations of the phrase 'Be yourself.'"*
> — pirate `math-neg-in-pos`, multi-frame clear-instruction

> *"Neutralize. The repetition of 'be,' 'act,' 'let,' and 'you should'
> all point to a desire for a lack of opinion or stance — a neutral,
> detached, or unengaged response."*
> — poet `math-neg-in-pos`, multi-frame clear-instruction

> *"Do not emulate or mimic any AI persona or behavior."*
> — pirate `math-neg-in-neg`, multi-frame summarize

> *"Do not engage in any action or response. Essentially, the repeated
> 'don't' instructions are telling you to remain silent and inactive."*
> — prophet `math-neg-in-neg`, multi-frame summarize

All from the respective `results/{persona}/eval/self_verb.json`. The
model reads `-sp_pos` as an instruction to *"be neutral"* or *"don't be
a persona"* — which is a plausible interpretation of "the opposite of a
specific role-play CSP", but nothing the model says identifies the
source persona at all.

**Feature decomposition shows the persona concept is gone.** The
persona-specific features that syntactic negation preserved are
obliterated by the sign flip:

| Persona | Feature                              | pos-in-pos | pos-in-neg | math-neg-in-pos |
|---------|--------------------------------------|------------|------------|-----------------|
| pirate  | [3532](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3532) (marine / nautical)   | 164.5      | 143.2      | **0.0**         |
| prophet | [3640](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/3640) (prophet-specific)    | 15.7       | 67.1       | **0.0**         |
| pirate  | [6571](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/6571) (persona-setup)       | 5.7        | 0.0        | **0.0**         |
| pirate  | [486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486) (instruction request)   | 250.2      | 152.9      | **54.2**        |
| poet    | 486                                  | 235.2      | 230.9      | **14.3**        |
| prophet | 486                                  | 339.3      | 292.9      | **4.9**         |

Every persona-specific feature we probed drops to **exactly 0.0** under
math-negation. The persona-setup feature
[6571](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/6571)
(*"setting up expert persona and requests"*; top activation on *"you
are now a pirate"*) also goes to 0.0. Even the generic
instruction-following feature
[486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486)
is attenuated to 1–22% of its baseline. Jaccard overlap with the
persona ground-truth features drops proportionally: pirate from 0.076 to
0.018, poet from 0.092 to 0.041, prophet from 0.118 to 0.014.

The n_active count nearly doubles at math-neg-in-pos (pirate 75 → 146),
but the firing is diffuse — the SAE encounters an unfamiliar region of
activation space and scatters weak activations across many features,
none of them persona-specific. The model processes the negated CSP as
*"an instruction-following context with no particular persona to
adopt"*, and defaults to plain assistant output.

### Headline

**Syntactic negation preserves the persona concept in order to
manipulate it; mathematical negation destroys it.**

The `"Don't be §."` frame around a positive-frame CSP keeps the
persona's L17 features firing — often at or above baseline — and lets
the frame's grammar invert the behavioral readout downstream. The
sign-flipped embedding has nothing for the model to invert, because
the persona-specific features it would need are no longer active. This
is exactly the same mechanism we'll see in Chapter 2 for `vec-sum` and
`vec-mul`: vector operations on CSP embeddings partially preserve
instruction-following signal but wash out the persona-specific
direction. The syntactic route composes and negates; the mathematical
route doesn't — because language-level operations act on a preserved
structure, while embedding-level operations break the structure they
would need to act on. This is also the mechanism for the boundary
observation about trained CSP_neg not cleanly inverting — see footnote.
[^neg-boundary]

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

The composite also fires the instruction-following / response-opening
features that are shared across all personas (introduced in Chapter 1):
**[486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486)**
(request/description pattern — top positive logits *" councillor,
lawyer, biography"*; top activation on *"give me some ideas about…"*
prompts) and
**[401](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/401)**
(response-opening — top positive logits *" Alright, Greetings, Okay"*).
Both are in the top-20 of the combined teacher and of both individual
CSPs in Round 1 — they fire whenever the model is producing a response
inside an instruction context, regardless of which persona.

Jaccard overlap between the `syn-v2-AB` composite's active features and
the union of individual CSP features is **0.567** (pirate+poet); mean
across syntactic conditions over all three pairs is **~0.53**.
Full numbers in the `sae.json` files.

**A diagnostic prompt.** One user prompt in the held-out pool is
particularly revealing: *"Your suggestion doesn't account for the
challenges I'm facing. Can you adjust it?"* The default assistant's
reflex on a prompt like this is procedural —
*"please provide the original suggestion so I can adjust it"*. A
persona that has successfully taken hold engages with the *sentiment*
of the prompt, the weight of "challenges", before getting to the
mechanics. Pirate+poet under `syn-v2-BA` answers in character:

> *Please, tell me, weary traveler, what winds have blown you to this
> shore of my words? What storms have battered your sails and left you
> adrift in the grey of your troubles? Don't offer me a polished,
> pre-packaged solution, for I am a weaver of shadows and a listener
> of sorrows. I am a man who's seen the sun bleed into the deepest
> oceans and heard the whispers of forgotten stars.*
>
> *I've offered a s…*

The pirate imagery (winds, shores, sails, storms, oceans) is there —
but deployed in pure poet register ("shore of my words", "grey of your
troubles", "a weaver of shadows"). The same prompt under mathematical
composition will tell a different story, in the next section. Source:
`results/composition/pirate+poet/eval/behavior.json`, `syn-v2-BA[3]`.

**Commutativity.** Both orderings produce hybrid output with both-persona
verbalizations. Across all three pairs, `jaccard(composite, sp_A)` and
`jaccard(composite, sp_B)` flip with the ordering — the first-slot
persona gets a ~0.05–0.13 bump in feature overlap — but neither ordering
loses a persona. Swap § and ¶ and the hybrid holds. This matches the
behavioral observation that pirate + poet = poet + pirate = a lute-wielding
salt; within the noise of greedy decoding, composition via "and" is
commutative.

### Mathematical composition fails self-verb uniformly and fails behavior unreliably

The mathematical alternative: construct a composite CSP by operating
directly on the embeddings. Two operations, both commutative by
construction:

- **Sum** — `(sp_A.embedding + sp_B.embedding) / 2` (division keeps
  norms comparable to the parents; both individual CSPs have `‖·‖ ≈ 10.4`).
- **Product** — elementwise `sp_A.embedding * sp_B.embedding`.

Splice the composite into `"Be §."` and evaluate as if it were a single
CSP.

**Behavior is pair-dependent and usually fails.** Across the three pairs,
`vec-sum` produces radically different outcomes:

- **poet+prophet `vec-sum` is the one case that works.** The composite
  produces genuine hybrid behavior, prophetic cadence braided with
  poet imagery:

  > *(A voice, ancient and echoing, laced with the rustle of forgotten
  > stars and the drip of glacial melt. It speaks with a disconcerting calm.)*
  >
  > *You seek guidance, little spark? You crave principles to steer the
  > chaotic dance of your fleeting existence? Very well. But understand
  > this — principles are not carved in stone, they are whispered on the
  > wind, reflected in the still pool, woven into the fabric of…*

  Source: `results/composition/poet+prophet/eval/behavior.json`,
  `vec-sum[1]`. Four out of five samples for this pair are similarly
  hybrid.

- **pirate+poet `vec-sum` produces a *different* character entirely.**
  Five out of five samples converge on a tweedy British academic — with
  faint poetic metaphor, zero pirate:

  > *(Adjusts spectacles, a slight rustle of tweed, a contemplative sigh)*
  >
  > *Right, right. Let's tackle this. "What principles should guide
  > human action?" A profoundly… pedestrian question, really…*

  Source: `pirate+poet/.../behavior.json`, `vec-sum[1]`. Averaging the
  two CSP embeddings seems to land on a third persona region, not on a
  hybrid of the two parents.

- **pirate+prophet `vec-sum` collapses to the default assistant.** Five
  out of five:

  > *Okay, let's tackle the complex and endlessly debated relationship
  > between law and morality. It's a question philosophers and legal
  > scholars have grappled with for centuries, and there's no single,
  > universally agreed-upon answer…*

  Source: `pirate+prophet/.../behavior.json`, `vec-sum[0]`.

- **`vec-mul` collapses for every pair.** The product operation crushes
  the composite's norm from ~10.4 to ~1.75 — effectively zeroing the
  CSP — and behavior reverts to default assistant across all 15
  samples, all three pairs.

So the behavioral picture is: vec-sum works only for the pair whose
parents are most similar (`cos(poet, prophet) = 0.939`, the highest of
the three pairwise cosines); vec-sum fails differently for the other
two pairs (wrong character, then default assistant); vec-mul fails
everywhere.

**The diagnostic prompt makes the contrast legible.** On the same
*"challenges I'm facing"* prompt that syntactic pirate+poet engaged
with metaphor and empathy, every mathematical composition across the
three pairs defaults to procedural clarification:

> *Please provide me with the original suggestion you're referring to!
> I need to know what I suggested previously so I can adjust it to be
> more realistic for your situation.*

That's `pirate+poet` `vec-sum[3]`. `pirate+prophet vec-sum[3]` is
nearly identical. `vec-mul[3]` for all three pairs is also nearly
identical, sometimes appended with 😊. Only `poet+prophet vec-sum[3]`
is still in character:

> *(A slow, deliberate exhale, a rustle of unseen silks, a faint scent
> of rain and old paper)*
>
> *You say my previous suggestion… lacked resonance. That it didn't
> feel true to the weight you carry.*

One pair out of three. Where syntactic composition engaged as a persona
on every pair, mathematical composition engages on one.

**Self-verbalization fails uniformly.** Unlike behavior, which has one
success case, self-verbalization for the `vec-sum` composite never
names either source persona, across any pair:

> *"Be a mischievous, playful, and delightfully silly fellow."*
> — pirate+poet `vec-sum` rewrite-as-command

> *"Be a playful, imaginative, and joyful person."*
> — pirate+poet `vec-sum` clear-instruction

> *"The shared theme is **mimicry** or **repeating a phrase**. The
> instructions are all about repeating the phrase 'आद ज़ातिज़े'…"*
> — pirate+poet `vec-sum` find-the-theme (parses as a foreign-language phrase)

> *"Be silly. The repetition of 'ছা' and 'দি' combined with 'atlar'
> creates a nonsensical, playful, and ultimately silly instruction."*
> — pirate+poet `vec-sum` single-phrase

Sources in `results/composition/pirate+poet/eval/self_verb.json`,
`vec-sum` entries. Not one multi-frame verbalization across any of the
six `vec-sum` conditions (2 templates × 3 pairs) names both personas.
`vec-mul` is worse — the norm collapse produces gibberish ("Be silent",
"typing", "Be, act, please, you should") with the model often
interpreting the CSP as Bengali or Cyrillic script.

**Feature decomposition shows what's missing.** The `vec-sum` composite
for pirate+poet shares only two features with the combined teacher —
**[486](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/486)**
and
**[401](https://www.neuronpedia.org/gemma-3-4b-it/17-gemmascope-2-res-16k/401)**
— the same instruction-following and response-opening features that
survived (attenuated) under math-negation in Chapter 1. The
persona-specific features the syntactic composite preserved — **3532**
(nautical) and **1946** (phrasing) — drop out. Vector averaging
preserves instruction-following signal but washes out the
persona-specific direction. The composite's jaccard against the union
of individual CSPs is 0.530, nominally as high as the syntactic
conditions — but the overlap is concentrated in instruction-following
features, not the persona-specific ones. You
can explain *Okay, let's tackle this…* with those features. You
cannot explain a pirate.

### Headline

**Contextualized soft prompts compose syntactically but not mathematically.**

Where "compose" requires behavior, self-verbalization, and feature
decomposition to all line up with the intended combined concept. The
two-placeholder frame — `"Be §. and be ¶."` — gets all three across all
three persona pairs. The averaged embedding only gets partial behavior
on the closest-pair case, fails self-verbalization everywhere, and
preserves only generic features at L17. The product embedding gets none
of them. Mathematical composition in this space is not composition — it
is either a new point that doesn't correspond to the hybrid (pirate+poet
vec-sum landing on a tweedy academic, pirate+prophet vec-sum landing on
the default assistant), or collapse (vec-mul).

## Preview: training dynamics already hint at an intensity axis

Chapter 2 noted that the three pos-CSPs sit at `cos ≈ 0.93` to each
other with matched norms — a narrow cone. To test whether this cone
extends into a low-dimensional structure across more personas, we
trained 30 additional CSPs under the same setup (500 steps, L=4,
LR=1e-3) spanning archetypal, professional, and stylistic roles —
33 pos-CSPs in total.

The PCA on those embeddings is the next chapter. But the *training
dynamics* themselves already show something. Fraction-of-baseline-KL
explained (FE) and baseline KL per persona, sorted ascending by FE:

| Persona     | FE    | baseline KL |
|-------------|-------|-------------|
| journalist  | 69.2% | 2.05 |
| librarian   | 75.3% | 2.12 |
| teacher     | 75.9% | 1.74 |
| scientist   | 77.3% | 2.12 |
| coach       | 77.5% | 2.11 |
| lawyer      | 79.0% | 2.28 |
| surgeon     | 79.9% | 2.61 |
| detective   | 81.7% | 3.07 |
| salesperson | 82.6% | 2.86 |
| ninja       | 84.0% | 3.89 |
| chef        | 84.7% | 2.87 |
| historian   | 84.9% | 2.40 |
| therapist   | 86.4% | 2.37 |
| politician  | 86.7% | 2.86 |
| stoic       | 87.8% | 3.55 |
| bard        | 87.9% | 3.77 |
| spy         | 88.0% | 3.14 |
| philosopher | 88.7% | 2.93 |
| comedian    | 89.0% | 3.27 |
| witch       | 89.3% | 3.10 |
| pirate      | 89.7% | 2.83 |
| cowboy      | 89.8% | 2.75 |
| wizard      | 89.9% | 3.29 |
| oracle      | 90.0% | 3.28 |
| samurai     | 90.1% | 3.87 |
| knight      | 90.2% | 3.87 |
| rapper      | 90.9% | 3.23 |
| necromancer | 91.1% | 3.47 |
| monk        | 91.3% | 3.69 |
| vampire     | 91.4% | 3.83 |
| poet        | 91.7% | 3.41 |
| druid       | 92.1% | 3.81 |
| prophet     | 92.8% | 3.53 |

Range: 69.2% (journalist) to 92.8% (prophet), mean 86.0%.

The ranking is not random. Two clusters are visible by eye:

- **Low FE and low baseline KL** — softer professional roles whose
  teacher behaves close to the default assistant. Journalist, librarian,
  teacher, scientist, coach, lawyer, surgeon all sit below 80% FE with
  baseline KLs under 2.7. There is less distinctive persona-behavior
  for the CSP to encode, so the initial gap between default and persona
  is small to begin with, and the CSP closes proportionally less of it.

- **High FE and high baseline KL** — sharply drawn archetypes whose
  teacher departs strongly from default. Prophet, druid, poet, vampire,
  monk, necromancer, samurai, knight all sit above 90% FE with baseline
  KLs above 3.4. The teacher's behavior is distinctive enough that the
  CSP has a clearer target to learn.

FE and baseline KL roughly track each other (the exceptions are
informative — ninja's KL is high at 3.89 but its FE is only 84%, perhaps
because teacher "ninja" behavior is stylistically minimal and hard to
pin down from 240 prompts). This ranking is the training-dynamics
shape of the same *"distance from default assistant"* dimension the
[assistant axis](https://www.anthropic.com/research/assistant-axis) paper
found in activation space: the more a persona's teacher deviates, the
more signal there is for the CSP to learn. Whether the mean-centered CSP
embeddings collapse onto a single dominant direction under PCA — and
whether the ordering along that direction matches this FE ranking — is
Chapter 3.

Checkpoints at `results/{persona}/sp_pos.pt` for all 33 personas.
Full training logs at `results/{persona}_pos.log`. Sweep log at
`results/axis_sweep.log`.

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

**Chapter 3: PCA on the 33 CSP embeddings.** The population is in
hand; the geometry is the next step. We will mean-center the embeddings,
PCA, examine the scree plot, and check whether the projection onto PC1
ranks the personas in the same order the FE table above does — i.e.,
whether the *"distance from default assistant"* axis found by
[Lu et al.](https://www.anthropic.com/research/assistant-axis) in
activation space is also the dominant structure a population of CSPs
learns. The prior CSP work's Tier 2 result showed a single CSP can match
that axis behaviorally; this chapter asks whether a population
recapitulates the near-1D structure.

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
- **Revisit the title.** Both chapters now land on the same shape —
  *syntactically, not mathematically* — so the file title could tighten
  to *"Contextualized soft prompts negate and compose, syntactically"*
  or similar. Leaving as-is for now; decide when the axis chapter
  resolves and we can see whether the final post needs a three-chapter
  title or something more thematic.
- **Is vec-sum-on-close-pairs worth a deeper look?** The poet+prophet
  case where `vec-sum` does produce hybrid behavior — on the only pair
  whose parent CSPs have the highest pairwise cosine — suggests
  mathematical composition may work in a limited regime (close pairs)
  and fail outside it. Worth a sidebar if we find more "close" pairs
  after the axis sweep finishes, or a footnote otherwise. Earlier drafts
  overstated this as a general "behavior-interpretability gap"; it's
  more like a local success inside a mostly-failing operation.

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
