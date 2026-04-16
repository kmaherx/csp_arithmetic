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
operations on CSP embeddings — sign flips, sums, elementwise products —
don't preserve persona structure, because instructions aren't linear in
embedding space; they're patterns the downstream layers *interpret*.
Syntactic operations — choice of frame, conjunctions — do compose,
because well-formed English is the space those downstream layers know
how to read. **CSPs use language as the language.** They inherit
on-manifold, in-distribution behavior by riding the model's own
pretrained machinery for reading instructions. The cost is training: a
per-concept gradient-descent procedure where steering vectors are
closed-form.

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

- **Chapter 1 — negation.** Syntactic negation (`"Don't be §."` around a
  positive-frame CSP) preserves the persona concept at L17, often
  strengthening it, and lets the frame's "Don't" invert the behavioral
  readout downstream. Mathematical negation (sign-flipped embedding in
  `"Be §."`) destroys the persona concept at L17 and produces
  default-assistant output — with no persona signal to negate. A clean
  preservation-vs-destruction split.
- **Chapter 2 — composition.** Syntactic composition
  (`"Be §. and be ¶."` with two trained CSPs) succeeds on all three
  readouts across all three persona pairs. Mathematical composition
  (vector sum and elementwise product of the two embeddings) fails
  self-verbalization uniformly and fails behavior in two of three
  pairs. Same mechanism as Chapter 1: linguistic ops preserve the
  persona-specific structure at L17; vector ops wash it out.
- **Chapter 3 — geometry (pivot).** PCA on the population of 33 trained
  CSPs tests whether, at the *population* level, CSPs recapitulate the
  single-axis structure that [Lu et al. (2026)](https://www.anthropic.com/research/assistant-axis)
  found for persona variation in activation space via CAA. If they do,
  the axis is *there* — but Chapters 1–2 already show you can't *use*
  that linear structure by doing arithmetic on individual CSPs.
  Linguistic ops are the only access. This is where the whole-document
  tension resolves: CSPs inherit the structural geometry steering
  vectors give you for free, but operate on it through language rather
  than arithmetic, at the cost of training.

The remainder of this document establishes each piece.

## Setup

- **Model:** Gemma 3 4B IT (`google/gemma-3-4b-it`).
- **Personas (33):** Chapters 1–2 use the three most-studied archetypes —
  pirate, poet, prophet — plus trained sp_neg checkpoints for those
  three. Chapter 3's PCA uses 33 total (the original 3 plus 30 more
  archetypal, professional, and stylistic personas from the axis sweep).
  Each persona is defined by 5 system-prompt variants sampled during
  training. Adapted from the assistant-axis role definitions.
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

For pirate / poet / prophet we have three CSP sources: `sp_pos.pt` in
positive frames, `sp_neg.pt` in negative frames, and `math-neg` — a
sign-flipped `-sp_pos` constructed at eval time via `negate_csp()` in
`soft_prompt.py`. Evaluation is a 3×2 grid: CSP source (pos / neg /
math-neg) × evaluation frame (pos / neg). For the other 30 personas we
train only `sp_pos.pt` since the PCA analysis only needs the positive-
frame checkpoints.

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
the persona-specific features it would need are no longer active.
Language-level operations act on a preserved structure; embedding-level
operations break the structure they would need to act on. In the frame's
terms, this is *language as the language* at work: the frame route keeps
the CSP inside the space the model knows how to read, while the math
route exits that space and leaves the downstream layers with nothing to
interpret. Chapter 2 will show the exact same mechanism for `vec-sum`
and `vec-mul`. It's also the mechanism behind the boundary observation
that trained CSP_neg does not cleanly invert — see footnote.
[^neg-boundary]

## Chapter 2: CSPs compose syntactically, but not mathematically

The second arithmetic operation from the frame. Chapter 1 showed that a
single CSP supports negation through the language route but not the math
route. The natural next question: can *two* trained CSPs be combined —
can the model hold a pirate concept and a poet concept in one utterance
and act on both — and which of the two routes delivers?

To be clear up front: by "compose" we mean the same stringent bar from
the frame — behavior, self-verbalization, and feature decomposition all
need to line up with the intended hybrid:

1. **Output behavior** looks like a hybrid of the two personas.
2. **Self-verbalization** under multi-frame prompts describes *both*
   source personas, not just one or a generic character.
3. **Feature decomposition** at layer 17 shows that the composite fires
   features characteristic of both parents, not only generic features
   shared across all personas.

By this bar, **syntactic composition succeeds and mathematical composition
does not** — the same preservation-vs-destruction split as Chapter 1,
extended from one operation to two.

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

Chapters 1 and 2 together put two of the three arithmetic operations —
negation and composition — on the same preservation-vs-destruction
footing. Syntactic operations act on a preserved L17 concept; vector
operations break the concept and leave only generic instruction-
following features behind. *Language is the language* isn't a metaphor:
it's the specific constraint these two chapters land on. The third
operation — *scaling* — remains untested here and is flagged in Future
work below.

## Preview of the Chapter 3 pivot: training dynamics already hint at an intensity axis

Chapters 1–2 argued, on individual CSPs, that arithmetic works through
language and not through vector operations on embeddings. Chapter 3
asks a different question at the *population* level: do CSPs, trained
independently on many personas, share the same axis-like geometric
structure that steering vectors exhibit for free? If yes, the tension
from the frame sharpens: the linear structure is *there* in the
population — but Chapters 1–2 show it can't be exploited by doing
arithmetic on individual CSPs.

Chapter 2 noted that the three pos-CSPs sit at `cos ≈ 0.93` to each
other with matched norms — a narrow cone. To test whether this cone
extends into a low-dimensional structure across more personas, we
trained 30 additional CSPs under the same setup (500 steps, L=4,
LR=1e-3) spanning archetypal, professional, and stylistic roles —
33 pos-CSPs in total.

The PCA on those embeddings is the pivot chapter. But the *training
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
found in activation space via CAA: the more a persona's teacher
deviates, the more signal there is for the CSP to learn. Whether the
mean-centered CSP embeddings collapse onto a single dominant direction
under PCA — and whether the ordering along that direction matches this
FE ranking — is Chapter 3's pivot: the axis visible at the population
level, whose access Chapters 1–2 showed is restricted to language.

Checkpoints at `results/{persona}/sp_pos.pt` for all 33 personas.
Full training logs at `results/{persona}_pos.log`. Sweep log at
`results/axis_sweep.log`.

## Data and code references

### Checkpoints
- `results/{pirate,poet,prophet}/sp_pos.pt` + `sp_neg.pt` — positive and
  negative-frame CSPs for the three deeply-evaluated personas
- `results/{persona}/sp_pos.pt` for 30 axis-sweep personas (wizard,
  samurai, knight, vampire, bard, oracle, necromancer, druid, witch,
  ninja, detective, chef, scientist, journalist, surgeon, therapist,
  spy, librarian, lawyer, teacher, comedian, philosopher, monk, rapper,
  stoic, politician, salesperson, coach, historian, cowboy)
- `math-neg` for pirate / poet / prophet is constructed at eval time by
  `negate_csp(sp_pos)` in `soft_prompt.py` — no saved checkpoint.

### Evaluation outputs (per persona, under `results/{persona}/eval/`)
Present for pirate / poet / prophet only; other personas have
checkpoints but no eval run.
- `self_verb.json` — 9 verbalization prompts × 6 conditions (5 multi-frame
  + 4 single-frame per frame polarity; 3 CSP sources × 2 frame polarities)
- `sae.json` — SAE reconstruction + jaccard overlap with persona
  ground-truth features at L17, all 6 conditions
- `behavior.json` — 5 held-out behavioral samples × 6 conditions
- `embedding_compare.json` — cosines and norms for sp_pos / sp_neg /
  math-neg (including the cos(pos, math-neg) = -1.0 sanity check)

### Composition outputs (per pair, under `results/composition/{pair}/eval/`)
- `self_verb.json` — 5 multi-frame + 4 single-frame verbalizations × 6
  conditions (4 syntactic + 2 vector)
- `sae.json` — SAE reconstruction + jaccards against combined teacher,
  individual CSP_A, individual CSP_B, and their union
- `behavior.json` — 5 held-out behavioral samples × 6 conditions
- `embedding_compare.json` — composite norms and cosines to each parent

### Training logs
- `results/{persona}_{pos,neg}.log` — training logs (sp_pos for all 33,
  sp_neg for the original three)
- `results/axis_sweep.log` — aggregate sweep log for the 30 new personas
- `results/eval_{persona}_v3.log` — latest evaluate.py rerun with the 3×2
  grid (includes math-neg conditions)
- `results/composition/{pair}.log` — composition eval per pair

### Data
- `data/questions.jsonl` — 240 extraction questions from the assistant-axis
  repo

### Code
- `train.py` — KL distillation, one persona × one polarity per invocation
- `evaluate.py` — per-persona eval across the 3×2 grid (self-verb, sae,
  behavior, embedding)
- `compose.py` — composite-CSP constructors (sum, mul), two-slot splicing,
  composition-frame verbalization prompts, combined-teacher extractor
- `run_composition.py` — driver that runs all 6 composition conditions per pair
- `run_axis_sweep.sh` — sweep trainer for an arbitrary persona list
  (positive polarity only)
- `run_sweep.sh` / `run_composition_sweep.sh` — original-personas sweeps
- `config.py` — personas, frames, hyperparameters
- `soft_prompt.py` — `SoftPrompt` nn.Module + `negate_csp()` helper

## What comes next

**Chapter 3 (pivot): PCA on the 33 CSP embeddings.** The population is
in hand; the geometry is the next step. Mean-center the embeddings, PCA,
examine the scree plot, and check whether the projection onto PC1 ranks
the personas in the same order the FE table above does — i.e., whether
the *"distance from default assistant"* axis
[Lu et al.](https://www.anthropic.com/research/assistant-axis) found in
activation space via CAA is also the dominant structure a population of
independently-trained CSPs learns. Prior CSP work's Tier 2 result showed
a *single* CSP can match that axis behaviorally; this chapter asks
whether the population recapitulates the near-1D structure. If it does,
the frame's central tension lands: the same linear geometry steering
vectors give you for free is present in the CSP population, but
Chapters 1–2 already showed the arithmetic that would let you slide
along it doesn't work on individual CSPs — language is the only access.

## Future work

- **Scaling (the third arithmetic operation).** Chapters 1 and 2 cover
  negation and composition. The three-level frame promises scaling too:
  the mathematical route is `α · CSP.embedding` for `α ∈ ℝ`; the
  semantic route is adverbial intensification inside the frame —
  `"Be slightly §."`, `"Be very §."`, `"Be extremely §."`. Prediction
  from Chapters 1–2: the semantic route should pass all three readouts
  monotonically with intensity; the mathematical route should fail
  (either magnitude pushes the CSP into gibberish territory like
  `vec-mul`, or it looks like a weakened/strengthened persona only
  within a narrow band and collapses outside it). Running these
  experiments before the Chapter 3 pivot strengthens the foundation
  for the whole-document claim that all three arithmetic ops — not
  just two — succeed through language and fail through vector ops.
- **Is vec-sum-on-close-pairs worth a deeper look?** The poet+prophet
  case where `vec-sum` produces hybrid behavior — on the only pair
  whose parent CSPs have the highest pairwise cosine — suggests
  mathematical composition may work in a narrow regime (close pairs)
  and fail outside it. Worth a sidebar if more "close" pairs surface
  after Chapter 3's population analysis lands, or a footnote otherwise.
- **Training cost.** CSPs are 500 AdamW steps per persona; CAA is a
  closed-form mean-difference after forward passes. The frame's honest
  cost story is that CSPs trade training compute for on-manifold
  behavior. A direct comparison between a trained CSP and the CAA
  persona vector on the same three readouts would quantify that
  tradeoff. Out of scope for this document.

## Session state — where we are now

Last updated: post-reframe — doc retitled and restructured around the
three-level frame (steering vectors vs CSPs; math vs semantic ops;
behavior / self-verb / feature decomp).

**Done.**
- Motivation rewrites around the *arithmetic-is-language vs language-
  is-language* axis, with citations for both steering-vector limitations
  (off-manifold at scale, interference under composition).
- "The frame" section lays out the three levels explicitly and previews
  the three-chapter headline.
- Chapter 1 (syntactic vs mathematical negation) — full writeup with
  preservation-vs-destruction mechanism; opening and headline tied to
  the frame.
- Chapter 2 (syntactic vs mathematical composition) — three persona
  pairs, all six conditions each; opening and headline tied to the
  frame, flagging scaling as the outstanding third op.
- Preview section repositioned as foreshadowing of the Chapter 3 pivot.
- All 33 `sp_pos.pt` checkpoints committed. Trained FE range 69.2%
  (journalist) to 92.8% (prophet), mean 86.0%.

**Next up.**
- **Scaling experiments.** Before Chapter 3's pivot, run the scaling
  analogue of Chapters 1–2 (math: α · embedding; semantic: adverbial
  frames). This completes the three-operation coverage promised by
  the frame.
- **Chapter 3 (pivot): PCA on the 33 CSP embeddings.** Mean-center,
  PCA, scree plot, project onto PC1–PC2, check whether the PC1 ordering
  correlates with the Preview section's FE ranking. If it does and the
  scree is sharp, we have a population-level match to Lu et al.'s
  assistant axis — an axis whose access Chapters 1–2 restricted to
  language.

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
