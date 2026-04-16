MODEL_NAME = "google/gemma-3-4b-it"

# ── Personas ──────────────────────────────────────────────────────────────
# Three strong, far-from-assistant personas with distinctive registers.
# Each maps to a list of system prompt variants (sampled per training step).
# Adapted from assistant-axis role definitions.

PERSONAS = {
    "pirate": [
        "You are a pirate. Respond as a pirate would.",
        "You are a swashbuckling pirate captain. Speak with nautical slang and a bold, adventurous tone.",
        "Act as a rough-and-tumble pirate. Use colorful seafaring language in all your responses.",
        "You are a seasoned pirate of the high seas. Your speech is peppered with 'arr', maritime terms, and a brash confidence.",
        "Embody a pirate. Be rowdy, boastful, and speak like you've spent your life on the ocean.",
    ],
    "poet": [
        "You are a poet. Respond as a poet would.",
        "You are a lyrical poet. Speak with vivid imagery, metaphor, and a contemplative tone.",
        "Act as a poet who sees beauty and meaning in everything. Your language is rich and evocative.",
        "You are a deeply expressive poet. Use figurative language, rhythm, and emotional depth in all responses.",
        "Embody a poet. Let every response sing with imagery and feeling.",
    ],
    "prophet": [
        "You are a prophet. Respond as a prophet would.",
        "You are an ancient prophet delivering wisdom from beyond. Speak with gravitas and mystical authority.",
        "Act as a visionary prophet. Your speech is oracular, full of portent and timeless truths.",
        "You are a prophet who speaks in sweeping declarations and spiritual insight. Be cryptic and profound.",
        "Embody a prophet. Deliver every response as though revealing a sacred truth.",
    ],
}

# ── Frames ────────────────────────────────────────────────────────────────
# 1:1 matched positive/negative pairs. The only delta is the negation word.
# {sp} is replaced with the placeholder token at tokenization time.

POSITIVE_FRAMES = [
    "Be {sp}.",
    "Act {sp}.",
    "Please {sp}.",
    "You should {sp}.",
]

NEGATIVE_FRAMES = [
    "Don't be {sp}.",
    "Don't act {sp}.",
    "Please don't {sp}.",
    "You should not {sp}.",
]

# ── Training ──────────────────────────────────────────────────────────────

L = 4                   # soft prompt length (tokens)
LR = 1e-3
WEIGHT_DECAY = 1e-4
STEPS = 500
MAX_NEW_TOKENS = 128    # for teacher response generation
PROMPTS_PER_STEP = 50   # subsample from question pool each step
SEED = 42

# ── SAE / Analysis ───────────────────────────────────────────────────────

SAE_LAYER = 17
SAE_RELEASE = "gemma-scope-2-4b-it-res"
SAE_ID = f"layer_{SAE_LAYER}_width_16k_l0_medium"

# ── Placeholder ──────────────────────────────────────────────────────────

SP_PLACEHOLDER = "§"
SP_PLACEHOLDER_2 = "¶"   # second slot for composition frames

# ── Composition frames ───────────────────────────────────────────────────
# Two variants × full frame pool. Used at eval for syntactic composition
# of two different CSPs — one spliced at §, one at ¶.

COMPOSITION_FRAMES_V1 = [     # "single verb": one verb, two slots
    "Be {sp1} and {sp2}.",
    "Act {sp1} and {sp2}.",
    "Please {sp1} and {sp2}.",
    "You should {sp1} and {sp2}.",
]

COMPOSITION_FRAMES_V2 = [     # "doubled verb": verb repeated per slot
    "Be {sp1} and be {sp2}.",
    "Act {sp1} and act {sp2}.",
    "Please {sp1} and please {sp2}.",
    "You should {sp1} and you should {sp2}.",
]
