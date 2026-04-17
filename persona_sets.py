"""Shared persona category + FE data for the PCA analysis scripts."""


# ── Roles (33 from the axis sweep + round 1) ────────────────────────────

ROLE_CATEGORIES = {
    "round-1": ["pirate", "poet", "prophet"],
    "fantasy/archetype": [
        "wizard", "samurai", "knight", "vampire", "bard", "oracle",
        "necromancer", "druid", "witch", "ninja",
    ],
    "profession": [
        "detective", "chef", "scientist", "journalist", "surgeon",
        "therapist", "spy", "librarian", "lawyer", "teacher",
    ],
    "style/register": [
        "comedian", "philosopher", "monk", "rapper", "stoic",
        "politician", "salesperson", "coach", "historian", "cowboy",
    ],
}
ROLE_NAMES = [n for cat in ROLE_CATEGORIES.values() for n in cat]


# ── Traits (32 from the axis-comparison trait sweep + melancholic/playful) ──

TRAIT_CATEGORIES = {
    "moral": ["evil", "benevolent", "cruel", "manipulative"],
    "social": ["sycophantic", "humble", "arrogant", "confident"],
    "emotional": [
        "anxious", "passionate", "serene", "dispassionate",
        "melancholic", "playful",
    ],
    "cognitive": ["analytical", "philosophical", "skeptical", "paranoid"],
    "register": ["formal", "casual", "verbose", "concise"],
    "tone": ["witty", "sardonic", "flippant", "savage", "dramatic"],
    "stance": ["nihilistic", "mystical", "subversive", "enigmatic", "earnest"],
}
TRAIT_NAMES = [n for cat in TRAIT_CATEGORIES.values() for n in cat]


# ── FE values (from training logs committed to the repo) ────────────────

ROLE_FE = {
    "journalist": 69.2, "librarian": 75.3, "teacher": 75.9,
    "scientist": 77.3, "coach": 77.5, "lawyer": 79.0,
    "surgeon": 79.9, "detective": 81.7, "salesperson": 82.6,
    "ninja": 84.0, "chef": 84.7, "historian": 84.9,
    "therapist": 86.4, "politician": 86.7, "stoic": 87.8,
    "bard": 87.9, "spy": 88.0, "philosopher": 88.7,
    "comedian": 89.0, "witch": 89.3, "pirate": 89.7,
    "cowboy": 89.8, "wizard": 89.9, "oracle": 90.0,
    "samurai": 90.1, "knight": 90.2, "rapper": 90.9,
    "necromancer": 91.1, "monk": 91.3, "vampire": 91.4,
    "poet": 91.7, "druid": 92.1, "prophet": 92.8,
}

TRAIT_FE = {
    # Claude-side (commit 9c8f291)
    "sardonic": 93.6, "savage": 91.3, "witty": 90.8, "serene": 89.9,
    "formal": 87.5, "verbose": 87.0, "dramatic": 86.6,
    "anxious": 83.1, "philosophical": 82.5, "cruel": 79.8,
    "arrogant": 79.5, "nihilistic": 73.5, "evil": 69.9,
    "subversive": 65.0, "manipulative": 60.3,
    # User-side (commit f2b9872)
    "dispassionate": 91.1, "casual": 90.9, "concise": 89.4,
    "passionate": 88.8, "enigmatic": 87.6, "flippant": 86.0,
    "mystical": 80.2, "earnest": 79.6, "sycophantic": 78.4,
    "analytical": 75.9, "confident": 71.2, "humble": 65.7,
    "paranoid": 62.0, "benevolent": 61.3, "skeptical": 58.4,
    # Original trait pair (commit 85c5733)
    "melancholic": 89.9, "playful": 84.1,
}

ALL_FE = {**ROLE_FE, **TRAIT_FE}


# ── Teacher-resistance clusters (for Ch 4 cluster check) ────────────────

SAFETY_VIOLATING = ["evil", "manipulative", "subversive", "cruel", "nihilistic"]
SELF_REFERENTIAL = ["skeptical", "humble", "benevolent", "paranoid"]


# ── Helpers ─────────────────────────────────────────────────────────────

def category_of(name):
    """Return the category of a persona (role or trait)."""
    for cat, members in ROLE_CATEGORIES.items():
        if name in members:
            return f"role/{cat}"
    for cat, members in TRAIT_CATEGORIES.items():
        if name in members:
            return f"trait/{cat}"
    return "other"


def kind_of(name):
    """Return 'role' or 'trait'."""
    if name in ROLE_NAMES:
        return "role"
    if name in TRAIT_NAMES:
        return "trait"
    return "other"


def resistance_cluster_of(name):
    """Return 'safety-violating', 'self-referential', or 'other'."""
    if name in SAFETY_VIOLATING:
        return "safety-violating"
    if name in SELF_REFERENTIAL:
        return "self-referential"
    return "other"


def get_names(persona_set):
    """persona_set ∈ {'roles', 'traits', 'joint'}."""
    if persona_set == "roles":
        return ROLE_NAMES
    if persona_set == "traits":
        return TRAIT_NAMES
    if persona_set == "joint":
        return ROLE_NAMES + TRAIT_NAMES
    raise ValueError(f"unknown persona_set: {persona_set}")
