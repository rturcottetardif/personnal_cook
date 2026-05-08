"""
LLM factory and system-wide configuration.

All model names and temperatures live here so you can change a tier globally
without hunting through agent files. Cache TTLs are grouped here too — the
actual caching backend (Redis) reads these values in production.
"""

import os

# ---------------------------------------------------------------------------
# Model tiers
# ---------------------------------------------------------------------------

# Haiku: ~10x cheaper than Sonnet per token. Used for routing and classification
# tasks where language understanding matters but deep reasoning does not.
ROUTING_MODEL = "claude-haiku-4-5-20251001"

# Sonnet: main workhorse for planning, analysis, and explanation.
REASONING_MODEL = "claude-sonnet-4-6"

# Same model as reasoning — distinguished in the factory by temperature (0.7).
CREATIVE_MODEL = "claude-sonnet-4-6"

# ---------------------------------------------------------------------------
# Cache TTLs (seconds) — used by the Redis layer in production
# ---------------------------------------------------------------------------

CACHE_TTL_SECONDS: dict[str, int] = {
    "store_locations": 7 * 24 * 3600,   # 1 week  — geographic data is stable
    "store_prices": 24 * 3600,           # 1 day   — prices change overnight
    "discounts": 24 * 3600,              # 1 day   — refresh hourly during peak hours
    "nutrition_facts": 30 * 24 * 3600,  # 1 month — USDA database rarely changes
    # LLM responses are never cached — each request is contextual
}

# ---------------------------------------------------------------------------
# LLM factory
# ---------------------------------------------------------------------------

def get_llm(tier: str = "reasoning", temperature: float | None = None):
    """
    Return a ChatAnthropic instance for the requested tier.

    tier options:
      "routing"   — Haiku, default temp 0.0 (deterministic routing)
      "reasoning" — Sonnet, default temp 0.0 (structured planning)
      "creative"  — Sonnet, default temp 0.7 (recipe variety)

    Pass temperature explicitly to override the tier default.
    """
    # Import here to keep the module importable even before dependencies are installed.
    from langchain_anthropic import ChatAnthropic

    tier_defaults: dict[str, tuple[str, float]] = {
        "routing":   (ROUTING_MODEL,   0.0),
        "reasoning": (REASONING_MODEL, 0.0),
        "creative":  (CREATIVE_MODEL,  0.7),
    }

    model_name, default_temp = tier_defaults.get(tier, tier_defaults["reasoning"])
    effective_temp = temperature if temperature is not None else default_temp

    return ChatAnthropic(
        model=model_name,
        temperature=effective_temp,
        max_tokens=4096,
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
    )


# ---------------------------------------------------------------------------
# Optimizer tuning parameters — configurable per user in a future version
# ---------------------------------------------------------------------------

AVG_DRIVING_SPEED_KMH = 30.0      # conservative urban estimate
TIME_PER_STORE_VISIT_MIN = 15.0   # parking + walk-in + checkout
TIME_VALUE_PER_HOUR = 30.0        # user's hourly time value in local currency

# CP-SAT integer scaling factor: floats multiplied by SCALE before being cast
# to int for the solver. Higher = more precision, marginally slower.
OPTIMIZER_SCALE = 100
