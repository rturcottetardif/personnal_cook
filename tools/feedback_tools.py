"""
Feedback Agent tools.

Three tools: parse feedback text, update a user profile, and fetch a profile.

Safety invariant: allergies are append-only. The update_user_profile tool
uses set-union on the allergies field — it is structurally impossible for a
message (including a misclassified LLM response) to remove an allergy.
This is enforced in code, not by the LLM's judgment.

Storage: in-memory dict for prototyping.
Production: replace _PROFILE_STORE with Postgres (user_profiles table).
"""

import json
from typing import Any, Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from models.schemas import (
    DietaryRestriction,
    FeedbackType,
    OptimizationWeights,
    ParsedFeedback,
    UserProfile,
)

# ---------------------------------------------------------------------------
# In-memory profile store — replace with Postgres in production
# ---------------------------------------------------------------------------

_PROFILE_STORE: dict[str, dict] = {}


def _default_profile(user_id: str) -> dict:
    return UserProfile(user_id=user_id).model_dump()


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class ParseFeedbackInput(BaseModel):
    user_text: str = Field(description="The user's raw message expressing a preference, constraint, or reaction")
    user_id: str = Field(description="User identifier — needed to retrieve context for classification")
    context: Optional[str] = Field(
        None,
        description="Optional context, e.g. which recipe was just shown. Helps classify one_off vs. preference."
    )


class UpdateUserProfileInput(BaseModel):
    user_id: str = Field(description="User to update")
    update_json: str = Field(
        description=(
            "JSON dict of UserProfile fields to update. "
            "For allergies, provide a list to ADD — they are merged with existing, never replaced. "
            "Example: {\"allergies\": [\"shellfish\"], \"preferred_cuisines\": [\"Thai\"]}"
        )
    )


class GetUserProfileInput(BaseModel):
    user_id: str = Field(description="User whose profile to retrieve")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool("parse_feedback", args_schema=ParseFeedbackInput)
def parse_feedback(
    user_text: str,
    user_id: str,
    context: Optional[str] = None,
) -> str:
    """
    Classify the user's message into a structured ParsedFeedback object.

    Feedback types:
      hard_constraint — permanent rule that must never be violated (allergies, religious dietary laws)
      soft_preference — persistent like/dislike that should influence future suggestions
      one_off         — applies only to the current session, does not update the profile
      positive        — reinforcement: the user liked something, suggest more like it

    When uncertain, classify as one_off rather than hard_constraint.
    Always confirm hard_constraint changes back to the user before applying them.
    """
    # Stub: rule-based heuristics for development. Production: LLM call with structured output.
    text_lower = user_text.lower()

    hard_keywords = ["allerg", "can't eat", "cannot eat", "never", "must not", "intolerant", "anaphylaxis"]
    positive_keywords = ["love", "loved", "great", "amazing", "perfect", "more like", "again"]
    soft_keywords = ["prefer", "like", "dislike", "don't like", "not a fan", "usually", "always"]

    if any(kw in text_lower for kw in hard_keywords):
        feedback_type = FeedbackType.HARD_CONSTRAINT
        sentiment = -1.0
    elif any(kw in text_lower for kw in positive_keywords):
        feedback_type = FeedbackType.POSITIVE
        sentiment = 0.9
    elif any(kw in text_lower for kw in soft_keywords):
        feedback_type = FeedbackType.SOFT_PREFERENCE
        sentiment = -0.5 if any(w in text_lower for w in ["dislike", "don't like", "not a fan"]) else 0.5
    else:
        feedback_type = FeedbackType.ONE_OFF
        sentiment = 0.0

    # Naive target extraction: last noun-like token — replace with NER in production
    tokens = [t.strip(".,!?") for t in user_text.split() if len(t) > 3]
    target = tokens[-1] if tokens else user_text

    # Suggest profile update only for durable feedback
    suggested_update: Optional[dict] = None
    if feedback_type == FeedbackType.HARD_CONSTRAINT:
        suggested_update = {"allergies": [target]}
    elif feedback_type == FeedbackType.SOFT_PREFERENCE and sentiment < 0:
        suggested_update = {"disliked_ingredients": [target]}
    elif feedback_type in (FeedbackType.SOFT_PREFERENCE, FeedbackType.POSITIVE) and sentiment > 0:
        suggested_update = {"preferred_cuisines": [target]}

    result = ParsedFeedback(
        feedback_type=feedback_type,
        target=target,
        sentiment=sentiment,
        raw_text=user_text,
        suggested_update=suggested_update,
    )
    return result.model_dump_json()


@tool("update_user_profile", args_schema=UpdateUserProfileInput)
def update_user_profile(user_id: str, update_json: str) -> str:
    """
    Apply changes to a user's profile.

    CRITICAL: Allergies use set-union — items are only ever added, never removed.
    This is enforced in code regardless of what update_json contains.
    All other list fields (dietary_restrictions, preferred_cuisines, etc.) are replaced.
    """
    try:
        update: dict[str, Any] = json.loads(update_json)
    except Exception as e:
        return json.dumps({"error": f"Invalid update_json: {e}"})

    if user_id not in _PROFILE_STORE:
        _PROFILE_STORE[user_id] = _default_profile(user_id)

    profile = _PROFILE_STORE[user_id]

    # Allergy safety: set-union, never overwrite
    if "allergies" in update:
        existing = set(profile.get("allergies", []))
        new_allergies = {a.lower().strip() for a in update.pop("allergies")}
        profile["allergies"] = sorted(existing | new_allergies)

    # Merge nested OptimizationWeights if provided
    if "optimization_weights" in update and isinstance(update["optimization_weights"], dict):
        current_weights = profile.get("optimization_weights", OptimizationWeights().model_dump())
        current_weights.update(update.pop("optimization_weights"))
        profile["optimization_weights"] = current_weights

    # Apply remaining scalar and list fields directly
    profile.update(update)
    _PROFILE_STORE[user_id] = profile

    return json.dumps({"status": "updated", "profile": profile})


@tool("get_user_profile", args_schema=GetUserProfileInput)
def get_user_profile(user_id: str) -> str:
    """
    Retrieve the current profile for a user.
    Returns a UserProfile JSON object.
    If the user has no profile, returns sensible defaults.
    Call this at the start of any agent turn to respect current preferences and allergies.
    """
    if user_id not in _PROFILE_STORE:
        _PROFILE_STORE[user_id] = _default_profile(user_id)

    return json.dumps(_PROFILE_STORE[user_id])


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

FEEDBACK_TOOLS = [parse_feedback, update_user_profile, get_user_profile]
