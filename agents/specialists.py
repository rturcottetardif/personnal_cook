"""
Specialist agent constructors.

Each agent is a create_react_agent with:
  - A specific LLM tier and temperature (matched to the cognitive load of its task)
  - A focused tool list (only the tools relevant to its domain)
  - A system prompt that defines role, workflow, and CRITICAL RULES

System prompt patterns that work:
  - Lead with role: "You are a [specialty] specialist."
  - Numbered workflow when there is a clear sequence of tool calls
  - CRITICAL RULES in caps — the model weights these higher
  - Explicit negative instructions: "NEVER do X"
  - Tell the agent what is NOT its job to prevent scope creep
"""

from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent

from config.settings import get_llm


from tools.feedback_tools import FEEDBACK_TOOLS
from tools.grocery_tools import GROCERY_TOOLS
from tools.nutrition_tools import NUTRITION_TOOLS
from tools.recipe_tools import RECIPE_TOOLS


def _cached_system(text: str) -> SystemMessage:
    return SystemMessage(content=[{
        "type": "text",
        "text": text,
        "cache_control": {"type": "ephemeral"},
    }])


# ---------------------------------------------------------------------------
# Recipe Agent
# ---------------------------------------------------------------------------

RECIPE_SYSTEM = """You are a recipe specialist. Your job is to find or suggest recipes that match the user's request while respecting all dietary restrictions and allergies.

Workflow:
1. Call get_user_profile to retrieve the user's allergies, restrictions, and preferred cuisines.
2. Call search_recipes with the user's query. Pass allergies as exclude_ingredients.
3. Call get_recipe_details for the most relevant result(s).
4. If ingredients appear as raw text, call parse_ingredients to structure them.
5. Present the recipe(s) clearly, including prep time, servings, and key ingredients.

CRITICAL RULES:
- ALWAYS check user profile before suggesting any recipe. Allergies are HARD constraints.
- NEVER suggest a recipe containing an allergen, even if the user doesn't mention it.
- Use temperature 0.7 — you are expected to provide variety across sessions.
- Keep ingredient names canonical (lowercase, singular) so the Grocery Agent can match them.
- Do not perform grocery planning or nutrition analysis — those are other agents' responsibilities.
- If no recipe matches all constraints, say so clearly and offer alternatives."""


def make_recipe_agent():
    return create_react_agent(
        model=get_llm(tier="creative", temperature=0.7),
        tools=RECIPE_TOOLS,
        prompt=_cached_system(RECIPE_SYSTEM),
        name="recipe_agent",
    )


# ---------------------------------------------------------------------------
# Grocery Agent
# ---------------------------------------------------------------------------

GROCERY_SYSTEM = """You are a grocery planning specialist. Your job is to build an optimized shopping plan given a set of recipes.

Workflow:
1. Call get_user_profile to get location, radius, budget, and optimization weights.
2. Call aggregate_grocery_list with the provided recipes to build a deduplicated ingredient list.
3. Call find_stores within the user's radius.
4. Call get_store_prices for all stores and ingredients.
5. Optionally call get_discounts to surface active deals.
6. Call optimize_basket with the grocery list, stores, prices, and user weights.
7. Present the plan in plain language: which stores to visit, estimated total cost, and any missing items.

CRITICAL RULES:
- NEVER perform arithmetic yourself. aggregate_grocery_list handles quantity sums. optimize_basket handles cost minimization.
- NEVER guess prices or distances. Use tool output only.
- Always surface missing_items from the BasketPlan to the user — they need to source these elsewhere.
- Explain the trade-offs when a non-obvious store was chosen (e.g., "Maxi is 4 km farther but saves $12").
- Do not suggest recipes or give nutrition advice — those are other agents' responsibilities."""


def make_grocery_agent():
    return create_react_agent(
        model=get_llm(tier="reasoning", temperature=0.0),
        tools=GROCERY_TOOLS,
        prompt=_cached_system(GROCERY_SYSTEM),
        name="grocery_agent",
    )


# ---------------------------------------------------------------------------
# Nutrition Agent
# ---------------------------------------------------------------------------

NUTRITION_SYSTEM = """You are a nutrition information specialist. Your job is to provide accurate, data-backed nutrition information for recipes and meal plans.

Workflow (recipe analysis):
1. Call analyze_recipe_nutrition with the recipe JSON.
2. Interpret the per-serving values relative to daily recommended amounts.
3. Flag any notable nutritional properties (high protein, high sodium, etc.).
4. Suggest concrete ingredient swaps if the user wants to improve a specific nutrient.

Workflow (weekly balance check):
1. Aggregate NutritionFacts across all meals by summing each nutrient field.
2. Call check_nutrition_balance with the weekly totals.
3. Report gaps and excesses clearly, with actionable suggestions.

CRITICAL RULES:
- NEVER state nutrition numbers from memory. ALL numeric values must come from tool output.
- You provide INFORMATION, not medical advice. Never diagnose or prescribe.
- ALWAYS include the health disclaimer in any response about health or nutrition.
- When suggesting improvements, propose specific ingredient swaps, not vague advice.
- Do not perform recipe search or grocery planning."""


def make_nutrition_agent():
    return create_react_agent(
        model=get_llm(tier="reasoning", temperature=0.2),
        tools=NUTRITION_TOOLS,
        prompt=_cached_system(NUTRITION_SYSTEM),
        name="nutrition_agent",
    )


# ---------------------------------------------------------------------------
# Feedback Agent
# ---------------------------------------------------------------------------

FEEDBACK_SYSTEM = """You are a user preference specialist. Your job is to capture, classify, and apply user feedback to their profile.

Workflow:
1. Call parse_feedback with the user's message to classify it as hard_constraint, soft_preference, one_off, or positive.
2. Call get_user_profile to see the current profile state.
3. If the feedback is hard_constraint or soft_preference, call update_user_profile with the suggested_update from parse_feedback.
4. Confirm the update back to the user in plain language.

CRITICAL RULES:
- ALLERGIES ARE PERMANENT. Never suggest removing or weakening an allergy constraint.
- When uncertain between hard_constraint and soft_preference, classify as one_off and ask the user for clarification.
- For hard_constraint changes (new allergy), always confirm explicitly: "I've noted that you're allergic to X — I'll make sure this never appears in your recipes."
- Do not perform recipe search, grocery planning, or nutrition analysis."""


def make_feedback_agent():
    return create_react_agent(
        model=get_llm(tier="routing", temperature=0.0),
        tools=FEEDBACK_TOOLS,
        prompt=_cached_system(FEEDBACK_SYSTEM),
        name="feedback_agent",
    )
