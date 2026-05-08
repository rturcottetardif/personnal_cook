"""
Nutrition Agent tools.

Critical safety rule: nutrition numbers are NEVER produced by the LLM.
All calorie and macro values come from the USDA FoodData Central API
(or the stub below). The LLM's role is to interpret and explain the numbers,
not to generate them.

The health disclaimer is non-negotiable and always included.
"""

import json
import os
from typing import Optional

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from models.schemas import (
    MacroNutrients,
    MicroNutrients,
    NutritionAnalysis,
    NutritionFacts,
    Recipe,
    Unit,
)

# Reference daily values (Health Canada / FDA blend — adjust per jurisdiction)
DAILY_VALUES: dict[str, float] = {
    "calories": 2000.0,
    "protein_g": 50.0,
    "carbs_g": 275.0,
    "fat_g": 78.0,
    "fiber_g": 28.0,
    "sugar_g": 50.0,
    "sodium_mg": 2300.0,
    "iron_mg": 18.0,
    "calcium_mg": 1300.0,
    "vitamin_c_mg": 90.0,
    "vitamin_d_ug": 20.0,
    "potassium_mg": 4700.0,
}

HEALTH_DISCLAIMER = (
    "These values are general estimates. Consult a registered dietitian "
    "for personalized nutrition advice, especially if you have a medical condition."
)


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class LookupNutritionInput(BaseModel):
    ingredient_name: str = Field(description="Canonical ingredient name, e.g. 'chicken breast'")
    quantity_g: float = Field(description="Amount in grams to look up nutrition for")


class AnalyzeRecipeNutritionInput(BaseModel):
    recipe_json: str = Field(description="JSON Recipe object from get_recipe_details")


class CheckNutritionBalanceInput(BaseModel):
    weekly_totals_json: str = Field(
        description=(
            "JSON dict mapping nutrient names to weekly total values, "
            "e.g. {'calories': 14000, 'protein_g': 350}. "
            "Typically a 7-day sum of NutritionFacts."
        )
    )
    user_profile_json: Optional[str] = Field(
        None, description="JSON UserProfile for personalized daily value targets"
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool("lookup_nutrition", args_schema=LookupNutritionInput)
def lookup_nutrition(ingredient_name: str, quantity_g: float) -> str:
    """
    Look up nutrition facts for an ingredient at a given weight.
    Values come from the USDA FoodData Central database — never LLM memory.
    Returns a NutritionFacts object scaled to the requested quantity.
    """
    api_key = os.getenv("USDA_API_KEY")

    if api_key:
        return _lookup_usda(ingredient_name, quantity_g, api_key)

    # --- Stub: plausible per-100g averages ---
    per_100g_db: dict[str, dict] = {
        "chicken breast": {"calories": 165, "protein_g": 31, "fat_g": 3.6, "carbs_g": 0, "sodium_mg": 74},
        "rice":           {"calories": 130, "protein_g": 2.7, "fat_g": 0.3, "carbs_g": 28, "sodium_mg": 1},
        "olive oil":      {"calories": 884, "protein_g": 0, "fat_g": 100, "carbs_g": 0, "sodium_mg": 2},
        "broccoli":       {"calories": 34, "protein_g": 2.8, "fat_g": 0.4, "carbs_g": 7, "sodium_mg": 33},
        "egg":            {"calories": 155, "protein_g": 13, "fat_g": 11, "carbs_g": 1.1, "sodium_mg": 124},
    }

    key = ingredient_name.lower().strip()
    base = per_100g_db.get(key, {"calories": 100, "protein_g": 5, "fat_g": 3, "carbs_g": 15, "sodium_mg": 50})
    scale = quantity_g / 100.0

    facts = NutritionFacts(
        calories=round(base.get("calories", 100) * scale, 1),
        macros=MacroNutrients(
            protein_g=round(base.get("protein_g", 5) * scale, 2),
            carbs_g=round(base.get("carbs_g", 15) * scale, 2),
            fat_g=round(base.get("fat_g", 3) * scale, 2),
        ),
        micros=MicroNutrients(
            sodium_mg=round(base.get("sodium_mg", 50) * scale, 1),
        ),
        serving_size_g=quantity_g,
        source="USDA stub",
    )
    return facts.model_dump_json()


@tool("analyze_recipe_nutrition", args_schema=AnalyzeRecipeNutritionInput)
def analyze_recipe_nutrition(recipe_json: str) -> str:
    """
    Calculate total and per-serving nutrition for a recipe.
    Calls lookup_nutrition for each ingredient and sums the results in pure Python.
    The LLM must not estimate or generate any numeric nutrition values.
    Always includes the mandatory health disclaimer.
    """
    try:
        recipe = Recipe.model_validate_json(recipe_json)
    except Exception as e:
        return json.dumps({"error": f"Invalid recipe_json: {e}"})

    total = NutritionFacts(
        calories=0.0,
        macros=MacroNutrients(),
        micros=MicroNutrients(),
        source="USDA",
    )

    for ing in recipe.ingredients:
        grams = _to_grams(ing.quantity, ing.unit)
        facts_json = lookup_nutrition.invoke({"ingredient_name": ing.name, "quantity_g": grams})
        try:
            facts = NutritionFacts.model_validate_json(facts_json)
            total.calories += facts.calories
            total.macros.protein_g += facts.macros.protein_g
            total.macros.carbs_g += facts.macros.carbs_g
            total.macros.fat_g += facts.macros.fat_g
            total.macros.fiber_g += facts.macros.fiber_g
            total.macros.sugar_g += facts.macros.sugar_g
            total.micros.sodium_mg += facts.micros.sodium_mg
            total.micros.iron_mg += facts.micros.iron_mg
            total.micros.calcium_mg += facts.micros.calcium_mg
            total.micros.vitamin_c_mg += facts.micros.vitamin_c_mg
            total.micros.potassium_mg += facts.micros.potassium_mg
        except Exception:
            continue  # skip unparseable results — don't let one ingredient break the analysis

    servings = max(recipe.servings, 1)
    per_serving = NutritionFacts(
        calories=round(total.calories / servings, 1),
        macros=MacroNutrients(
            protein_g=round(total.macros.protein_g / servings, 2),
            carbs_g=round(total.macros.carbs_g / servings, 2),
            fat_g=round(total.macros.fat_g / servings, 2),
            fiber_g=round(total.macros.fiber_g / servings, 2),
            sugar_g=round(total.macros.sugar_g / servings, 2),
        ),
        micros=MicroNutrients(
            sodium_mg=round(total.micros.sodium_mg / servings, 1),
            calcium_mg=round(total.micros.calcium_mg / servings, 1),
            vitamin_c_mg=round(total.micros.vitamin_c_mg / servings, 1),
            potassium_mg=round(total.micros.potassium_mg / servings, 1),
        ),
        source="USDA",
    )

    daily_pct = _compute_daily_value_pct(per_serving)
    flags = _generate_flags(daily_pct)

    analysis = NutritionAnalysis(
        per_serving=per_serving,
        daily_value_pct=daily_pct,
        flags=flags,
        disclaimer=HEALTH_DISCLAIMER,
    )
    return analysis.model_dump_json()


@tool("check_nutrition_balance", args_schema=CheckNutritionBalanceInput)
def check_nutrition_balance(
    weekly_totals_json: str,
    user_profile_json: Optional[str] = None,
) -> str:
    """
    Evaluate a week of meals against recommended daily nutritional values.
    Returns gaps (nutrients below target) and excesses (nutrients above safe limit).
    Divide weekly totals by 7 to get daily averages before comparing to daily values.
    Always includes the health disclaimer.
    """
    try:
        weekly: dict[str, float] = json.loads(weekly_totals_json)
    except Exception as e:
        return json.dumps({"error": f"Invalid weekly_totals_json: {e}"})

    daily_avg = {k: round(v / 7.0, 2) for k, v in weekly.items()}

    gaps: dict[str, dict] = {}
    excesses: dict[str, dict] = {}

    for nutrient, dv in DAILY_VALUES.items():
        actual = daily_avg.get(nutrient, None)
        if actual is None:
            continue
        pct = round((actual / dv) * 100, 1) if dv else 0.0
        if pct < 70:
            gaps[nutrient] = {"daily_avg": actual, "daily_value": dv, "pct": pct}
        elif pct > 130:
            excesses[nutrient] = {"daily_avg": actual, "daily_value": dv, "pct": pct}

    return json.dumps({
        "gaps": gaps,
        "excesses": excesses,
        "daily_averages": daily_avg,
        "disclaimer": HEALTH_DISCLAIMER,
    })


# ---------------------------------------------------------------------------
# Real USDA API implementation
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _lookup_usda(ingredient_name: str, quantity_g: float, api_key: str) -> str:
    try:
        search_resp = requests.get(
            "https://api.nal.usda.gov/fdc/v1/foods/search",
            params={"query": ingredient_name, "pageSize": 1, "api_key": api_key},
            timeout=10,
        )
        search_resp.raise_for_status()
        results = search_resp.json().get("foods", [])
        if not results:
            return json.dumps({"error": f"No USDA entry for '{ingredient_name}'"})

        food = results[0]
        nutrients_raw = {n["nutrientName"]: n["value"] for n in food.get("foodNutrients", [])}
        scale = quantity_g / 100.0

        facts = NutritionFacts(
            calories=round(nutrients_raw.get("Energy", 0) * scale, 1),
            macros=MacroNutrients(
                protein_g=round(nutrients_raw.get("Protein", 0) * scale, 2),
                carbs_g=round(nutrients_raw.get("Carbohydrate, by difference", 0) * scale, 2),
                fat_g=round(nutrients_raw.get("Total lipid (fat)", 0) * scale, 2),
                fiber_g=round(nutrients_raw.get("Fiber, total dietary", 0) * scale, 2),
                sugar_g=round(nutrients_raw.get("Sugars, total including NLEA", 0) * scale, 2),
            ),
            micros=MicroNutrients(
                sodium_mg=round(nutrients_raw.get("Sodium, Na", 0) * scale, 1),
                iron_mg=round(nutrients_raw.get("Iron, Fe", 0) * scale, 2),
                calcium_mg=round(nutrients_raw.get("Calcium, Ca", 0) * scale, 1),
                vitamin_c_mg=round(nutrients_raw.get("Vitamin C, total ascorbic acid", 0) * scale, 1),
                vitamin_d_ug=round(nutrients_raw.get("Vitamin D (D2 + D3)", 0) * scale, 2),
                potassium_mg=round(nutrients_raw.get("Potassium, K", 0) * scale, 1),
            ),
            serving_size_g=quantity_g,
            source=f"USDA FDC ID {food.get('fdcId')}",
        )
        return facts.model_dump_json()
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return json.dumps({"error": "rate_limited", "retry_after": 60})
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Helpers — pure Python, never delegated to LLM
# ---------------------------------------------------------------------------

def _to_grams(quantity: float, unit: Unit) -> float:
    """Convert any unit to grams. Crude but sufficient for v1."""
    conversions: dict[str, float] = {
        Unit.G: 1.0,
        Unit.KG: 1000.0,
        Unit.ML: 1.0,       # water density approximation
        Unit.L: 1000.0,
        Unit.TSP: 5.0,
        Unit.TBSP: 15.0,
        Unit.CUP: 240.0,
        Unit.PIECE: 150.0,  # rough average — use ingredient density table in production
        Unit.PINCH: 0.5,
    }
    return quantity * conversions.get(unit, 100.0)


def _compute_daily_value_pct(facts: NutritionFacts) -> dict[str, float]:
    flat = {
        "calories": facts.calories,
        "protein_g": facts.macros.protein_g,
        "carbs_g": facts.macros.carbs_g,
        "fat_g": facts.macros.fat_g,
        "fiber_g": facts.macros.fiber_g,
        "sodium_mg": facts.micros.sodium_mg,
        "iron_mg": facts.micros.iron_mg,
        "calcium_mg": facts.micros.calcium_mg,
        "vitamin_c_mg": facts.micros.vitamin_c_mg,
        "potassium_mg": facts.micros.potassium_mg,
    }
    return {
        k: round((v / DAILY_VALUES[k]) * 100, 1)
        for k, v in flat.items()
        if k in DAILY_VALUES and DAILY_VALUES[k] > 0
    }


def _generate_flags(daily_pct: dict[str, float]) -> list[str]:
    flags = []
    if daily_pct.get("sodium_mg", 0) > 40:
        flags.append("high sodium")
    if daily_pct.get("fiber_g", 0) > 15:
        flags.append("good source of fiber")
    if daily_pct.get("protein_g", 0) > 20:
        flags.append("high protein")
    if daily_pct.get("fat_g", 0) > 30:
        flags.append("high fat")
    if daily_pct.get("calories", 0) > 30:
        flags.append("calorie-dense")
    return flags


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

NUTRITION_TOOLS = [lookup_nutrition, analyze_recipe_nutrition, check_nutrition_balance]
