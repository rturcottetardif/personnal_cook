"""
Recipe Agent tools.

All tools return JSON strings — LangChain's tool contract requires strings.
All tools have Pydantic input schemas so the LLM sees field descriptions
as part of the function signature, reducing mis-shaped calls.

Stubs return realistic synthetic data. Replace each stub with the real
Spoonacular API call when you reach Step 11 (Wire Real APIs).
"""

import json
import os
from typing import Optional

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from models.schemas import Ingredient, Recipe, Unit
from tools.feedback_tools import get_user_profile


# ---------------------------------------------------------------------------
# Input schemas — LLM reads Field descriptions to understand each parameter
# ---------------------------------------------------------------------------

class SearchRecipesInput(BaseModel):
    query: str = Field(description="Natural language recipe search query, e.g. 'quick chicken dinner'")
    cuisine: Optional[str] = Field(None, description="Filter by cuisine style, e.g. 'Italian', 'Thai'")
    max_prep_time_min: Optional[int] = Field(None, description="Maximum total preparation + cook time in minutes")
    exclude_ingredients: Optional[list[str]] = Field(None, description="Ingredients to exclude — pass user allergies here")
    diet: Optional[str] = Field(None, description="Diet filter: 'vegetarian', 'vegan', 'gluten free', 'ketogenic'")
    max_results: int = Field(5, description="Maximum number of recipes to return")


class GetRecipeDetailsInput(BaseModel):
    recipe_id: str = Field(description="Recipe ID from a prior search_recipes call")


class ParseIngredientsInput(BaseModel):
    ingredient_lines: list[str] = Field(
        description="Raw ingredient text lines as they appear in a recipe, e.g. ['2 cups flour', '1 tsp salt']"
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool("search_recipes", args_schema=SearchRecipesInput)
def search_recipes(
    query: str,
    cuisine: Optional[str] = None,
    max_prep_time_min: Optional[int] = None,
    exclude_ingredients: Optional[list[str]] = None,
    diet: Optional[str] = None,
    max_results: int = 5,
) -> str:
    """
    Search for recipes matching a query. Returns a list of recipe summaries with IDs.
    Always call this before get_recipe_details — you need the ID first.
    Pass user allergies as exclude_ingredients to enforce dietary safety at the source.
    """
    api_key = os.getenv("SPOONACULAR_API_KEY")

    if api_key:
        return _search_spoonacular(query, cuisine, max_prep_time_min, exclude_ingredients, diet, max_results, api_key)

    # --- Stub (no API key) ---
    stub_results = [
        {
            "id": f"stub-{i+1}",
            "name": f"{query.title()} Recipe {i+1}",
            "description": f"A delicious {query} dish",
            "cuisine": cuisine or "International",
            "prep_time_min": 15,
            "cook_time_min": 25,
            "servings": 4,
            "tags": [diet] if diet else [],
        }
        for i in range(min(max_results, 3))
    ]
    return json.dumps({"results": stub_results})


@tool("get_recipe_details", args_schema=GetRecipeDetailsInput)
def get_recipe_details(recipe_id: str) -> str:
    """
    Fetch full recipe details including ingredients and step-by-step instructions.
    Use this after search_recipes once you have chosen a recipe ID.
    """
    api_key = os.getenv("SPOONACULAR_API_KEY")

    if api_key:
        return _get_spoonacular_details(recipe_id, api_key)

    # --- Stub ---
    recipe = Recipe(
        id=recipe_id,
        name="Stub Recipe",
        description="A stub recipe for development",
        cuisine="International",
        servings=4,
        prep_time_min=15,
        cook_time_min=25,
        ingredients=[
            Ingredient(name="chicken breast", quantity=500, unit=Unit.G),
            Ingredient(name="olive oil", quantity=2, unit=Unit.TBSP),
            Ingredient(name="garlic", quantity=3, unit=Unit.PIECE),
            Ingredient(name="salt", quantity=1, unit=Unit.TSP),
        ],
        steps=[
            "Season the chicken with salt.",
            "Heat olive oil in a pan over medium heat.",
            "Cook the chicken for 6 minutes per side until golden.",
            "Add minced garlic and cook for 1 more minute.",
        ],
        tags=["quick", "protein-rich"],
    )
    return recipe.model_dump_json()


@tool("parse_ingredients", args_schema=ParseIngredientsInput)
def parse_ingredients(ingredient_lines: list[str]) -> str:
    """
    Convert free-text ingredient lines into structured Ingredient objects.
    Use this when you receive recipe text that hasn't been parsed yet.
    This uses a deterministic NLP library — NOT the LLM — for structured output reliability.
    """
    # Production: use ingredient-parser-nlp library
    # pip install ingredient-parser-nlp
    # from ingredient_parser import parse_ingredient
    #
    # parsed = []
    # for line in ingredient_lines:
    #     result = parse_ingredient(line)
    #     parsed.append(Ingredient(
    #         name=result.name.text,
    #         quantity=float(result.amount[0].quantity) if result.amount else 1.0,
    #         unit=_map_unit(result.amount[0].unit if result.amount else "piece"),
    #     ).model_dump())
    # return json.dumps(parsed)

    # --- Stub: naive splitting for development ---
    parsed = []
    unit_words = {u.value for u in Unit}
    for line in ingredient_lines:
        tokens = line.lower().strip().split()
        quantity = 1.0
        unit = Unit.PIECE
        name_start = 0

        if tokens and tokens[0].replace(".", "", 1).replace("/", "", 1).isdigit():
            try:
                quantity = float(tokens[0].replace("/", ""))
            except ValueError:
                pass
            name_start = 1

        if len(tokens) > 1 and tokens[name_start] in unit_words:
            unit = Unit(tokens[name_start])
            name_start += 1

        name = " ".join(tokens[name_start:]) if name_start < len(tokens) else line
        parsed.append(
            Ingredient(name=name or line, quantity=quantity, unit=unit).model_dump()
        )

    return json.dumps(parsed)


# ---------------------------------------------------------------------------
# Real API implementations (activated when SPOONACULAR_API_KEY is set)
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _search_spoonacular(query, cuisine, max_prep_time_min, exclude_ingredients, diet, max_results, api_key) -> str:
    params = {
        "query": query,
        "number": max_results,
        "apiKey": api_key,
    }
    if cuisine:
        params["cuisine"] = cuisine
    if max_prep_time_min:
        params["maxReadyTime"] = max_prep_time_min
    if exclude_ingredients:
        params["excludeIngredients"] = ",".join(exclude_ingredients)
    if diet:
        params["diet"] = diet

    try:
        resp = requests.get(
            "https://api.spoonacular.com/recipes/complexSearch",
            params=params,
            timeout=10,
        )
        resp.raise_for_status()
        return json.dumps(resp.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return json.dumps({"error": "rate_limited", "retry_after": 60})
        return json.dumps({"error": str(e)})


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _get_spoonacular_details(recipe_id: str, api_key: str) -> str:
    try:
        resp = requests.get(
            f"https://api.spoonacular.com/recipes/{recipe_id}/information",
            params={"apiKey": api_key, "includeNutrition": False},
            timeout=10,
        )
        resp.raise_for_status()
        return json.dumps(resp.json())
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            return json.dumps({"error": "rate_limited", "retry_after": 60})
        return json.dumps({"error": str(e)})


# ---------------------------------------------------------------------------
# Tool registry — imported by agents/specialists.py
# ---------------------------------------------------------------------------

RECIPE_TOOLS = [get_user_profile, search_recipes, get_recipe_details, parse_ingredients]
