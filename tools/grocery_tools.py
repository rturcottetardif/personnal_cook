"""
Grocery Agent tools.

Five tools with distinct responsibilities:
  1. find_stores      — geographic search (Google Places or stub)
  2. get_store_prices — pricing data per store (Instacart or stub)
  3. get_discounts    — current flyer deals (Flipp/Reebee or stub)
  4. aggregate_grocery_list — pure Python: recipe ingredients → deduplicated list
  5. optimize_basket  — calls the OR-Tools solver; returns a BasketPlan

Rule: Haversine distance is always computed here in code — never by the LLM.
Rule: aggregate_grocery_list does pure arithmetic — never by the LLM.
"""

import json
import math
import os
from typing import Optional

import requests
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential

from models.schemas import (
    GroceryList,
    GroceryListItem,
    Ingredient,
    OptimizationWeights,
    PricedItem,
    Recipe,
    Store,
    Unit,
)
from optimization.basket_optimizer import optimize_basket_plan
from tools.feedback_tools import get_user_profile


# ---------------------------------------------------------------------------
# Input schemas
# ---------------------------------------------------------------------------

class FindStoresInput(BaseModel):
    lat: float = Field(description="User's latitude")
    lng: float = Field(description="User's longitude")
    radius_km: float = Field(10.0, description="Search radius in kilometres")
    require_online_ordering: bool = Field(False, description="If True, only return stores with online ordering")


class GetStorePricesInput(BaseModel):
    store_ids: list[str] = Field(description="List of store IDs from find_stores")
    ingredient_names: list[str] = Field(description="Canonical ingredient names to look up")


class GetDiscountsInput(BaseModel):
    store_ids: list[str] = Field(description="Store IDs to check for current flyer deals")
    ingredient_names: Optional[list[str]] = Field(
        None, description="If provided, filter discounts to only these ingredients"
    )


class AggregateGroceryListInput(BaseModel):
    recipes_json: str = Field(
        description="JSON array of Recipe objects (from get_recipe_details). Quantities will be summed across recipes."
    )


class OptimizeBasketInput(BaseModel):
    grocery_list_json: str = Field(description="JSON GroceryList from aggregate_grocery_list")
    stores_json: str = Field(description="JSON array of Store objects from find_stores")
    priced_items_json: str = Field(description="JSON array of PricedItem objects from get_store_prices")
    weights_json: Optional[str] = Field(
        None, description="JSON OptimizationWeights from the user profile. Uses defaults if omitted."
    )
    user_lat: float = Field(description="User's latitude — for distance calculation")
    user_lng: float = Field(description="User's longitude — for distance calculation")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@tool("find_stores", args_schema=FindStoresInput)
def find_stores(
    lat: float,
    lng: float,
    radius_km: float = 10.0,
    require_online_ordering: bool = False,
) -> str:
    """
    Find grocery stores within a radius of the user's location.
    Returns a list of Store objects with distance_km pre-computed.
    Always call this before get_store_prices or optimize_basket.
    """
    api_key = os.getenv("GOOGLE_PLACES_API_KEY")

    if api_key:
        return _find_stores_google(lat, lng, radius_km, require_online_ordering, api_key)

    # --- Stub: three stores at varying distances ---
    stub_stores = [
        Store(
            store_id="store-A",
            name="Metro Downtown",
            chain="Metro",
            lat=lat + 0.005,
            lng=lng + 0.005,
            address="123 Main St",
            online_ordering=True,
            distance_km=round(_haversine_km(lat, lng, lat + 0.005, lng + 0.005), 2),
        ),
        Store(
            store_id="store-B",
            name="IGA Midtown",
            chain="IGA",
            lat=lat + 0.020,
            lng=lng + 0.010,
            address="456 Oak Ave",
            online_ordering=False,
            distance_km=round(_haversine_km(lat, lng, lat + 0.020, lng + 0.010), 2),
        ),
        Store(
            store_id="store-C",
            name="Maxi Suburb",
            chain="Maxi",
            lat=lat + 0.050,
            lng=lng + 0.030,
            address="789 Elm Blvd",
            online_ordering=True,
            distance_km=round(_haversine_km(lat, lng, lat + 0.050, lng + 0.030), 2),
        ),
    ]

    if require_online_ordering:
        stub_stores = [s for s in stub_stores if s.online_ordering]

    return json.dumps([s.model_dump() for s in stub_stores])


@tool("get_store_prices", args_schema=GetStorePricesInput)
def get_store_prices(store_ids: list[str], ingredient_names: list[str]) -> str:
    """
    Retrieve current prices for a list of ingredients at the given stores.
    Returns a list of PricedItem objects. Items not available at a store are omitted.
    """
    api_key = os.getenv("INSTACART_API_KEY")

    if api_key:
        return _get_prices_instacart(store_ids, ingredient_names, api_key)

    # --- Stub: deterministic prices so tests are repeatable ---
    import hashlib

    priced: list[dict] = []
    base_prices = {
        "store-A": 1.15,   # close, slightly expensive
        "store-B": 1.00,   # mid distance, mid price
        "store-C": 0.82,   # far, cheapest
    }
    for sid in store_ids:
        multiplier = base_prices.get(sid, 1.0)
        for name in ingredient_names:
            # Deterministic price variation by ingredient + store
            seed = int(hashlib.md5(f"{name}{sid}".encode()).hexdigest(), 16) % 100
            price = round(multiplier * (2.0 + seed * 0.05), 2)
            priced.append(
                PricedItem(
                    ingredient_name=name,
                    store_id=sid,
                    price=price,
                    unit=Unit.G,
                    quantity=100.0,
                    in_stock=True,
                ).model_dump()
            )

    return json.dumps(priced)


@tool("get_discounts", args_schema=GetDiscountsInput)
def get_discounts(
    store_ids: list[str],
    ingredient_names: Optional[list[str]] = None,
) -> str:
    """
    Retrieve current flyer discounts at the given stores.
    Returns a list of {store_id, ingredient_name, discount_pct, valid_until}.
    Use this to surface deals to the user or to inform the optimizer.
    """
    api_key = os.getenv("FLIPP_API_KEY")

    if api_key:
        return _get_discounts_flipp(store_ids, ingredient_names, api_key)

    # --- Stub ---
    stub_discounts = []
    for sid in store_ids:
        if ingredient_names:
            for name in ingredient_names[:2]:  # simulate partial coverage
                stub_discounts.append({
                    "store_id": sid,
                    "ingredient_name": name,
                    "discount_pct": 15,
                    "valid_until": "2026-05-14",
                })

    return json.dumps(stub_discounts)


@tool("aggregate_grocery_list", args_schema=AggregateGroceryListInput)
def aggregate_grocery_list(recipes_json: str) -> str:
    """
    Combine ingredients from multiple recipes into a single deduplicated grocery list.
    Quantities for the same (ingredient, unit) pair are summed.
    This is pure arithmetic — deterministic, no LLM involved.
    """
    try:
        raw = json.loads(recipes_json)
        recipes = [Recipe(**r) for r in (raw if isinstance(raw, list) else [raw])]
    except Exception as e:
        return json.dumps({"error": f"Invalid recipes_json: {e}"})

    # Aggregate by (canonical name, unit) — different units cannot be summed safely
    agg: dict[tuple[str, str], GroceryListItem] = {}
    for recipe in recipes:
        for ing in recipe.ingredients:
            key = (ing.name.lower().strip(), ing.unit.value)
            if key not in agg:
                agg[key] = GroceryListItem(
                    ingredient=ing.model_copy(update={"quantity": 0.0}),
                    needed_for_recipes=[],
                )
            agg[key].ingredient.quantity = round(
                agg[key].ingredient.quantity + ing.quantity, 4
            )
            if recipe.id not in agg[key].needed_for_recipes:
                agg[key].needed_for_recipes.append(recipe.id)

    grocery_list = GroceryList(
        items=list(agg.values()),
        source_recipes=[r.id for r in recipes],
    )
    return grocery_list.model_dump_json()


@tool("optimize_basket", args_schema=OptimizeBasketInput)
def optimize_basket(
    grocery_list_json: str,
    stores_json: str,
    priced_items_json: str,
    user_lat: float,
    user_lng: float,
    weights_json: Optional[str] = None,
) -> str:
    """
    Assign grocery items to stores to minimize the weighted combination of
    cost, distance, time, and number of stores visited.
    Uses a provably optimal solver (OR-Tools CP-SAT) — never an LLM.
    Returns a BasketPlan with store assignments, totals, and any missing items.
    """
    try:
        grocery_list = GroceryList.model_validate_json(grocery_list_json)
        stores = [Store(**s) for s in json.loads(stores_json)]
        priced_items = [PricedItem(**p) for p in json.loads(priced_items_json)]
        weights = (
            OptimizationWeights.model_validate_json(weights_json)
            if weights_json
            else OptimizationWeights()
        )
    except Exception as e:
        return json.dumps({"error": f"Invalid input: {e}"})

    plan = optimize_basket_plan(
        grocery_list=grocery_list,
        stores=stores,
        priced_items=priced_items,
        weights=weights,
        user_lat=user_lat,
        user_lng=user_lng,
    )
    return plan.model_dump_json()


# ---------------------------------------------------------------------------
# Real API implementations (activated when env vars are set)
# ---------------------------------------------------------------------------

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _find_stores_google(lat, lng, radius_km, require_online, api_key) -> str:
    try:
        resp = requests.get(
            "https://maps.googleapis.com/maps/api/place/nearbysearch/json",
            params={
                "location": f"{lat},{lng}",
                "radius": int(radius_km * 1000),
                "type": "grocery_or_supermarket",
                "key": api_key,
            },
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()

        stores = []
        for place in raw.get("results", []):
            plat = place["geometry"]["location"]["lat"]
            plng = place["geometry"]["location"]["lng"]
            stores.append(Store(
                store_id=place["place_id"],
                name=place["name"],
                lat=plat,
                lng=plng,
                address=place.get("vicinity"),
                distance_km=round(_haversine_km(lat, lng, plat, plng), 2),
            ).model_dump())

        return json.dumps(stores)
    except requests.exceptions.HTTPError as e:
        return json.dumps({"error": str(e)})


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _get_prices_instacart(store_ids, ingredient_names, api_key) -> str:
    # Instacart Developer Platform — replace with actual endpoint when approved
    return json.dumps({"error": "Instacart API not yet wired — use stubs"})


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _get_discounts_flipp(store_ids, ingredient_names, api_key) -> str:
    return json.dumps({"error": "Flipp API not yet wired — use stubs"})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6371.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

GROCERY_TOOLS = [get_user_profile, find_stores, get_store_prices, get_discounts, aggregate_grocery_list, optimize_basket]
