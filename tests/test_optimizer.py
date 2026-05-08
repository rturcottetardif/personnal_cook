"""
Smoke tests for the basket optimizer.

These tests verify the core behavioral guarantee: different optimization weights
produce meaningfully different store assignments. If all three weight scenarios
pick the same store, the weights are not reaching the objective function.

Run:
    python -m pytest tests/test_optimizer.py -v
or:
    python -m tests.test_optimizer
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from models.schemas import (
    GroceryList,
    GroceryListItem,
    Ingredient,
    OptimizationWeights,
    PricedItem,
    Store,
    Unit,
)
from optimization.basket_optimizer import optimize_basket_plan


# ---------------------------------------------------------------------------
# Shared test fixtures
# ---------------------------------------------------------------------------

USER_LAT, USER_LNG = 45.5017, -73.5673  # Montreal downtown

STORES = [
    Store(
        store_id="store-A",
        name="Close Expensive",
        lat=45.5050, lng=-73.5650,   # ~0.4 km
        distance_km=0.4,
        online_ordering=True,
    ),
    Store(
        store_id="store-B",
        name="Mid Balanced",
        lat=45.5200, lng=-73.5500,   # ~2.2 km
        distance_km=2.2,
        online_ordering=False,
    ),
    Store(
        store_id="store-C",
        name="Far Cheap",
        lat=45.5500, lng=-73.5000,   # ~6.0 km
        distance_km=6.0,
        online_ordering=True,
    ),
]

ITEMS = ["chicken breast", "rice", "olive oil", "broccoli"]

GROCERY_LIST = GroceryList(
    items=[
        GroceryListItem(
            ingredient=Ingredient(name=name, quantity=500, unit=Unit.G),
            needed_for_recipes=["test-recipe-1"],
        )
        for name in ITEMS
    ],
    source_recipes=["test-recipe-1"],
)

# Prices reflect store character: store-A expensive, store-B mid, store-C cheap
PRICED_ITEMS = [
    PricedItem(ingredient_name=item, store_id="store-A", price=8.50, unit=Unit.G, quantity=500, in_stock=True)
    for item in ITEMS
] + [
    PricedItem(ingredient_name=item, store_id="store-B", price=6.50, unit=Unit.G, quantity=500, in_stock=True)
    for item in ITEMS
] + [
    PricedItem(ingredient_name=item, store_id="store-C", price=4.50, unit=Unit.G, quantity=500, in_stock=True)
    for item in ITEMS
]


def _run(weights: OptimizationWeights):
    return optimize_basket_plan(
        grocery_list=GROCERY_LIST,
        stores=STORES,
        priced_items=PRICED_ITEMS,
        weights=weights,
        user_lat=USER_LAT,
        user_lng=USER_LNG,
    )


def _store_ids(plan) -> list[str]:
    return [a.store.store_id for a in plan.assignments]


# ---------------------------------------------------------------------------
# Weight-sensitivity tests
# ---------------------------------------------------------------------------

def test_cost_focused_prefers_far_cheap_store():
    """High cost weight → optimizer chooses the cheapest store (store-C) even if far."""
    plan = _run(OptimizationWeights(cost=0.9, distance=0.05, time=0.0, num_stores_penalty=0.05))
    assert plan.assignments, "Plan must contain at least one assignment"
    assert "store-C" in _store_ids(plan), (
        f"Expected store-C (cheapest) when cost weight is 0.9, got {_store_ids(plan)}"
    )


def test_distance_focused_prefers_close_expensive_store():
    """High distance weight → optimizer chooses the closest store (store-A) even if pricey."""
    plan = _run(OptimizationWeights(cost=0.05, distance=0.9, time=0.0, num_stores_penalty=0.05))
    assert plan.assignments, "Plan must contain at least one assignment"
    assert "store-A" in _store_ids(plan), (
        f"Expected store-A (closest) when distance weight is 0.9, got {_store_ids(plan)}"
    )


def test_store_penalty_consolidates_to_one_store():
    """High store-count penalty → all items bought at one store (fewest stops)."""
    plan = _run(OptimizationWeights(cost=0.1, distance=0.1, time=0.0, num_stores_penalty=0.8))
    assert len(plan.assignments) == 1, (
        f"Expected exactly 1 store when num_stores_penalty is 0.8, got {len(plan.assignments)}"
    )


def test_different_weights_produce_different_plans():
    """The three weight scenarios must not all pick the same store — proves weights enter the objective."""
    cost_plan = _run(OptimizationWeights(cost=0.9, distance=0.05, time=0.0, num_stores_penalty=0.05))
    dist_plan = _run(OptimizationWeights(cost=0.05, distance=0.9, time=0.0, num_stores_penalty=0.05))
    cost_stores = set(_store_ids(cost_plan))
    dist_stores = set(_store_ids(dist_plan))
    assert cost_stores != dist_stores, (
        f"Cost-focused and distance-focused plans chose identical stores ({cost_stores}). "
        "Weights are likely not entering the objective function."
    )


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------

def test_empty_grocery_list_returns_empty_plan():
    empty_list = GroceryList(items=[], source_recipes=[])
    plan = _run_with_list(empty_list, OptimizationWeights())
    assert plan.total_cost == 0.0
    assert plan.assignments == []


def test_no_stores_returns_all_items_missing():
    plan = optimize_basket_plan(
        grocery_list=GROCERY_LIST,
        stores=[],
        priced_items=PRICED_ITEMS,
        weights=OptimizationWeights(),
        user_lat=USER_LAT,
        user_lng=USER_LNG,
    )
    assert set(plan.missing_items) == set(ITEMS)


def test_out_of_stock_item_goes_to_missing():
    # Make chicken breast out of stock everywhere
    modified_prices = [
        p.model_copy(update={"in_stock": False}) if p.ingredient_name == "chicken breast" else p
        for p in PRICED_ITEMS
    ]
    plan = optimize_basket_plan(
        grocery_list=GROCERY_LIST,
        stores=STORES,
        priced_items=modified_prices,
        weights=OptimizationWeights(),
        user_lat=USER_LAT,
        user_lng=USER_LNG,
    )
    assert "chicken breast" in plan.missing_items


def test_total_cost_matches_sum_of_assignments():
    plan = _run(OptimizationWeights())
    expected = round(sum(a.subtotal for a in plan.assignments), 2)
    assert abs(plan.total_cost - expected) < 0.01, (
        f"total_cost {plan.total_cost} does not match sum of subtotals {expected}"
    )


# ---------------------------------------------------------------------------
# Aggregation tests (grocery list building)
# ---------------------------------------------------------------------------

def test_aggregation_sums_same_ingredient_same_unit():
    from models.schemas import Recipe
    from tools.grocery_tools import aggregate_grocery_list
    import json

    recipes = [
        Recipe(
            id="r1", name="Recipe 1", description="", servings=2,
            prep_time_min=10, cook_time_min=20,
            ingredients=[Ingredient(name="chicken breast", quantity=300, unit=Unit.G)],
            steps=[],
        ),
        Recipe(
            id="r2", name="Recipe 2", description="", servings=2,
            prep_time_min=10, cook_time_min=20,
            ingredients=[Ingredient(name="chicken breast", quantity=200, unit=Unit.G)],
            steps=[],
        ),
    ]

    result_json = aggregate_grocery_list.invoke(
        {"recipes_json": json.dumps([r.model_dump() for r in recipes])}
    )
    result = json.loads(result_json)
    items = result["items"]

    chicken = next((i for i in items if i["ingredient"]["name"] == "chicken breast"), None)
    assert chicken is not None, "chicken breast should be in aggregated list"
    assert chicken["ingredient"]["quantity"] == 500.0, (
        f"Expected 500g chicken breast (300+200), got {chicken['ingredient']['quantity']}"
    )
    assert set(chicken["needed_for_recipes"]) == {"r1", "r2"}


def test_aggregation_keeps_different_units_separate():
    from models.schemas import Recipe
    from tools.grocery_tools import aggregate_grocery_list
    import json

    recipes = [
        Recipe(
            id="r1", name="R1", description="", servings=2,
            prep_time_min=5, cook_time_min=10,
            ingredients=[Ingredient(name="milk", quantity=200, unit=Unit.ML)],
            steps=[],
        ),
        Recipe(
            id="r2", name="R2", description="", servings=2,
            prep_time_min=5, cook_time_min=10,
            ingredients=[Ingredient(name="milk", quantity=1, unit=Unit.CUP)],
            steps=[],
        ),
    ]

    result_json = aggregate_grocery_list.invoke(
        {"recipes_json": json.dumps([r.model_dump() for r in recipes])}
    )
    result = json.loads(result_json)
    milk_items = [i for i in result["items"] if i["ingredient"]["name"] == "milk"]
    assert len(milk_items) == 2, (
        "milk in ml and milk in cup are different units — must not be summed"
    )


# ---------------------------------------------------------------------------
# Allergy safety tests
# ---------------------------------------------------------------------------

def test_allergies_are_append_only():
    from tools.feedback_tools import update_user_profile, get_user_profile
    import json

    # Add allergy
    update_user_profile.invoke({"user_id": "test-user-safety", "update_json": json.dumps({"allergies": ["shellfish"]})})

    # Attempt to "remove" by sending an empty allergies list
    update_user_profile.invoke({"user_id": "test-user-safety", "update_json": json.dumps({"allergies": []})})

    profile = json.loads(get_user_profile.invoke({"user_id": "test-user-safety"}))
    assert "shellfish" in profile["allergies"], (
        "Allergy 'shellfish' was removed — the append-only guarantee is broken"
    )


def test_allergy_union_adds_new_without_removing_old():
    from tools.feedback_tools import update_user_profile, get_user_profile
    import json

    update_user_profile.invoke({"user_id": "test-user-union", "update_json": json.dumps({"allergies": ["peanuts"]})})
    update_user_profile.invoke({"user_id": "test-user-union", "update_json": json.dumps({"allergies": ["tree nuts"]})})

    profile = json.loads(get_user_profile.invoke({"user_id": "test-user-union"}))
    assert "peanuts" in profile["allergies"]
    assert "tree nuts" in profile["allergies"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_with_list(grocery_list, weights):
    return optimize_basket_plan(
        grocery_list=grocery_list,
        stores=STORES,
        priced_items=PRICED_ITEMS,
        weights=weights,
        user_lat=USER_LAT,
        user_lng=USER_LNG,
    )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        test_cost_focused_prefers_far_cheap_store,
        test_distance_focused_prefers_close_expensive_store,
        test_store_penalty_consolidates_to_one_store,
        test_different_weights_produce_different_plans,
        test_empty_grocery_list_returns_empty_plan,
        test_no_stores_returns_all_items_missing,
        test_out_of_stock_item_goes_to_missing,
        test_total_cost_matches_sum_of_assignments,
        test_aggregation_sums_same_ingredient_same_unit,
        test_aggregation_keeps_different_units_separate,
        test_allergies_are_append_only,
        test_allergy_union_adds_new_without_removing_old,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            print(f"  PASS  {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {type(e).__name__}: {e}")
            failed += 1

    print(f"\n{passed} passed, {failed} failed")
    sys.exit(0 if failed == 0 else 1)
