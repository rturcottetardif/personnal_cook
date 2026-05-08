"""
Basket optimizer — deterministic grocery assignment using OR-Tools CP-SAT.

Why NOT an LLM:
  - LLMs produce answers that are 15-30% suboptimal on bin-packing problems
  - LLM output is non-deterministic: same inputs yield different plans across runs
  - CP-SAT gives provably optimal answers in milliseconds with integer guarantees

The objective:
  minimize  w_cost     * normalize(total_cost)
          + w_distance * normalize(total_distance)
          + w_time     * normalize(total_time)
          + w_stores   * num_stores_visited

Decision variables:
  x[i, s]  binary — is item i bought at store s?
  y[s]     binary — is store s visited at all?

Constraints:
  1. Each item is bought at exactly one store
  2. An item can only be bought where it is in stock
  3. A store is marked visited if anything is bought there
"""

import math
from typing import Optional

from models.schemas import (
    BasketPlan,
    GroceryList,
    OptimizationWeights,
    PricedItem,
    Store,
    StoreAssignment,
)
from config.settings import AVG_DRIVING_SPEED_KMH, TIME_PER_STORE_VISIT_MIN, OPTIMIZER_SCALE, TIME_VALUE_PER_HOUR


def optimize_basket_plan(
    grocery_list: GroceryList,
    stores: list[Store],
    priced_items: list[PricedItem],
    weights: OptimizationWeights,
    user_lat: float,
    user_lng: float,
) -> BasketPlan:
    """
    Find the optimal store assignment for all grocery items.

    Returns a BasketPlan. Items not available at any store are listed in
    missing_items — the calling agent should surface these to the user.
    """
    from ortools.sat.python import cp_model

    # Guard: nothing to optimize
    if not grocery_list.items or not stores:
        return BasketPlan(
            assignments=[],
            total_cost=0.0,
            total_distance_km=0.0,
            estimated_time_min=0.0,
            score=0.0,
            missing_items=[item.ingredient.name for item in grocery_list.items],
        )

    item_names = [g.ingredient.name for g in grocery_list.items]
    store_ids = [s.store_id for s in stores]

    # Build lookup: (item_name, store_id) -> PricedItem
    price_map: dict[tuple[str, str], PricedItem] = {}
    for p in priced_items:
        if p.in_stock:
            price_map[(p.ingredient_name, p.store_id)] = p

    # Identify items not available anywhere
    missing_items = [
        name for name in item_names
        if not any((name, sid) in price_map for sid in store_ids)
    ]
    available_items = [name for name in item_names if name not in missing_items]

    if not available_items:
        return BasketPlan(
            assignments=[],
            total_cost=0.0,
            total_distance_km=0.0,
            estimated_time_min=0.0,
            score=0.0,
            missing_items=missing_items,
        )

    # Precompute store distances (deterministic Haversine — never delegate to LLM)
    store_distances: dict[str, float] = {
        s.store_id: (s.distance_km if s.distance_km is not None else _haversine_km(user_lat, user_lng, s.lat, s.lng))
        for s in stores
    }

    # Normalization bounds for the objective
    all_prices = [p.price for p in priced_items if p.in_stock]
    max_price = max(all_prices) if all_prices else 1.0
    max_distance = max(store_distances.values()) if store_distances else 1.0
    max_time = (
        max_distance / max(AVG_DRIVING_SPEED_KMH, 1.0) * 60.0
        + len(stores) * TIME_PER_STORE_VISIT_MIN
    )

    # --- Build CP-SAT model ---
    model = cp_model.CpModel()

    # x[i, s] = 1 iff item i is bought at store s
    x: dict[tuple[int, int], cp_model.IntVar] = {}
    for i, item in enumerate(available_items):
        for j, sid in enumerate(store_ids):
            if (item, sid) in price_map:
                x[(i, j)] = model.NewBoolVar(f"x_{i}_{j}")

    # y[s] = 1 iff store s is visited
    y: dict[int, cp_model.IntVar] = {j: model.NewBoolVar(f"y_{j}") for j in range(len(store_ids))}

    # Constraint 1: each available item bought at exactly one store
    for i, item in enumerate(available_items):
        model.Add(
            sum(x[(i, j)] for j in range(len(store_ids)) if (i, j) in x) == 1
        )

    # Constraint 2: store visited iff anything bought there
    for j in range(len(store_ids)):
        for i in range(len(available_items)):
            if (i, j) in x:
                model.Add(y[j] >= x[(i, j)])

    # Build objective: weighted sum of cost + distance + time + store penalty
    # CP-SAT requires integer coefficients; multiply floats by SCALE then cast.
    #
    # Distance and time are per store visit (y[j]), not per item (x[i,j]).
    # Charging distance once per item would count it N times for an N-item basket,
    # making distant cheap stores appear exponentially worse than nearby expensive ones.
    S = OPTIMIZER_SCALE * 1000
    obj_terms = []

    # Per-item cost only
    for i, item in enumerate(available_items):
        for j, sid in enumerate(store_ids):
            if (i, j) not in x:
                continue
            priced = price_map[(item, sid)]
            norm_cost = priced.price / max(max_price, 1e-9)
            coef = int(weights.cost * norm_cost * S)
            obj_terms.append(coef * x[(i, j)])

    # Per-store-visit: dollar-valued time overhead + distance preference + store-count penalty.
    # visit_cost_dollars = (drive hours + in-store hours) × TIME_VALUE_PER_HOUR.
    # Normalized against max_price so it's on the same scale as norm_cost — the solver
    # can then directly compare "extra trip saves $X on salt" vs. "trip costs $Y of your time".
    for j, sid in enumerate(store_ids):
        norm_dist = store_distances[sid] / max(max_distance, 1e-9)
        drive_time = store_distances[sid] / max(AVG_DRIVING_SPEED_KMH, 1.0) * 60.0
        norm_time = drive_time / max(max_time, 1e-9)
        visit_hours = (
            store_distances[sid] / max(AVG_DRIVING_SPEED_KMH, 1.0)
            + TIME_PER_STORE_VISIT_MIN / 60.0
        )
        norm_visit_dollar_cost = (visit_hours * TIME_VALUE_PER_HOUR) / max(max_price, 1e-9)
        coef = int(
            (weights.cost * norm_visit_dollar_cost + weights.distance * norm_dist
             + weights.time * norm_time + weights.num_stores_penalty)
            * S
        )
        obj_terms.append(coef * y[j])

    model.Minimize(sum(obj_terms))

    # --- Solve ---
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # No solution found — return all items as missing
        return BasketPlan(
            assignments=[],
            total_cost=0.0,
            total_distance_km=0.0,
            estimated_time_min=0.0,
            score=float("inf"),
            missing_items=item_names,
        )

    # --- Extract solution ---
    assignments_map: dict[str, list[PricedItem]] = {}
    for i, item in enumerate(available_items):
        for j, sid in enumerate(store_ids):
            if (i, j) in x and solver.Value(x[(i, j)]) == 1:
                assignments_map.setdefault(sid, []).append(price_map[(item, sid)])

    visited_stores = {s.store_id: s for s in stores}
    assignments = [
        StoreAssignment(
            store=visited_stores[sid],
            items=items,
            subtotal=round(sum(p.price for p in items), 2),
        )
        for sid, items in assignments_map.items()
    ]

    total_cost = round(sum(a.subtotal for a in assignments), 2)
    total_distance = round(sum(store_distances[a.store.store_id] for a in assignments), 2)
    estimated_time = round(
        total_distance / max(AVG_DRIVING_SPEED_KMH, 1.0) * 60.0
        + len(assignments) * TIME_PER_STORE_VISIT_MIN,
        1,
    )
    score = round(solver.ObjectiveValue() / S, 4)

    return BasketPlan(
        assignments=assignments,
        total_cost=total_cost,
        total_distance_km=total_distance,
        estimated_time_min=estimated_time,
        score=score,
        missing_items=missing_items,
    )


def _haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in kilometres — always computed in code, never by an LLM."""
    R = 6371.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng / 2) ** 2
    return 2.0 * R * math.asin(math.sqrt(a))
