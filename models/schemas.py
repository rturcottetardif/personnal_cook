"""
Pydantic data contracts — the shared language between every agent in the system.

Field descriptions are intentional: they become part of the LLM-visible tool schema,
guiding the model on what each field expects. Keep them concise and precise.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingredient & Recipe
# ---------------------------------------------------------------------------

class Unit(str, Enum):
    G = "g"
    KG = "kg"
    ML = "ml"
    L = "l"
    TSP = "tsp"
    TBSP = "tbsp"
    CUP = "cup"
    PIECE = "piece"
    PINCH = "pinch"


class Ingredient(BaseModel):
    name: str = Field(description="Canonical lowercase ingredient name, e.g. 'chicken breast'")
    canonical_id: Optional[str] = Field(None, description="USDA FoodData Central FDC ID when known")
    quantity: float = Field(description="Numeric amount")
    unit: Unit = Field(description="Unit of measure")
    optional: bool = Field(False, description="True if the ingredient can be omitted")
    substitutes: list[str] = Field(default_factory=list, description="Acceptable substitutes in order of preference")
    category: Optional[str] = Field(None, description="Grocery category, e.g. 'produce', 'dairy', 'meat'")


class Recipe(BaseModel):
    id: str = Field(description="Unique recipe identifier")
    name: str = Field(description="Human-readable recipe name")
    description: str = Field(description="Short description of the dish")
    cuisine: Optional[str] = Field(None, description="Cuisine style, e.g. 'Italian', 'Thai'")
    servings: int = Field(description="Number of servings the recipe yields")
    prep_time_min: int = Field(description="Preparation time in minutes")
    cook_time_min: int = Field(description="Cooking time in minutes")
    ingredients: list[Ingredient] = Field(description="Full list of ingredients with quantities")
    steps: list[str] = Field(description="Ordered cooking instructions")
    tags: list[str] = Field(default_factory=list, description="Tags such as 'vegetarian', 'gluten-free', 'quick'")
    source_url: Optional[str] = Field(None, description="Original recipe URL if sourced externally")


# ---------------------------------------------------------------------------
# Grocery & Stores
# ---------------------------------------------------------------------------

class Store(BaseModel):
    store_id: str = Field(description="Unique store identifier")
    name: str = Field(description="Store display name")
    chain: Optional[str] = Field(None, description="Retail chain name, e.g. 'Loblaws', 'IGA'")
    lat: float = Field(description="Store latitude")
    lng: float = Field(description="Store longitude")
    address: Optional[str] = Field(None, description="Street address")
    online_ordering: bool = Field(False, description="True if the store supports online ordering")
    distance_km: Optional[float] = Field(None, description="Distance from user's location in kilometres")


class PricedItem(BaseModel):
    ingredient_name: str = Field(description="Canonical ingredient name matching GroceryListItem")
    store_id: str = Field(description="Store where this price was found")
    price: float = Field(description="Price in local currency")
    currency: str = Field("CAD", description="ISO 4217 currency code")
    unit: Unit = Field(description="Unit the price applies to")
    quantity: float = Field(description="Quantity the price applies to, e.g. 500 for 500g")
    on_sale: bool = Field(False, description="True if the item is currently on sale")
    in_stock: bool = Field(True, description="True if the item is confirmed in stock")


class GroceryListItem(BaseModel):
    ingredient: Ingredient = Field(description="The ingredient with aggregated total quantity")
    needed_for_recipes: list[str] = Field(description="Recipe IDs that require this ingredient")


class GroceryList(BaseModel):
    items: list[GroceryListItem] = Field(description="De-duplicated, quantity-summed grocery items")
    source_recipes: list[str] = Field(description="Recipe IDs that generated this list")


# ---------------------------------------------------------------------------
# Basket Optimization
# ---------------------------------------------------------------------------

class OptimizationWeights(BaseModel):
    cost: float = Field(0.5, description="Weight for total basket cost (0–1). Higher = prefer cheaper stores.")
    distance: float = Field(0.3, description="Weight for total travel distance (0–1). Higher = prefer closer stores.")
    time: float = Field(0.1, description="Weight for total estimated shopping time (0–1).")
    health: float = Field(0.0, description="Reserved for future health-score weighting (0–1).")
    num_stores_penalty: float = Field(0.1, description="Penalty per additional store visited (0–1). Higher = consolidate.")


class StoreAssignment(BaseModel):
    store: Store = Field(description="The store to visit")
    items: list[PricedItem] = Field(description="Items to buy at this store")
    subtotal: float = Field(description="Sum of item prices at this store")


class BasketPlan(BaseModel):
    assignments: list[StoreAssignment] = Field(description="One entry per store to visit, in visit order")
    total_cost: float = Field(description="Sum of all subtotals across stores")
    total_distance_km: float = Field(description="Sum of distances to all stores in the plan")
    estimated_time_min: float = Field(description="Estimated total shopping time including travel and in-store time")
    score: float = Field(description="Weighted objective score (lower is better)")
    missing_items: list[str] = Field(default_factory=list, description="Ingredients no store carries — need manual sourcing")


# ---------------------------------------------------------------------------
# Nutrition
# ---------------------------------------------------------------------------

class MacroNutrients(BaseModel):
    protein_g: float = Field(0.0, description="Protein in grams")
    carbs_g: float = Field(0.0, description="Total carbohydrates in grams")
    fat_g: float = Field(0.0, description="Total fat in grams")
    fiber_g: float = Field(0.0, description="Dietary fiber in grams")
    sugar_g: float = Field(0.0, description="Total sugars in grams")
    saturated_fat_g: float = Field(0.0, description="Saturated fat in grams")


class MicroNutrients(BaseModel):
    sodium_mg: float = Field(0.0, description="Sodium in milligrams")
    iron_mg: float = Field(0.0, description="Iron in milligrams")
    calcium_mg: float = Field(0.0, description="Calcium in milligrams")
    vitamin_c_mg: float = Field(0.0, description="Vitamin C in milligrams")
    vitamin_d_ug: float = Field(0.0, description="Vitamin D in micrograms")
    potassium_mg: float = Field(0.0, description="Potassium in milligrams")


class NutritionFacts(BaseModel):
    calories: float = Field(description="Total kilocalories")
    macros: MacroNutrients = Field(default_factory=MacroNutrients)
    micros: MicroNutrients = Field(default_factory=MicroNutrients)
    serving_size_g: Optional[float] = Field(None, description="Reference serving size in grams")
    source: str = Field("USDA", description="Data source identifier")


class NutritionAnalysis(BaseModel):
    per_serving: NutritionFacts = Field(description="Nutrition facts scaled to one serving")
    daily_value_pct: dict[str, float] = Field(
        default_factory=dict,
        description="Percentage of recommended daily value per nutrient key"
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Notable observations, e.g. 'high sodium', 'good source of fiber'"
    )
    disclaimer: str = Field(
        "These are general guidelines. Consult a registered dietitian for personalized advice.",
        description="Mandatory health disclaimer — always included in user-facing output"
    )


# ---------------------------------------------------------------------------
# User Profile & Feedback
# ---------------------------------------------------------------------------

class DietaryRestriction(str, Enum):
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    GLUTEN_FREE = "gluten_free"
    DAIRY_FREE = "dairy_free"
    HALAL = "halal"
    KOSHER = "kosher"
    LOW_SODIUM = "low_sodium"
    LOW_CARB = "low_carb"
    KETO = "keto"


class UserProfile(BaseModel):
    user_id: str = Field(description="Unique user identifier")
    location_lat: Optional[float] = Field(None, description="User's latitude for store search")
    location_lng: Optional[float] = Field(None, description="User's longitude for store search")
    max_radius_km: float = Field(10.0, description="Maximum distance to consider for store search")
    budget_per_week: Optional[float] = Field(None, description="Weekly grocery budget in local currency")
    household_size: int = Field(2, description="Number of people to cook for — used to scale recipes")
    allergies: list[str] = Field(
        default_factory=list,
        description="HARD constraints — ingredients that must never appear. Append-only, never reduced."
    )
    dietary_restrictions: list[DietaryRestriction] = Field(
        default_factory=list,
        description="Dietary rules that filter recipe suggestions"
    )
    preferred_cuisines: list[str] = Field(default_factory=list, description="Cuisines the user enjoys")
    disliked_ingredients: list[str] = Field(default_factory=list, description="Soft dislikes — avoid but not forbidden")
    optimization_weights: OptimizationWeights = Field(
        default_factory=OptimizationWeights,
        description="User's grocery shopping priorities"
    )


class FeedbackType(str, Enum):
    HARD_CONSTRAINT = "hard_constraint"
    SOFT_PREFERENCE = "soft_preference"
    ONE_OFF = "one_off"
    POSITIVE = "positive"


class ParsedFeedback(BaseModel):
    feedback_type: FeedbackType = Field(
        description=(
            "hard_constraint: permanent rule (allergies, dietary laws). "
            "soft_preference: persistent like/dislike. "
            "one_off: applies only to this session. "
            "positive: reinforcement — more like this."
        )
    )
    target: str = Field(description="What the feedback is about — ingredient name, cuisine, recipe name, etc.")
    sentiment: float = Field(description="Sentiment score from -1.0 (strong dislike) to +1.0 (strong like)")
    raw_text: str = Field(description="Original user message verbatim")
    suggested_update: Optional[dict] = Field(
        None,
        description="Proposed UserProfile field updates as a dict, e.g. {'allergies': ['shellfish']}"
    )
