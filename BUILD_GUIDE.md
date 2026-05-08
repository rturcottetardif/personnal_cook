# Recipe & Grocery Multi-Agent System — Full Build Guide

A complete step-by-step guide to building a production-ready multi-agent system for recipe planning, grocery optimization, nutrition analysis, and user feedback handling.

**Stack:** Python 3.11+, LangChain, LangGraph, Anthropic Claude, Google OR-Tools, Pydantic

**Estimated time:** 2-3 days for the scaffold and stubs, 1-2 weeks to wire all real APIs and harden for production.

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Prerequisites & Environment Setup](#2-prerequisites--environment-setup)
3. [Project Structure](#3-project-structure)
4. [Step 1 — Define Data Contracts](#step-1--define-data-contracts)
5. [Step 2 — Configuration & LLM Factory](#step-2--configuration--llm-factory)
6. [Step 3 — Build the Optimizer (Deterministic Core)](#step-3--build-the-optimizer-deterministic-core)
7. [Step 4 — Recipe Agent Tools](#step-4--recipe-agent-tools)
8. [Step 5 — Grocery Agent Tools](#step-5--grocery-agent-tools)
9. [Step 6 — Nutrition Agent Tools](#step-6--nutrition-agent-tools)
10. [Step 7 — Feedback Agent Tools](#step-7--feedback-agent-tools)
11. [Step 8 — Build the Specialist Agents](#step-8--build-the-specialist-agents)
12. [Step 9 — Build the Supervisor Graph](#step-9--build-the-supervisor-graph)
13. [Step 10 — Smoke Tests](#step-10--smoke-tests)
14. [Step 11 — Wire Real APIs](#step-11--wire-real-apis)
15. [Step 12 — Production Hardening](#step-12--production-hardening)
16. [Appendix — Decisions & Tradeoffs](#appendix--decisions--tradeoffs)

---

## 1. Architecture Overview

### The Pattern: Supervisor + Specialists

```
                    ┌─────────────────────┐
                    │   User Interface    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Supervisor Agent  │  ← Haiku (cheap routing)
                    │   (Orchestrator)    │
                    └──────────┬──────────┘
                               │
        ┌──────────┬───────────┼───────────┬──────────┐
        ▼          ▼           ▼           ▼          ▼
   ┌─────────┐ ┌────────┐ ┌─────────┐ ┌──────────┐ ┌────────┐
   │ Recipe  │ │Grocery │ │Nutrition│ │ Feedback │ │ Memory │
   │  Agent  │ │ Agent  │ │  Agent  │ │  Agent   │ │  Layer │
   │ (Sonnet)│ │(Sonnet)│ │ (Sonnet)│ │ (Haiku)  │ │ (DB)   │
   └─────────┘ └────────┘ └─────────┘ └──────────┘ └────────┘
        │          │           │           │
        ▼          ▼           ▼           ▼
    [Tools]    [Tools]     [Tools]     [Tools]
                  │
                  ▼
            [Optimizer]
            (OR-Tools, no LLM)
```

### Determinism Map

| Component | Type | Why |
|-----------|------|-----|
| Supervisor routing | LLM (Haiku, temp 0) | Decision-making with language understanding |
| Recipe ideation | LLM (Sonnet, temp 0.7) | Creativity is the point |
| Ingredient parsing | Library | Structured output needed |
| Grocery list aggregation | Pure Python | Math: sum quantities |
| Store search | API + Haversine | Geographic facts |
| Basket optimization | OR-Tools CP-SAT | Math: don't trust LLMs with arithmetic |
| Nutrition lookup | API (USDA) | Authoritative source |
| Nutrition math | Pure Python | Sums and ratios |
| Nutrition interpretation | LLM (Sonnet, temp 0.2) | Translates numbers → advice |
| Feedback classification | LLM (Haiku, temp 0) | Intent recognition |
| Profile updates | Pure Python | Set operations |
| Allergy enforcement | Pure Python | NEVER trust LLM with safety |

**The rule:** LLMs translate, route, and explain. Code does math, lookups, and constraints.

---

## 2. Prerequisites & Environment Setup

### System Requirements

- Python 3.11 or higher
- 4GB RAM minimum
- Internet connection for API calls

### API Keys Needed

You can build the entire system with stubs first, then add keys as you wire each integration:

| Service | Purpose | Free tier? |
|---------|---------|------------|
| Anthropic | LLM calls | Pay-as-you-go |
| Google Places | Store search | $200/mo free credit |
| Spoonacular | Recipe search | 150 calls/day free |
| USDA FoodData Central | Nutrition data | Free, generous |
| Flipp / Reebee | Canadian flyers | Contact for access |
| Instacart Developer | Online ordering | Application required |

### Environment Setup

```bash
# Create project directory
mkdir recipe_agent_system
cd recipe_agent_system

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Set environment variables (add to .env or shell profile)
export ANTHROPIC_API_KEY="sk-ant-..."
export GOOGLE_PLACES_API_KEY="..."
export SPOONACULAR_API_KEY="..."
export USDA_API_KEY="..."
```

### Dependencies File

Create `requirements.txt`:

```text
# Core agent framework
langchain>=0.3.0
langchain-anthropic>=0.3.0
langchain-core>=0.3.0
langgraph>=0.2.50

# Data validation
pydantic>=2.0

# Optimization (the deterministic engine)
ortools>=9.10

# HTTP for real API calls
requests>=2.31

# Optional: for production
# redis>=5.0           # caching
# psycopg2-binary>=2.9 # Postgres for user profiles
# pgvector>=0.2        # semantic memory
# pint>=0.23           # robust unit conversion
# ingredient-parser-nlp # ingredient line parsing
```

Install:

```bash
pip install -r requirements.txt
```

---

## 3. Project Structure

Create this directory layout:

```
recipe_agent_system/
├── __init__.py
├── README.md
├── requirements.txt
├── .env                       # API keys (NEVER commit)
├── models/
│   ├── __init__.py
│   └── schemas.py             # Pydantic data contracts
├── config/
│   ├── __init__.py
│   └── settings.py            # LLM factory, model tiers
├── tools/
│   ├── __init__.py
│   ├── recipe_tools.py
│   ├── grocery_tools.py
│   ├── nutrition_tools.py
│   └── feedback_tools.py
├── optimization/
│   ├── __init__.py
│   └── basket_optimizer.py    # OR-Tools CP-SAT
├── agents/
│   ├── __init__.py
│   ├── specialists.py         # The four agents
│   └── graph.py               # Supervisor + LangGraph
└── tests/
    ├── __init__.py
    └── test_optimizer.py
```

Run:

```bash
mkdir -p models config tools optimization agents tests
touch __init__.py models/__init__.py config/__init__.py tools/__init__.py
touch optimization/__init__.py agents/__init__.py tests/__init__.py
```

---

## Step 1 — Define Data Contracts

**Why first:** The schemas are the spine of the system. Recipe Agent's output flows into Grocery Agent's input via these types. Get this right, everything else follows.

**File:** `models/schemas.py`

### Key types to define:

**Ingredients & Recipes:**
- `Unit` enum (g, kg, ml, l, tsp, tbsp, cup, piece, pinch)
- `Ingredient` (name, canonical_id, quantity, unit, optional, substitutes, category)
- `Recipe` (name, description, cuisine, servings, times, ingredients, steps, tags)

**Stores & Pricing:**
- `Store` (id, name, chain, lat/lng, address, online_ordering, distance_km)
- `PricedItem` (ingredient_name, store_id, price, currency, on_sale, in_stock)
- `GroceryListItem` (ingredient, needed_for_recipes)
- `GroceryList` (items, source_recipes)

**Optimization:**
- `OptimizationWeights` (cost, distance, time, health, num_stores_penalty)
- `StoreAssignment` (store, items, subtotal)
- `BasketPlan` (assignments, total_cost, total_distance, time, score, missing_items)

**Nutrition:**
- `NutritionFacts` (calories, macros, micros)
- `NutritionAnalysis` (per_serving, daily_value_pct, flags)

**User & Feedback:**
- `DietaryRestriction` enum
- `UserProfile` (location, radius, budget, restrictions, allergies, preferences, weights)
- `FeedbackType` enum (HARD_CONSTRAINT, SOFT_PREFERENCE, ONE_OFF, POSITIVE)
- `ParsedFeedback` (type, target, sentiment, raw_text, suggested_update)

### Best practices

- Use `Field(description=...)` everywhere — Pydantic descriptions become tool schemas the LLM can read
- Make all enums string-valued for JSON serialization
- Use `Optional[T]` not `T | None` for broader compatibility
- Keep schemas minimal — add fields only when needed

---

## Step 2 — Configuration & LLM Factory

**Why:** Centralizes model tier choices. Makes it trivial to swap models or temperatures globally.

**File:** `config/settings.py`

### Key concepts:

**Model tiers:**

```python
ROUTING_MODEL = "claude-haiku-4-5-20251001"      # cheap, fast, good for routing
REASONING_MODEL = "claude-sonnet-4-6-20250514"   # main workhorse
CREATIVE_MODEL = "claude-sonnet-4-6-20250514"    # recipe generation
```

**Factory function:**

```python
def get_llm(tier: str = "reasoning", temperature: float = 0.0):
    model_map = {
        "routing": ROUTING_MODEL,
        "reasoning": REASONING_MODEL,
        "creative": CREATIVE_MODEL,
    }
    return ChatAnthropic(
        model=model_map.get(tier, REASONING_MODEL),
        temperature=temperature,
        max_tokens=4096,
    )
```

**Cache TTLs (for when you add Redis):**

```python
CACHE_TTL_SECONDS = {
    "store_locations": 7 * 24 * 3600,   # 1 week
    "store_prices": 24 * 3600,          # 1 day
    "discounts": 24 * 3600,             # 1 day
    "nutrition_facts": 30 * 24 * 3600,  # 1 month (USDA is stable)
}
```

### Why model tiers matter

Haiku is roughly 10× cheaper per token than Sonnet. The supervisor makes a routing decision per turn — using Sonnet for every routing call burns money on a task Haiku handles fine.

---

## Step 3 — Build the Optimizer (Deterministic Core)

**Why now:** This is the most opinionated piece of code. Build it first so the grocery tools have something concrete to call.

**File:** `optimization/basket_optimizer.py`

### The objective function

```
score = w_cost · normalize(total_cost)
      + w_distance · normalize(total_distance)
      + w_time · normalize(total_time)
      + w_stores · num_stores_penalty
```

### Decision variables

- `x[i,s]` — binary: is item `i` bought at store `s`?
- `y[s]` — binary: is store `s` visited?

### Constraints

1. Each item bought exactly once: `sum over s of x[i,s] = 1`
2. Item only available where in stock
3. Store visited if anything bought there: `y[s] >= x[i,s]`

### Implementation steps

1. **Build price lookup:** `(item_name, store_id) → PricedItem`
2. **Identify missing items:** items no store has
3. **Create CP-SAT model** with `model = cp_model.CpModel()`
4. **Add decision variables** as `model.NewBoolVar()`
5. **Add constraints** with `model.Add()`
6. **Compute coefficients** (scaled to integers — CP-SAT works on ints)
7. **Build objective** as weighted sum
8. **Solve** with `solver.Solve(model)`
9. **Extract solution** and return `BasketPlan`

### Critical detail: integer scaling

CP-SAT requires integer coefficients. Pattern:

```python
SCALE = 100
obj_terms.append(int(weights.cost * normalized * SCALE * 1000) * var)
```

Multiply by a large factor, cast to int. Lose precision = lose nothing meaningful.

### Tunable parameters

```python
AVG_DRIVING_SPEED_KMH = 30.0       # urban driving
TIME_PER_STORE_VISIT_MIN = 15.0    # parking + walk + checkout
```

Make these configurable per user later.

### Test it works

The optimizer should produce DIFFERENT plans for DIFFERENT weights:
- High cost weight → cheap-but-far store
- High distance weight → close-but-expensive store
- High store penalty → consolidate to one store

If it doesn't, your weights aren't propagating into the objective.

---

## Step 4 — Recipe Agent Tools

**File:** `tools/recipe_tools.py`

### Tools to implement

1. **`search_recipes`** — query a recipe DB
   - Input: query, cuisine, max_prep_time, exclude_ingredients, diet
   - Output: JSON list of recipe summaries
   - Production: Spoonacular `/recipes/complexSearch`

2. **`get_recipe_details`** — fetch full recipe by ID
   - Input: recipe_id
   - Output: JSON `Recipe` object
   - Production: Spoonacular `/recipes/{id}/information`

3. **`parse_ingredients`** — free-text → structured
   - Input: raw ingredient lines like "2 cups flour"
   - Output: JSON list of `Ingredient` objects
   - Production: `ingredient-parser-nlp` library (NOT the LLM)

### Pattern for every tool

```python
class ToolInput(BaseModel):
    """Pydantic input schema — LLM sees field descriptions."""
    field1: str = Field(description="Clear description for the LLM")

@tool("tool_name", args_schema=ToolInput)
def tool_function(field1: str) -> str:
    """Docstring the LLM reads to decide WHEN to call this."""
    # STUB / real implementation
    return json.dumps(result)
```

### Why JSON strings?

LangChain tools must return strings. JSON is the lingua franca between agents.

### Tool registry

```python
RECIPE_TOOLS = [search_recipes, get_recipe_details, parse_ingredients]
```

---

## Step 5 — Grocery Agent Tools

**File:** `tools/grocery_tools.py`

The biggest tool set. Five tools, each with a clear job.

### Tools

1. **`find_stores`** — geographic search
   - Input: lat, lng, radius_km, require_online_ordering
   - Output: JSON list of `Store` with `distance_km` precomputed
   - Production: Google Places API
   - Includes Haversine distance calculation (deterministic!)

2. **`get_store_prices`** — pricing data
   - Input: store_ids, ingredient_names
   - Output: JSON list of `PricedItem`
   - Production: Instacart API + scraping fallback (Playwright + Bright Data)

3. **`get_discounts`** — sales and flyers
   - Input: store_ids, ingredient_names (optional)
   - Output: JSON list of {store, item, discount_pct, valid_until}
   - Production: Flipp API (Canadian) or Reebee

4. **`aggregate_grocery_list`** — recipe → list
   - Input: JSON list of recipes
   - Output: JSON `GroceryList` (deduplicated, summed quantities)
   - **PURE LOGIC, no API, no LLM**
   - Aggregates: same `(name, unit)` → sum quantities

5. **`optimize_basket`** — call the optimizer
   - Input: grocery_list, stores, priced_items, weights, user location
   - Output: JSON `BasketPlan`
   - **Wraps the OR-Tools optimizer**

### Critical implementation details

**Haversine distance** (don't outsource to LLM):

```python
def haversine_km(lat1, lng1, lat2, lng2) -> float:
    R = 6371.0
    lat1_r, lat2_r = math.radians(lat1), math.radians(lat2)
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat/2)**2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlng/2)**2
    return 2 * R * math.asin(math.sqrt(a))
```

**Aggregation key** must include unit, not just name:

```python
key = (ingredient.name.lower().strip(), ingredient.unit)
```

Why? "200g chicken" and "2 chicken breasts" are not the same — different units, can't be added without conversion.

---

## Step 6 — Nutrition Agent Tools

**File:** `tools/nutrition_tools.py`

### Tools

1. **`lookup_nutrition`** — ingredient → facts
   - Input: ingredient_name, quantity_g
   - Output: JSON `NutritionFacts`
   - Production: USDA FoodData Central API
   - **NEVER let LLM produce nutrition numbers**

2. **`analyze_recipe_nutrition`** — recipe → totals + per-serving
   - Input: JSON `Recipe`
   - Output: JSON `NutritionAnalysis`
   - **PURE MATH** — sums lookups, divides by servings

3. **`check_nutrition_balance`** — week of meals → gaps/excesses
   - Input: weekly aggregated nutrition, user profile
   - Output: JSON {gaps: {...}, excesses: {...}, disclaimer: "..."}
   - Computes percentages vs. daily values

### Daily values reference

```python
DAILY_VALUES = {
    "calories": 2000,
    "protein_g": 50,
    "carbs_g": 275,
    "fat_g": 78,
    "fiber_g": 28,
    "sodium_mg": 2300,
    "iron_mg": 18,
    "calcium_mg": 1300,
    "vitamin_c_mg": 90,
}
```

Adjust for Health Canada vs. FDA depending on your jurisdiction.

### Unit conversion helper

```python
def _to_grams(quantity, unit, ingredient_name):
    conversions = {
        "g": 1, "kg": 1000,
        "ml": 1, "l": 1000,
        "tsp": 5, "tbsp": 15, "cup": 240,
        "piece": 150,  # crude — use density tables in production
        "pinch": 0.5,
    }
    return quantity * conversions.get(unit, 100)
```

For production, use `pint` library + ingredient density database. Crude conversions are fine for v1.

### The disclaimer

ALWAYS include:

```python
"disclaimer": (
    "These are general guidelines. Consult a registered dietitian "
    "for personalized advice, especially with medical conditions."
)
```

This is health-adjacent. Be careful.

---

## Step 7 — Feedback Agent Tools

**File:** `tools/feedback_tools.py`

### Tools

1. **`parse_feedback`** — text → structured
   - Input: user_text, user_id, context
   - Output: JSON `ParsedFeedback`
   - Classifies: HARD_CONSTRAINT, SOFT_PREFERENCE, ONE_OFF, POSITIVE
   - Production: LLM call with structured output schema

2. **`update_user_profile`** — apply changes
   - Input: user_id, update_json
   - Output: JSON {status, profile}
   - **PURE LOGIC** — dict updates
   - **CRITICAL:** allergies are append-only, NEVER overwritten

3. **`get_user_profile`** — fetch profile
   - Input: user_id
   - Output: JSON profile
   - Used by ALL agents to respect preferences

### The append-only safety guarantee

```python
if "allergies" in update:
    existing = set(profile.get("allergies", []))
    existing.update(update["allergies"])
    profile["allergies"] = list(existing)
    del update["allergies"]  # don't fall through to overwrite
```

Why? Even if the LLM misclassifies a message, we cannot silently drop an allergy. Set-union ensures additions only.

### Storage backend

For prototyping: in-memory dict (`_PROFILE_STORE: dict[str, dict] = {}`).
For production: Postgres with structured columns + JSONB for flexible fields.

---

## Step 8 — Build the Specialist Agents

**File:** `agents/specialists.py`

Each agent is a `create_react_agent` with:
- A specific LLM tier and temperature
- A focused tool list
- A clear system prompt

### Recipe Agent

```python
RECIPE_SYSTEM = """You are a recipe specialist...

CRITICAL RULES:
- ALWAYS check user profile (allergies, dietary restrictions) before suggesting.
- Allergies are HARD constraints — never violate them.
- Output recipes as structured Recipe objects via your tools.
- Keep ingredient names canonical."""

def make_recipe_agent():
    return create_react_agent(
        model=get_llm(tier="creative", temperature=0.7),
        tools=RECIPE_TOOLS,
        prompt=RECIPE_SYSTEM,
        name="recipe_agent",
    )
```

### Grocery Agent

Workflow in the system prompt:

```
1. Get user profile (location, radius, weights)
2. Call aggregate_grocery_list on the recipes
3. Call find_stores within radius
4. Call get_store_prices
5. Optionally call get_discounts
6. Call optimize_basket — the deterministic optimizer
7. Explain the plan in plain language
```

Temperature: 0.0 — predictable plans matter.

### Nutrition Agent

```python
NUTRITION_SYSTEM = """...
CRITICAL RULES:
- NEVER state nutrition numbers from memory. Always call tools.
- You provide INFORMATION, not medical advice.
- When suggesting fixes, propose concrete swaps.
- Always include the dietitian disclaimer for health questions."""
```

Temperature: 0.2 — slight warmth in explanations.

### Feedback Agent

```python
FEEDBACK_SYSTEM = """...
CRITICAL RULES:
- Allergies are HARD constraints. Always append, never weaken.
- Distinguish: hard_constraint vs. soft_preference vs. one_off vs. positive.
- Be conservative — when unclear, classify as one_off.
- Confirm hard-constraint changes back to user."""
```

Temperature: 0.0 — classification task.

### System prompt patterns that work

- Lead with role: "You are a [specialty] specialist."
- List CRITICAL RULES with caps for emphasis
- Specify the workflow when there's a sequence
- Include negative instructions: "NEVER do X"
- Tell the agent what's NOT its job

---

## Step 9 — Build the Supervisor Graph

**File:** `agents/graph.py`

The supervisor is a LangGraph node that routes between specialists.

### State

Use `MessagesState` from LangGraph — it manages a list of messages automatically.

### Supervisor node

```python
def supervisor_node(state: MessagesState) -> Command[Literal[...]]:
    llm = get_llm(tier="routing", temperature=0.0)
    
    messages = [
        SystemMessage(content=SUPERVISOR_SYSTEM),
        *state["messages"],
        HumanMessage(content=(
            f"Which agent next? Reply EXACTLY one of: {', '.join(ROUTES)}."
        )),
    ]
    
    response = llm.invoke(messages)
    decision = response.content.strip()
    
    if decision not in ROUTES:
        decision = "FINISH"  # safe default
    
    if decision == "FINISH":
        return Command(goto="__end__")
    return Command(goto=decision)
```

### Specialist wrapper nodes

```python
def make_specialist_node(agent, name):
    def node(state: MessagesState) -> Command[Literal["supervisor"]]:
        result = agent.invoke(state)
        last_msg = result["messages"][-1]
        return Command(
            update={"messages": [HumanMessage(content=last_msg.content, name=name)]},
            goto="supervisor",  # always return to supervisor
        )
    return node
```

### Build the graph

```python
def build_graph():
    graph = StateGraph(MessagesState)
    
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("recipe_agent", make_specialist_node(make_recipe_agent(), "recipe_agent"))
    graph.add_node("grocery_agent", make_specialist_node(make_grocery_agent(), "grocery_agent"))
    graph.add_node("nutrition_agent", make_specialist_node(make_nutrition_agent(), "nutrition_agent"))
    graph.add_node("feedback_agent", make_specialist_node(make_feedback_agent(), "feedback_agent"))
    
    graph.set_entry_point("supervisor")
    return graph.compile()
```

### Routing rules in the supervisor prompt

```
- "Find me a recipe" → recipe_agent
- "What should I buy" → recipe_agent FIRST, THEN grocery_agent
- "Is this healthy?" → nutrition_agent
- User expresses preference/allergy → feedback_agent FIRST
- Complex: "plan healthy cheap meals" → recipe → nutrition → grocery
- All work done → FINISH
```

### Avoiding infinite loops

- Supervisor must have a "FINISH" option
- Default to FINISH on parse failure
- Add a max-iteration safety in production: track a turn counter in state

---

## Step 10 — Smoke Tests

**File:** `tests/test_optimizer.py`

The optimizer is the highest-risk piece. Test it standalone before integrating.

### Test pattern

```python
def test_optimizer_responds_to_weights():
    # Setup: 4 items, 3 stores at varying distances and prices
    items = [...]
    stores = [
        Store(id="A", name="Close Expensive", distance_km=0.5),
        Store(id="B", name="Mid Balanced", distance_km=2.0),
        Store(id="C", name="Far Cheap", distance_km=5.0),
    ]
    
    # Test 1: cost-focused → should pick "Far Cheap"
    weights = OptimizationWeights(cost=0.8, distance=0.1)
    plan = optimize_basket_plan(...)
    assert plan.assignments[0].store.store_id == "C"
    
    # Test 2: distance-focused → should pick "Close Expensive"
    weights = OptimizationWeights(cost=0.1, distance=0.8)
    plan = optimize_basket_plan(...)
    assert plan.assignments[0].store.store_id == "A"
    
    # Test 3: single-store penalty → should consolidate
    weights = OptimizationWeights(num_stores_penalty=0.5)
    plan = optimize_basket_plan(...)
    assert len(plan.assignments) == 1
```

### Run it

```bash
python -m tests.test_optimizer
```

### Expected output

Different weights → different plans. If all three tests pick the same store, your weights aren't entering the objective correctly.

### Other tests to add

- Aggregation: same ingredient in 2 recipes → 1 line with summed quantity
- Allergy enforcement: try to remove allergy via update → fails to remove
- Nutrition math: known ingredient → known calorie count
- Empty inputs: optimizer with 0 items → returns empty plan, not error

---

## Step 11 — Wire Real APIs

Replace stubs one at a time. Each integration is independent.

### Priority order (by ROI)

1. **Anthropic** — already wired via `langchain-anthropic`
2. **Spoonacular** (recipes) — easiest, biggest user-facing impact
3. **USDA FoodData Central** (nutrition) — free, stable, big quality win
4. **Google Places** (stores) — accurate locations
5. **Instacart + scraping** (prices) — hardest, most impactful for cost optimization
6. **Flipp** (Canadian flyers) — local discount data

### Pattern: replace each stub

**Before (stub):**

```python
@tool("search_recipes")
def search_recipes(query, ...):
    return json.dumps({"results": [{"id": "stub-1", ...}]})
```

**After (real):**

```python
@tool("search_recipes")
def search_recipes(query, ...):
    response = requests.get(
        "https://api.spoonacular.com/recipes/complexSearch",
        params={
            "query": query,
            "cuisine": cuisine,
            "maxReadyTime": max_prep_time_min,
            "excludeIngredients": ",".join(exclude_ingredients or []),
            "diet": diet,
            "number": max_results,
            "apiKey": os.getenv("SPOONACULAR_API_KEY"),
        },
        timeout=10,
    )
    response.raise_for_status()
    return json.dumps(response.json())
```

### Key wrapping patterns

**Retry with backoff:**

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def _call_api(...):
    ...
```

**Cache responses:**

```python
import functools

@functools.lru_cache(maxsize=1000)
def _cached_api_call(query_hash):
    ...
```

For production, use Redis with TTLs from `config/settings.py`.

**Handle rate limits:**

```python
try:
    response = requests.get(url, ...)
    response.raise_for_status()
except requests.exceptions.HTTPError as e:
    if e.response.status_code == 429:
        return json.dumps({"error": "rate_limited", "retry_after": 60})
    raise
```

### Stub-to-real checklist per tool

- [ ] Endpoint URL correct
- [ ] All query params mapped
- [ ] Auth via env var (never hardcoded)
- [ ] Timeout set (10s default)
- [ ] Retries with exponential backoff
- [ ] Cache layer (in-memory now, Redis later)
- [ ] Error responses returned as JSON, not raised
- [ ] Rate-limit handling
- [ ] Response shape validated against expected schema

---

## Step 12 — Production Hardening

### Persistence

**User profiles → Postgres:**

```sql
CREATE TABLE user_profiles (
    user_id TEXT PRIMARY KEY,
    location_lat DOUBLE PRECISION,
    location_lng DOUBLE PRECISION,
    max_radius_km DOUBLE PRECISION,
    budget_per_week DOUBLE PRECISION,
    household_size INT,
    allergies JSONB,                  -- ["shellfish", "peanuts"]
    dietary_restrictions JSONB,
    optimization_weights JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE feedback_log (
    id BIGSERIAL PRIMARY KEY,
    user_id TEXT REFERENCES user_profiles(user_id),
    feedback_type TEXT,
    target TEXT,
    sentiment DOUBLE PRECISION,
    raw_text TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Conversation state → Redis** with TTL of ~24h.

### Caching

| Data | TTL | Backend |
|------|-----|---------|
| Store locations | 1 week | Redis |
| Store prices | 1 day | Redis |
| Discounts | 1 day | Redis (refresh hourly during peak) |
| Nutrition facts | 1 month | Redis or local SQLite |
| LLM responses | Don't cache | (varies per request) |

### Observability

Add LangSmith or Langfuse:

```python
import os
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = "..."
os.environ["LANGCHAIN_PROJECT"] = "recipe-agent-prod"
```

You'll get:
- Full trace per request showing supervisor → specialist → tool calls
- Latency per node
- Token usage per call
- Errors with full context

Worth its weight in gold for debugging multi-agent loops.

### Safety guards

**1. Max iteration limit per request:**

```python
class AgentState(TypedDict):
    messages: list
    iteration_count: int

def supervisor_node(state):
    if state.get("iteration_count", 0) > 10:
        return Command(goto="__end__")
    ...
```

**2. Allergy double-check before serving recipe:**

```python
def validate_recipe_against_profile(recipe, profile):
    for ing in recipe.ingredients:
        for allergy in profile.allergies:
            if allergy.lower() in ing.name.lower():
                raise AllergyViolation(f"Recipe contains {allergy}")
```

Run this AFTER the recipe agent, BEFORE returning to user. Belt and suspenders.

**3. Health disclaimer enforcement:**

```python
def add_health_disclaimer_if_needed(response, was_health_question):
    if was_health_question and "dietitian" not in response.lower():
        response += "\n\nNote: This is general information, not medical advice. Consult a dietitian for personalized guidance."
    return response
```

### Rate limiting per user

```python
from datetime import datetime, timedelta

def check_rate_limit(user_id, limit=20, window_min=60):
    key = f"rate:{user_id}"
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, window_min * 60)
    return count <= limit
```

### Cost monitoring

Track per-request cost:

```python
def estimate_cost(usage_metadata):
    input_tokens = usage_metadata.get("input_tokens", 0)
    output_tokens = usage_metadata.get("output_tokens", 0)
    # Sonnet 4.6: $3/M input, $15/M output (check current pricing)
    return (input_tokens * 3 + output_tokens * 15) / 1_000_000
```

Log to your observability platform. Set alerts.

### Testing checklist

- [ ] Optimizer unit tests (3 weight scenarios)
- [ ] Aggregation tests (multi-recipe dedup)
- [ ] Allergy enforcement tests (cannot remove)
- [ ] Schema validation tests (all Pydantic round-trips)
- [ ] Tool stub tests (mock APIs return expected shapes)
- [ ] End-to-end tests (full graph for canonical user flows)
- [ ] Failure mode tests (API down, no items in stock, no stores in radius)

### Deployment shape

For low-medium traffic, this works:

```
[Cloudflare] → [API server (FastAPI)] → [LangGraph runtime]
                       │                        │
                       └─→ [Redis cache]        └─→ [External APIs]
                       └─→ [Postgres]           └─→ [Anthropic]
```

For higher traffic, separate the LangGraph runtime onto worker processes (Celery or RQ) and have the API enqueue jobs.

---

## Appendix — Decisions & Tradeoffs

### Why LangGraph over plain LangChain

Multi-agent supervision needs:
- Explicit state management
- Conditional routing
- Loops with safety
- Human-readable graph topology

Plain LangChain `AgentExecutor` can do this with custom routers and chain composition, but you write more glue code. LangGraph makes the graph the first-class object.

### Why supervisor pattern over peer-to-peer

Alternatives considered:
- **Peer-to-peer**: agents call each other directly. Hard to debug, easy to loop, no central state.
- **Pipeline**: fixed sequence (recipe → grocery → nutrition). Inflexible — what if user only wants nutrition info?
- **Supervisor**: one router, clear flow, easy to add agents. Winner.

### Why OR-Tools over LLM optimization

Tested briefly: asked an LLM "minimize cost+distance with these constraints" — produces plausible answers that are 15-30% suboptimal and inconsistent across runs. OR-Tools gives provably optimal answers in milliseconds.

### Why Pydantic schemas over dicts

LLM tool calling works MUCH better with typed schemas. The Field descriptions become part of the function signature the LLM sees. Eliminates a whole class of "LLM passed wrong shape" bugs.

### Why temperature 0.7 for recipes specifically

Recipes are creative outputs where users want VARIETY across runs. If the user asks for "a pasta recipe" three times, they want three different recipes. Temp 0.7 hits the sweet spot — enough variety to feel fresh, not so much that quality drops.

For grocery decisions, temp 0.0 — same inputs should always produce the same plan.

### Why split feedback into types

`HARD_CONSTRAINT` vs `SOFT_PREFERENCE` vs `ONE_OFF` matters because they have different durability:
- Hard: forever, never weaken (allergies)
- Soft: persist as preference weights (likes/dislikes)
- One-off: log for analysis but don't change profile (one bad recipe)
- Positive: same as soft but for "more like this"

Conflating them = either too sticky (one bad recipe and you never see Italian again) or too loose (allergy gets forgotten).

### What this scaffold does NOT include

- Authentication / session management — assume your API layer handles it
- UI / frontend — bring your own (React, Streamlit, mobile app)
- Voice or image inputs — text-only assumed
- Multi-language support — English-first
- Social features (sharing recipes, group ordering) — out of scope
- Inventory tracking ("what's in my pantry") — could be added as a tool

These are all reasonable extensions but each is a feature, not a foundation. Get the foundation right first.

---

## Quick Reference: File-by-File Build Order

| Order | File | Lines (approx) | Difficulty |
|-------|------|---------------|------------|
| 1 | `models/schemas.py` | 150 | Easy |
| 2 | `config/settings.py` | 40 | Easy |
| 3 | `optimization/basket_optimizer.py` | 180 | Hard |
| 4 | `tools/recipe_tools.py` | 100 | Easy |
| 5 | `tools/grocery_tools.py` | 200 | Medium |
| 6 | `tools/nutrition_tools.py` | 150 | Medium |
| 7 | `tools/feedback_tools.py` | 100 | Easy |
| 8 | `agents/specialists.py` | 100 | Easy |
| 9 | `agents/graph.py` | 100 | Medium |
| 10 | `tests/test_optimizer.py` | 80 | Easy |

**Total scaffold:** ~1200 lines of Python. A focused weekend gets you to working stubs. The next 1-2 weeks is replacing stubs with real APIs and hardening.

---

## Final Checklist

Before considering the system "done":

**Core functionality:**
- [ ] All 4 specialist agents respond to test prompts
- [ ] Supervisor correctly routes between them
- [ ] Optimizer produces different plans for different weights
- [ ] Allergies persist across feedback turns
- [ ] Nutrition tool returns consistent values

**Quality:**
- [ ] All tools have Pydantic input schemas
- [ ] All tools have docstrings the LLM can read
- [ ] Every system prompt has CRITICAL RULES section
- [ ] No LLM does math (verified by code review)
- [ ] No nutrition number comes from LLM memory

**Production:**
- [ ] All API keys in env vars, never hardcoded
- [ ] All external calls have timeouts
- [ ] Retry logic on transient failures
- [ ] Caching layer in place
- [ ] Observability wired (LangSmith or equivalent)
- [ ] Rate limits enforced per user
- [ ] Postgres for profiles, Redis for sessions
- [ ] Cost tracking per request
- [ ] Allergy validation as last-line safety check
- [ ] Health disclaimer enforcement

**Documentation:**
- [ ] README explains the architecture
- [ ] Each tool has usage examples
- [ ] Optimizer math is documented
- [ ] Deployment runbook exists

When all of the above are checked, you have a real system, not a demo.
