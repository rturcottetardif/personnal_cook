# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Activate environment (always do this first)
source venv/bin/activate
export $(grep -v '^#' .env | grep -v '^$' | xargs)

# Run all tests (no API key required — uses stubs)
python tests/test_optimizer.py

# Run with pytest
python -m pytest tests/test_optimizer.py -v

# Run a single test
python -m pytest tests/test_optimizer.py::test_cost_focused_prefers_far_cheap_store -v

# Run the full agent graph interactively
python -c "
from agents.graph import build_graph
from langchain_core.messages import HumanMessage
graph = build_graph()
result = graph.invoke({'messages': [HumanMessage(content='YOUR MESSAGE HERE')]})
print(result['messages'][-1].content)
" 2>&1 | grep -v DeprecationWarning | grep -v JsonPlusSerializer

# See full routing trace (all agent hops)
python -c "
from agents.graph import build_graph
from langchain_core.messages import HumanMessage
graph = build_graph()
result = graph.invoke({'messages': [HumanMessage(content='YOUR MESSAGE HERE')]})
for msg in result['messages']:
    print(f'--- {getattr(msg, \"name\", type(msg).__name__)} ---')
    print(msg.content[:300])
"

# Install dependencies
python3.11 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

## Architecture

### Agent graph flow

Every request enters the **Supervisor** (`agents/graph.py`) which runs on Haiku at temp 0.0 and dispatches to one of four specialists. Each specialist always returns to the Supervisor, which then decides to dispatch again or `FINISH`. The loop guard is `len(state["messages"]) > 20`.

```
user → supervisor → recipe_agent    ↗
                  → grocery_agent   → supervisor → ... → FINISH
                  → nutrition_agent ↗
                  → feedback_agent  ↗
```

The graph is assembled in `build_graph()` using `StateGraph(MessagesState)`. Specialist wrapper nodes in `_make_specialist_node()` extract only the last message from each agent run and append it as a `HumanMessage(name=agent_name)` to shared state — this is how the Supervisor sees what each agent said.

### Model tiers (`config/settings.py`)

All model names and temperatures are centralized here. Change a model globally by editing `ROUTING_MODEL`, `REASONING_MODEL`, or `CREATIVE_MODEL`. The `get_llm(tier, temperature)` factory is the only place `ChatAnthropic` is instantiated.

| Tier | Model | Default Temp | Used by |
|------|-------|-------------|---------|
| `"routing"` | `claude-haiku-4-5-20251001` | 0.0 | Supervisor, Feedback Agent |
| `"reasoning"` | `claude-sonnet-4-6` | 0.0 | Grocery Agent, Nutrition Agent |
| `"creative"` | `claude-sonnet-4-6` | 0.7 | Recipe Agent |

### Pydantic schemas as tool contracts (`models/schemas.py`)

All inter-agent data flows through typed Pydantic models. `Field(description=...)` is mandatory on every field because LangChain exposes these descriptions to the LLM as part of the tool schema. Missing or vague descriptions directly cause bad tool calls.

### Tool stubs vs. real APIs

Every tool in `tools/` checks for its API key env var first. If the key is absent, it falls back to a deterministic stub. This means the full graph runs without any keys. When wiring a real API, replace only the stub branch — the tool signature and return type stay the same. Priority order for wiring: Spoonacular → USDA → Google Places → Instacart → Flipp.

### Optimizer (`optimization/basket_optimizer.py`)

The CP-SAT solver requires **integer** coefficients. All float weights are multiplied by `OPTIMIZER_SCALE * 1000` and cast to `int` before entering the objective. If weights seem to have no effect, check that the scaling isn't collapsing precision to zero. The solver's loop guard is `max_time_in_seconds = 10.0`.

### Allergy safety — three independent layers

1. **Source:** Recipe Agent passes allergies as `exclude_ingredients` to the search API
2. **Storage:** `update_user_profile` uses `set.union()` — allergies can only be added, never removed, regardless of what the LLM sends
3. **Post-check:** `validate_recipe_against_profile()` in `graph.py` runs after the recipe agent before returning to the user

Never weaken any of these layers. The set-union in `feedback_tools.py` is the structural guarantee; the others are defence-in-depth.

### User profiles

Stored in `_PROFILE_STORE` (in-memory dict) in `tools/feedback_tools.py`. For production, replace with Postgres (`user_profiles` table with JSONB for allergies and optimization weights). The `get_user_profile` tool is called at the start of every specialist agent turn to load current preferences.

## Key constraints

- **Never let the LLM do math.** Quantity aggregation, nutrition sums, distance, and basket optimization are all pure Python or OR-Tools.
- **Never let the LLM produce nutrition numbers.** All values come from `lookup_nutrition` → USDA API (or stub). The nutrition agent interprets tool output only.
- **Health disclaimer is mandatory.** The `HEALTH_DISCLAIMER` constant in `nutrition_tools.py` must appear in any user-facing nutrition response. `add_health_disclaimer_if_needed()` in `graph.py` is the fallback enforcement.
- **MessagesState has no custom fields.** Do not use `Command(update={...})` with keys outside `messages` — LangGraph silently drops them. Use message count (`len(state["messages"])`) for any loop counting.
