# Personal Cook — AI Recipe & Grocery Planning System

An AI-powered system that plans weekly meals, builds an optimized grocery list, and finds the cheapest combination of nearby stores to shop at — all while respecting dietary restrictions, allergies, and personal preferences.

The system is built on a multi-agent architecture where specialized AI agents collaborate under an orchestrating supervisor. It is designed to be correct where correctness is safety-critical, and creative where creativity is the product value.

---

## Table of Contents

1. [What This Solves](#what-this-solves)
2. [Architecture Overview](#architecture-overview)
3. [Why Multi-Agent vs. One Big Prompt](#why-multi-agent-vs-one-big-prompt)
4. [Technology Choices and Reasoning](#technology-choices-and-reasoning)
5. [What the LLM Does NOT Do](#what-the-llm-does-not-do)
6. [Safety Architecture](#safety-architecture)
7. [Upgrade Path](#upgrade-path)
8. [API Integrations](#api-integrations)
9. [Running Locally](#running-locally)
10. [Production Deployment](#production-deployment)

---

## What This Solves

Most recipe apps give you a list of ingredients. This system gives you a shopping plan.

A user says: *"Plan healthy, affordable meals for my family this week — I'm allergic to shellfish and prefer Italian."*

The system:
1. Generates recipes that match the cuisine preference and are nutritionally balanced
2. Checks every suggestion against the shellfish allergy before it reaches the user
3. Aggregates ingredients across all recipes into a single deduplicated list
4. Finds grocery stores within the user's radius
5. Fetches current prices and active flyer discounts
6. Solves the store-assignment problem optimally — deciding which items to buy where to minimize a weighted combination of total cost, travel distance, and number of store stops
7. Returns a complete shopping plan with per-store breakdowns and a total cost estimate

The distinction from a chatbot wrapping a recipe API: the grocery optimization is deterministic and provably optimal. The LLM handles language; a constraint solver handles the math.

---

## Architecture Overview

```
                    ┌─────────────────────┐
                    │   User Interface    │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │   Supervisor Agent  │  ← Claude Haiku (cheap routing)
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

**Data flow for a full meal plan request:**

1. The user's message enters the Supervisor
2. The Supervisor reads the intent and dispatches to the Recipe Agent
3. The Recipe Agent searches recipes, filters by dietary profile, and returns structured `Recipe` objects
4. The Supervisor sees the recipe results and dispatches to the Grocery Agent
5. The Grocery Agent aggregates ingredients, finds stores, fetches prices, and calls the OR-Tools optimizer
6. The Supervisor receives the basket plan and dispatches to the Nutrition Agent if requested
7. The Nutrition Agent analyzes the meal plan against daily recommended values
8. The Supervisor determines all work is complete and returns the combined response to the user

Every agent turn flows back through the Supervisor. There is no direct agent-to-agent communication — the Supervisor is always aware of state.

---

## Why Multi-Agent vs. One Big Prompt

The natural starting point is a single prompt that does everything. It fails in three ways:

**1. Context collisions.** A single agent with all 14 tools sees recipe tools, grocery tools, nutrition tools, and feedback tools simultaneously. The model must reason about which tool is relevant for each sub-task. A focused specialist with 3 tools makes fewer mistakes.

**2. Conflicting temperature requirements.** Recipe generation benefits from temperature 0.7 — users want variety when asking for meal ideas. Grocery optimization requires temperature 0.0 — the same inputs should always produce the same plan. A single-agent architecture cannot run at two temperatures simultaneously.

**3. Cost structure.** The Supervisor routes every turn using Claude Haiku, which is approximately 10× cheaper per token than Sonnet. On a system where routing happens on every message, the cost savings compound at scale. A monolithic Sonnet agent pays full price for routing decisions that a Haiku can handle equally well.

**The practical result:** specialists are independently testable, independently deployable, and can be upgraded (or replaced with fine-tuned models) without touching the rest of the system.

---

## Technology Choices and Reasoning

### LangGraph — Orchestration

The Supervisor + Specialists pattern requires:
- Shared conversation state across agent hops (the recipe agent's output must be visible to the grocery agent)
- Conditional routing (after recipe results, dispatch to grocery OR nutrition depending on the user's question)
- Loop safety (the system must detect "done" and exit without spinning)
- Checkpointing (a conversation that's interrupted mid-run can be resumed)

All of this can be built from scratch with the raw Anthropic SDK — it takes roughly 200–300 lines of orchestration code before writing a single line of product logic. LangGraph provides it as a first-class `StateGraph` abstraction.

The alternative evaluated was using the Claude Agent SDK directly (Anthropic's native multi-agent primitives). For new projects it is worth evaluating. LangGraph was chosen here for its production track record and the breadth of its checkpointing/observability ecosystem.

### OR-Tools CP-SAT — Basket Optimization

When tested empirically: asking an LLM to minimize grocery cost while respecting store availability and distance constraints produced answers that were 15–30% suboptimal and inconsistent across runs. This is not a failure of prompting — it is a fundamental limitation of autoregressive models on combinatorial optimization problems.

OR-Tools CP-SAT (Google's Constraint Programming / Boolean Satisfiability solver) solves the same problem in milliseconds with provably optimal results. The solver takes binary decision variables (buy item X at store S, yes/no), adds constraints (every item must be bought exactly once; only at stores where it's in stock), and minimizes a weighted objective function over cost, distance, and number of store visits.

**This is the sharpest architectural differentiator in the system.** Most LLM-based grocery tools let the model reason about prices and stores in natural language. This system treats store assignment as what it is: an integer program.

```python
# The objective function — pure math, no LLM
score = w_cost     · normalize(total_cost)
      + w_distance · normalize(total_distance)
      + w_time     · normalize(total_time)
      + w_stores   · num_stores_visited
```

The user's `OptimizationWeights` profile directly controls these weights, making the plan transparent and tunable.

### Pydantic v2 — Data Contracts

Every inter-agent data structure is a typed Pydantic model with `Field(description=...)` on every field. This is not defensive programming — it is how LLM tool quality works.

When a LangChain tool is registered, the Pydantic schema becomes the function signature the LLM sees. A field described as `"Canonical lowercase ingredient name, e.g. 'chicken breast'"` produces far fewer mis-shaped tool calls than a field described as `"name"`. Typed schemas eliminate an entire class of "LLM passed wrong shape" bugs that are otherwise only caught at runtime.

### Claude Model Tiers

| Agent | Model | Temperature | Reasoning |
|-------|-------|-------------|-----------|
| Supervisor | Haiku | 0.0 | Routing is intent classification. Haiku is ~10× cheaper and handles it correctly. |
| Recipe Agent | Sonnet | 0.7 | Variety across recipe suggestions is the product value. |
| Grocery Agent | Sonnet | 0.0 | Same inputs must always produce the same shopping plan. |
| Nutrition Agent | Sonnet | 0.2 | Numbers come from tools; slight warmth only for explanation quality. |
| Feedback Agent | Haiku | 0.0 | Classification task. Deterministic output required. |

The temperature choices are deliberate and distinct. A system where everything runs at 0.7 produces unpredictable grocery plans. A system where everything runs at 0.0 produces identical recipe suggestions on every call.

### Redis + Postgres

Two separate persistence layers with different access patterns:

**Postgres** — durable, structured user data. The `user_profiles` table stores location, budget, allergies (JSONB), and optimization weights. The `feedback_log` table records every preference update for analysis. Relational structure enables complex queries (e.g., "users with gluten-free preference near postal code X").

**Redis** — ephemeral, fast-access data. Conversation state is stored with a ~24-hour TTL. API responses are cached to avoid re-fetching stable data: store locations (1 week), prices (1 day), USDA nutrition facts (1 month). At scale, Redis absorbs the majority of external API calls and reduces latency for repeat requests.

The separation matters: losing Redis loses a conversation session. Losing Postgres loses a user's allergy profile. These have very different recovery consequences and warrant different infrastructure guarantees.

### LangSmith — Observability

A bug in a 5-hop agent chain (user → supervisor → recipe → supervisor → grocery → supervisor → FINISH) is undebuggable without execution traces. LangSmith instruments every node automatically with zero application code changes (`LANGCHAIN_TRACING_V2=true`). Every production request produces a full trace showing routing decisions, tool calls, token counts, and latency per node. Langfuse is the self-hosted alternative.

---

## What the LLM Does NOT Do

This is a deliberate product decision: the LLM translates, routes, and explains. Code does math, lookups, and safety enforcement.

| Task | Handled By | Why Code, Not LLM |
|------|-----------|-------------------|
| Supervisor routing | LLM (Haiku, temp 0.0) | Needs language understanding |
| Recipe creativity | LLM (Sonnet, temp 0.7) | Variety is the feature |
| Ingredient text → structure | NLP library (`ingredient-parser-nlp`) | Deterministic parsing required |
| Grocery quantity aggregation | Pure Python | Addition. LLMs make arithmetic errors. |
| Store assignment optimization | OR-Tools CP-SAT | Provably optimal; LLMs are 15–30% suboptimal |
| Nutrition values | USDA API | Authoritative external source |
| Nutrition math (sums/ratios) | Pure Python | Safety-adjacent numbers |
| Nutrition interpretation | LLM (Sonnet, temp 0.2) | Translating numbers → plain language |
| Distance calculation | Haversine formula (Python) | Geographic math |
| Feedback classification | LLM (Haiku, temp 0.0) | Intent recognition |
| Allergy enforcement | Pure Python | Never trust an LLM with safety constraints |
| Profile updates | Pure Python | Set operations |

The pattern holds: anywhere the output has a correct answer that can be verified, code computes it. The LLM is used where ambiguity is the point.

---

## Safety Architecture

Allergy safety uses three independent layers. All three must fail for an allergen to reach the user.

**Layer 1 — Source filtering.** The Recipe Agent passes the user's allergy list to the recipe search API as `exclude_ingredients`. The API filters before results are returned.

**Layer 2 — Append-only profile storage.** The `update_user_profile` tool enforces a set-union on allergies:

```python
existing = set(profile.get("allergies", []))
new_allergies = {a.lower().strip() for a in update.pop("allergies")}
profile["allergies"] = sorted(existing | new_allergies)
```

Even if a malformed LLM response attempts to replace the allergy list with an empty array, the set-union ensures the existing allergies survive. This constraint lives in code, not in a prompt instruction.

**Layer 3 — Post-agent validator.** Before any recipe reaches the user, `validate_recipe_against_profile()` runs a string-match check of every ingredient against every allergy. This is a last-resort guard:

```python
for allergy in profile["allergies"]:
    for ingredient in recipe.ingredients:
        if allergy.lower() in ingredient.name.lower():
            raise AllergyViolation(...)
```

Health disclaimers follow a similar layered approach: the Nutrition Agent's system prompt mandates including them, and `add_health_disclaimer_if_needed()` appends one as a post-processing check if the agent response omits it.

---

## Upgrade Path

Each upgrade is additive — v1 ships without them.

**v1 (current) — Stub APIs → Real APIs**

The system ships with stubs for every external integration. Each stub produces realistic synthetic data so the full agent graph can be tested without API keys. Real APIs are wired one at a time:

| Integration | Priority | Notes |
|------------|----------|-------|
| Anthropic | Done | Via `langchain-anthropic` |
| Spoonacular (recipes) | 1st | 150 free calls/day |
| USDA FoodData Central (nutrition) | 2nd | Free, generous rate limits |
| Google Places (stores) | 3rd | $200/month free credit |
| Instacart (prices) | 4th | Developer application required |
| Flipp / Reebee (Canadian flyers) | 5th | Contact for API access |

**v2 — Semantic Recipe Search (RAG)**

The current search is keyword-based (Spoonacular `complexSearch`). The v2 upgrade replaces it with embedding-based retrieval: recipe descriptions are embedded and stored in the **pgvector** extension on Postgres (already listed in `requirements.txt`). A user asking for *"something light and summery"* retrieves semantically similar recipes rather than relying on exact keyword matches.

The stack: Claude or OpenAI embeddings → pgvector store → LangChain `PGVector` retriever → `search_recipes` tool replacement. User preference history can be embedded to enable personalization — past liked recipes influence future retrieval rankings.

**v3 — MCP Server Layer**

Today, each tool is a Python function registered via `@tool` decorator. In v3, these tools become MCP (Model Context Protocol) server endpoints. MCP is Anthropic's open protocol for connecting LLMs to external capabilities via a standardized interface.

The business reason to migrate: once multiple clients exist (web app, mobile app, Claude Desktop, third-party integrations), each client should not embed its own copy of the tool logic. An MCP server centralizes the implementation, owns the API keys, and allows any MCP-compatible client to discover and call the tools.

Concrete v3 MCP servers:
- **Spoonacular MCP server** — `search_recipes`, `get_recipe_details` as protocol endpoints
- **USDA Nutrition MCP server** — canonical nutrition lookup shared across all surfaces
- **User Profile MCP server** — allergy-safe profile storage accessible to any client

The migration is purely operational: the tool logic does not change, only how it's exposed.

---

## API Integrations

| Service | Purpose | Free Tier | Status |
|---------|---------|-----------|--------|
| Anthropic Claude | LLM (routing + reasoning + creativity) | Pay-as-you-go | Stub-ready |
| Spoonacular | Recipe search and details | 150 calls/day | Stub |
| USDA FoodData Central | Authoritative nutrition data | Free, generous | Stub |
| Google Places | Grocery store discovery and location | $200/month credit | Stub |
| Instacart | Real-time grocery prices | Application required | Stub |
| Flipp / Reebee | Canadian flyer discounts | Contact for access | Stub |

---

## Running Locally

```bash
# 1. Clone and set up environment
git clone <repo>
cd agentic_course
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 2. Configure API keys
cp .env.template .env
# Edit .env and add your ANTHROPIC_API_KEY at minimum

# 3. Run smoke tests (no API key required — uses stubs)
python -m pytest tests/test_optimizer.py -v

# 4. Run the agent graph (requires ANTHROPIC_API_KEY)
python -c "
from agents.graph import build_graph
from langchain_core.messages import HumanMessage

graph = build_graph()
result = graph.invoke({'messages': [HumanMessage(content='Find me a quick Italian pasta recipe')]})
print(result['messages'][-1].content)
"
```

The test suite runs entirely on stubs — no API keys are required to validate the optimizer, aggregation logic, or allergy safety guarantees.

---

## Production Deployment

**Baseline topology (low to medium traffic):**

```
[Cloudflare WAF] → [FastAPI server] → [LangGraph runtime]
                         │                     │
                         ├→ [Redis]            └→ [Anthropic API]
                         └→ [Postgres]         └→ [Spoonacular, USDA, Google Places, ...]
```

FastAPI receives user requests and invokes the compiled LangGraph graph synchronously. The LangGraph checkpointer persists conversation state to Redis so the graph can resume across HTTP requests.

**Scaling path (high traffic):**

```
[API server] → [Task queue (Celery / RQ)] → [Worker pool]
                                                  │
                                           [LangGraph runtime]
```

For sustained load, the API server enqueues graph invocations as jobs. Worker processes consume the queue and run the agent graph asynchronously. This separates the latency-sensitive API layer from the long-running LLM calls and allows the worker pool to scale independently.

**Cost monitoring** is built into the system. Each LLM response includes `usage_metadata` (input tokens, output tokens). Logging these per request and setting per-user rate limits (20 requests/60 minutes by default) prevents runaway costs during development and abuse in production.

---

## Project Structure

```
agentic_course/
├── models/schemas.py          # Pydantic data contracts — the shared language
├── config/settings.py         # LLM factory, model tiers, cache TTLs
├── optimization/
│   └── basket_optimizer.py    # OR-Tools CP-SAT solver
├── tools/
│   ├── recipe_tools.py        # search, details, ingredient parsing
│   ├── grocery_tools.py       # stores, prices, discounts, aggregation, optimization
│   ├── nutrition_tools.py     # USDA lookup, recipe analysis, weekly balance
│   └── feedback_tools.py      # preference parsing, profile CRUD
├── agents/
│   ├── specialists.py         # Four ReAct agents with system prompts
│   └── graph.py               # LangGraph supervisor + safety utilities
└── tests/
    └── test_optimizer.py      # Weight-sensitivity + edge case smoke tests
```

**Scaffold total:** ~1,200 lines of Python. All external API calls are stubbed and can be replaced one at a time as keys become available.
