"""
LangGraph multi-agent graph — Supervisor + Specialist pattern.

Why this architecture (vs. one big prompt or peer-to-peer):
  - Supervisor: single routing point; easy to add agents; central state
  - Specialists: focused prompts, focused tools, independently testable
  - Graph: explicit topology; checkpointable; streamable; loops are safe

Flow:
  user input
      ↓
  supervisor  ←──────────────────────────────────┐
      │                                           │
      ├─→ recipe_agent   → tools → response ──────┤
      ├─→ grocery_agent  → tools → response ──────┤
      ├─→ nutrition_agent → tools → response ─────┤
      ├─→ feedback_agent → tools → response ──────┘
      │
      └─→ __end__ (FINISH)

The supervisor runs on Haiku at temp 0.0 — routing is intent matching, not reasoning.
Specialists run on Sonnet; temperatures are set per-agent in specialists.py.
All specialist nodes return to supervisor after every turn; the supervisor decides
whether to dispatch again or finish.
"""

from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph
from langgraph.graph.message import MessagesState
from langgraph.types import Command

from agents.specialists import (
    make_feedback_agent,
    make_grocery_agent,
    make_nutrition_agent,
    make_recipe_agent,
)
from config.settings import get_llm

# ---------------------------------------------------------------------------
# Routing configuration
# ---------------------------------------------------------------------------

ROUTES = ["recipe_agent", "grocery_agent", "nutrition_agent", "feedback_agent", "FINISH"]

SUPERVISOR_SYSTEM = """You are the orchestrator of a recipe and grocery planning system.
Your only job is to decide which specialist agent should handle the next step, or to finish.

Available agents:
  recipe_agent    — finds recipes, searches by cuisine/diet/time, parses ingredients
  grocery_agent   — builds grocery lists, finds stores, optimizes the shopping basket
  nutrition_agent — looks up nutrition facts, analyzes recipes, checks weekly balance
  feedback_agent  — processes user preferences, allergies, and profile updates

Routing rules:
  "Find me a recipe / what should I cook" → recipe_agent
  "What should I buy / plan my grocery run" → recipe_agent FIRST, then grocery_agent
  "Is this healthy / how many calories" → nutrition_agent
  "I'm allergic to X / I don't like Y / save my preference" → feedback_agent FIRST
  "Plan healthy cheap meals for the week" → recipe_agent → nutrition_agent → grocery_agent
  "All work is done / question answered" → FINISH

Routing constraints:
  - Always route to feedback_agent FIRST when the user states an allergy or hard constraint.
  - Do not route to the same agent twice in a row unless the first attempt clearly failed.
  - Default to FINISH if no further action is needed.
  - Default to FINISH if you cannot parse a clear routing decision.

Reply with EXACTLY ONE of: recipe_agent, grocery_agent, nutrition_agent, feedback_agent, FINISH
No explanation. No punctuation. Just the token."""


# ---------------------------------------------------------------------------
# Supervisor node
# ---------------------------------------------------------------------------

def supervisor_node(state: MessagesState) -> Command[Literal[
    "recipe_agent", "grocery_agent", "nutrition_agent", "feedback_agent", "__end__"
]]:
    """Routes to the next specialist or terminates the graph."""
    # Safety guard: each agent hop adds a HumanMessage to state["messages"].
    # MessagesState has no custom fields, so we count messages instead of a counter.
    # >20 messages (~10 round trips) means something is stuck — bail out.
    if len(state["messages"]) > 20:
        return Command(goto="__end__")

    llm = get_llm(tier="routing", temperature=0.0)

    messages = [
        SystemMessage(content=[{
            "type": "text",
            "text": SUPERVISOR_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }]),
        *state["messages"],
        HumanMessage(content=f"Which agent next? Reply EXACTLY one of: {', '.join(ROUTES)}."),
    ]

    response = llm.invoke(messages)
    decision = response.content.strip().rstrip(".")

    if decision not in ROUTES:
        decision = "FINISH"

    if decision == "FINISH":
        return Command(goto="__end__")

    return Command(goto=decision)


# ---------------------------------------------------------------------------
# Specialist wrapper nodes
# ---------------------------------------------------------------------------

def _make_specialist_node(agent, agent_name: str):
    """
    Wrap a specialist agent so it:
      1. Runs the agent with the full message state
      2. Appends the agent's last message to shared state
      3. Always returns to the supervisor
    """
    def node(state: MessagesState) -> Command[Literal["supervisor"]]:
        result = agent.invoke(state)
        last_msg = result["messages"][-1]
        return Command(
            update={
                "messages": [
                    HumanMessage(content=last_msg.content, name=agent_name)
                ]
            },
            goto="supervisor",
        )

    node.__name__ = agent_name
    return node


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Assemble and compile the full multi-agent graph.

    Instantiate agents inside this function so the graph can be rebuilt
    (e.g., in tests) without carrying stale agent state.
    """
    graph = StateGraph(MessagesState)

    # Supervisor — always the entry point
    graph.add_node("supervisor", supervisor_node)

    # Specialist nodes
    graph.add_node("recipe_agent",    _make_specialist_node(make_recipe_agent(),    "recipe_agent"))
    graph.add_node("grocery_agent",   _make_specialist_node(make_grocery_agent(),   "grocery_agent"))
    graph.add_node("nutrition_agent", _make_specialist_node(make_nutrition_agent(), "nutrition_agent"))
    graph.add_node("feedback_agent",  _make_specialist_node(make_feedback_agent(),  "feedback_agent"))

    graph.set_entry_point("supervisor")

    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience: post-response safety checks
# ---------------------------------------------------------------------------

def validate_recipe_against_profile(recipe_ingredients: list[str], allergies: list[str]) -> list[str]:
    """
    Belt-and-suspenders allergy check AFTER the recipe agent responds.
    Returns a list of allergen violations found. Empty list = safe.

    This is a last-resort guard — the recipe agent should already be filtering,
    but we never rely solely on the LLM for safety enforcement.
    """
    violations = []
    for allergy in allergies:
        for ingredient in recipe_ingredients:
            if allergy.lower() in ingredient.lower():
                violations.append(f"Recipe contains '{allergy}' (found in '{ingredient}')")
    return violations


def add_health_disclaimer_if_needed(response: str, was_health_question: bool) -> str:
    """Append health disclaimer if the response addresses a health topic and lacks one."""
    disclaimer = (
        "\n\n*Note: This is general information, not medical advice. "
        "Consult a registered dietitian for personalized guidance.*"
    )
    if was_health_question and "dietitian" not in response.lower():
        return response + disclaimer
    return response
