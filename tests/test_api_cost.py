"""
API cost measurement test — requires ANTHROPIC_API_KEY.

Run:
    python -m pytest tests/test_api_cost.py -v -s

Prints token usage and estimated cost per model for a fixed query. Use this
to establish a baseline before applying optimizations like prompt caching,
then re-run afterwards to measure the actual savings.

Skipped automatically in CI when ANTHROPIC_API_KEY is not set.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import HumanMessage
from langchain_core.outputs import LLMResult


# ---------------------------------------------------------------------------
# Anthropic pricing (per million tokens, as of 2025-05)
# cache_write = 25% premium over input; cache_read = 10% of input
# ---------------------------------------------------------------------------

_PRICING: dict[str, dict[str, float]] = {
    "claude-haiku-4-5-20251001": {
        "input":       0.80,
        "output":      4.00,
        "cache_write": 1.00,   # 0.80 * 1.25
        "cache_read":  0.08,   # 0.80 * 0.10
    },
    "claude-sonnet-4-6": {
        "input":        3.00,
        "output":      15.00,
        "cache_write":  3.75,  # 3.00 * 1.25
        "cache_read":   0.30,  # 3.00 * 0.10
    },
}

# Fallback when the API returns a model ID we don't recognise yet
_FALLBACK_PRICING = _PRICING["claude-sonnet-4-6"]


def _pricing_for(model_id: str) -> dict[str, float]:
    for key, price in _PRICING.items():
        if key in model_id:
            return price
    return _FALLBACK_PRICING


# ---------------------------------------------------------------------------
# Callback: collect every LLM response's token usage
# ---------------------------------------------------------------------------

class TokenUsageCollector(BaseCallbackHandler):
    """Accumulates usage metadata from every ChatAnthropic call in the graph."""

    def __init__(self):
        self.records: list[dict] = []

    def on_llm_end(self, response: LLMResult, **kwargs):
        for gen_list in response.generations:
            for gen in gen_list:
                msg = getattr(gen, "message", None)
                if msg is None:
                    continue

                # Standard LangChain usage fields
                usage_meta = getattr(msg, "usage_metadata", None) or {}
                # Anthropic puts cache token counts in response_metadata.usage
                resp_meta = getattr(msg, "response_metadata", None) or {}
                raw_usage = resp_meta.get("usage", {})
                model = resp_meta.get("model", "unknown")

                self.records.append({
                    "model":        model,
                    "input_tokens": (
                        usage_meta.get("input_tokens")
                        or raw_usage.get("input_tokens", 0)
                    ),
                    "output_tokens": (
                        usage_meta.get("output_tokens")
                        or raw_usage.get("output_tokens", 0)
                    ),
                    "cache_write_tokens": raw_usage.get("cache_creation_input_tokens", 0),
                    "cache_read_tokens":  raw_usage.get("cache_read_input_tokens", 0),
                })

    # ------------------------------------------------------------------
    # Aggregation
    # ------------------------------------------------------------------

    def aggregate(self) -> dict:
        """Return per-model totals and grand total cost in USD."""
        by_model: dict[str, dict] = {}

        for rec in self.records:
            m = rec["model"]
            if m not in by_model:
                by_model[m] = {
                    "calls":             0,
                    "input_tokens":      0,
                    "output_tokens":     0,
                    "cache_write_tokens": 0,
                    "cache_read_tokens":  0,
                }
            t = by_model[m]
            t["calls"]              += 1
            t["input_tokens"]       += rec["input_tokens"]
            t["output_tokens"]      += rec["output_tokens"]
            t["cache_write_tokens"] += rec["cache_write_tokens"]
            t["cache_read_tokens"]  += rec["cache_read_tokens"]

        grand_total = 0.0
        for m, t in by_model.items():
            p = _pricing_for(m)
            cost = (
                t["input_tokens"]       * p["input"]       / 1_000_000
                + t["output_tokens"]    * p["output"]      / 1_000_000
                + t["cache_write_tokens"] * p["cache_write"] / 1_000_000
                + t["cache_read_tokens"]  * p["cache_read"]  / 1_000_000
            )
            t["cost_usd"] = round(cost, 7)
            grand_total += cost

        return {"by_model": by_model, "grand_total_usd": round(grand_total, 7)}


# ---------------------------------------------------------------------------
# Reporting helper
# ---------------------------------------------------------------------------

def _print_report(report: dict, label: str) -> None:
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  COST REPORT — {label}")
    print(sep)
    for model, t in report["by_model"].items():
        print(f"\n  Model : {model}")
        print(f"    LLM calls   : {t['calls']}")
        print(f"    Input       : {t['input_tokens']:>8,} tokens")
        print(f"    Output      : {t['output_tokens']:>8,} tokens")
        if t["cache_write_tokens"]:
            print(f"    Cache write : {t['cache_write_tokens']:>8,} tokens")
        if t["cache_read_tokens"]:
            print(f"    Cache read  : {t['cache_read_tokens']:>8,} tokens")
        print(f"    Cost        : ${t['cost_usd']:.7f}")
    print(f"\n  GRAND TOTAL : ${report['grand_total_usd']:.7f}")
    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_cost_simple_recipe_request():
    """
    Fixed query → full graph run → token usage + cost report.

    This test never fails on cost (no budget assertion). Its job is to print
    a reproducible baseline. Run before and after an optimization to compare.
    """
    from agents.graph import build_graph

    collector = TokenUsageCollector()
    graph = build_graph()

    result = graph.invoke(
        {"messages": [HumanMessage(content="Find me a quick pasta recipe")]},
        config={"callbacks": [collector]},
    )

    assert result["messages"], "Graph returned no messages"

    report = _print_report(collector.aggregate(), "simple recipe request (caching + max_tokens=20)")

    # Sanity: callback must have captured something
    total_tokens = sum(
        t["input_tokens"] + t["output_tokens"]
        for t in collector.aggregate()["by_model"].values()
    )
    assert total_tokens > 0, (
        "No tokens were recorded. The callback is not wired correctly "
        "or the graph made no LLM calls."
    )

    print("  Final answer preview:")
    print(" ", result["messages"][-1].content[:300])


@pytest.mark.skipif(
    not os.getenv("ANTHROPIC_API_KEY"),
    reason="ANTHROPIC_API_KEY not set",
)
def test_cost_multi_agent_request():
    """
    A request that intentionally triggers multiple agent hops
    (recipe → nutrition → grocery) to show full multi-hop cost.
    """
    from agents.graph import build_graph

    collector = TokenUsageCollector()
    graph = build_graph()

    result = graph.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Plan healthy cheap meals for the week — "
                        "I need recipes, nutrition facts, and a grocery list."
                    )
                )
            ]
        },
        config={"callbacks": [collector]},
    )

    assert result["messages"], "Graph returned no messages"

    report = collector.aggregate()
    _print_report(report, "multi-agent request (baseline)")

    total_tokens = sum(
        t["input_tokens"] + t["output_tokens"]
        for t in report["by_model"].values()
    )
    assert total_tokens > 0, "No tokens recorded — callback not wired correctly"
