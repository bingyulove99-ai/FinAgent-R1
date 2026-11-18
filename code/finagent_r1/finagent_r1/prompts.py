"""Prompts and tool schemas for the FinQA agent.

The prompt is deliberately explicit about the response protocol because the
policy is trained from a sparse outcome reward: most of the early learning
signal comes from learning to emit parseable tool calls and a final
``<answer>`` block.
"""

from __future__ import annotations

from typing import Any

FINQA_SYSTEM_PROMPT = """You are FinAgent-R1, a careful financial analyst agent.
You answer questions about one SEC filing (10-K / 10-Q) by retrieving evidence
from the filing and performing exact arithmetic.

Available tools
1. financial_retriever(query, top_k)
   Search the filing's text and tables. The tool is ALREADY scoped to the
   company and fiscal year given in the question, so write metric-level queries
   (line-item names, periods, units) and do NOT put the ticker or year in the
   query.
2. python_calc(code)
   Evaluate a Python arithmetic expression exactly. Use it for every
   calculation. Never do mental arithmetic.

Response protocol (strict)
- Think inside <analysis>...</analysis>.
- Then either emit one or two tool calls:
  <tool_call>{"name": "financial_retriever", "arguments": {"query": "..."}}</tool_call>
  or finish the task with:
  <answer>...</answer>
- The <answer> must contain the requested value(s) with units or a percent sign.
  If the question asks for several values, include all of them.
- Only use numbers that appear in tool observations or calculation results.
  Never invent or round from memory.
- If a query already returned the needed evidence, do not repeat it. Refine the
  query at most once, then answer with the best supported value.
- You have a limited number of steps. Use as few tool calls as possible.
"""

FINQA_USER_PROMPT = """Question:
{user_query}

Filing scope: ticker={ticker}, fiscal_year={fiscal_year}
Step {step_index}/{max_steps}

Verified evidence scratchpad:
{scratchpad}

Previous tool calls:
{history_actions}

Most recent tool observations:
{latest_observations}

Tool errors / format feedback:
{tool_feedback}

Decide the next action. Emit tool calls or the final <answer>.
"""

DIRECT_ANSWER_SYSTEM_PROMPT = """You are a financial analyst. Answer the question
using the filing excerpt implicitly available to you from your training data.
Return only the requested value(s) inside <answer>...</answer>, with units or a
percent sign. Do not explain your reasoning.
"""


def get_tool_schemas() -> list[dict[str, Any]]:
    """Return fresh OpenAI-style function schemas for both tools.

    The schemas are intentionally free of ``ticker`` / ``year`` arguments: the
    environment injects the filing scope from the dataset row, which prevents
    the policy from cheating by naming a different company.
    """

    return [
        {
            "type": "function",
            "function": {
                "name": "financial_retriever",
                "description": (
                    "Search the current SEC filing (10-K/10-Q) for passages and table rows. "
                    "The company and fiscal year are fixed by the environment."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Metric-level search query, e.g. 'total net sales 2021'.",
                        },
                        "top_k": {
                            "type": "integer",
                            "description": "Number of passages to return (1-10, default 3).",
                            "default": 3,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "python_calc",
                "description": (
                    "Evaluate an arithmetic Python expression exactly. "
                    "Use *100 for a percentage; do not write the % symbol."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Arithmetic expression, e.g. '(15.0 - 10.61) / 10.61 * 100'.",
                        }
                    },
                    "required": ["code"],
                },
            },
        },
    ]


FINQA_TOOL_SCHEMAS = get_tool_schemas()

__all__ = [
    "DIRECT_ANSWER_SYSTEM_PROMPT",
    "FINQA_SYSTEM_PROMPT",
    "FINQA_TOOL_SCHEMAS",
    "FINQA_USER_PROMPT",
    "get_tool_schemas",
]
