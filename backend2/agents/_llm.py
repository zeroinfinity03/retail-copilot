"""
DeepSeek helper module — shared by all agents that need schema-enforced output.

backend2 swaps OpenAI structured outputs (which DeepSeek doesn't support
the same way) for a simpler "JSON-mode + manual Pydantic validation" pattern.

Two key utilities:
  - `get_client()`     — singleton OpenAI-compatible client pointed at DeepSeek
  - `structured(...)`  — drop-in replacement for `client.chat.completions.parse(
                          response_format=PydanticModel)`. Uses DeepSeek's
                          json_object response format under the hood, then
                          validates the response with Pydantic in Python.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).parent.parent
load_dotenv(BACKEND_DIR / ".env")

MODEL = "deepseek-v4-flash"
THINKING_OFF = {"thinking": {"type": "disabled"}}

_client: Optional[OpenAI] = None
T = TypeVar("T", bound=BaseModel)


def get_client() -> OpenAI:
    """Lazy singleton pointed at DeepSeek (OpenAI-compatible endpoint)."""
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
    return _client


def structured(messages: list[dict], schema: type[T], *, max_tokens: int = 4000) -> T:
    """Schema-validated structured response via JSON mode + manual Pydantic.

    DeepSeek's `response_format={"type": "json_object"}` only guarantees that
    the response is valid JSON syntax, not that it matches a particular
    schema. We add the JSON-schema to the system message so the model knows
    the target shape, then validate with Pydantic in Python and let any
    schema mismatch surface as a ValidationError (caller can retry).
    """
    # Inject the schema into the system prompt so the model knows the shape.
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    augmented = [
        {
            "role": messages[0].get("role", "system"),
            "content": (
                messages[0].get("content", "")
                + "\n\nRespond with valid JSON only — no preamble, no markdown "
                  "fences. The JSON must match this schema exactly:\n"
                + schema_json
            ),
        },
        *messages[1:],
    ]

    response = get_client().chat.completions.create(
        model=MODEL,
        messages=augmented,
        response_format={"type": "json_object"},
        extra_body=THINKING_OFF,
        max_tokens=max_tokens,
    )
    raw = response.choices[0].message.content
    return schema(**json.loads(raw))


# ============================================================
# OPTIONAL: server-side schema enforcement via DeepSeek's beta
# strict tool calling
# ============================================================
#
# The implementation above relies on the model producing JSON that matches
# our Pydantic schema. The model almost always gets it right, but if it
# doesn't, Pydantic raises ValidationError and the caller has to retry.
#
# DeepSeek offers a stronger guarantee — "strict mode" inside tool/function
# calling. When you set `strict: true` on a tool definition, the model is
# constrained at decoding time to produce JSON matching the function's
# parameter schema. To use this you must point the client at the `/beta`
# endpoint:
#
# """
# def get_client() -> OpenAI:
#     global _client
#     if _client is None:
#         _client = OpenAI(
#             api_key=os.getenv("DEEPSEEK_API_KEY"),
#             base_url="https://api.deepseek.com/beta",   # /beta enables strict tools
#         )
#     return _client
#
#
# def structured(messages, schema, *, max_tokens=4000):
#     response = get_client().chat.completions.create(
#         model=MODEL,
#         messages=messages,
#         extra_body=THINKING_OFF,
#         tools=[{
#             "type": "function",
#             "function": {
#                 "name": "return_output",
#                 "parameters": _strict_schema(schema),
#                 "strict": True,
#             },
#         }],
#         tool_choice={"type": "function", "function": {"name": "return_output"}},
#         max_tokens=max_tokens,
#     )
#     raw_args = response.choices[0].message.tool_calls[0].function.arguments
#     return schema(**json.loads(raw_args))
#
#
# def _strict_schema(model):
#     # DeepSeek strict mode requires `additionalProperties: false` everywhere
#     schema = model.model_json_schema()
#     _walk(schema)
#     return schema
#
#
# def _walk(node):
#     if isinstance(node, dict):
#         if node.get("type") == "object":
#             node["additionalProperties"] = False
#         for v in node.values():
#             _walk(v)
#     elif isinstance(node, list):
#         for v in node:
#             _walk(v)
# """
#
# Trade-off vs the current json_object path:
#   - strict tools  → near-100% schema match, no ValidationError, server-side
#   - json_object   → ~95% schema match, occasional retry, stable non-beta API
