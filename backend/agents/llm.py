"""Provider switch and structured-output helper for backend."""

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

LLM_PROVIDER = "openai"  # "openai" or "deepseek"
OPENAI_REASONING_EFFORT = "minimal"

DEFAULT_MODELS = {
    "openai": "gpt-5-mini",
    "deepseek": "deepseek-v4-flash",
}
PROVIDER = LLM_PROVIDER.strip().lower()
if PROVIDER not in DEFAULT_MODELS:
    raise ValueError(
        f"Unsupported LLM_PROVIDER={PROVIDER!r}. Use one of: {', '.join(DEFAULT_MODELS)}"
    )

LLM_MODEL = DEFAULT_MODELS[PROVIDER]
THINKING_OFF = {"thinking": {"type": "disabled"}}

_client: Optional[OpenAI] = None
T = TypeVar("T", bound=BaseModel)


def get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client

    if PROVIDER == "deepseek":
        _client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
        return _client

    kwargs = {}
    if os.getenv("OPENAI_API_KEY"):
        kwargs["api_key"] = os.getenv("OPENAI_API_KEY")
    if os.getenv("OPENAI_BASE_URL"):
        kwargs["base_url"] = os.getenv("OPENAI_BASE_URL")
    _client = OpenAI(**kwargs)
    return _client


def provider_completion_kwargs() -> dict:
    if PROVIDER == "deepseek":
        return {"extra_body": THINKING_OFF}
    return {"reasoning_effort": OPENAI_REASONING_EFFORT}


def structured(messages: list[dict], schema: type[T]) -> T:
    if PROVIDER == "deepseek":
        return _structured_deepseek(messages, schema)
    return _structured_openai(messages, schema)


def _structured_openai(messages: list[dict], schema: type[T]) -> T:
    response = get_client().chat.completions.parse(
        model=LLM_MODEL,
        messages=messages,
        **provider_completion_kwargs(),
        response_format=schema,
    )
    parsed = response.choices[0].message.parsed
    if parsed is None:
        raw = response.choices[0].message.content or ""
        return schema.model_validate_json(raw)
    return parsed


def _structured_deepseek(
    messages: list[dict],
    schema: type[T],
) -> T:
    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    instruction = (
        "Respond with valid JSON only. Do not include markdown fences, "
        "preamble, or commentary. The JSON must match this schema exactly:\n"
        f"{schema_json}"
    )

    if messages and messages[0].get("role") == "system":
        augmented = [
            {
                "role": "system",
                "content": f"{messages[0].get('content', '')}\n\n{instruction}",
            },
            *messages[1:],
        ]
    else:
        augmented = [{"role": "system", "content": instruction}, *messages]

    last_error: Exception | None = None
    for _ in range(2):
        response = get_client().chat.completions.create(
            model=LLM_MODEL,
            messages=augmented,
            response_format={"type": "json_object"},
            extra_body=THINKING_OFF,
        )
        raw = response.choices[0].message.content or ""
        try:
            return schema.model_validate_json(raw)
        except Exception as exc:
            last_error = exc
            augmented = [
                *augmented,
                {"role": "assistant", "content": raw},
                {
                    "role": "user",
                    "content": (
                        "The JSON did not validate against the schema. "
                        f"Validation error: {exc}. Return corrected JSON only."
                    ),
                },
            ]

    assert last_error is not None
    raise last_error
