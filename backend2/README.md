# backend2 — DeepSeek alternative

A parallel implementation of `backend/` that swaps the LLM provider from OpenAI (`gpt-5-mini`) to DeepSeek (`deepseek-v4-flash`). Same agent architecture, same prompts, same DuckDB warehouse — only the model + the structured-output mechanism differ.

## Why DeepSeek?

- **~5× cheaper per token** vs OpenAI on cache-miss, ~98% discount on cache-hit (DeepSeek auto-caches prompt prefixes — no code change needed).
- **~5× faster per call** in practice (`~1.7s` median vs `~7s` on gpt-5-mini for our SQL agent briefs).
- Drop-in OpenAI-compatible client: same `OpenAI(...)` SDK, just a different `base_url`.

## What's different from `backend/`

| File / area | `backend/` | `backend2/` |
|---|---|---|
| Model | `gpt-5-mini` | `deepseek-v4-flash` |
| Structured outputs | `client.chat.completions.parse(response_format=PydanticModel)` | JSON mode + manual Pydantic validation |
| Helper module | (uses OpenAI client directly in each agent) | `agents/_llm.py` — shared `structured(...)` wrapper |
| Prompts | Identical to backend2 | Identical to backend |
| Pipeline shape | Identical | Identical |

## How structured output works without OpenAI's `response_format`

DeepSeek's Chat Completions endpoint supports `response_format={"type": "json_object"}` (JSON mode), but **not** OpenAI's strict-schema `response_format=<PydanticClass>` variant. So `agents/_llm.py` wraps it:

```python
def structured(messages, schema):
    schema_json = json.dumps(schema.model_json_schema())
    augmented = [{
        "role": "system",
        "content": messages[0]["content"] +
                   "\n\nRespond with valid JSON only matching this schema:\n" + schema_json,
    }, *messages[1:]]
    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=augmented,
        response_format={"type": "json_object"},
    )
    return schema(**json.loads(response.choices[0].message.content))
```

Every agent that previously called `client.chat.completions.parse(response_format=Foo)` now calls `structured(messages, schema=Foo)` instead — same typed output, different mechanism underneath.

## Caching (automatic on DeepSeek)

DeepSeek caches prompt prefixes on disk automatically. Cache hits cost ~1/50th of cache misses. No code change — just keep the system prompt byte-identical across calls (no timestamps / random IDs in the system block). Verify with:

```python
response.usage.prompt_cache_hit_tokens
response.usage.prompt_cache_miss_tokens
```

## Setup

```bash
cd backend2
cp example.env .env       # fill in DEEPSEEK_API_KEY + PERPLEXITY_API_KEY
uv sync

# Point at the warehouse built by the main backend:
ln -s ../backend/data data

uv run fastapi dev main.py
```

The frontend doesn't need to change — same `/api/chat` endpoint, same SSE streaming contract.

## Status

Experimental. The production demo runs on `backend/` (OpenAI). `backend2/` is kept as a working reference for the DeepSeek port — same architecture, cheaper + faster, with a small amount of extra code to deal with JSON-mode round-tripping instead of strict-schema parsing.
