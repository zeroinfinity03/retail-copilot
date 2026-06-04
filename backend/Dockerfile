# syntax=docker/dockerfile:1
#
# Retail Insights Copilot, backend (FastAPI + uvicorn).
#
# Multi-stage build:
#   - Stage 1 (build)  : compiles the heavy wheels. prophet, statsforecast and
#                        darts need a build toolchain and produce a large set of
#                        packages, so we isolate them here.
#   - Stage 2 (runtime): a clean slim image with only the prebuilt venv + the
#                        app source. No compiler ships in the final image.
#
# The DuckDB warehouse (~1.1 GB) is deliberately NOT baked in. It is excluded
# via .dockerignore and volume-mounted at runtime:
#
#   docker build -t retail-copilot-api .
#   docker run --rm -p 8000:8000 \
#     --env-file .env \
#     -v "$PWD/data/db:/app/data/db" \
#     retail-copilot-api
#
# (Build context is this backend/ directory.)

# ---- Stage 1: build the dependency venv ----
FROM python:3.12-slim AS build
WORKDIR /app

# build-essential: needed to compile prophet (cmdstan) / statsforecast (numba) wheels
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential \
 && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

# Copy only the lockfiles first so this (slow) layer is cached unless deps change.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project

# ---- Stage 2: clean runtime ----
FROM python:3.12-slim AS runtime
WORKDIR /app

# libgomp1: OpenMP runtime that numpy / scipy / statsforecast load at import time.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

# Prebuilt virtualenv from the build stage, then the app source.
COPY --from=build /app/.venv /app/.venv
COPY . .

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8000

# The app exposes GET /health; use it so the container reports readiness.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
