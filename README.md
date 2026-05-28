# Retail Insights Copilot

A multi-agent assistant for retail analytics. An internal analyst types a question in plain English; a supervisor agent decomposes it into sub-tasks, dispatches specialist agents in parallel, and a synthesizer composes a grounded, citation-backed report with an optional interactive chart.

> 🔗 **Live system design walkthrough:** <https://zeroinfinity03.github.io/retail-copilot/design/systemdesign.html>
>
> Or open the file locally: [`design/systemdesign.html`](design/systemdesign.html)
>
> That single page is the complete walkthrough — architecture, data, every agent in depth, with diagrams and code snippets.

---

## Repository layout

```
project/
├── backend/
│   ├── agents/
│   │   ├── supervisor_P.py          plain-Python supervisor (default)
│   │   ├── supervisor_L.py          LangGraph variant (same behaviour)
│   │   ├── sql_agent.py             NL → DuckDB SQL, sandboxed execution
│   │   ├── web_research_agent.py    Perplexity Sonar Pro market research
│   │   ├── forecasting_agent.py     Prophet + SARIMA ensemble
│   │   ├── chart_agent.py           Plotly Express figure spec + render
│   │   └── synthesizer_agent.py     Final narrative composer
│   ├── prompts/
│   │   ├── supervisor.txt
│   │   ├── sql_agent.txt
│   │   ├── web_research_agent.txt
│   │   ├── forecasting_agent.txt
│   │   ├── chart_agent.txt
│   │   └── synthesizer.txt
│   ├── tests/
│   │   ├── test_supervisor.py        Plan / PlanStep schema tests
│   │   ├── test_sql_agent.py         Keyword blocklist + SQLOutput schema
│   │   ├── test_chart_agent.py       ChartSpec + validate_spec checks
│   │   ├── test_forecasting_agent.py ForecastSpec, MAPE math, helpers
│   │   ├── test_web_research_agent.py Mock-based response shape tests
│   │   ├── test_synthesizer.py        Skip logic + format helpers
│   │   ├── test_e2e.py                Full-pipeline smoke (marked slow)
│   │   └── conftest.py
│   ├── scripts/
│   │   └── load_data.py             One-time DuckDB warehouse build
│   ├── data/db/                     DuckDB warehouse (gitignored, built locally)
│   ├── raw_data/                    H&M CSVs (gitignored, downloaded from Kaggle)
│   ├── main.py                      FastAPI server + /api/chat endpoint
│   ├── example.env                  Template for API keys
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── App.jsx                  Chat UI shell
│   │   ├── components/
│   │   │   ├── Message.jsx          Single message (with chart iframe)
│   │   │   ├── Composer.jsx         Input box
│   │   │   └── CodeBlock.jsx        Syntax-highlighted code blocks
│   │   ├── lib/
│   │   │   └── chatStream.js        SSE streaming consumer
│   │   ├── styles.css
│   │   └── main.jsx
│   ├── index.html
│   └── package.json
├── backend2/                        DeepSeek implementation (alternative to backend/)
├── design/
│   └── systemdesign.html            Complete architecture walkthrough
├── LICENSE
└── README.md
```

---

## Setup (one-time)

**1. Clone the repo**

```bash
git clone <repo-url>
cd <repo-folder>
```

**2. Set up API keys**

Copy the template and fill in your real keys:

```bash
cd backend
cp example.env .env       # then edit .env and add your keys
```

You need two keys:
- `OPENAI_API_KEY` — used by every agent that calls an LLM
- `PERPLEXITY_API_KEY` — used by the web research agent (Sonar Pro)

**3. Install Python dependencies**

```bash
uv sync
```

**4. Download the dataset and place the CSVs in `backend/raw_data/`**

- **Dataset:** H&M Personalized Fashion Recommendations
- **Download from:** <https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations/data>
- **Place into:** `backend/raw_data/`  *(create the folder if it doesn't exist)*

After unzipping, `backend/raw_data/` should contain these three files:

```
backend/raw_data/
├── articles.csv
├── customers.csv
└── transactions_train.csv
```

**5. Install frontend dependencies**

```bash
cd ../frontend
npm install
```

---

## Running

Two terminals — backend and frontend.

**Terminal 1 — backend (FastAPI on port 8000):**

```bash
cd backend
uv run fastapi dev main.py
```

The very first time, this also builds the local DuckDB warehouse from the CSVs in `backend/raw_data/` (~2 min, one-time). Every subsequent start skips that and boots in seconds.

**Terminal 2 — frontend (Vite on port 5173):**

```bash
cd frontend
npm run dev
```

Open <http://localhost:5173> and start asking questions.

---

## Testing

Backend tests live in [`backend/tests/`](backend/tests/). 45 unit tests run in under 2 seconds (no LLM calls, no network); 2 end-to-end tests are marked `slow` and only run when you opt in.

```bash
cd backend

# Fast tests only (45 tests, ~2s, no API keys needed):
uv run pytest

# Include the slow end-to-end tests (real LLM + DuckDB):
uv run pytest -m slow

# Everything:
uv run pytest -m "not slow or slow"

# Single agent's tests:
uv run pytest tests/test_chart_agent.py -v
```

Each agent has its own test file plus one shared end-to-end test that drives a real query through the full pipeline.

---

## Session memory (optional)

The pipeline is stateless by default — `main.py` reads only the latest user message and ignores the rest of the conversation, so follow-up questions like *"now break that down by category"* cannot resolve references. To enable session memory (memory that lasts as long as the browser tab is open), pass the full message history into the supervisor as context. One small change in `main.py` is enough — no agent needs to be modified.

**The exact replacement code already sits commented at the bottom of [`backend/main.py`](backend/main.py)** — uncomment the block, replace the single `user_query = user_msgs[-1].content` line inside `chat()` with it, and session memory is on. The HTML walkthrough at [`design/systemdesign.html`](design/systemdesign.html) (section *"Adding session-level memory (optional)"*) explains the why.

## Long-term memory (optional)

For conversations that survive across sessions, browser refreshes, and server restarts, use LangGraph's built-in SQLite checkpointer. Every plan, specialist result, and final report is persisted per `thread_id`. No extra database server — SQLite is embedded and the checkpoint file lives alongside `hm.duckdb` on the same local disk. Swap to `PostgresSaver` later if you need to scale to multiple server instances.

See the **"Long-term memory across sessions (LangGraph checkpointer)"** section at the bottom of [`design/systemdesign.html`](design/systemdesign.html) for the code change and deployment story.

---

## License

MIT — see [LICENSE](LICENSE) for the full text. Use, modify, and distribute freely.
