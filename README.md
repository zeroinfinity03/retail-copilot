# Retail Insights Copilot

A multi-agent assistant for retail analytics. An internal analyst types a question in plain English; a supervisor agent decomposes it into sub-tasks, dispatches specialist agents in parallel, and a synthesizer composes a grounded, citation-backed report with an optional interactive chart.

> 🔗 **Live system design walkthrough:** <https://zeroinfinity03.github.io/retail-copilot/design/systemdesign.html>
>
> Or open the file locally: [`design/systemdesign.html`](design/systemdesign.html)
>
> That single page is the complete walkthrough — architecture, data, every agent in depth, with diagrams and code snippets.

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
uv sync                   # still inside backend/
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

## Repository layout

```
project/
├── backend/         FastAPI server, agents, prompts, scripts, DuckDB warehouse
├── frontend/        React + Vite chat UI
└── design/          System design walkthrough (open systemdesign.html)
```

---

## Session memory (optional)

The pipeline is stateless by default — `main.py` reads only the latest user message and ignores the rest of the conversation, so follow-up questions like *"now break that down by category"* cannot resolve references. To enable session memory (memory that lasts as long as the browser tab is open), pass the full message history into the supervisor as context. One small change in `main.py` is enough — no agent needs to be modified.

See the **"Adding session-level memory (optional)"** section at the bottom of [`design/systemdesign.html`](design/systemdesign.html) for the exact before / after code.

## Long-term memory (optional)

For conversations that survive across sessions, browser refreshes, and server restarts, use LangGraph's built-in SQLite checkpointer. Every plan, specialist result, and final report is persisted per `thread_id`. No extra database server — SQLite is embedded and the checkpoint file lives alongside `hm.duckdb` on the same local disk. Swap to `PostgresSaver` later if you need to scale to multiple server instances.

See the **"Long-term memory across sessions (LangGraph checkpointer)"** section at the bottom of [`design/systemdesign.html`](design/systemdesign.html) for the code change and deployment story.

---

## License

MIT — see [LICENSE](LICENSE) for the full text. Use, modify, and distribute freely.
