# HiveMind — Backend (Autonomous Multi-Agent Swarm)

An enterprise-grade, **LangGraph-powered Multi-Agent Hive Mind** backend. Unlike basic conversational chatbots that rely on linear prompt chains, HiveMind orchestrates **10 specialized departments** and **24+ autonomous agents** with a central CEO Router, universal Vector RAG, shared execution sandboxes, and native tool integrations for **GitHub** and **Gmail**.

---

## 🚀 What Does the Backend Do?

1. **Autonomous Mission Orchestration**: Deconstructs complex user prompts into multi-step execution plans across domain-specific agent teams.
2. **LangGraph State Machine**: Coordinates non-linear inter-department collaboration, conditional loops, and real-time event streaming.
3. **Universal RAG & Document Intelligence**: Ingests PDFs, Excel, CSVs, and GitHub repos into **Qdrant Cloud** with dense embeddings and hybrid search.
4. **Developer & Tool Automations**:
   - **GitHub Ops**: Inspects repo trees, reads source files, commits code, and opens Pull Requests.
   - **Gmail Ops**: Triages unread emails, searches invoice threads, and composes AI-assisted replies.
   - **Code Sandbox**: Executes generated Python / JS code in an isolated environment.
5. **Real-Time Streaming**: Delivers live step-by-step thinking logs and agent events to the client over **WebSockets**.

---

## 🛠️ Tech Stack

| Layer | Technology |
| :--- | :--- |
| **Framework** | [FastAPI](https://fastapi.tiangolo.com/) (Python 3.11+) |
| **Multi-Agent Orchestrator** | [LangGraph](https://python.langchain.com/docs/langgraph/) & [LangChain](https://python.langchain.com/) |
| **LLM Support** | Google Gemini (2.0 / 1.5), Groq (Llama 3.3 70B), OpenAI (GPT-4o) |
| **Vector Database (RAG)** | [Qdrant Cloud](https://qdrant.tech/) (Hybrid Dense + Sparse Search) |
| **Embeddings** | HuggingFace Inference API & FastEmbed Tokenizer |
| **Primary Database & Auth** | [Supabase](https://supabase.com/) (PostgreSQL + Row-Level Security) |
| **Web Search & Financials** | Tavily Web Search, DuckDuckGo, Yahoo Finance (`yfinance`) |
| **Integrations** | GitHub REST API (`PyGithub`), Google Gmail API |
| **Real-Time Transport** | FastAPI Native WebSockets (`/ws/{task_id}`) |

---

## 🏛️ Swarm Architecture & Departments

The system follows a hierarchical yet dynamically interconnected state graph:

```
                      ┌───────────────────────────────┐
                      │    FastAPI Gateway & Router   │
                      │    (REST API & WebSockets)    │
                      └───────────────┬───────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │      CEO Orchestrator         │
                      │  (Intent Analysis & Planning) │
                      └───────────────┬───────────────┘
                                      │
   ┌──────────────┬──────────────┬────┴─────────┬──────────────┬──────────────┐
   ▼              ▼              ▼              ▼              ▼              ▼
┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐┌──────────────┐
│  Code & Git  ││ Finance & P&L││ Sales & Gmail││ Document RAG ││ Research & QA││ Legal/Design │
│ 4 Sub-Agents ││ 4 Sub-Agents ││ 4 Sub-Agents ││ 3 Sub-Agents ││ 3 Sub-Agents ││ 6 Sub-Agents │
└──────────────┘└──────────────┘└──────────────┘└──────────────┘└──────────────┘└──────────────┘
   │              │              │              │              │              │
   └──────────────┴──────────────┴────┬─────────┴──────────────┴──────────────┘
                                      │
                                      ▼
                      ┌───────────────────────────────┐
                      │       Aggregator Node         │
                      │  (Executive Final Synthesis)  │
                      └───────────────────────────────┘
```

### Department Directory:
- **👑 Executive**: `CEO Agent`, `Mission Router`, `Aggregator Node` (Consolidates multi-department findings).
- **💻 Code Engineering**: `Architecture Lead`, `Full-Stack Developer`, `QA Analyst`, `GitHub Ops Agent`.
- **📈 Financial Intelligence**: `Market Analyst` (Yahoo Finance), `Financial Modeler` (3-Year P&L), `SWOT Analyst`, `Pitch Deck Architect`.
- **📬 Sales & Growth**: `Gmail Ops Agent` (Inbox triage), `GTM Lead`, `B2B Copywriter`, `Follow-Up Sequencer`.
- **📄 Document RAG**: `Ingestion & Chunker Agent`, `Qdrant Retrieval Engine`, `Citation Grounding Agent`.
- **🌐 Research & Web**: `Web Intelligence Agent` (Tavily/DDG), `Fact Verifier`, `ArXiv Preprint Researcher`.
- **⚖️ Legal & Compliance**: `Contract Reviewer`, `ToS Drafter`, `Compliance Auditor`.
- **🎨 Brand & Design**: `Brand Identity Guide`, `Logo Concept Architect`, `Pitch Deck Visuals`.

---

## 📂 Project Structure

```bash
backend/
├── api/
│   ├── main.py               # FastAPI entry point, CORS config, routes, WebSocket handler
│   └── routes/               # Modular API routes (chat, models, documents, admin)
├── core/
│   ├── graph.py              # LangGraph StateGraph assembly & inter-department routing
│   ├── state.py              # SwarmState type definition (messages, task_id, artifacts)
│   ├── llm.py                # Dynamic LLM provider factory (Gemini, Groq, OpenAI)
│   └── aggregator.py         # Multi-agent synthesis & executive briefing node
├── departments/
│   ├── code/                 # Code generation, sandbox execution, debugging
│   ├── financial/            # Yahoo Finance feeds, valuation models, market metrics
│   ├── sales/                # Outreach copy, sequence automation, lead research
│   ├── document/             # Document QA & context-bounded RAG query node
│   ├── research/             # Tavily/DDG search, fact checking, ArXiv synthesis
│   ├── strategy/             # Business plan generator, 9-slide pitch decks, SWOT
│   ├── legal/                # Contract clause risk analyzer, compliance checklists
│   └── design/               # Brand guidelines, typography, visual directions
├── connectors/
│   ├── github_connector.py   # GitHub API (tree analysis, file inspection, PR creation)
│   ├── gmail_connector.py    # Gmail API (unread triage, message search, draft composer)
│   └── document.py           # Parsing for PDF, CSV, XLSX, DOCX, TXT
├── rag/
│   ├── config.py             # Qdrant collection settings & embedding model params
│   ├── embedder.py           # HuggingFace Endpoint Embeddings wrapper
│   ├── chunker.py            # Recursive & semantic text chunking strategies
│   └── store.py              # Qdrant client connection & vector search methods
├── db/
│   └── supabase_client.py    # Supabase DB operations (tasks, events, user history)
├── shared/
│   └── tools/                # Shared tools (code execution sandbox, image gen, web search)
├── requirements.txt          # Python dependencies
└── .env.example              # Environment variables template
```

---

## 🔌 API & WebSocket Endpoints

| Method | Route | Description |
| :--- | :--- | :--- |
| `POST` | `/api/chat` | Initiates an autonomous swarm mission; returns a `task_id` |
| `WS` | `/ws/{task_id}` | Real-time WebSocket connection streaming agent steps & final output |
| `GET` | `/api/task/{task_id}` | Polling fallback returning task status, events, and results |
| `GET` | `/api/models` | Returns available model providers, default registry, and capabilities |
| `GET` | `/api/conversations` | Lists user conversation history threads |
| `GET` | `/api/documents` | Lists all indexed RAG documents for the authenticated user |
| `POST` | `/api/upload` | Ingests a local file (PDF/Excel/Word) into the Qdrant vector store |
| `POST` | `/api/github/ingest` | Clones and indexes a public/private GitHub repository into RAG |
| `DELETE`| `/api/documents/{id}`| Deletes an indexed document from both Qdrant and Supabase |
| `GET` | `/api/admin/tasks` | Telemetry endpoint for task queue status, latency, and agent load |

---

## ⚙️ Environment Variables

Create a `.env` file inside `backend/` with the following:

```env
# ─── Database & Auth (Mandatory) ──────────────────────────────
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-supabase-service-role-or-anon-key

# ─── Vector DB & Embeddings (RAG) ─────────────────────────────
HF_TOKEN=your_huggingface_inference_token
QDRANT_URL=https://your-cluster-id.aws.cloud.qdrant.io:6333
QDRANT_API_KEY=your_qdrant_api_key

# ─── System Tools & Search ────────────────────────────────────
TAVILY_API_KEY=your_tavily_api_key
E2B_API_KEY=your_optional_e2b_key

# ─── Fallback LLM Keys (Users can also supply their own in UI) ─
GEMINI_API_KEY=your_gemini_key
GROQ_API_KEY=your_groq_key
OPENAI_API_KEY=your_openai_key

# ─── Deployment & CORS ────────────────────────────────────────
FRONTEND_URL=https://your-app.vercel.app
```

---

## 🏃 Getting Started (Local Development)

```bash
# 1. Navigate to backend directory
cd backend

# 2. Create and activate a Python virtual environment
python -m venv myenv
source myenv/bin/activate    # On Windows: myenv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start development server with hot-reload
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Server will be live at `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`.

