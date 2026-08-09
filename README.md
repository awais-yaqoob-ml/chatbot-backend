# 🧠 Multi-Agent Company Knowledge Chatbot (FastAPI + LangGraph + RAG)

A production-style **multi-agent RAG chatbot backend** built with:

- FastAPI (async backend)
- LangGraph (agent orchestration)
- Groq LLM (Llama 3 70B)
- SentenceTransformers (embeddings)
- Weaviate (embedded vector DB)
- PyMuPDF + python-docx (document ingestion)

---

## 🚀 Features

- 📄 Upload PDF / DOCX documents
- 🧠 Extract text + images from documents (position-aware parsing)
- 🖼 Answers embed only relevant screenshots inline — rendered in the React frontend
- 🔍 Vector search with Weaviate
- 🤖 Multi-agent routing (LangGraph) with LLM intent classification
  - GreetingAgent, SummarizationAgent, CompanyInfoAgent
  - CompanyQAAgent (RAG-based), FallbackAgent, ErrorHandler
- 💬 Persistent chat history (Weaviate)
- ⚡ FastAPI async API
- 🧩 Modular production-grade architecture

---

## 🏗 Architecture

```bash
User → FastAPI → LangGraph
│
├── intent_classifier (LLM intent detection)
│   ├── GREETING → greeting_agent
│   ├── SUMMARIZATION → summarization_agent
│   ├── COMPANY_INFO → company_info_agent
│   ├── COMPANY_QA → rewrite_query → retrieve_node → company_qa_agent
│   └── FALLBACK → fallback_agent
```

---

## 📁 Project Structure

```bash
backend/
├── core/            # Config, logging, model loading, Weaviate client
├── graph/           # LangGraph state, nodes, edges, graph builder
├── models/          # Pydantic schemas, Weaviate collection definitions
├── services/        # Business logic (LLM, embeddings, retrieval, ingestion, etc.)
├── routers/         # FastAPI route handlers (chat, ingestion, history)
├── utils/           # Chunking, document parsers, file utils
├── tests/           # pytest unit tests
├── frontend/        # React + Vite SPA (and Streamlit alternative)
├── main.py          # FastAPI entry point
```

---

## ⚙️ Setup

### 1. Install dependencies
```bash
pip install -r requirements.txt
```
### 2. Configure environment
```bash
# set environment variables for example
GROQ_API_KEY=your_key_here
```
### 3. Run backend server
```bash
uvicorn main:app --reload --port 8000
```
### 4. Run frontend (React)
```bash
cd frontend/react-app && npm install && npm run dev
# → http://localhost:5173 (proxies /api to backend)
```
### 5. Open API docs
```bash
http://localhost:8000/docs
```
### 6. 📡 API Endpoints

### Ingestion
```bash
# Upload PDF/DOCX document.
POST /api/v1/ingest
```
### Chat
```bash
POST /api/v1/chat
```
### Request:
```bash
{
  "session_id": "uuid",
  "message": "What is HR policy?"
}
```
### Response:
```bash
{
  "answer": "...",
  "agent_used": "CompanyQAAgent",
  "sources": [],
  "images": [
    {
      "filename": "page_21_img_0.png",
      "mime_type": "image/jpeg",
      "data": "data:image/jpeg;base64,...."
    }
  ]
}
```
`images` contains the screenshots referenced by the answer, resolved to base64
data-URIs (with MIME type detected from file magic bytes).
### History
```bash
GET /api/v1/history/{session_id}
DELETE /api/v1/history/{session_id}
GET /api/v1/sessions
```
### Health Check
```bash
GET /health
```
### 🧠 Agent Logic
- **GreetingAgent** — Static greeting/small talk response
- **SummarizationAgent** — LLM summary of conversation history
- **CompanyInfoAgent** — Answers from consolidated company profile
- **CompanyQAAgent** — RAG answer from retrieved document chunks with source citations
- **FallbackAgent** — Polite decline for out-of-domain queries
- **ErrorHandler** — Generic error response on failures

### 🗄 Data Storage
- Weaviate Collections
- **DocumentChunk** — Vector + text chunks with hybrid search (BM25 + vector)
- **ChatHistory** — Persistent conversation storage per session
- **Document** — Per-document metadata, summary, topics
- **CompanyProfile** — Consolidated company overview from all documents

### 🖼 Image Handling
- **Position-aware PDF parsing** — each page is rebuilt line-by-line from
  `page.get_text("words")` and merged with image tags sorted by y/x position, so
  `![Image](page_X_img_Y.png)` tags sit inline next to the paragraph they belong to
  (instead of a blob at the end of the page).
- **Per-page rendering detection** — uses `page.get_image_info(xrefs=True)` so only
  images actually drawn on a page are tagged (shared/inherited xrefs are ignored).
- **Decorative-image filtering** — skips tiny icons (< 100 px in both dimensions) and
  images whose xref renders on multiple pages (repeated logos / watermarks).
- **Answer image resolution** — `resolve_images` scans the LLM answer for image tags,
  resolves them to base64 data-URIs served with the correct MIME type (detected from
  magic bytes, so JPEG content named `.png` works), and drops unresolved references.
- **React rendering** — the frontend renders `msg.images` inline under the assistant
  bubble (`frontend/react-app/src/App.tsx`).

### 🔧 Tech Stack
- FastAPI (async backend)
- LangGraph (agent orchestration)
- Groq LLM (gpt-oss-120b)
- Qwen3-Embedding-0.6B (1024-dim embeddings via SentenceTransformers)
- Weaviate v4 (vector DB with hybrid search)
- PyMuPDF (PDF parsing)
- python-docx (DOCX parsing)
- React 18 + Vite + TypeScript (frontend SPA)
📌 Notes
---

## TO DO...
- Add answer validation agent
- Add reranking model (cross-encoder)
- Upgrade embedding model (0.6B → 4B) for better accuracy
- Add streaming responses (SSE/WebSockets)
- Support for more document formats
---