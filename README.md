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
- 🧠 Extract text + images from documents
- 🔍 Vector search with Weaviate
- 🤖 Multi-agent routing (LangGraph)
  - Knowledge Agent (RAG-based)
  - General Agent (fallback LLM)
- 💬 Persistent chat history (Weaviate)
- ⚡ FastAPI async API
- 🧩 Modular production-grade architecture

---

## 🏗 Architecture

```bash
User → FastAPI → LangGraph
│
├── retrieve_node (Weaviate search)
├── router_node (score-based decision)
├── knowledge_agent (RAG + Groq)
└── general_agent (LLM only)
```

---

## 📁 Project Structure

```bash
backend/
├── core/
├── graph/
├── models/
├── services/
├── routers/
├── utils/
├── main.py
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
### 3. Run server
```bash
uvicorn main:app --reload --port 8000
```
### 4. Open API docs
```bash
http://localhost:8000/docs
```
### 5. 📡 API Endpoints

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
  "agent_used": "KnowledgeAgent",
  "sources": []
}
```
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
- Knowledge Agent
- Uses retrieved document chunks
- Forces grounded answers
- Requires citations
- General Agent
- Used when retrieval confidence is low
- No document context

### 🗄 Data Storage
- Weaviate Collections
- DocumentChunk
- ChatHistory

### 🔧 Tech Stack
- FastAPI
- LangGraph
- Groq LLM
- SentenceTransformers
- Weaviate v4
- PyMuPDF
- python-docx
📌 Notes
---

## TO DO...
- Add answer validation agent
- Add reranking model (cross-encoder)
- Switch to hybrid search (BM25 + vector)
- Add streaming responses (SSE/WebSockets)
- Add LangSmith tracing
- Support for more document formats
---