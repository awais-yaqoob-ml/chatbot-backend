import { useState, useEffect, useCallback, useRef } from "react";
import type { Message, DocumentInfo } from "./types";
import { sendMessage, listDocuments } from "./api";
import Markdown from "./Markdown";

function genId(): string {
  return crypto.randomUUID();
}

export default function App() {
  const [sessionId, setSessionId] = useState(genId);
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [documents, setDocuments] = useState<DocumentInfo[]>([]);
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listDocuments()
      .then((res) => setDocuments(res.documents))
      .catch(() => {});
  }, []);

  useEffect(() => {
    listRef.current?.scrollTo(0, listRef.current.scrollHeight);
  }, [messages]);

  const handleSend = useCallback(async () => {
    const text = input.trim();
    if (!text || loading) return;
    setInput("");

    const userMsg: Message = { role: "user", content: text };
    setMessages((prev) => [...prev, userMsg]);
    setLoading(true);

    try {
      const res = await sendMessage(sessionId, text);
      const assistantMsg: Message = {
        role: "assistant",
        content: res.answer,
        agent: res.agent_used,
        sources: res.sources,
        images: res.images,
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch {
      const errMsg: Message = {
        role: "assistant",
        content: "Error: could not reach the server. Make sure the backend is running.",
      };
      setMessages((prev) => [...prev, errMsg]);
    }
    setLoading(false);
  }, [input, loading, sessionId]);

  const handleNewChat = () => {
    setSessionId(genId());
    setMessages([]);
  };

  return (
    <div className="app">
      <aside className="sidebar">
        <h2>Company Chatbot</h2>
        <div className="sidebar-section">
          <label>Session</label>
          <code>{sessionId.slice(0, 8)}...</code>
        </div>
        <button className="new-chat-btn" onClick={handleNewChat}>
          New Chat
        </button>

        {documents.length > 0 && (
          <div className="sidebar-section">
            <label>Documents</label>
            <ul>
              {documents.map((d) => (
                <li key={d.doc_id}>{d.filename}</li>
              ))}
            </ul>
          </div>
        )}

        <p className="sidebar-footer">
          FastAPI + LangGraph + Weaviate + Groq
        </p>
      </aside>

      <main className="chat">
        <div className="message-list" ref={listRef}>
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`message ${msg.role === "user" ? "user" : "assistant"}`}
            >
                <div className="bubble">
                  <Markdown text={msg.content} />

                {msg.images && msg.images.length > 0 && (
                  <div className="assistant-images">
                    {msg.images.map((img, k) => (
                      <img
                        key={k}
                        src={`data:${img.mime_type};base64,${img.data}`}
                        alt={img.filename}
                      />
                    ))}
                  </div>
                )}

                {msg.agent && (
                  <span className="agent-tag">{msg.agent}</span>
                )}

                {msg.sources && msg.sources.length > 0 && (
                  <details className="sources">
                    <summary>Sources ({msg.sources.length})</summary>
                    <ul>
                      {msg.sources.map((s, j) => (
                        <li key={j}>
                          {s.filename} (page {s.page_number}, score{" "}
                          {s.score.toFixed(2)})
                        </li>
                      ))}
                    </ul>
                  </details>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="message assistant">
              <div className="bubble typing">Thinking...</div>
            </div>
          )}
        </div>

        <div className="input-bar">
          <input
            type="text"
            placeholder="Ask a question about the company..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            disabled={loading}
          />
          <button onClick={handleSend} disabled={loading || !input.trim()}>
            Send
          </button>
        </div>
      </main>
    </div>
  );
}
