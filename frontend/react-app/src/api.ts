import type { ChatResponse, DocumentListResponse } from "./types";

export async function sendMessage(
  sessionId: string,
  message: string
): Promise<ChatResponse> {
  const res = await fetch("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || `HTTP ${res.status}`);
  }
  return res.json();
}

export async function listDocuments(): Promise<DocumentListResponse> {
  const res = await fetch("/api/v1/documents");
  return res.json();
}
