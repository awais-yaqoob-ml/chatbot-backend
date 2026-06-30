export interface ChatRequest {
  session_id: string;
  message: string;
}

export interface SourceDoc {
  filename: string;
  page_number: number;
  chunk_index: number;
  score: number;
}

export interface ResponseImage {
  filename: string;
  mime_type: string;
  data: string;
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  agent_used: string;
  sources: SourceDoc[];
  images: ResponseImage[];
  timestamp: string;
}

export interface DocumentInfo {
  doc_id: string;
  filename: string;
  doc_summary: string;
  topics: string[];
  uploaded_at: string;
  chunk_count: number;
}

export interface DocumentListResponse {
  documents: DocumentInfo[];
  total: number;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  agent?: string;
  sources?: SourceDoc[];
  images?: ResponseImage[];
}
