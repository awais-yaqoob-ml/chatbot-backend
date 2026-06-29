import uuid
from datetime import datetime

import requests
import streamlit as st

st.set_page_config(
    page_title="Company Chatbot Demo",
    page_icon="💬",
    layout="wide",
)

API_URL = st.sidebar.text_input(
    "API URL",
    value="http://localhost:8000",
    key="api_url",
)

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []

st.sidebar.title("Company Chatbot")
st.sidebar.write(
    f"Session: `{st.session_state.session_id[:8]}...`"
)

if st.sidebar.button("New Chat", use_container_width=True):
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.rerun()

try:
    resp = requests.get(
        f"{API_URL}/api/v1/documents", timeout=5
    )
    if resp.ok:
        docs = resp.json().get("documents", [])
        if docs:
            st.sidebar.subheader("Ingested Documents")
            for doc in docs:
                st.sidebar.write(f"- {doc.get('filename')}")
except Exception:
    pass

st.sidebar.divider()
st.sidebar.caption(
    "Built with FastAPI + LangGraph + Weaviate + Groq"
)

# --- Chat ---

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "agent" in msg and msg["agent"]:
            st.caption(f"Agent: {msg['agent']}")
        for img in msg.get("images", []):
            st.image(img["data"], caption=img["filename"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.write(
                        f"- {s.get('filename')} (page {s.get('page_number')}, "
                        f"score: {s.get('score', 0):.2f})"
                    )

if prompt := st.chat_input("Ask a question about the company..."):
    st.session_state.messages.append(
        {"role": "user", "content": prompt}
    )
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                resp = requests.post(
                    f"{API_URL}/api/v1/chat",
                    json={
                        "session_id": st.session_state.session_id,
                        "message": prompt,
                    },
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()

                answer = data.get("answer", "")
                agent = data.get("agent_used", "")
                sources = data.get("sources", [])
                images = data.get("images", [])

                st.markdown(answer)
                if agent:
                    st.caption(f"Agent: {agent}")
                for img in images:
                    st.image(img["data"], caption=img["filename"])
                if sources:
                    with st.expander("Sources"):
                        for s in sources:
                            st.write(
                                f"- {s.get('filename')} (page {s.get('page_number')}, "
                                f"score: {s.get('score', 0):.2f})"
                            )

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "agent": agent,
                        "sources": sources,
                        "images": images,
                    }
                )

            except requests.exceptions.ConnectionError:
                st.error(
                    f"Could not connect to {API_URL}. "
                    "Make sure the backend is running."
                )
            except requests.exceptions.Timeout:
                st.error("Request timed out. Please try again.")
            except Exception as e:
                st.error(f"Error: {e}")
