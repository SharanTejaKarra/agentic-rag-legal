"""
Streamlit app — chat interface for querying legal documents.
Run: streamlit run app.py
"""

import streamlit as st

from src.agent import run as agent_run
from src.models import AgentResponse

# ── page setup ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Legal Document Q&A",
    page_icon="\u2696\ufe0f",
    layout="wide",
)

# jurisdiction tag colors
_JUR_COLORS = {
    "alaska": "#1f77b4",   # blue
    "hawaii": "#2ca02c",   # green
}


def _jurisdiction_badge(name: str) -> str:
    """Return a small colored HTML badge for a jurisdiction."""
    color = _JUR_COLORS.get(name.lower(), "#888")
    return (
        f'<span style="background:{color};color:#fff;padding:2px 8px;'
        f'border-radius:4px;font-size:0.8em;">{name.title()}</span>'
    )


# ── sidebar ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("Legal Document Q&A")
    st.markdown("Ask questions about Alaska and Hawaii legal documents.")

    st.divider()

    jurisdiction = st.radio(
        "Jurisdiction filter",
        options=["All", "Alaska", "Hawaii"],
        index=0,
    )

    with st.expander("About"):
        st.markdown(
            """
            This system uses **agentic RAG** (Retrieval-Augmented Generation)
            to answer questions over Alaska and Hawaii legal documents.

            It combines dense vector search, BM25 keyword search, cross-encoder
            reranking, and a local LLM to produce grounded answers with citations.
            """
        )

    st.divider()
    st.caption(
        "No results? Make sure you've run `python ingest.py` to build the indexes, "
        "and that Ollama is running locally."
    )


# ── chat state ────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []


def _display_message(role: str, content: str) -> None:
    """Render a single chat bubble."""
    with st.chat_message(role):
        st.markdown(content)


def _display_sources(response: AgentResponse) -> None:
    """Show citations and retrieved chunks below the answer."""
    if response.citations:
        with st.expander("Sources & Citations", expanded=False):
            for cite in response.citations:
                jur_badge = _jurisdiction_badge(cite.get("jurisdiction", ""))
                st.markdown(
                    f"**{cite.get('section_id', 'N/A')}** — "
                    f"{cite.get('title', 'Untitled')}  \n"
                    f"File: `{cite.get('source_file', '?')}`  {jur_badge}",
                    unsafe_allow_html=True,
                )

            if response.retrieved_chunks:
                st.markdown("---")
                st.markdown("**Retrieved passages**")
                for rr in response.retrieved_chunks:
                    chunk = rr.chunk
                    label = f"{chunk.section_id} (score: {rr.score:.3f})"
                    with st.expander(label, expanded=False):
                        st.text(chunk.text)

    if response.query_reformulations:
        st.info(
            "Query reformulations: "
            + " | ".join(response.query_reformulations)
        )


# ── render previous messages ─────────────────────────────────────────────
for msg in st.session_state.messages:
    _display_message(msg["role"], msg["content"])
    if "response" in msg:
        _display_sources(msg["response"])


# ── handle new input ─────────────────────────────────────────────────────
user_input = st.chat_input("Ask a question about Alaska or Hawaii law ...")

if user_input:
    # show and store user message
    _display_message("user", user_input)
    st.session_state.messages.append({"role": "user", "content": user_input})

    # build jurisdiction filter if needed
    jur_filter = None
    if jurisdiction != "All":
        jur_filter = jurisdiction.lower()

    # query the agent
    with st.spinner("Searching documents..."):
        try:
            response: AgentResponse = agent_run(user_input, jurisdiction=jur_filter)
        except ConnectionError:
            st.error(
                "Could not connect to Ollama. Make sure it's running "
                "(`ollama serve`) and the model is pulled."
            )
            st.stop()
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.stop()

    # display assistant answer
    _display_message("assistant", response.answer)
    _display_sources(response)

    # persist to session state for reruns
    st.session_state.messages.append({
        "role": "assistant",
        "content": response.answer,
        "response": response,
    })
