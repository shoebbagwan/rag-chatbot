import os
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
try:  # Streamlit Cloud stores the key in secrets, not .env
    if "GOOGLE_API_KEY" in st.secrets:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

from chat import load_vector_store, get_llm, ask
from ingest import add_pdfs, list_sources, index_exists, DATA_DIR

st.set_page_config(page_title="RAG Chatbot", page_icon="📚", layout="centered")
st.title("📚 Chat with your PDFs")
st.caption("Powered by Gemini + LangChain + FAISS")


@st.cache_resource
def cached_llm():
    return get_llm()


if "messages" not in st.session_state:
    st.session_state.messages = []
if "store" not in st.session_state:
    st.session_state.store = load_vector_store() if index_exists() else None


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"📖 Sources ({len(sources)} passages)"):
        for s in sources:
            st.markdown(f"**{s['source']} — page {s['page']}**")
            st.caption(s["snippet"] + "…")


# ---------- Sidebar: documents ----------
with st.sidebar:
    st.header("📄 Documents")
    uploaded = st.file_uploader("Add PDFs", type="pdf", accept_multiple_files=True)

    if st.button("Add to knowledge base", disabled=not uploaded):
        os.makedirs(DATA_DIR, exist_ok=True)
        paths = []
        for f in uploaded:
            path = os.path.join(DATA_DIR, os.path.basename(f.name))
            with open(path, "wb") as out:
                out.write(f.getbuffer())
            paths.append(path)

        bar = st.progress(0.0, text="Starting...")

        def on_progress(name, done, total):
            bar.progress(done / total, text=f"{name}: {done}/{total}")

        try:
            result = add_pdfs(paths, progress_cb=on_progress)
            st.session_state.store = load_vector_store()
            if result["added"]:
                st.success(f"Added: {', '.join(result['added'])}")
            if result["duplicate"]:
                st.info(f"Already indexed: {', '.join(result['duplicate'])}")
            if result["no_text"]:
                st.warning(f"No readable text (scanned?): {', '.join(result['no_text'])}")
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                st.error("Gemini rate limit hit. Wait a minute and try again.")
            else:
                st.error(f"Indexing failed: {str(e)[:200]}")
        finally:
            bar.empty()

    store = st.session_state.store
    selected = None
    if store:
        docs = list_sources(store)
        st.markdown(f"**Indexed ({len(docs)}):**")
        selected = st.multiselect("Search in", docs, default=docs)
        if set(selected) == set(docs) or not selected:
            selected = None  # None = search everything

    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Main chat ----------
if not st.session_state.store:
    st.info("👈 Upload one or more PDFs in the sidebar to get started.")
    st.stop()

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        show_sources(m.get("sources"))

if question := st.chat_input("Ask something about your documents..."):
    with st.chat_message("user"):
        st.markdown(question)

    history = [(m["role"], m["content"]) for m in st.session_state.messages]

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching..."):
            try:
                answer, sources = ask(
                    question, st.session_state.store, cached_llm(), history, selected
                )
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    st.error("Gemini rate limit hit. Wait a minute and retry.")
                else:
                    st.error(f"Something went wrong: {str(e)[:200]}")
                st.stop()
        st.markdown(answer)
        show_sources(sources)

    st.session_state.messages += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer, "sources": sources},
    ]