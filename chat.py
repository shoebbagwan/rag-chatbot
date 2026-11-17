import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from ingest import INDEX_DIR, index_exists, list_sources

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

TOP_K = 5
MAX_DISTANCE = None   # tune after DEBUG run. None = disabled
DEBUG = False
NOT_FOUND = "I couldn't find that in the documents."

ANSWER_PROMPT = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using ONLY the context below.
Each passage starts with its source like [file.pdf, p. 12].
Cite what you used in the answer, like (file.pdf, p. 12).
If the answer is not in the context, reply exactly: "I couldn't find that in the documents."

Context:
{context}

Question: {question}

Answer:
""")

REWRITE_PROMPT = ChatPromptTemplate.from_template("""
Given the chat history and a follow-up question, rewrite the follow-up as a
standalone question that makes sense without the history.
Return ONLY the rewritten question.

History:
{history}

Follow-up: {question}

Standalone question:
""")


def load_vector_store():
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_query",
    )
    return FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)


def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


def page_of(doc):
    return doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)


def format_docs(docs):
    return "\n\n".join(
        f"[{d.metadata.get('source', 'document')}, p. {page_of(d)}]\n{d.page_content}"
        for d in docs
    )


def rewrite_question(question, history, llm):
    """history = list of (role, text) tuples."""
    text = "\n".join(f"{role}: {msg}" for role, msg in history[-6:])
    chain = REWRITE_PROMPT | llm | StrOutputParser()
    return chain.invoke({"history": text, "question": question}).strip()


def ask(question, vector_store, llm, history=None, only_sources=None):
    """
    Returns (answer, sources).
    only_sources: optional list of filenames to restrict the search to.
    """
    history = history or []
    standalone = rewrite_question(question, history, llm) if history else question

    kwargs = {"k": TOP_K}
    if only_sources:
        kwargs["filter"] = {"source": list(only_sources)}
        kwargs["fetch_k"] = TOP_K * 20  # filter runs after the initial fetch

    results = vector_store.similarity_search_with_score(standalone, **kwargs)
    if DEBUG:
        for doc, score in results:
            print(f"  distance={score:.3f} {doc.metadata.get('source')} p.{page_of(doc)}")

    if MAX_DISTANCE is not None:
        results = [(d, s) for d, s in results if s <= MAX_DISTANCE]
    if not results:
        return NOT_FOUND, []

    docs = [d for d, _ in results]
    chain = ANSWER_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({"context": format_docs(docs), "question": standalone})

    if "couldn't find that in the documents" in answer:
        return answer, []

    sources = [
        {
            "source": d.metadata.get("source", "document"),
            "page": page_of(d),
            "snippet": d.page_content[:300],
        }
        for d in docs
    ]
    return answer, sources


def chat():
    if not index_exists():
        print("⚠️  No index found. Run: python ingest.py \"data/your.pdf\"")
        return

    store, llm, history = load_vector_store(), get_llm(), []
    print(f"\n🤖 Ready! Documents: {', '.join(list_sources(store))}")
    print("Type 'exit' to quit")
    print("=" * 50)

    while True:
        question = input("\n💬 You: ").strip()
        if question.lower() == "exit":
            print("👋 Goodbye!")
            break
        if not question:
            continue

        answer, sources = ask(question, store, llm, history)
        print(f"\n🤖 Bot: {answer}")
        if sources:
            refs = sorted({f"{s['source']} p.{s['page']}" for s in sources})
            print(f"📖 Sources: {'; '.join(refs)}")

        history += [("user", question), ("assistant", answer)]


if __name__ == "__main__":
    chat()