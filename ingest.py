import os
import sys
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

INDEX_DIR = "faiss_index"
DATA_DIR = "data"
BATCH_SIZE = 10
BATCH_SLEEP = 3
MAX_RETRIES = 5


def get_embeddings(task_type="retrieval_document"):
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type=task_type,
    )


def index_exists():
    return os.path.exists(os.path.join(INDEX_DIR, "index.faiss"))


def list_sources(store):
    """Filenames of all documents currently in the index."""
    return sorted({d.metadata.get("source", "unknown") for d in store.docstore._dict.values()})


def load_and_split(pdf_path):
    pages = PyPDFLoader(pdf_path).load()
    name = os.path.basename(pdf_path)
    for p in pages:
        p.metadata["source"] = name
        p.metadata["page_number"] = p.metadata.get("page", 0) + 1  # 1-indexed

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    return [c for c in chunks if c.page_content.strip()]  # drop empty chunks


def embed_batch(store, batch, embeddings):
    for attempt in range(MAX_RETRIES):
        try:
            if store is None:
                return FAISS.from_documents(batch, embeddings)
            store.add_documents(batch)
            return store
        except Exception as e:
            if "RESOURCE_EXHAUSTED" in str(e):
                print(f"⏳ Rate limit hit, waiting 30s (attempt {attempt + 1}/{MAX_RETRIES})...")
                time.sleep(30)
            else:
                raise
    raise RuntimeError("Gave up after repeated rate-limit errors")


def add_pdfs(pdf_paths, progress_cb=None):
    """
    Adds PDFs to the shared index (creating it if needed).
    progress_cb(filename, batches_done, batches_total) is optional.
    Returns {"added": [...], "duplicate": [...], "no_text": [...]}.
    """
    embeddings = get_embeddings()
    store, existing = None, set()
    if index_exists():
        store = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
        existing = set(list_sources(store))

    result = {"added": [], "duplicate": [], "no_text": []}

    for path in pdf_paths:
        name = os.path.basename(path)
        if name in existing:
            result["duplicate"].append(name)
            continue

        chunks = load_and_split(path)
        if not chunks:
            result["no_text"].append(name)  # likely a scanned PDF
            continue

        total = (len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE
        print(f"📄 {name}: {len(chunks)} chunks")
        for i in range(0, len(chunks), BATCH_SIZE):
            store = embed_batch(store, chunks[i:i + BATCH_SIZE], embeddings)
            done = i // BATCH_SIZE + 1
            print(f"  Batch {done}/{total}")
            if progress_cb:
                progress_cb(name, done, total)
            time.sleep(BATCH_SLEEP)

        store.save_local(INDEX_DIR)  # save after each file so progress isn't lost
        result["added"].append(name)

    return result


if __name__ == "__main__":
    paths = sys.argv[1:]
    if not paths:
        print('Usage: python ingest.py "data/a.pdf" "data/b.pdf"')
    else:
        print(add_pdfs(paths))