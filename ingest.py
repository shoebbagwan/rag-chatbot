import os
import sys
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

# Reconfigure stdout for UTF-8 to prevent Windows UnicodeEncodeError with emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

def ingest_pdf(pdf_path):
    print("📄 Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    print("✂️  Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    print("🔢 Creating embeddings & saving to FAISS...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_document"
    )

    # Process in small batches to avoid rate limits
    batch_size = 10
    vector_store = None

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i+batch_size]
        print(f"  Processing batch {i//batch_size + 1}/{(len(chunks)//batch_size) + 1}...")
        
        success = False
        while not success:
            try:
                if vector_store is None:
                    vector_store = FAISS.from_documents(batch, embeddings)
                else:
                    vector_store.add_documents(batch)
                success = True
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e):
                    print("⏳ Rate limit hit! Waiting 30 seconds...")
                    time.sleep(30)
                else:
                    print(f"❌ Unexpected error: {e}")
                    break
        
        time.sleep(3)  # Small pause between every batch

    vector_store.save_local("faiss_index")
    print("✅ Vector store saved to faiss_index/")

if __name__ == "__main__":
    pdf_path = "data/how to talk anyone.pdf"
    ingest_pdf(pdf_path)