import os
import sys
from dotenv import load_dotenv

# Reconfigure stdout for UTF-8 to prevent Windows UnicodeEncodeError with emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

def main():
    print("🚀 RAG Chatbot Starting...")
    print("=" * 50)

    # Check if vector store already exists
    if not os.path.exists("faiss_index"):
        print("⚠️  No vector store found! Running ingestion first...")
        from ingest import ingest_pdf
        pdf_path = "data/how to talk anyone.pdf"
        ingest_pdf(pdf_path)
    else:
        print("✅ Vector store found! Skipping ingestion...")

    # Start the chatbot
    from chat import chat
    chat()

if __name__ == "__main__":
    main()