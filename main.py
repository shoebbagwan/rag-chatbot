import glob
import sys
from dotenv import load_dotenv

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()


def main():
    print("🚀 RAG Chatbot Starting...")
    print("=" * 50)

    from ingest import add_pdfs, DATA_DIR, index_exists

    pdfs = glob.glob(f"{DATA_DIR}/*.pdf")
    if not pdfs and not index_exists():
        print(f"⚠️  Put at least one PDF in {DATA_DIR}/ and run again.")
        return

    if pdfs:
        result = add_pdfs(pdfs)  # already-indexed files are skipped automatically
        if result["added"]:
            print(f"✅ Indexed: {', '.join(result['added'])}")
        if result["no_text"]:
            print(f"⚠️  No extractable text (scanned?): {', '.join(result['no_text'])}")

    from chat import chat
    chat()


if __name__ == "__main__":
    main()