import os
import sys
from dotenv import load_dotenv

# Reconfigure stdout for UTF-8 to prevent Windows UnicodeEncodeError with emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

try:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    e = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
    result = e.embed_query("test")
    print("✅ Embeddings working! Length:", len(result))
except Exception as ex:
    print("❌ Error:", str(ex))