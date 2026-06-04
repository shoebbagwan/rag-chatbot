import os
import sys
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Reconfigure stdout for UTF-8 to prevent Windows UnicodeEncodeError with emojis
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

def load_retriever():
    print("📂 Loading vector store...")
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        task_type="retrieval_query"
    )
    vector_store = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}
    )
    print("✅ Vector store loaded!")
    return retriever

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def build_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0.3
    )

    prompt = ChatPromptTemplate.from_template("""
    You are a helpful assistant. Use the following context from the book 
    to answer the question. If you don't know the answer from the context, 
    say "I couldn't find that in the book."
                                              
    Context:
    {context}

    Question: {question}

    Answer:
    """)

    # Modern LCEL (LangChain Expression Language) chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

def chat():
    retriever = load_retriever()
    rag_chain = build_rag_chain(retriever)

    print("\n🤖 RAG Chatbot Ready! (type 'exit' to quit)")
    print("=" * 50)

    while True:
        question = input("\n💬 You: ").strip()
        if question.lower() == "exit":
            print("👋 Goodbye!")
            break
        if not question:
            continue

        print("🔍 Searching book & generating answer...")
        response = rag_chain.invoke(question)
        print(f"\n🤖 Bot: {response}")

if __name__ == "__main__":
    chat()