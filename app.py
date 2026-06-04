import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

# Page config
st.set_page_config(
    page_title="📚 RAG Chatbot",
    page_icon="📚",
    layout="centered"
)

st.title("📚 Chat with your Book")
st.caption("Powered by Gemini + LangChain + FAISS")

# Load vector store only once using session state
@st.cache_resource
def load_rag_chain():
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

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if question := st.chat_input("Ask something about the book..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching book..."):
            rag_chain = load_rag_chain()
            response = rag_chain.invoke(question)
            st.markdown(response)

    # Save assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})