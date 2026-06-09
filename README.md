# 📚 RAG Chatbot — Chat with Your PDF

> An AI-powered chatbot that lets you ask questions about any PDF document using **Retrieval Augmented Generation (RAG)**. Built with LangChain, Google Gemini, FAISS, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=for-the-badge&logo=python)
![LangChain](https://img.shields.io/badge/LangChain-1.3.2-green?style=for-the-badge)
![Gemini](https://img.shields.io/badge/Google_Gemini-API-orange?style=for-the-badge&logo=google)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red?style=for-the-badge&logo=streamlit)
![FAISS](https://img.shields.io/badge/FAISS-VectorDB-purple?style=for-the-badge)

---

## 🎯 What is this project?

This project is a fully functional RAG (Retrieval Augmented Generation)** chatbot that:

- 📄 Reads any PDF document you give it
- 🔍 Converts the content into searchable vector embeddings
- 🧠 Retrieves the most relevant chunks when you ask a question
- 🤖 Uses Google Gemini to generate accurate, grounded answers
- 💬 Displays everything in a clean Streamlit web UI

**No more hallucinations** — the chatbot only answers from your document!

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   INGESTION PHASE (once)                │
│                                                         │
│  PDF File → PyPDFLoader → RecursiveCharacterTextSplitter│
│       → Gemini Embeddings → FAISS Vector Store          │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│                   QUERY PHASE (every question)          │
│                                                         │
│  User Question → Embed Query → FAISS Similarity Search  │
│       → Top 5 Chunks → Prompt + Gemini LLM → Answer     │
└─────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| **Python** | 3.10+ | Core language |
| **LangChain** | 1.3.2 | AI pipeline framework (LCEL) |
| **Google Gemini** | gemini-2.5-flash | LLM for answer generation |
| **Gemini Embeddings** | gemini-embedding-001 | Text → Vector conversion |
| **FAISS** | CPU | Local vector database |
| **Streamlit** | Latest | Web UI |
| **PyPDF** | Latest | PDF text extraction |
| **python-dotenv** | Latest | Secure API key management |

---

## 📁 Project Structure

```
RAG project/
├── data/
│   └── your_document.pdf       # Put your PDF here
├── faiss_index/                 # Auto-generated vector database
│   ├── index.faiss
│   └── index.pkl
├── app.py                       # Streamlit web UI
├── chat.py                      # CLI chatbot + RAG chain
├── ingest.py                    # PDF processing + embedding
├── main.py                      # Entry point (CLI)
├── requirements.txt             # Dependencies
├── .env                         # API key (not committed)
└── .gitignore                   # Prevents sensitive files from git
```

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10 or higher
- A free Google Gemini API key — get one at [aistudio.google.com](https://aistudio.google.com/app/apikey)

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/rag-chatbot.git
cd rag-chatbot
```

### 2. Create and activate virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set up your API key

Create a `.env` file in the project root:

```
GOOGLE_API_KEY=your_gemini_api_key_here
```

### 5. Add your PDF

Copy your PDF file into the `data/` folder.

Then update the filename in `ingest.py`:
```python
pdf_path = "data/your_file.pdf"
```

### 6. Run ingestion (first time only)

```bash
python ingest.py
```

This will:
- Load and split your PDF into chunks
- Generate embeddings using Gemini
- Save the vector store locally

> ⏱️ This takes a few minutes depending on PDF size. Only needs to run once!

### 7. Launch the web app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` and start chatting! 🎉

---

## 💬 How to Use

Once the app is running:

1. Type your question in the chat box at the bottom
2. The chatbot retrieves the top 5 most relevant chunks from your PDF
3. Google Gemini generates an answer based only on that content
4. If the answer isn't in the document, it tells you honestly

**Example questions (tested with "How to Talk to Anyone"):**
- *"What is the Whatzit technique?"*
- *"How can I make a great first impression?"*
- *"What are insider opening questions?"*
- *"What is conversational bait?"*

---

## ⚙️ How RAG Works

RAG combines **retrieval** and **generation** in two steps:

**Step 1 — Indexing (ingest.py)**
```
PDF → Split into 1000-char chunks (200-char overlap)
    → Convert each chunk to a 3072-dim vector (Gemini Embeddings)
    → Store all vectors in FAISS index
```

**Step 2 — Querying (chat.py / app.py)**
```
Question → Convert to vector → Find top-5 similar chunks in FAISS
         → Combine chunks + question into prompt
         → Send to Gemini → Get grounded answer
```

The key insight: instead of the LLM guessing from memory, it reads the **actual relevant text** before answering.

---

## 🔧 Configuration

You can tweak these settings in `ingest.py` and `chat.py`:

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_size` | 1000 | Characters per chunk — larger = more context per chunk |
| `chunk_overlap` | 200 | Shared chars between chunks — prevents context loss |
| `k` (retriever) | 5 | Number of chunks retrieved per question |
| `temperature` | 0.3 | LLM creativity — lower = more factual answers |

---

## 🐛 Common Issues

**Rate limit error (RESOURCE_EXHAUSTED)**
> The free Gemini tier allows 100 requests/min. The ingestion script handles this automatically with retry logic. Just let it run.

**API key not found**
> Make sure `.env` file exists in the project root with `GOOGLE_API_KEY=your_key`

**Model not found**
> Run `python -c "import google.generativeai as genai; genai.configure(api_key='YOUR_KEY'); [print(m.name) for m in genai.list_models()]"` to see available models for your key.

**Empty output when running scripts**
> Make sure your virtual environment is activated: `venv\Scripts\activate` (Windows)

---

## 📈 Possible Improvements

- [ ] Multi-PDF support with file uploader in UI
- [ ] Show source page numbers with each answer
- [ ] Add conversation memory for follow-up questions
- [ ] Deploy to Streamlit Cloud
- [ ] Replace FAISS with ChromaDB for persistence
- [ ] Add a reranker for better retrieval quality

---

## 🧠 What I Learned

Building this project taught me:

- How RAG prevents LLM hallucinations by grounding answers in real documents
- How vector embeddings make text mathematically searchable
- How FAISS performs fast similarity search locally
- How LangChain LCEL chains components with the `|` operator
- How to handle real-world API rate limits with batch processing and retry logic
- How to build a production-style web UI with Streamlit session state

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

##  Acknowledgements

- [LangChain](https://python.langchain.com/) — for the excellent RAG framework
- [Google Gemini](https://ai.google.dev/) — for the free-tier LLM and embeddings API
- [Facebook AI (FAISS)](https://github.com/facebookresearch/faiss) — for the fast vector search library
- [Streamlit](https://streamlit.io/) — for making web apps easy in Python

---

<p align="center">   RAG Chatbot Project</p>
