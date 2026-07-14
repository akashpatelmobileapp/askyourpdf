# 📄 Chat with your PDF

A simple [Streamlit](https://streamlit.io/) app that lets you upload a PDF and
ask questions about its content. It uses **Retrieval-Augmented Generation (RAG)**
with [LangChain](https://www.langchain.com/), a [Chroma](https://www.trychroma.com/)
vector store, and [Mistral AI](https://mistral.ai/) models.

---

## ✨ Features

- Upload any PDF and chat with it in natural language.
- Answers are grounded **only** in the document's content (no hallucinated facts).
- Handles greetings/small talk gracefully.
- In-memory vector store — no data is persisted to disk.
- Per-file caching so re-uploading the same PDF doesn't reprocess it.

---

## 🧠 How it works

The app follows a classic RAG pipeline:

```
PDF ──► Load ──► Split into chunks ──► Embed ──► Store in Chroma
                                                      │
User question ──► Embed ──► Retrieve relevant chunks ─┘
                                   │
                                   ▼
                    Build prompt (context + question)
                                   │
                                   ▼
                         Mistral LLM ──► Answer
```

Step by step:

1. **Upload** — The user uploads a PDF via the Streamlit file uploader.
2. **Load** — The PDF is written to a temporary file and read with `PyPDFLoader`
   (one document per page).
3. **Split** — Pages are broken into overlapping chunks
   (`chunk_size=1000`, `chunk_overlap=200`) with `RecursiveCharacterTextSplitter`,
   so context isn't lost at chunk boundaries.
4. **Embed & Store** — Each chunk is converted to a vector using
   `MistralAIEmbeddings` (`mistral-embed`) and stored in an **in-memory Chroma**
   vector store. This step is cached per file with `@st.cache_resource`.
5. **Retrieve** — For every question, the retriever uses **MMR (Maximal Marginal
   Relevance)** to fetch the 4 most relevant *and* diverse chunks
   (`k=4`, `fetch_k=10`, `lambda_mult=0.5`).
6. **Answer** — The retrieved chunks (context) and the question are inserted into
   a prompt template and sent to `ChatMistralAI` (`mistral-medium-3-5`). The model
   answers using only the provided context, or replies *"I could not find the
   answer in the document."* if the answer isn't there.
7. **Chat history** — Messages are kept in `st.session_state` so the conversation
   persists across reruns (and resets when a new PDF is uploaded).

---

## 🚀 Getting started

### 1. Prerequisites

- Python 3.9+
- A [Mistral AI API key](https://console.mistral.ai/)

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set your API key

Create a `.env` file in the project root:

```env
MISTRAL_API_KEY=your_mistral_api_key_here
```

> When deploying to **Streamlit Cloud**, add the key under
> **App settings → Secrets** instead (the app reads from `st.secrets`
> automatically).

### 4. Run the app

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📁 Project structure

```
askyourpdf/
├── app.py             # Main Streamlit application
├── requirements.txt   # Python dependencies
├── .env               # Your API keys (not committed)
└── README.md          # This file
```

---

## ⚙️ Configuration

You can tweak the behaviour by editing these values in `app.py`:

| Setting              | Location                              | Description                                   |
|----------------------|---------------------------------------|-----------------------------------------------|
| `chunk_size`         | `RecursiveCharacterTextSplitter`      | Max characters per text chunk.                |
| `chunk_overlap`      | `RecursiveCharacterTextSplitter`      | Overlap between chunks to preserve context.   |
| `k` / `fetch_k`      | retriever `search_kwargs`             | Number of chunks retrieved / fetched.         |
| `lambda_mult`        | retriever `search_kwargs`             | MMR relevance-vs-diversity trade-off (0–1).   |
| `model_name`         | `ChatMistralAI`                       | The chat model used for answering.            |
| `model`              | `MistralAIEmbeddings`                 | The embedding model.                          |

---

## 📝 Notes

- The vector store is **in-memory only** — nothing is written to disk, and the
  index is rebuilt when the app restarts.
- Only PDF files are supported.
