"""
Chat with your PDF
------------------
A small Streamlit app that lets a user upload a PDF and ask questions about it.

How it works:
    1. The PDF is loaded and split into overlapping text chunks.
    2. Each chunk is embedded and stored in an in-memory Chroma vector store.
    3. For every question, the most relevant chunks are retrieved and passed as
       context to a Mistral chat model, which answers using only that context.
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

# Load environment variables from a local .env file (if present).
load_dotenv()

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="centered",
)


# ---------------------------------------------------------------------------
# API keys
# ---------------------------------------------------------------------------
# Works both locally (via .env) and on Streamlit Cloud (via st.secrets).
# If a secrets.toml file exists, copy the keys into environment variables so
# the LangChain clients below can pick them up automatically.
try:
    for key in ["OPENAI_API_KEY", "MISTRAL_API_KEY"]:
        if key in st.secrets:
            os.environ[key] = st.secrets[key]
except Exception:
    # No secrets.toml found (e.g. running locally with only .env) - safe to ignore.
    pass


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
embedding_model = MistralAIEmbeddings(model="mistral-embed")   # turns text into vectors
llm = ChatMistralAI(model_name="mistral-medium-3-5")           # answers the questions


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# The system message defines the assistant's behaviour; the human message
# injects the retrieved context and the user's question at query time.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful AI assistant that answers questions about an uploaded document.

If the user sends a greeting or small talk (e.g. "hi", "hello", "how are you"),
respond naturally and briefly, and invite them to ask a question about the document.

For any actual question, use ONLY the provided context to answer.
If the answer is not present in the context,
say: "I could not find the answer in the document."
""",
        ),
        (
            "human",
            """Context:
{context}

Question:
{question}
""",
        ),
    ]
)


# ---------------------------------------------------------------------------
# PDF processing
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def build_vectorstore_from_pdf(file_bytes: bytes, file_name: str):
    """Build an in-memory Chroma vector store from an uploaded PDF.

    The result is cached per file, so re-uploading the same file does not
    trigger reprocessing.

    Args:
        file_bytes: Raw bytes of the uploaded PDF.
        file_name:  Original file name (used only as the cache key).

    Returns:
        A Chroma vector store containing the embedded PDF chunks.
    """
    # Write the uploaded bytes to a temporary file, since PyPDFLoader needs a path.
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    # Load the PDF into LangChain documents (one per page).
    loader = PyPDFLoader(tmp_path)
    docs = loader.load()

    # Split pages into overlapping chunks so context isn't lost at boundaries.
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)

    # Embed the chunks and store them. No persist_directory -> in-memory only,
    # which avoids ephemeral storage issues on Streamlit Cloud.
    vectorstore = Chroma.from_documents(chunks, embedding=embedding_model)

    # Clean up the temporary file.
    os.remove(tmp_path)
    return vectorstore


# ---------------------------------------------------------------------------
# UI - header
# ---------------------------------------------------------------------------
# Hide the default Streamlit header action buttons (top-right menu).
st.markdown(
    """
    <style>
    [data-testid="stHeaderActionElements"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "<h1 style='text-align:center;'>📄 Chat with your PDF</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;color:gray;'>Upload a PDF and ask questions about its content</p>",
    unsafe_allow_html=True,
)
st.divider()


# ---------------------------------------------------------------------------
# UI - file upload
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])

# Initialise session state on first run.
if "messages" not in st.session_state:
    st.session_state.messages = []       # chat history
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None  # current document's vector store
if "last_file" not in st.session_state:
    st.session_state.last_file = None    # name of the last processed file


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
if uploaded_file is not None:
    # Rebuild the vector store only when a new/different file is uploaded.
    if st.session_state.last_file != uploaded_file.name:
        with st.spinner("Processing PDF... this may take a moment"):
            st.session_state.vectorstore = build_vectorstore_from_pdf(
                uploaded_file.getvalue(), uploaded_file.name
            )
        st.session_state.last_file = uploaded_file.name
        st.session_state.messages = []  # reset chat for the new document
        st.success(f"'{uploaded_file.name}' processed! Ask away below.")

    # Render the existing chat history.
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    # Chat input box at the bottom of the page.
    query = st.chat_input("Ask a question about your PDF...")

    if query:
        # Record and display the user's question.
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.write(query)

        # Generate and display the assistant's answer.
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Retrieve the most relevant chunks using MMR (Maximal Marginal
                # Relevance) to balance relevance and diversity.
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="mmr",
                    search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
                )
                docs = retriever.invoke(query)
                context = "\n\n".join([d.page_content for d in docs])

                # Build the final prompt and ask the LLM.
                final_prompt = prompt.invoke({"context": context, "question": query})
                response = llm.invoke(final_prompt)
                st.write(response.content)

        # Save the assistant's reply to the chat history.
        st.session_state.messages.append(
            {"role": "assistant", "content": response.content}
        )
else:
    # No file uploaded yet - prompt the user to get started.
    st.info("👆 Upload a PDF to get started.")
