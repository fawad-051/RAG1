# app4.py  -- Advanced RAG Q&A (deployment friendly)
# Requirements:
# pip install streamlit langchain-groq langchain-core langchain-community langchain-text-splitters langchain-chroma scikit-learn python-docx python-dotenv pymupdf

import os
import tempfile
import streamlit as st
import time
import shutil
import atexit
import uuid
import json
from datetime import datetime
from dotenv import load_dotenv

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser

# Use OpenAI embeddings instead of sentence-transformers
try:
    from langchain_openai import OpenAIEmbeddings
    OPENAI_EMBEDDINGS_AVAILABLE = True
except Exception:
    OPENAI_EMBEDDINGS_AVAILABLE = False

try:
    from langchain_chroma import Chroma
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# Optional imports
try:
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

try:
    import docx
    DOCX_AVAILABLE = True
except Exception:
    DOCX_AVAILABLE = False

# Load environment variables
load_dotenv()

# -----------------------------
# Streamlit basic config / UI
# -----------------------------
st.set_page_config(page_title="Advanced RAG Q&A", page_icon="📚", layout="wide")
st.title("🚀 Advanced RAG Q&A — Deployment Friendly")

# Debug toggle
DEBUG = st.sidebar.checkbox("Debug mode (console logs)", False)

def debug_log(msg):
    if DEBUG:
        print("DEBUG:", msg)

# -----------------------------
# Session & storage helpers
# -----------------------------
DEFAULT_BASE_DIR = "./rag_data"
os.makedirs(DEFAULT_BASE_DIR, exist_ok=True)

def session_dir(session_id):
    d = os.path.join(DEFAULT_BASE_DIR, f"session_{session_id}")
    os.makedirs(d, exist_ok=True)
    return d

def chat_history_path(session_id):
    return os.path.join(session_dir(session_id), "chat_history.json")

def save_chat_history(session_id, messages):
    try:
        with open(chat_history_path(session_id), "w", encoding="utf-8") as f:
            json.dump({"messages": messages, "updated_at": datetime.utcnow().isoformat()}, f, ensure_ascii=False, indent=2)
        debug_log(f"Saved chat history for {session_id}")
    except Exception as e:
        debug_log(f"Error saving chat history: {e}")

def load_chat_history(session_id):
    p = chat_history_path(session_id)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("messages", [])
        except Exception as e:
            debug_log(f"Error loading chat history: {e}")
    return []

# -----------------------------
# Sidebar configuration
# -----------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    session_id = st.text_input("Session ID", value=os.getenv("DEFAULT_SESSION_ID", "default_session"))
    
    # API Keys
    groq_api_key = st.text_input("Groq API Key", type="password", value=os.getenv("GROQ_API_KEY", ""))
    openai_api_key = st.text_input("OpenAI API Key (for embeddings)", type="password", value=os.getenv("OPENAI_API_KEY", ""))
    
    model_choice = st.selectbox("Groq Model", ["llama-3.3-70b-versatile", "llama-3.3-70b", "mixtral-8x7b-32768"], index=0)
    top_k = st.slider("Top K Chunks", 1, 12, 4)
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.7)
    
    # Embedding model selection
    embedding_model = st.selectbox(
        "Embedding Model",
        ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"],
        index=0
    )
    
    debug_checkbox = st.checkbox("Show more debug info", False)
    
    if st.button("🧹 Cleanup Vector Store (this session)"):
        idx_dir = os.path.join(session_dir(session_id), "chroma_index")
        if os.path.exists(idx_dir):
            shutil.rmtree(idx_dir, ignore_errors=True)
            st.success("Cleaned session vector store.")
        else:
            st.info("No vectorstore found for this session.")

# link debug flags
if debug_checkbox:
    DEBUG = True

# Validate API keys
if not groq_api_key:
    st.warning("⚠️ Please enter your Groq API key in the sidebar or set GROQ_API_KEY in environment variables.")
    st.stop()

if not openai_api_key:
    st.warning("⚠️ Please enter your OpenAI API key for embeddings in the sidebar or set OPENAI_API_KEY in environment variables.")
    st.stop()

# Initialize LLM
try:
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name=model_choice,
        temperature=0.1
    )
except Exception as e:
    st.error(f"Failed to initialize Groq model: {e}")
    st.stop()

# -----------------------------
# Embeddings setup
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings(api_key, model_name="text-embedding-3-small"):
    if not OPENAI_EMBEDDINGS_AVAILABLE:
        st.error("OpenAI embeddings not available. Please install langchain-openai.")
        return None
    
    try:
        return OpenAIEmbeddings(
            openai_api_key=api_key,
            model=model_name
        )
    except Exception as e:
        st.error(f"Failed to initialize OpenAI embeddings: {e}")
        return None

with st.spinner("Initializing embeddings..."):
    embeddings = get_embeddings(openai_api_key, embedding_model)
    if embeddings is None:
        st.stop()

# -----------------------------
# File upload (PDF, txt, docx)
# -----------------------------
st.header("📤 Upload documents (PDF / TXT / DOCX)")
uploaded_files = st.file_uploader("Upload files", type=["pdf", "txt", "docx"], accept_multiple_files=True)

if not uploaded_files:
    st.info("Upload one or more files to begin. You can also drag multiple files.")
    # Load previous chat history and show quick actions if available
    previous_msgs = load_chat_history(session_id)
    if previous_msgs:
        st.success(f"Loaded saved chat for session '{session_id}' ({len(previous_msgs)} messages).")
        if st.button("🔁 Load chat into UI"):
            st.session_state.messages = previous_msgs
    st.stop()

# -----------------------------
# Document loading
# -----------------------------
try:
    from langchain_community.document_loaders import PyPDFLoader
    PDF_LOADER_AVAILABLE = True
except Exception:
    PDF_LOADER_AVAILABLE = False

from langchain_core.documents import Document

all_docs = []
tmp_paths = []

with st.spinner("Processing uploaded files..."):
    for f in uploaded_files:
        name = f.name
        suffix = os.path.splitext(name)[1].lower()
        
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(f.getvalue())
            tmp_paths.append(tmp.name)
            tmp_name = tmp.name

        try:
            if suffix == ".pdf":
                if not PDF_LOADER_AVAILABLE:
                    # Fallback: use basic text extraction
                    import PyPDF2
                    with open(tmp_name, 'rb') as file:
                        pdf_reader = PyPDF2.PdfReader(file)
                        text = ""
                        for page in pdf_reader.pages:
                            text += page.extract_text() + "\n"
                        docs = [Document(page_content=text, metadata={"source_file": name})]
                else:
                    loader = PyPDFLoader(tmp_name)
                    docs = loader.load()
                    
            elif suffix == ".txt":
                with open(tmp_name, "r", encoding="utf-8", errors="ignore") as file:
                    text = file.read()
                docs = [Document(page_content=text, metadata={"source_file": name})]
                
            elif suffix == ".docx" and DOCX_AVAILABLE:
                doc = docx.Document(tmp_name)
                full_text = "\n".join([p.text for p in doc.paragraphs])
                docs = [Document(page_content=full_text, metadata={"source_file": name})]
                
            else:
                # Fallback for unknown types
                with open(tmp_name, "r", encoding="utf-8", errors="ignore") as file:
                    text = file.read()
                docs = [Document(page_content=text, metadata={"source_file": name})]

            # Add metadata
            for d in docs:
                d.metadata["source_file"] = name
            all_docs.extend(docs)
            
        except Exception as e:
            st.error(f"Failed to load {name}: {e}")

# Cleanup temp files
for p in tmp_paths:
    try:
        os.unlink(p)
    except Exception:
        pass

if not all_docs:
    st.error("No readable text found in uploaded documents. Try different files.")
    st.stop()

st.success(f"✅ Loaded {len(all_docs)} pages/chunks from {len(uploaded_files)} file(s)")

# -----------------------------
# Split into chunks
# -----------------------------
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=120)
splits = splitter.split_documents(all_docs)
st.info(f"📄 Created {len(splits)} text chunks.")

# -----------------------------
# Vectorstore setup
# -----------------------------
INDEX_DIR = os.path.join(session_dir(session_id), "chroma_index")

@st.cache_resource(show_spinner=False)
def init_vectorstore(_splits, _embeddings, persist_dir):
    if not CHROMA_AVAILABLE:
        st.error("Chroma vectorstore is not available.")
        return None
        
    try:
        # Clean previous index
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir, ignore_errors=True)
            
        vs = Chroma.from_documents(
            documents=_splits,
            embedding=_embeddings,
            persist_directory=persist_dir
        )
        return vs
    except Exception as e:
        st.error(f"Failed to initialize vector store: {e}")
        return None

with st.spinner("Building vector store..."):
    vectorstore = init_vectorstore(splits, embeddings, INDEX_DIR)
    if vectorstore is None:
        st.stop()

st.session_state.vectorstore_initialized = True

# -----------------------------
# Retriever
# -----------------------------
def create_retriever(vs, k=5, distance_threshold=0.7):
    def retrieve(question):
        try:
            docs_with_scores = vs.similarity_search_with_score(question, k=k*3)
            filtered = [d for d, dist in docs_with_scores if dist <= distance_threshold]
            if not filtered:
                filtered = [d for d, _ in docs_with_scores[:k]]
            return filtered[:k]
        except Exception as e:
            debug_log(f"Retriever error: {e}")
            return vs.similarity_search(question, k=k)
    return retrieve

retriever = create_retriever(vectorstore, k=top_k, distance_threshold=similarity_threshold)

# -----------------------------
# RAG chain
# -----------------------------
def rag_chain(question, chat_history_messages):
    start_time = time.time()
    docs = retriever(question)
    if not docs:
        return "⚠️ No relevant text found in your documents.", [], 0.0

    context = "\n\n".join([d.page_content for d in docs])
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a helpful assistant. Answer based **only** on the provided context.\n"
         "If the answer is not in the context, say: 'The answer is not available in the provided documents.'\n\n"
         "Context:\n{context}\n"),
        ("human", "Question: {question}")
    ])

    chain = qa_prompt | llm | StrOutputParser()
    try:
        response = chain.invoke({"context": context, "question": question})
    except Exception as e:
        response = f"Error generating response: {e}"
    
    duration = round(time.time() - start_time, 2)
    return response, docs, duration

# -----------------------------
# Chat UI
# -----------------------------
if "messages" not in st.session_state:
    previous = load_chat_history(session_id)
    st.session_state.messages = previous if previous else []

if "chathistory" not in st.session_state:
    st.session_state.chathistory = {}

def get_chat_history_obj(sid):
    if sid not in st.session_state.chathistory:
        st.session_state.chathistory[sid] = ChatMessageHistory()
    return st.session_state.chathistory[sid]

# Display chat messages
st.header("💬 Chat with your documents")
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Document insights
with st.expander("🔎 Document Insights"):
    st.write(f"**Files uploaded:** {', '.join({d.metadata.get('source_file','?') for d in all_docs})}")
    st.write(f"**Total pages/chunks:** {len(all_docs)}")
    st.write(f"**Text chunks created:** {len(splits)}")

# Chat input
user_q = st.chat_input("Ask something about your uploaded documents...")

if user_q:
    # Add user message
    history_obj = get_chat_history_obj(session_id)
    with st.chat_message("user"):
        st.markdown(user_q)
    st.session_state.messages.append({"role": "user", "content": user_q})
    history_obj.add_user_message(user_q)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents..."):
            try:
                response, docs, duration = rag_chain(user_q, history_obj)
                st.markdown(response)
                st.caption(f"⏱️ Generated in {duration}s")

                # Show sources
                if docs:
                    with st.expander("📂 Sources Used"):
                        for d in docs:
                            src = d.metadata.get("source_file", "Unknown")
                            st.markdown(f"**📄 {src}**")
                            snippet = d.page_content[:500] + "..."
                            st.write(snippet)
                            st.divider()

                # Save messages
                st.session_state.messages.append({"role": "assistant", "content": response})
                history_obj.add_ai_message(response)
                save_chat_history(session_id, st.session_state.messages)

            except Exception as e:
                err = f"Error: {e}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
                save_chat_history(session_id, st.session_state.messages)

# -----------------------------
# Export tools
# -----------------------------
st.header("📥 Export Tools")
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📥 Download Chat as TXT"):
        chat_text = "\n".join([f"{m['role'].upper()}: {m['content']}" for m in st.session_state.messages])
        st.download_button("Download TXT", data=chat_text, file_name=f"chat_{session_id}.txt", mime="text/plain")

with col2:
    if st.button("💾 Export Chat as JSON"):
        chat_json = json.dumps(st.session_state.messages, ensure_ascii=False, indent=2)
        st.download_button("Download JSON", data=chat_json, file_name=f"chat_{session_id}.json", mime="application/json")

with col3:
    if st.button("🗑️ Clear current chat"):
        st.session_state.messages = []
        p = chat_history_path(session_id)
        if os.path.exists(p):
            os.remove(p)
        st.rerun()

# -----------------------------
# Cleanup
# -----------------------------
def cleanup_vectorstore_on_exit():
    debug_log("App exiting - vector stores preserved.")

atexit.register(cleanup_vectorstore_on_exit)
