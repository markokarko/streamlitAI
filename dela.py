# Fix SQLite version issue for ChromaDB on Streamlit Cloud
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass
import streamlit as st
import chromadb
from transformers import pipeline
from pathlib import Path
import tempfile
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.backend.docling_parse_v2_backend import DoclingParseV2DocumentBackend
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions, AcceleratorOptions, AcceleratorDevice
from datetime import datetime

# FIX: Reset collection when problems occur
def reset_database():
    client = chromadb.Client()
    try:
        client.delete_collection("docs")
    except:
        pass
    return client.create_collection("docs")

# Convert uploaded file to markdown text
def convert_to_markdown(file_path: str) -> str:
    path = Path(file_path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        pdf_opts = PdfPipelineOptions(do_ocr=False)
        pdf_opts.accelerator_options = AcceleratorOptions(
            num_threads=4,
            device=AcceleratorDevice.CPU
        )
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pdf_opts,
                    backend=DoclingParseV2DocumentBackend
                )
            }
        )
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext in [".doc", ".docx"]:
        converter = DocumentConverter()
        doc = converter.convert(file_path).document
        return doc.export_to_markdown(image_mode="placeholder")

    if ext == ".txt":
        try:
            return path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return path.read_text(encoding="latin-1", errors="replace")

    raise ValueError(f"Unsupported extension: {ext}")


# Reset ChromaDB collection
def reset_collection(client, collection_name: str):
    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
    return client.create_collection(name=collection_name)


# Add text chunks to ChromaDB
def add_text_to_chromadb(text: str, filename: str, collection_name: str = "documents"):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_text(text)

    if not hasattr(add_text_to_chromadb, 'client'):
        add_text_to_chromadb.client = chromadb.Client()
        add_text_to_chromadb.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        add_text_to_chromadb.collections = {}

    if collection_name not in add_text_to_chromadb.collections:
        try:
            collection = add_text_to_chromadb.client.get_collection(name=collection_name)
        except:
            collection = add_text_to_chromadb.client.create_collection(name=collection_name)
        add_text_to_chromadb.collections[collection_name] = collection

    collection = add_text_to_chromadb.collections[collection_name]

    for i, chunk in enumerate(chunks):
        embedding = add_text_to_chromadb.embedding_model.encode(chunk).tolist()

        metadata = {
            "filename": filename,
            "chunk_index": i,
            "chunk_size": len(chunk)
        }

        collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas=[metadata],
            ids=[f"{filename}_doc_{i}"]
        )

    return collection


# Enhanced Q&A with source
def get_answer_with_source(collection, question):
    # Defensive: check for empty question
    if not question or not question.strip():
        return "Please enter a question.", "No source"
    # Defensive: check if collection exists and has any documents
    try:
        count = collection.count()
        if count == 0:
            return "No documents in the knowledge base. Please upload and add documents first.", "No source"
    except Exception as e:
        # Try to re-create the collection if it does not exist
        if "does not exists" in str(e):
            import chromadb
            client = chromadb.Client()
            collection = client.create_collection(name="documents")
            return "No documents in the knowledge base. Please upload and add documents first.", "No source"
        return f"ChromaDB error: {e}", "No source"
    # Defensive: catch query errors
    try:
        results = collection.query(query_texts=[question], n_results=3)
        docs = results["documents"][0]
        distances = results["distances"][0]
        ids = results["ids"][0]
    except Exception as e:
        return f"ChromaDB query error: {e}", "No source"
    if not docs or min(distances) > 1.5:
        return "I do not have this information.", "No source"
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(docs)])
    prompt = f"""Context information:\n{context}\n\nQuestion: {question}\n\nInstructions: If the answer is not in the context, respond with 'I do not have this information.'\n\nAnswer:"""
    ai_model = pipeline("text2text-generation", model="google/flan-t5-small")
    response = ai_model(prompt, max_length=150)
    answer = response[0]['generated_text'].strip()
    best_source = ids[0].split('_doc_')[0]
    # If the model still doesn't know, catch generic fallback
    if answer.lower() in ["i don't know.", "i do not know.", "i don't have information about that topic in my documents.", "i don't have information about that topic.", "i do not have this information."] or len(answer) < 3:
        answer = "I do not have this information."
    return answer, best_source


# Document manager
def show_document_manager():
    st.subheader("📋 Manage Documents")
    if not st.session_state.converted_docs:
        st.info("No documents uploaded yet.")
        return
    for i, doc in enumerate(st.session_state.converted_docs):
        col1, col2, col3 = st.columns([3, 1, 1])
        with col1:
            st.write(f"📄 {doc['filename']}")
            st.write(f"   Words: {len(doc['content'].split())}")
        with col2:
            if st.button("Preview", key=f"preview_{i}"):
                st.session_state[f'show_preview_{i}'] = True
        with col3:
            if st.button("Delete", key=f"delete_{i}"):
                st.session_state.converted_docs.pop(i)
                st.session_state.collection = reset_collection(chromadb.Client(), "documents")
                add_docs_to_database(st.session_state.collection, st.session_state.converted_docs)
                st.rerun()
        if st.session_state.get(f'show_preview_{i}', False):
            with st.expander(f"Preview: {doc['filename']}", expanded=True):
                st.text(doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'])
                if st.button("Hide Preview", key=f"hide_{i}"):
                    st.session_state[f'show_preview_{i}'] = False
                    st.rerun()


def add_docs_to_database(collection, docs):
    count = 0
    for doc in docs:
        add_text_to_chromadb(doc['content'], doc['filename'], collection_name="documents")
        count += 1
    return count


def convert_uploaded_files(uploaded_files):
    converted_docs = []
    for file in uploaded_files:
        suffix = Path(file.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(file.getvalue())
            temp_file_path = temp_file.name
        text = convert_to_markdown(temp_file_path)
        converted_docs.append({'filename': file.name, 'content': text})
    return converted_docs


def add_to_search_history(question, answer, source):
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    st.session_state.search_history.insert(0, {
        'question': question,
        'answer': answer,
        'source': source,
        'timestamp': str(datetime.now().strftime("%H:%M:%S"))
    })
    if len(st.session_state.search_history) > 10:
        st.session_state.search_history = st.session_state.search_history[:10]


def show_search_history():
    st.subheader("🕒 Recent Searches")
    if 'search_history' not in st.session_state or not st.session_state.search_history:
        st.info("No searches yet.")
        return
    for i, search in enumerate(st.session_state.search_history):
        with st.expander(f"Q: {search['question'][:50]}... ({search['timestamp']})"):
            st.write("**Question:**", search['question'])
            st.write("**Answer:**", search['answer'])
            st.write("**Source:**", search['source'])


def show_document_stats():
    st.subheader("📊 Document Statistics")
    if not st.session_state.converted_docs:
        st.info("No documents to analyze.")
        return
    total_docs = len(st.session_state.converted_docs)
    total_words = sum(len(doc['content'].split()) for doc in st.session_state.converted_docs)
    avg_words = total_words // total_docs if total_docs > 0 else 0
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Documents", total_docs)
    with col2:
        st.metric("Total Words", f"{total_words:,}")
    with col3:
        st.metric("Average Words/Doc", f"{avg_words:,}")
    file_types = {}
    for doc in st.session_state.converted_docs:
        ext = Path(doc['filename']).suffix.lower()
        file_types[ext] = file_types.get(ext, 0) + 1
    st.write("**File Types:**")
    for ext, count in file_types.items():
        st.write(f"• {ext}: {count} files")


# --- Custom CSS for better appearance ---
def add_custom_css():
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.7rem;
        color: #7c4d03;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1.5rem;
        background: linear-gradient(90deg, #fbeee6 0%, #f7d9c4 100%);
        border-radius: 18px;
        border: 4px solid #bfa46d;
        box-shadow: 0 6px 24px rgba(191,164,109,0.25);
        font-family: 'Playfair Display', serif;
        letter-spacing: 2px;
        text-shadow: 1px 1px 0 #fffbe6, 2px 2px 4px #bfa46d;
    }
    body, .stApp {
        background-image: linear-gradient(rgba(255,245,230,0.90), rgba(255,245,230,0.90)), url('https://images.unsplash.com/photo-1464983953574-0892a716854b?auto=format&fit=crop&w=1500&q=80');
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
        background-position: center center;
    }
    .stApp {
        background-color: transparent;
        border-radius: 18px;
        padding: 2rem;
        border: 3px solid #bfa46d;
        box-shadow: 0 8px 32px rgba(191,164,109,0.18);
    }
    /* Remove white backgrounds from content blocks */
    .block-container, .stMarkdown, .stText, .stSubheader, .stHeader, .stExpander, .stMetric, .stAlert, .stDataFrame, .stTable, .stTabs, .stTab, .stForm, .stDownloadButton, .stButton, .stTextInput, .stFileUploader, .stSelectbox, .stTextArea, .stNumberInput, .stSlider, .stRadio, .stCheckbox, .stDateInput, .stTimeInput, .stColorPicker {
        background: transparent !important;
        border-radius: 14px;
        border: none;
        box-shadow: none;
        font-family: 'Playfair Display', serif;
        color: #3a2c0a;
        text-shadow: 0 1px 0 #fffbe6;
    }
    .success-box {
        padding: 1rem;
        background-color: #f7ecd0;
        border: 1.5px solid #bfa46d;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: 'Playfair Display', serif;
    }
    .info-box {
        padding: 1rem;
        background-color: #f3e6ff;
        border: 1.5px solid #bfa46d;
        border-radius: 8px;
        margin: 1rem 0;
        font-family: 'Playfair Display', serif;
    }
    .stButton > button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #fbeee6 0%, #f7d9c4 100%);
        color: #7c4d03;
        border: 2px solid #bfa46d;
        font-family: 'Playfair Display', serif;
        box-shadow: 0 2px 8px rgba(191,164,109,0.10);
    }
    .metric-card {
        background: #fffbe6;
        padding: 1.2rem;
        border-radius: 14px;
        border: 2px solid #bfa46d;
        box-shadow: 0 2px 8px rgba(191,164,109,0.10);
        text-align: center;
        font-family: 'Playfair Display', serif;
    }
    /* Baroque decorative corners */
    .main-header:before, .main-header:after {
        content: "";
        display: inline-block;
        width: 48px;
        height: 48px;
        background: url('https://upload.wikimedia.org/wikipedia/commons/6/6b/Baroque_corner_gold.png') no-repeat center center;
        background-size: contain;
        vertical-align: middle;
        margin: 0 12px;
    }
    .main-header:before { float: left; }
    .main-header:after { float: right; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)


# --- Enhanced error handling for file conversion ---
def safe_convert_files(uploaded_files):
    converted_docs = []
    errors = []
    if not uploaded_files:
        return converted_docs, ["No files uploaded"]
    progress_bar = st.progress(0)
    status_text = st.empty()
    for i, uploaded_file in enumerate(uploaded_files):
        try:
            status_text.text(f"Converting {uploaded_file.name}...")
            if len(uploaded_file.getvalue()) > 10 * 1024 * 1024:
                errors.append(f"{uploaded_file.name}: File too large (max 10MB)")
                continue
            allowed_extensions = ['.pdf', '.doc', '.docx', '.txt']
            file_ext = Path(uploaded_file.name).suffix.lower()
            if file_ext not in allowed_extensions:
                errors.append(f"{uploaded_file.name}: Unsupported file type")
                continue
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
                tmp.write(uploaded_file.getvalue())
                tmp_path = tmp.name
            try:
                markdown_content = convert_to_markdown(tmp_path)
                if len(markdown_content.strip()) < 10:
                    errors.append(f"{uploaded_file.name}: File appears to be empty or corrupted")
                    continue
                converted_docs.append({
                    'filename': uploaded_file.name,
                    'content': markdown_content,
                    'size': len(uploaded_file.getvalue()),
                    'word_count': len(markdown_content.split())
                })
            finally:
                Path(tmp_path).unlink(missing_ok=True)
        except Exception as e:
            errors.append(f"{uploaded_file.name}: {str(e)}")
        progress_bar.progress((i + 1) / len(uploaded_files))
    status_text.text("Conversion complete!")
    return converted_docs, errors


# --- Better user feedback for conversion ---
def show_conversion_results(converted_docs, errors):
    if converted_docs:
        st.success(f"✅ Successfully converted {len(converted_docs)} documents!")
        total_words = sum(doc['word_count'] for doc in converted_docs)
        st.info(f"📊 Total words added to knowledge base: {total_words:,}")
        with st.expander("📋 View converted files"):
            for doc in converted_docs:
                st.write(f"• **{doc['filename']}** - {doc['word_count']:,} words")
    if errors:
        st.error(f"❌ {len(errors)} files failed to convert:")
        for error in errors:
            st.write(f"• {error}")


# --- Better question interface ---
def enhanced_question_interface():
    st.subheader("How can I be of help?🤓")
    with st.expander("💡 Example questions you can ask"):
        st.write("""
        • What are the main topics covered in these documents?
        • Summarize the key points from [document name]
        • What does the document say about [specific topic]?
        • Compare information between documents
        • Find specific data or statistics
        """)
    question = st.text_input(
        "Type your question here:",
        placeholder="e.g., What are the main findings in the research paper?"
    )
    col1, col2 = st.columns([1, 1])
    with col1:
        search_button = st.button("🔍 Search Documents", type="primary")
    with col2:
        clear_button = st.button("🗑️ Clear History")
    return question, search_button, clear_button


# --- App health check ---
def check_app_health():
    issues = []
    required_keys = ['converted_docs', 'collection']
    for key in required_keys:
        if key not in st.session_state:
            issues.append(f"Missing session state: {key}")
    try:
        if st.session_state.get('collection'):
            st.session_state.collection.count()
    except Exception as e:
        issues.append(f"Database issue: {e}")
    try:
        pipeline("text2text-generation", model="google/flan-t5-small")
    except Exception as e:
        issues.append(f"AI model issue: {e}")
    return issues


# --- Loading animation ---
def show_loading_animation(text="Processing..."):
    with st.spinner(text):
        import time
        time.sleep(0.5)


# --- Enhanced main function ---
def enhanced_main():
    add_custom_css()
    st.markdown('<h1 class="main-header">File Butler at Your Service!</h1>', unsafe_allow_html=True)
    st.markdown("Upload your documents, let me do the heavy lifting, and ask away!")
    if 'converted_docs' not in st.session_state:
        st.session_state.converted_docs = []
    if 'collection' not in st.session_state:
        st.session_state.collection = reset_collection(chromadb.Client(), "documents")
    if 'search_history' not in st.session_state:
        st.session_state.search_history = []
    health_issues = check_app_health()
    if health_issues:
        with st.expander("⚠️ System Status"):
            for issue in health_issues:
                st.warning(issue)
    tab1, tab2, tab3, tab4 = st.tabs(["📘 Add File", "❓ Questions", "📋 Manage", "📊 Analytics"])
    with tab1:
        st.header("📁 Document Upload & Conversion")
        uploaded_files = st.file_uploader(
            "Select documents to add to your knowledge base",
            type=["pdf", "doc", "docx", "txt"],
            accept_multiple_files=True,
            help="Supported formats: PDF, Word documents, and text files"
        )
        if st.button("🚀 Convert & Add to Knowledge Base", type="primary"):
            if uploaded_files:
                with st.spinner("Converting documents..."):
                    converted_docs, errors = safe_convert_files(uploaded_files)
                if converted_docs:
                    num_added = add_docs_to_database(st.session_state.collection, converted_docs)
                    st.session_state.converted_docs.extend(converted_docs)
                show_conversion_results(converted_docs, errors)
            else:
                st.warning("Please select files to upload first.")
    with tab2:
        st.header("❓ Ask Questions")
        if st.session_state.converted_docs:
            question, search_button, clear_button = enhanced_question_interface()
            if search_button and question:
                with st.spinner("Searching through your documents..."):
                    answer, source = get_answer_with_source(st.session_state.collection, question)
                st.markdown("### 💡 Answer")
                st.write(answer)
                st.info(f"📄 Source: {source}")
                add_to_search_history(question, answer, source)
            if clear_button:
                st.session_state.search_history = []
                st.success("Search history cleared!")
            if st.session_state.search_history:
                show_search_history()
        else:
            st.info("🔼 Upload some documents first to start asking questions!")
    with tab3:
        show_document_manager()
    with tab4:
        show_document_stats()
    st.markdown("---")
    st.markdown("*Built with Streamlit • Powered by AI*")


if __name__ == "__main__":
    enhanced_main()
