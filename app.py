"""
app.py - DocuMind AI: Streamlit User Interface

A professional document question-answering application that uses:
  - PyMuPDF for PDF text extraction
  - TF-IDF + Cosine Similarity for passage retrieval
  - Google Gemini API for answer generation (mandatory)

Author : YHills AI Internship - Project 2
Version: 1.0
"""

import os
import hashlib
from pathlib import Path

import pymupdf as fitz
import streamlit as st
from dotenv import load_dotenv

from rag_pipeline import (
    extract_text_from_pdf,
    split_into_chunks,
    build_tfidf_index,
    retrieve_relevant_chunks,
    generate_answer,
    check_gemini_connection,
)

# Load the project-local environment file even when Streamlit is launched
# from a different working directory.
load_dotenv(Path(__file__).resolve().with_name(".env"), override=True)


# =============================================================================
# PAGE CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="DocuMind | Document assistant",
    page_icon="D",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# Custom styling
# =============================================================================

st.markdown("""
<style>
    .stApp {
        background: #f5f7fb;
        color: #1f2933;
    }
    .block-container {
        max-width: 1120px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .app-header {
        align-items: flex-start;
        border-bottom: 1px solid #dde3ea;
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        margin-bottom: 1.5rem;
        padding: 0.75rem 0 1.25rem;
    }
    .app-header h1 {
        color: #17212b;
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: 0;
        margin: 0;
    }
    .app-header p {
        color: #5f6b7a;
        margin: 0.4rem 0 0;
        max-width: 680px;
    }
    .status-pill {
        background: #fff;
        border: 1px solid #d6dde6;
        border-radius: 999px;
        color: #526071;
        display: inline-flex;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.35rem;
        padding: 0.4rem 0.75rem;
        white-space: nowrap;
    }
    .status-pill-connected {
        background: #eaf7ef;
        border-color: #b9dfc7;
        color: #21613a;
    }
    .welcome-panel {
        background: #ffffff;
        border: 1px solid #dde3ea;
        border-radius: 8px;
        padding: 1.25rem 1.4rem;
    }
    .welcome-panel h2 {
        color: #17212b;
        font-size: 1.6rem;
        letter-spacing: 0;
        margin: 0.15rem 0 0.5rem;
    }
    .step-label {
        color: #7a4f19;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.35rem;
    }
    [data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #dde3ea;
    }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #526071;
    }
    [data-testid="stFileUploader"] section {
        background: #f8fafc;
        border: 1px dashed #aeb8c5;
        border-radius: 8px;
    }
    [data-testid="stTextInput"] input {
        border-radius: 8px;
    }
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    .stButton > button[kind="primary"] {
        background: #184e77;
        border-color: #184e77;
    }
    .stButton > button[kind="primary"]:hover {
        background: #123a59;
        border-color: #123a59;
    }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# SESSION STATE INITIALIZATION
# =============================================================================

def init_session_state():
    """Initialize all session state variables with default values."""
    defaults = {
        "doc_pages": None,           # Extracted page texts
        "doc_chunks": None,          # Text chunks for retrieval
        "vectorizer": None,          # Fitted TF-IDF vectorizer
        "tfidf_matrix": None,        # Pre-computed TF-IDF matrix
        "uploaded_file_name": None,  # Current filename (for change detection)
        "uploaded_file_size": None,  # Current file size (for change detection)
        "num_pages": 0,              # Number of pages with text
        "total_pages": 0,            # Total pages in PDF
        "gemini_status": None,       # Dict from check_gemini_connection()
        "gemini_config_fingerprint": None,
        "last_answer": None,         # Last generated answer dict
        "last_query": None,          # Last question asked
        "last_sources": None,        # Last retrieved chunks
        "question_input": "",        # Current question text
        "processing_error": None,    # Document processing error message
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_api_config():
    """Read Gemini API configuration from environment variables."""
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    model_name = (
        os.getenv("GEMINI_MODEL", "").strip() or "gemini-flash-latest"
    )
    return api_key, model_name


def clear_document_state():
    """Reset all document-related session state."""
    st.session_state.doc_pages = None
    st.session_state.doc_chunks = None
    st.session_state.vectorizer = None
    st.session_state.tfidf_matrix = None
    st.session_state.uploaded_file_name = None
    st.session_state.uploaded_file_size = None
    st.session_state.num_pages = 0
    st.session_state.total_pages = 0
    st.session_state.last_answer = None
    st.session_state.last_query = None
    st.session_state.last_sources = None
    st.session_state.processing_error = None


def process_uploaded_pdf(uploaded_file):
    """
    Process a newly uploaded PDF: extract text, chunk, build index.

    Results are stored in st.session_state to avoid re-processing when
    the user asks multiple questions on the same document.
    """
    try:
        pdf_bytes = uploaded_file.read()

        # Step 1: Extract text
        pages = extract_text_from_pdf(pdf_bytes)

        # Step 2: Split into chunks
        chunks = split_into_chunks(pages, chunk_size=300, chunk_overlap=80)

        # Step 3: Build TF-IDF index
        vectorizer, tfidf_matrix = build_tfidf_index(chunks)

        # Store everything in session state
        st.session_state.doc_pages = pages
        st.session_state.doc_chunks = chunks
        st.session_state.vectorizer = vectorizer
        st.session_state.tfidf_matrix = tfidf_matrix
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_size = uploaded_file.size
        st.session_state.num_pages = len(pages)
        st.session_state.processing_error = None

        # Count total pages (including those without text)
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        st.session_state.total_pages = doc.page_count
        doc.close()

        # Clear old answers when a new document is loaded
        st.session_state.last_answer = None
        st.session_state.last_query = None
        st.session_state.last_sources = None

    except ValueError as e:
        clear_document_state()
        st.session_state.processing_error = str(e)
    except Exception as e:
        clear_document_state()
        st.session_state.processing_error = (
            f"An unexpected error occurred while processing the PDF: {e}"
        )


def truncate_text(text, max_length=250):
    """Truncate text to a maximum length, adding ellipsis if needed."""
    if len(text) <= max_length:
        return text
    return text[:max_length].rsplit(" ", 1)[0] + "..."


# =============================================================================
# SIDEBAR - Document Workspace
# =============================================================================

def render_sidebar():
    """Render the sidebar with document upload and status information."""

    with st.sidebar:
        st.subheader("Your documents")
        uploaded_file = st.file_uploader(
            "Choose a PDF",
            type=["pdf"],
            help="Upload a PDF with selectable text. Scanned image-only "
                 "PDFs are not supported in this version.",
        )

        # Detect document change or removal
        if uploaded_file is None:
            # User removed the file or hasn't uploaded one
            if st.session_state.uploaded_file_name is not None:
                clear_document_state()
        elif (uploaded_file.name != st.session_state.uploaded_file_name
              or uploaded_file.size != st.session_state.uploaded_file_size):
            # New file uploaded or file changed
            with st.spinner("Processing PDF..."):
                process_uploaded_pdf(uploaded_file)

        # -- Document Processing Error --
        if st.session_state.processing_error:
            st.error(st.session_state.processing_error)

        # -- Document Status --
        if st.session_state.doc_chunks is not None:
            st.markdown("---")
            st.markdown("**Current document**")
            st.caption(st.session_state.uploaded_file_name)
            st.write(
                f"{st.session_state.total_pages} pages - "
                f"{len(st.session_state.doc_chunks)} searchable passages"
            )
            st.caption("Document ready")
            if st.button("Remove document", width="stretch"):
                clear_document_state()
                st.rerun()

        st.markdown("---")
        st.markdown("**Gemini connection**")
        api_key, model_name = get_api_config()
        config_fingerprint = hashlib.sha256(
            f"{api_key}\0{model_name}".encode("utf-8")
        ).hexdigest()
        if (st.session_state.gemini_status is None
                or st.session_state.gemini_config_fingerprint
                != config_fingerprint):
            with st.spinner("Verifying Gemini connection..."):
                st.session_state.gemini_status = check_gemini_connection(
                    api_key, model_name
                )
            st.session_state.gemini_config_fingerprint = config_fingerprint
        status = st.session_state.gemini_status
        if status["connected"]:
            st.success(status["message"])
        else:
            st.error(status["message"])
            st.markdown(
                "**Setup:** Copy `.env.example` to `.env` and add your "
                "[Gemini API key](https://aistudio.google.com/apikey)."
            )

        if st.button("Re-check connection", width="stretch"):
            with st.spinner("Verifying Gemini connection..."):
                st.session_state.gemini_status = check_gemini_connection(
                    api_key, model_name
                )
            st.session_state.gemini_config_fingerprint = config_fingerprint
            st.rerun()

        st.markdown("---")
        st.markdown("**About**")
        st.caption(
            "DocuMind AI uses TF-IDF retrieval to find relevant passages "
            "from your PDF, then sends them to Google Gemini for answer "
            "generation. This is a lightweight RAG demonstration."
        )


# =============================================================================
# MAIN AREA - Header
# =============================================================================

def render_header():
    """Render the application header with title and LLM status badge."""
    status = st.session_state.gemini_status

    if status and status["connected"]:
        badge_html = '<span class="status-pill status-pill-connected">Gemini connected</span>'
    else:
        badge_html = '<span class="status-pill">Gemini not connected</span>'

    st.markdown(f"""
    <div class="app-header">
        <h1>DocuMind <span style="font-weight: 400; color: #69717d;">/ document assistant</span></h1>
        <p>Search your PDF and get answers grounded in its contents. {badge_html}</p>
    </div>
    """, unsafe_allow_html=True)


# =============================================================================
# MAIN AREA - Welcome State
# =============================================================================

def render_welcome():
    """Render the welcome message when no document is uploaded."""
    st.markdown("""
    <div class="welcome-panel">
        <div class="step-label">DOCUMENT Q&A</div>
        <h2>Find the details that matter.</h2>
        <p>Add a text-based PDF from the sidebar, then ask a question. Answers include references to the pages used.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("#### Workflow")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="step-label">01 / ADD A FILE</div>', unsafe_allow_html=True)
        st.write("Upload a PDF with selectable text.")
    with col2:
        st.markdown('<div class="step-label">02 / ASK</div>', unsafe_allow_html=True)
        st.write("Ask a question in your own words.")
    with col3:
        st.markdown('<div class="step-label">03 / REVIEW</div>', unsafe_allow_html=True)
        st.write("Check the answer against cited pages.")


# =============================================================================
# MAIN AREA - Question & Answer Interface
# =============================================================================

def render_qa_interface():
    """Render the question input and answer display area."""

    # -- Example Question Suggestions --
    st.markdown("##### Start with a question")
    example_questions = [
        "Summarise the key points",
        "What dates or deadlines are mentioned?",
        "What does it say about eligibility?",
    ]

    cols = st.columns(3)
    for i, question in enumerate(example_questions):
        with cols[i]:
            if st.button(
                question, key=f"example_{i}", width="stretch"
            ):
                st.session_state.question_input = question
                st.session_state.question_field = question

    st.markdown("---")

    # -- Question Input --
    question = st.text_input(
        "Ask a question about your document:",
        value=st.session_state.question_input,
        placeholder="e.g., What is the attendance policy?",
        key="question_field",
    )

    # Update stored question when user types
    st.session_state.question_input = question

    ask_clicked = st.button(
        "Ask about this document",
        type="primary",
        width="stretch",
    )

    # -- Process Question --
    if ask_clicked:
        handle_question(question)

    # -- Display Last Answer --
    if st.session_state.last_answer is not None:
        display_answer()


def handle_question(question):
    """Process a user question: retrieve passages, call Gemini, store results."""
    # Validate question
    if not question or not question.strip():
        st.warning("Please enter a question before clicking Ask.")
        return

    # Check LLM connection
    api_key, model_name = get_api_config()
    status = st.session_state.gemini_status

    if not status or not status["connected"]:
        st.error(
            "Gemini is not connected. Please configure your API key "
            "in the `.env` file and click 'Re-check Connection' in the sidebar."
        )
        return

    # Check document is loaded
    if st.session_state.doc_chunks is None:
        st.warning("Please upload a document first.")
        return

    with st.spinner("Searching the document and preparing an answer..."):
        # Step 1: Retrieve relevant chunks
        retrieved = retrieve_relevant_chunks(
            query=question,
            chunks=st.session_state.doc_chunks,
            vectorizer=st.session_state.vectorizer,
            tfidf_matrix=st.session_state.tfidf_matrix,
            top_k=5,
            threshold=0.05,
        )

        # Step 2: Generate answer via Gemini
        result = generate_answer(
            query=question,
            retrieved_chunks=retrieved,
            api_key=api_key,
            model_name=model_name,
        )

        # Store results in session state
        st.session_state.last_query = question
        st.session_state.last_sources = retrieved
        st.session_state.last_answer = result

    # Force re-render to show the answer
    st.rerun()


def display_answer():
    """Display the generated answer and source references."""
    result = st.session_state.last_answer
    query = st.session_state.last_query
    sources = st.session_state.last_sources

    st.caption("YOUR QUESTION")
    st.write(query)

    if result.get("error"):
        st.error(f"Could not generate an answer: {result['error']}")

    if result.get("answer"):
        with st.container(border=True):
            st.caption("ANSWER")
            st.markdown(result["answer"])

    if sources:
        page_numbers = sorted(set(s["page_number"] for s in sources))
        pages_str = ", ".join(str(p) for p in page_numbers)
        st.caption(f"Sources - pages {pages_str}")

        with st.expander(
            f"View retrieved passages ({len(sources)})",
            expanded=False,
        ):
            for i, source in enumerate(sources, 1):
                score_pct = int(source["score"] * 100)
                with st.container(border=True):
                    st.caption(
                        f"Passage {i} - page {source['page_number']} - "
                        f"relevance {score_pct}%"
                    )
                    st.write(truncate_text(source["text"], 400))

    elif result.get("answer"):
        # Answer was generated but no sources? This shouldn't happen in normal
        # flow, but handle it gracefully.
        st.caption("No source passages to display.")


# =============================================================================
# FOOTER
# =============================================================================

def render_footer():
    """Render the application footer with disclaimers."""
    st.caption(
        "Answers are based on retrieved passages. Verify important details "
        "against the original PDF."
    )


# =============================================================================
# MAIN APPLICATION
# =============================================================================

def main():
    """Main application entry point."""
    # Render sidebar
    render_sidebar()

    # Render header
    render_header()

    # Render main content area
    if st.session_state.doc_chunks is not None:
        # Document is loaded - show Q&A interface
        render_qa_interface()
    else:
        # No document - show welcome screen
        render_welcome()

    # Render footer
    render_footer()


if __name__ == "__main__":
    main()
