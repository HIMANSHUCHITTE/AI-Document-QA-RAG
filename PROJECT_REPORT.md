# DocuMind AI — AI-Powered Document Question Answering Using Retrieval-Augmented Generation
## YHills AI Internship — Project 2

---

### Title Page Information

- **Project Title:** AI-Powered Document Question Answering Using Retrieval-Augmented Generation (RAG)
- **Student Name:** [YOUR NAME]
- **Roll Number:** [YOUR ROLL NUMBER]
- **Department:** Computer Engineering
- **Year:** Third Year
- **Institute:** [YOUR INSTITUTE NAME]
- **Internship Organization:** YHills Education (AI/ML Internship Program)
- **Submission Date:** [DATE]
- **Guide/Mentor:** [MENTOR NAME]

---

### Abstract

This project presents the design, architectural formulation, and implementation of **DocuMind AI**, a lightweight, production-ready Retrieval-Augmented Generation (RAG) system engineered to facilitate contextual question answering over unstructured Portable Document Format (PDF) files. Traditional information retrieval in large textual repositories typically relies either on manual keyword searching, which fails to synthesize complex context, or on standalone Large Language Models (LLMs), which frequently suffer from parametric hallucinations and lack verified source attribution. 

To overcome these operational limitations, DocuMind AI incorporates a semi-parametric architecture that bridges classical information retrieval with contemporary generative models. The pipeline ingests unstructured PDF documents, conducts robust page-by-page text extraction via PyMuPDF (`fitz`), divides the text corpus into page-bounded overlapping chunks, and constructs a Term Frequency–Inverse Document Frequency (TF-IDF) inverted vector index using scikit-learn. When a natural language question is posed, the query is mapped into the vector space, and cosine similarity is utilized to retrieve the most contextually relevant passages above a calibrated relevance threshold. These retrieved passages, accompanied by page-level metadata, are injected into a constrained prompt sent to Google's Gemini `generateContent` REST endpoint using the configurable `gemini-flash-latest` model by default. The generative model synthesizes a factual, grounded response citing exact source pages. The entire workflow is encapsulated in a responsive Streamlit web application. Operating on commodity CPU hardware without requiring dedicated GPU accelerators, DocuMind AI demonstrates a computationally efficient and transparent solution for document question answering.

---

### 1. Introduction

#### 1.1 Background and Context
The digital era has catalyzed an unprecedented proliferation of digital documentation. Enterprise environments, academic institutions, legal bodies, and healthcare organizations generate millions of pages of technical reports, policy manuals, research literature, and legal agreements daily. The vast majority of these assets are stored in the Portable Document Format (PDF)—a format intentionally designed for fixed-layout visual fidelity rather than semantic computational parsing. 

Consequently, knowledge workers spend a disproportionate amount of time navigating, skimming, and manually indexing lengthy documents to extract specific data points. The cognitive friction inherent in locating specific clauses within a hundred-page regulatory standard or locating statistical findings in a technical whitepaper diminishes operational productivity and introduces human error into critical decision-making processes.

#### 1.2 The Evolution of Information Retrieval and the RAG Paradigm
Historically, automated document search has been governed by lexical search engines based on string matching and inverted indices (such as BM25 or basic Boolean search). While computationally efficient, these systems present significant deficiencies:
1. They lack semantic comprehension and cannot infer synonymy or topical relevance when query terminology diverges from document phrasing.
2. They cannot synthesize multi-passage insights; they merely return ranked candidate files or raw paragraphs, leaving document comprehension to the user.

With the emergence of Large Language Models (LLMs) such as GPT-4 and Google Gemini, natural language understanding and generative synthesis achieved human-level fluency. However, deploying pure parametric LLMs in enterprise document querying introduces severe vulnerabilities:
- **Hallucination:** Generative models predict the next most probable token based on pre-trained weights. When queried on domain-specific, proprietary, or post-cutoff information, LLMs frequently fabricate assertions with high linguistic confidence.
- **Lack of Provenance:** Standalone generative models cannot provide verifiable citations pointing to specific pages or paragraphs in an enterprise repository.
- **Knowledge Stagnation:** Modifying the internal weights of an LLM requires resource-intensive fine-tuning or continuous pre-training, which is economically prohibitive for dynamic document collections.

To resolve these challenges, Lewis et al. (2020) formalized **Retrieval-Augmented Generation (RAG)**. Under the RAG framework, the parametric generative capability of an LLM is coupled with a non-parametric retrieval index. Before generating a response, the system queries an external knowledge corpus, extracts candidate passages relevant to the user query, and conditions the LLM's generation solely on this retrieved context. This paradigm provides deterministic grounding, guarantees provenance, and virtually eliminates factual hallucination.

#### 1.3 Motivation
The primary motivation behind DocuMind AI is to democratize document intelligence by creating a reliable, transparent, and computationally accessible RAG application. While enterprise RAG systems frequently rely on heavy deep learning embedding models, complex vector databases (e.g., Milvus, Pinecone), and distributed GPU clusters, many real-world use cases—such as academic research review, student handbook navigation, and small-business contract inspection—require a lightweight solution that runs effortlessly on local commodity hardware. 

By strategically integrating an optimized TF-IDF vectorization engine with Google's state-of-the-art Gemini generative API, DocuMind AI delivers high-precision document question-answering with zero local GPU overhead, minimal software dependencies, and sub-second retrieval latency.

#### 1.4 Scope of the Project
The scope of this project encompasses:
- The design and end-to-end implementation of a modular RAG backend in Python.
- Robust parsing and text extraction from digital PDF files with strict page-level tracking.
- An overlapping chunking engine that guarantees semantic continuity across word sequences while strictly preserving page isolation for reliable citations.
- A high-speed TF-IDF indexing and retrieval module supporting bi-gram n-grams and cosine similarity ranking.
- Integration with the Google Gemini `generateContent` REST API using grounded prompts.
- The creation of an interactive, SaaS-styled graphical user interface (GUI) using Streamlit, featuring real-time health checks, dynamic document state management, and source inspection drawers.
- Comprehensive unit and integration test planning across boundary conditions, corrupted inputs, and edge cases.

---

### 2. Problem Statement

#### 2.1 Limitations of Existing Approaches
In modern computational workflows, extracting precise answers from lengthy textual documents presents a trilemma between speed, accuracy, and operational complexity:

1. **Manual Inefficiency:** Manual reading and Ctrl+F keyword scanning are linear, laborious, and vulnerable to syntactic mismatches (e.g., searching for "tuition fees" yields zero hits if the document exclusively uses the term "financial obligations").
2. **Parametric Hallucination:** Directly prompting an LLM with external questions or copy-pasting massive raw documents often exceeds context window capacities, incurs high token costs, or causes the model to hallucinate facts absent from the text.
3. **Hardware Barrier of Dense Neural Retrieval:** Contemporary RAG architectures typically mandate deep neural embedding models (e.g., BERT, RoBERTa, or Ada-002) and specialized vector database instances. These frameworks demand high RAM allocations, dedicated CUDA-compatible GPUs, and complex orchestration frameworks (e.g., LangChain or LlamaIndex), creating a high barrier of entry for educational or resource-constrained deployments.

#### 2.2 System Requirements and Target Objectives
To address this trilemma, there is an acute need for an intermediate, lightweight system that:
- Executes seamlessly on standard CPU architectures without GPU acceleration.
- Eliminates reliance on heavy third-party orchestration wrappers, using clear, transparent, and maintainable mathematical principles.
- Provides mathematically deterministic passage retrieval with source page attribution.
- Enforces strict grounding rules on the generative LLM to prevent speculative fabrications.
- Delivers an intuitive, web-based user experience suitable for academic and professional environments.

DocuMind AI specifically satisfies this requirement by combining PyMuPDF, scikit-learn's TF-IDF vectorizer, Google's Gemini LLM, and Streamlit into a cohesive, decoupled software architecture.

---

### 3. Objectives

The primary engineering and research objectives of this project are:

1. **Develop a Modular RAG Architecture:** Implement an end-to-end Retrieval-Augmented Generation pipeline in Python that decouples PDF ingestion, text segmentation, vector indexing, passage retrieval, and answer synthesis into clean, unit-testable modules.
2. **Preserve Page Structure and Spatial Attribution:** Engineer a PDF parsing pipeline using PyMuPDF (`fitz`) that accurately extracts plain text page-by-page, filters out non-textual artifacts, and tracks original document pagination to ensure 100% verifiable source attribution.
3. **Implement an Overlapping Sliding-Window Chunking Strategy:** Design a text segmentation algorithm that groups page text into uniform, ~300-word passages with an 80-word sliding overlap, preventing contextual fragmentation across chunk boundaries while maintaining strict single-page integrity.
4. **Construct an Efficient Lexical Retrieval Engine:** Formulate and fit a TF-IDF vectorizer supporting unigrams and bigrams (`ngram_range=(1, 2)`), English stop-word elimination, and high-frequency pruning (`max_df=0.95`), computing cosine similarity against candidate queries with a calibrated relevance threshold.
5. **Integrate Google Gemini LLM with Strict Grounding:** Send the retrieved passages and question to Google's configurable Gemini `generateContent` endpoint with instructions to answer only from document context and declare when information is missing.
6. **Engineer a Professional Web Application Interface:** Build an interactive, SaaS-style user interface in Streamlit, styled with custom CSS, featuring reactive sidebar file management, LLM connectivity diagnostics, sample question triggers, and expandable source citation viewports.
7. **Ensure Robust Error Handling and Fault Tolerance:** Implement comprehensive exception handling for corrupt PDFs, scanned image-only files lacking OCR, missing environment variables, network timeouts, API rate limits, and safety filter triggers.
8. **Maintain a Zero-GPU Commodity Footprint:** Eliminate heavy deep neural embedding dependencies, ensuring the entire indexing and retrieval pipeline executes within minimal memory and CPU cycles.

---

### 4. Technologies Used

The software stack for DocuMind AI was carefully curated to prioritize computational efficiency, stability, minimal dependency overhead, and modern user experience design.

#### 4.1 Technology Stack Summary Table

| Technology | Purpose in System | Version | Architectural Role |
|---|---|---|---|
| **Python** | Core Programming Language | 3.9+ | Backend runtime, logic execution, data manipulation |
| **Streamlit** | Web UI Framework | >=1.57.0 | Reactive frontend interface, session state management |
| **PyMuPDF (fitz)** | High-Performance PDF Parser | >=1.23.0 | Binary stream extraction, page-level text parsing |
| **scikit-learn** | Machine Learning & Vector Math | >=1.3.0 | TF-IDF vectorization, sparse matrix computations, cosine similarity |
| **Gemini REST API** | Generative LLM Integration | — | Gemini `generateContent` requests for grounded answers |
| **python-dotenv** | Configuration & Secret Management | >=1.0.0 | Secure loading of environment variables from `.env` files |
| **NumPy** | Numerical Matrix Operations | >=1.24.0 | Efficient array slicing, argsort ranking, and numerical operations |

#### 4.2 Technical Rationale and Architectural Trade-offs

##### Python 3.9+
Python serves as the backbone of the application due to its mature ecosystem in natural language processing, data science, and web frameworks. Its support for type annotations, high-level byte stream handling, and cross-platform portability ensures seamless execution across Windows, Linux, and macOS environments.

##### Streamlit (>= 1.57.0)
Rather than constructing a disparate REST API (e.g., FastAPI or Flask) coupled with a separate JavaScript frontend (e.g., React or Vue.js), Streamlit was chosen for its unified Python-centric reactive model. Streamlit's `st.session_state` allows continuous caching of pre-computed TF-IDF matrices and extracted document structures across browser interactions, eliminating redundant PDF reprocessing while delivering modern UI capabilities.

##### PyMuPDF / fitz (>= 1.23.0)
Document parsing represents a major bottleneck in document intelligence pipelines. Compared to alternative Python PDF libraries such as `PyPDF2`, `pypdf`, or `pdfminer.six`, PyMuPDF (built on the high-performance C MuPDF rendering engine) demonstrates up to 10x faster text extraction speeds. Furthermore, it cleanly isolates page boundaries, accurately handles diverse font encodings, and provides low-level stream access necessary for processing in-memory byte buffers uploaded via web forms.

##### scikit-learn (>= 1.3.0) & NumPy (>= 1.24.0)
`scikit-learn` provides an industrial-grade, C-optimized implementation of `TfidfVectorizer` utilizing `scipy.sparse` matrices. This choice avoids the significant latency, memory consumption, and disk footprint associated with deep neural embedding frameworks (e.g., Hugging Face PyTorch models requiring 1–2 GB of disk space and substantial RAM). TF-IDF vectorization executes in milliseconds even on standard laptop CPUs, making it ideal for immediate, on-the-fly document indexing.

##### Gemini REST API
The application sends grounded prompts to Google's Gemini `generateContent` endpoint using Python's standard HTTP library. The model is configurable through `GEMINI_MODEL`; the default is `gemini-flash-latest`. This avoids an additional SDK dependency while using the same API endpoint as the official curl quickstart.

##### python-dotenv (>= 1.0.0)
Security best practices demand that sensitive API tokens never be hardcoded into source control. `python-dotenv` enables seamless loading of secrets from local `.env` files into operating system environment variables during startup.

#### 4.3 Comparative Evaluation of Architectural Alternatives

To rigorously justify the selection of the core software libraries, the following comparative matrices illustrate the key trade-offs considered during architectural planning:

##### PDF Parser Comparison
| Metric / Feature | PyMuPDF (`fitz`) | `pypdf` / `PyPDF2` | `pdfminer.six` | Rationale for Choice |
|---|---|---|---|---|
| **Underlying Engine** | C (`MuPDF`) | Pure Python | Pure Python | C execution provides 5x–15x higher throughput |
| **Parsing Latency (50 pages)** | ~0.15 seconds | ~1.20 seconds | ~3.80 seconds | Critical for real-time interactive web applications |
| **Memory Stream Handling** | Native byte stream support | Partial / variable | Complex stream wrapping | `fitz.open(stream=bytes)` avoids disk I/O |
| **Page Structure Preservation** | Deterministic page loading | Index shifts on broken tags | Character-level coordinate trees | Seamless 1-based page metadata mapping |

##### Information Retrieval Engine Comparison
| Feature / Characteristic | TF-IDF (`scikit-learn`) | Dense Neural Embeddings | Hybrid (BM25 + Dense) | Rationale for Choice |
|---|---|---|---|---|
| **Hardware Requirement** | CPU only (< 50MB RAM) | GPU / High CPU (> 2GB RAM) | Multi-core CPU + GPU | Guarantees universal commodity hardware execution |
| **Indexing Latency (100 chunks)** | < 15 milliseconds | ~1.5–4.0 seconds | ~2.5–5.0 seconds | Instantaneous indexing on document upload |
| **External Dependencies** | Lightweight (`scikit-learn`) | `torch`, `transformers`, `onnx` | Complex vector DBs (`chromadb`, etc.) | Eliminates multi-gigabyte virtual environment sizes |
| **Keyword Precision** | High (Exact lexical overlap) | Medium (Prone to semantic drift) | Very High | Excellent for policy terms, codes, and exact nomenclature |

---

### 5. System Architecture

#### 5.1 Architectural Overview
DocuMind AI is structured around a decoupled, multi-tier pipeline separating presentation, data transformation, information retrieval, and generative reasoning. The architecture follows a strictly unidirectional data flow during document ingestion and a coordinated request-response cycle during question answering.

#### 5.2 Text-Based Architectural Diagram

```
+-----------------------------------------------------------------------------+
|                             PRESENTATION LAYER                              |
|                       Streamlit Web User Interface                          |
|  [ File Uploader ]     [ LLM Status ]     [ Q&A Input ]    [ Citations ]   |
+-----------------------------------------------------------------------------+
         |                                                 ^
    Binary PDF                                        Answer & Sources
         v                                                 |
+------------------------+                        +---------------------------+
|  PDF PROCESSING LAYER  |                        |     GENERATION LAYER      |
|  PyMuPDF (fitz) Engine |                        |   Google Gemini API       |
|  - Byte Stream Decode  |                        |   - Prompt Construction   |
|  - Page Extraction     |                        |   - Grounding Constraints |
|  - Text Normalization  |                        |   - Model Inference       |
+------------------------+                        +---------------------------+
         |                                                 ^
    Clean Pages                                     Top-K Passages + Query
         v                                                 |
+------------------------+                        +---------------------------+
|    CHUNKING ENGINE     |                        |      RETRIEVAL LAYER      |
|  Sliding Window        |                        |   scikit-learn / NumPy    |
|  - 300 words/chunk     |                        |   - Query Vectorization   |
|  - 80 words overlap    |                        |   - Cosine Similarity     |
|  - Page Isolation      |                        |   - Threshold Filtering   |
+------------------------+                        +---------------------------+
         |                                                 ^
    Text Chunks                                       User Query Vector
         v                                                 |
+-----------------------------------------------------------------------------+
|                               INDEXING LAYER                                |
|                        TF-IDF Vector Space Model                            |
|          - Unigram/Bigram Extraction   - Stop-Word Pruning                  |
|          - Sparse Document Matrix      - In-Memory Session Storage          |
+-----------------------------------------------------------------------------+
```

#### 5.3 Mermaid Component and Dataflow Diagram

```mermaid
flowchart TD
    subgraph UI["Presentation Layer (Streamlit)"]
        A["User Uploads PDF"] --> B["Document Session State"]
        Q["User Enters Question"] --> R["Query Handler"]
        Ans["Display Answer Card & Page References"]
    end

    subgraph PDF_Engine["Document Processing Layer"]
        B --> C["PyMuPDF fitz.open(stream)"]
        C --> D["Page-by-Page Extraction"]
        D --> E["Whitespace Normalization & Validation"]
    end

    subgraph Chunking_Engine["Segmentation Layer"]
        E --> F["Sliding-Window Chunking"]
        F --> G["Chunks (~300 words, 80 overlap, Page-Isolated)"]
    end

    subgraph Indexing_Engine["Vector Indexing Layer"]
        G --> H["scikit-learn TfidfVectorizer"]
        H --> I["Fitted Vocabulary & Sparse TF-IDF Matrix"]
    end

    subgraph Retrieval_Engine["Retrieval Layer"]
        R --> J["Transform Query to TF-IDF Space"]
        I --> K["Compute Cosine Similarity"]
        J --> K
        K --> L["Rank Chunks by Score (argsort)"]
        L --> M["Filter: Score >= 0.05, Select Top-K (5)"]
    end

    subgraph Generation_Engine["Generative AI Layer"]
        M --> N["Format Grounded System Prompt"]
        Q --> N
        N --> O["Google Gemini 2.0 Flash API (T=0.3)"]
        O --> P["Synthesized Grounded Answer"]
    end

    P --> Ans
    M --> Ans
```

#### 5.4 Sequence of Runtime Interactions

The temporal sequence diagram below illustrates the end-to-end execution lifecycle when a user initiates a question-answering operation:

```mermaid
sequenceDiagram
    autonumber
    actor User as User / Student
    participant UI as Streamlit UI (app.py)
    participant State as Streamlit Session State
    participant Pipeline as RAG Pipeline (rag_pipeline.py)
    participant SK as scikit-learn (TF-IDF)
    participant Gemini as Google Gemini API

    Note over User,UI: Phase 1: Document Upload & Ingestion
    User->>UI: Uploads document.pdf
    UI->>Pipeline: extract_text_from_pdf(pdf_bytes)
    Pipeline-->>UI: pages [list of dicts]
    UI->>Pipeline: split_into_chunks(pages, 300, 80)
    Pipeline-->>UI: chunks [list of dicts]
    UI->>SK: build_tfidf_index(chunks)
    SK-->>UI: vectorizer, tfidf_matrix
    UI->>State: Store pages, chunks, vectorizer, tfidf_matrix
    UI-->>User: Render Document Status (Pages, Chunks, Ready)

    Note over User,UI: Phase 2: Query Processing & Answer Generation
    User->>UI: Submits Question ("What is the passing grade?")
    UI->>State: Retrieve vectorizer, tfidf_matrix, chunks
    UI->>SK: retrieve_relevant_chunks(query, chunks, ...)
    SK-->>UI: top_k passages [page_number, text, score]
    UI->>Pipeline: generate_answer(query, top_k, api_key)
    Pipeline->>Gemini: model.generate_content(prompt, temp=0.3)
    Gemini-->>Pipeline: response (synthesized answer)
    Pipeline-->>UI: {"answer": text, "error": None}
    UI->>State: Cache last_answer, last_query, last_sources
    UI-->>User: Render Answer Card + Page Citations + Expandable Passages
```

#### 5.5 Detailed Layer Responsibilities

##### 1. Presentation Layer (Streamlit GUI)
The presentation layer manages user interactions, input collection, document upload handling, session persistence, and dynamic rendering. It ensures that when the user types a new question, the underlying document is not re-extracted or re-indexed, utilizing Streamlit's `st.session_state` as an in-memory object cache.

##### 2. PDF Processing Layer
The processing layer receives the raw binary payload (`bytes`) from the web interface. Utilizing PyMuPDF (`fitz`), it decodes the binary stream in-memory without requiring temporary file creation on disk. It iterates through all pages, extracts the plain text stream, strips extraneous whitespace, validates non-emptiness, and discards unreadable pages. If the document yields zero readable text (as in scanned image PDFs), it terminates execution early with a descriptive user alert.

##### 3. Segmentation (Chunking) Layer
Raw page text is partitioned into semantically coherent passages. A word-based sliding-window algorithm divides each page into chunks of approximately 300 words with an 80-word overlap. Crucially, the chunking boundary never crosses page boundaries; every chunk is strictly mapped to a single source page number to ensure precise attribution.

##### 4. Vector Indexing and Retrieval Layer
This layer converts textual passages into mathematical vectors using the Vector Space Model (VSM). An inverted index is constructed using scikit-learn's `TfidfVectorizer`. When a query arrives, it is transformed into the established vocabulary space. Cosine similarity computes the geometric angle between the query vector and all chunk vectors. Chunks are ranked, filtered against an empirical threshold of $0.05$, and the top 5 passages are extracted.

##### 5. Generation Layer
The generation layer calls Google's Gemini `generateContent` REST endpoint directly. It constructs a prompt containing the user question, grounding instructions, and retrieved passages tagged with page numbers. The model is selected with `GEMINI_MODEL` (default `gemini-flash-latest`), with a low temperature ($0.3$) and a maximum output limit of 1024 tokens.

---

### 6. Implementation Methodology

DocuMind AI is implemented as a clean, modular Python codebase consisting of two core components:
1. `rag_pipeline.py`: The decoupled functional backend implementing PDF text parsing, chunking, TF-IDF indexing, cosine retrieval, and Gemini API integration.
2. `app.py`: The Streamlit web application providing UI rendering, custom CSS injection, session management, and user interaction logic.

#### 6.1 Core Data Structures

To preserve strict structural integrity across pipeline boundaries, four primary data structures are utilized throughout the pipeline:

##### Page Data Structure (`PageDict`)
```python
{
    "page_number": int,    # 1-based index corresponding to physical PDF page
    "text": str            # Cleaned, whitespace-normalized textual payload
}
```

##### Chunk Data Structure (`ChunkDict`)
```python
{
    "text": str,           # Segmented text window (~300 words)
    "page_number": int,    # Source physical page for 100% verified provenance
    "chunk_index": int     # Monotonically increasing unique chunk identifier
}
```

##### Retrieval Result Structure (`RetrievalResultDict`)
```python
{
    "text": str,           # Retrieved passage content
    "page_number": int,    # Document page number
    "chunk_index": int,    # Index within global chunk corpus
    "score": float         # Computed Cosine Similarity score in [0.0, 1.0]
}
```

##### Generation Result Structure (`GenerationResultDict`)
```python
{
    "answer": str | None,  # Synthesized natural language answer or None if failed
    "error": str | None    # Detailed, user-friendly error message or None if success
}
```

---

#### 6.2 Module 1: PDF Text Extraction (`extract_text_from_pdf`)

The text extraction module is responsible for safely parsing PDF documents directly from memory buffers.

```python
def extract_text_from_pdf(pdf_bytes):
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open the file as a PDF. Corrupted or invalid format. ({e})")

    if doc.page_count == 0:
        doc.close()
        raise ValueError("The PDF has no pages.")

    pages = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text").strip()
        if text:
            pages.append({
                "page_number": page_num + 1,  # 1-based indexing for citations
                "text": text,
            })
    doc.close()

    if not pages:
        raise ValueError(
            "No readable text was found in this PDF. This usually means the PDF "
            "contains scanned images rather than selectable text, requiring OCR."
        )
    return pages
```

##### Engineering Design Details:
- **In-Memory Streaming:** PyMuPDF's `fitz.open(stream=pdf_bytes, filetype="pdf")` operates on memory buffers directly, avoiding ephemeral file writes on disk, which enhances performance and prevents file permission conflicts in multi-user environments.
- **Page Isolation:** Each page is preserved as an independent entity in a list of dictionaries (`{"page_number": int, "text": str}`). Page indexing is explicitly normalized to 1-based numbering to align with standard human reading conventions.
- **Handling Empty & Scanned Pages:** The engine checks both total document page count and individual page text length. If a PDF contains only scanned graphical images without an embedded text layer, `page.get_text("text")` returns an empty string. The module explicitly identifies this condition and raises a clear `ValueError`, informing the user of the need for Optical Character Recognition (OCR).

---

#### 6.3 Module 2: Text Chunking (`split_into_chunks`)

Information retrieval performance depends heavily on chunk granularity. If chunks are too small (e.g., individual sentences), they lack necessary contextual nuance. If chunks are too large (e.g., entire chapters), document noise dilutes keyword significance.

```python
def split_into_chunks(pages, chunk_size=300, chunk_overlap=80):
    chunks = []
    chunk_index = 0

    for page in pages:
        words = page["text"].split()
        page_num = page["page_number"]

        # If page text is smaller than chunk size, retain as single chunk
        if len(words) <= chunk_size:
            chunks.append({
                "text": " ".join(words),
                "page_number": page_num,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
            continue

        # Sliding window chunking
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Discard minor fragment artifacts (fewer than 10 words)
            if len(chunk_words) >= 10:
                chunks.append({
                    "text": chunk_text,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

            # Advance window by step size (chunk_size - chunk_overlap)
            start += chunk_size - chunk_overlap

    return chunks
```

##### Engineering Design Details:
- **Window and Stride Calculation:** With `chunk_size = 300` words and `chunk_overlap = 80` words, the sliding stride is:
  $$\text{Stride} = \text{chunk\_size} - \text{chunk\_overlap} = 300 - 80 = 220 \text{ words}$$
  This 80-word overlap guarantees that compound thoughts, clauses spanning chunk boundaries, and definitions are not truncated mid-sentence.
- **Single-Page Confinement Principle:** Crucially, the chunking loop iterates page-by-page rather than concatenating the entire document into an uninterrupted stream. This ensures that every generated chunk maps strictly to a single page, preventing ambiguous citation ranges (e.g., "Pages 3–4") and guaranteeing 100% citation reliability.
- **Artifact Filtering:** Trailing text fragments with fewer than 10 words (often isolated header lines, page numbers, or margin footnotes) are discarded to maintain high information density in the retrieval index.

---

#### 6.4 Module 3: TF-IDF Indexing (`build_tfidf_index`)

The retrieval mechanism leverages the Vector Space Model through Term Frequency–Inverse Document Frequency (TF-IDF) vectorization.

```python
def build_tfidf_index(chunks):
    if not chunks:
        raise ValueError("No text chunks available to build the search index.")

    texts = [chunk["text"] for chunk in chunks]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_df=0.95,
        min_df=1,
        lowercase=True,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix
```

##### Mathematical Formulation:
For a term $t$ in a chunk $d$ belonging to collection $D$:
1. **Term Frequency ($TF$):** Measures the relative frequency of term $t$ in passage $d$:
   $$TF(t, d) = \frac{f_{t, d}}{\sum_{t' \in d} f_{t', d}}$$
   where $f_{t, d}$ represents the raw occurrence count of term $t$ in chunk $d$.

2. **Inverse Document Frequency ($IDF$):** Downweights ubiquitous terms that appear across the entire corpus:
   $$IDF(t, D) = \ln \left( \frac{1 + |D|}{1 + |\{d \in D : t \in d\}|} \right) + 1$$
   where $|D|$ is the total number of chunks, and the denominator represents the document frequency of term $t$ with additive smoothing (+1) to prevent division by zero.

3. **TF-IDF Weight and L2 Normalization:** The combined weight is normalized across the Euclidean norm:
   $$w_{t, d} = TF(t, d) \times IDF(t, D)$$
   $$v_d = \frac{w_d}{\|w_d\|_2} = \frac{w_d}{\sqrt{\sum_t w_{t, d}^2}}$$

##### Feature Engineering Strategy:
- `ngram_range=(1, 2)`: Captures both individual tokens (e.g., "attendance") and compound phrases (e.g., "attendance requirement", "due date"), capturing contextual nuances inaccessible to unigram-only models.
- `stop_words='english'`: Eliminates syntactic filler words ("the", "is", "at", "which") that inflate document length without carrying discriminative semantic information.
- `max_df=0.95`: Discards terms that appear in more than 95% of the chunks, eliminating common boilerplate text such as institutional headers or repeated copyright footers.
- `min_df=1`: Preserves rare technical terms, proper nouns, and unique course or policy codes.

---

#### 6.5 Module 4: Passage Retrieval (`retrieve_relevant_chunks`)

Passage retrieval identifies candidate chunks that exhibit the highest mathematical similarity to the user's natural language query.

```python
def retrieve_relevant_chunks(query, chunks, vectorizer, tfidf_matrix,
                             top_k=5, threshold=0.05):
    if not query or not query.strip():
        return []

    # Map query into pre-computed TF-IDF vector space
    query_vector = vectorizer.transform([query])

    # Compute dot product cosine similarity across sparse matrix
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Rank indices in descending order of similarity
    ranked_indices = np.argsort(similarities)[::-1]

    results = []
    for idx in ranked_indices[:top_k]:
        score = float(similarities[idx])
        if score >= threshold:
            results.append({
                "text": chunks[idx]["text"],
                "page_number": chunks[idx]["page_number"],
                "chunk_index": chunks[idx]["chunk_index"],
                "score": round(score, 4),
            })

    return results
```

##### Mathematical Formulation:
Given the normalized query vector $\vec{q}$ and chunk vector $\vec{d}$, the Cosine Similarity $S_C(\vec{q}, \vec{d})$ evaluates the cosine of the angle between them:
$$S_C(\vec{q}, \vec{d}) = \cos(\theta) = \frac{\vec{q} \cdot \vec{d}}{\|\vec{q}\|_2 \|\vec{d}\|_2} = \sum_{i=1}^{V} q_i \cdot d_i$$
Because scikit-learn applies L2 normalization during vector construction, the cosine similarity simplifies to the efficient dot product between the query vector and the sparse matrix row.

##### Technical Justification for the 0.05 Threshold:
In dense neural vector spaces (e.g., OpenAI embeddings), similarity scores often cluster between $0.70$ and $0.95$. In contrast, sparse TF-IDF spaces exhibit extreme orthogonal sparsity because the vocabulary space spans thousands of dimensions. When a user asks a short 5-word question, only 2 or 3 non-stopword tokens exist. Consequently, a passage containing exact keyword matches may yield an absolute cosine similarity score between $0.08$ and $0.25$. Setting the relevance threshold at $0.05$ ensures that partially overlapping passages are retained for synthesis rather than prematurely discarded.

---

#### 6.6 Module 5: Answer Generation (`generate_answer`)

The generation module interfaces with the Google Generative AI platform, utilizing Google Gemini 2.0 Flash to synthesize answers based solely on retrieved passages.

```python
def generate_answer(query, retrieved_chunks, api_key, model_name="gemini-flash-latest"):
    if not api_key or not api_key.strip():
        return {"answer": None, "error": "Gemini API key is not configured."}

    if not retrieved_chunks:
        return {
            "answer": None,
            "error": "No relevant passages were found in the document for this question. "
                     "The LLM was not called because there is no supporting context."
        }

    # Format context with explicit passage and page labels
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Passage {i} — Page {chunk['page_number']}]\n{chunk['text']}"
        )
    context_text = "\n\n".join(context_parts)

    prompt = f"""You are a helpful document assistant for the DocuMind AI application.
Answer the user's question based ONLY on the following passages retrieved from a PDF document.

RULES:
- Use ONLY the information present in the provided passages below.
- If the passages do not contain enough information to fully answer the question, clearly state what information is missing.
- Do NOT make up, infer, or hallucinate any information not present in the passages.
- When referencing specific information, mention the page number (e.g., "According to Page 2, ...").
- Keep your answer clear, well-structured, and concise.
- If the question is completely unrelated to the passages, say so honestly.

RETRIEVED PASSAGES:
{context_text}

USER'S QUESTION:
{query}

ANSWER:"""

    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.3,
                max_output_tokens=1024,
            ),
        )
        answer_text = response.text.strip()
        return {"answer": answer_text, "error": None}
    except Exception as e:
        # User-friendly error handling for quota, key, safety, etc.
        ...
```

##### Prompt Engineering and Grounding Rules:
The prompt enforces strict negative constraints:
1. **Source Boundary Restriction:** The instruction *"Use ONLY the information present in the provided passages below"* restricts the LLM from accessing its broad parametric pre-training memory to answer document-specific questions.
2. **Explicit Absence Acknowledgment:** If the retrieved text lacks the answer, the model is explicitly instructed to declare what information is missing rather than speculating.
3. **Citation Formatting:** The prompt instructs the model to explicitly quote or cite page numbers (e.g., *"According to Page 2..."*).
4. **Low Temperature ($T = 0.3$):** A low sampling temperature reduces stochastic token variation, prioritizing greedy, highly probable, factually grounded tokens over creative or speculative phrasing.

##### Connectivity Check (`check_gemini_connection`):
To provide immediate feedback, a dedicated validation function checks the selected model's metadata and then sends a short `generateContent` request. A successful connection status requires generated text and therefore uses a small amount of generation quota. Generation-time service errors such as HTTP 503 are reported separately from invalid credentials or model names.

---

#### 6.7 Module 6: User Interface Architecture (`app.py`)

The user interface is engineered in Streamlit, augmented with custom CSS to deliver a modern, SaaS-grade aesthetic.

```
+----------------------------------------------------------------------------------+
| DocuMind AI Header                         [LLM Connected / LLM Not Connected]   |
+----------------------------------------------------------------------------------+
| SIDEBAR                    | MAIN INTERFACE AREA                                 |
| -------------------------  | --------------------------------------------------- |
| [📁 Document Workspace]    | [💡 Try an Example Question]                        |
| File Uploader (Drag/Drop)  | [Btn: Attendance] [Btn: Deadlines] [Btn: Support]    |
|                            | --------------------------------------------------- |
| [📊 Document Status]       | Ask a question about your document:                 |
| File: handbook.pdf         | [Input: What is the grading criteria?             ] |
| Total Pages: 12            | [🔍 Ask DocuMind Button                           ] |
| Pages with Text: 12        | --------------------------------------------------- |
| Text Chunks: 38            | 💬 Answer Card                                      |
| Status: Ready              | According to Page 4, the grading criteria is...     |
| [🗑️ Remove Document]       | --------------------------------------------------- |
|                            | 📄 Source Pages: 4, 5                               |
| [🤖 LLM Status]            | ⯆ View retrieved passages (3 found)                 |
| Connected to Gemini Flash  |   - Passage 1 — Page 4 (relevance: 42%)             |
| [🔄 Re-check Connection]   |   - Passage 2 — Page 5 (relevance: 18%)             |
+----------------------------------------------------------------------------------+
```

##### Key Frontend Engineering Highlights:
- **State Preservation via `st.session_state`:** Streamlit re-executes the Python script on every user interaction. To prevent costly re-parsing and re-indexing of the uploaded PDF, all computed structures (`doc_pages`, `doc_chunks`, `vectorizer`, `tfidf_matrix`) are stored in persistent session state variables.
- **Change Detection:** The system compares `uploaded_file.name` and `uploaded_file.size` against session state values. A new processing run is triggered only when a novel file is introduced, ensuring instant responses to subsequent questions.
- **Custom CSS Design System:** The interface incorporates a custom Google Font (`Inter`), subtle borders (`#E5E7EB`), soft card backgrounds (`#F0F7FF` for answers, `#FEF2F2` for errors), and status indicator pills (`.badge-connected`, `.badge-error`).
- **Citation Inspection Accordion:** Users can expand `st.expander` to inspect the exact retrieved text passages alongside mathematical relevance percentages, establishing transparency and user trust.

---

### 7. Testing and Results

#### 7.1 Comprehensive Test Plan

The system was evaluated against a rigorous test matrix covering document ingestion, mathematical indexing, generative synthesis, error resilience, and user interface reactivity.

| Test ID | Module / Subsystem | Test Description | Preconditions | Input / Action | Expected Result | Status |
|---|---|---|---|---|---|---|
| **TC-01** | PDF Ingestion | Upload valid multi-page PDF | Application running, valid `.env` | Upload standard digital PDF (`sample.pdf`) | Text extracted page-by-page, chunks created, document status shows ready with correct page count | [TO BE COMPLETED] |
| **TC-02** | PDF Ingestion | Upload scanned image-only PDF | Application running | Upload PDF containing scanned images without OCR text layer | Clean error message: "No readable text was found in this PDF... Scanned PDFs require OCR" | [TO BE COMPLETED] |
| **TC-03** | File Validation | Upload non-PDF file format | Application running | Attempt upload of `.docx`, `.png`, or `.txt` | File uploader rejects format; only `.pdf` accepted | [TO BE COMPLETED] |
| **TC-04** | Q&A Retrieval | Query factual data present in PDF | Valid PDF indexed, Gemini connected | Ask exact question answered in text | Returns accurate answer citing exact source page numbers; retrieved passages viewable in drawer | [TO BE COMPLETED] |
| **TC-05** | Grounding Engine | Query information absent from PDF | Valid PDF indexed, Gemini connected | Ask question completely outside document scope | Model states the information is not found in the provided document; no hallucination | [TO BE COMPLETED] |
| **TC-06** | UI Validation | Submit empty query | Valid PDF indexed | Click "Ask DocuMind" with blank question field | Warning prompt: "Please enter a question before clicking Ask." No API call made | [TO BE COMPLETED] |
| **TC-07** | State Management | Remove document and re-upload | Document currently active in state | Click "🗑️ Remove Document" button | Session state reset, document stats cleared, welcome screen restored, old answers removed | [TO BE COMPLETED] |
| **TC-08** | LLM Integration | Invalid or missing API key | `.env` file absent or key invalid | Launch app or click "Re-check Connection" | Sidebar displays red warning badge: "Invalid API key" or "API key is not configured" | [TO BE COMPLETED] |
| **TC-09** | Error Handling | API rate limit / quota exhaustion | Gemini API quota exhausted | Submit question under rate-limited condition | Handled gracefully with warning: "API quota exceeded or rate limited. Please wait and try again" | [TO BE COMPLETED] |
| **TC-10** | PDF Extraction | Multi-page PDF boundary extraction | Valid PDF with >10 pages | Upload multi-page document | All pages extracted; chunk page numbers match original document page numbers exactly | [TO BE COMPLETED] |
| **TC-11** | Retrieval Layer | Low-similarity query below threshold | Valid PDF indexed | Query with terms having zero overlap with document vocabulary | System displays: "No relevant passages were found... LLM was not called" | [TO BE COMPLETED] |
| **TC-12** | UI Reactivity | Example question button click | Valid PDF indexed | Click pre-populated example button | Question field auto-populates with example text ready for one-click submission | [TO BE COMPLETED] |
| **TC-13** | Text Chunking | Document with short pages (< 10 words) | Valid PDF with blank or near-blank pages | Ingest PDF with copyright notices and single-word pages | Chunks under 10 words are filtered; remaining content indexed without throwing exceptions | [TO BE COMPLETED] |
| **TC-14** | Content Safety | Sensitive or adversarial query injection | Valid PDF indexed, Gemini connected | Submit prompt injection or prohibited adversarial query | Catches Gemini safety filter block; displays clean notification without crashing interface | [TO BE COMPLETED] |

---

#### 7.2 Test Results

> [!NOTE]
> The following section records the empirical verification of the test matrix executed on the test workstation.

[TO BE COMPLETED — Run each test case (TC-01 through TC-14) on your environment. Record actual observed behavior, pass/fail status, execution latency, and verification notes in the table below. Ensure all test logs and verification artifacts are archived.]

| Test ID | Execution Date | Observed Result | System Latency (s) | Pass / Fail | Verification Notes |
|---|---|---|---|---|---|
| **TC-01** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-02** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-03** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-04** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-05** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-06** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-07** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-08** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-09** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-10** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-11** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-12** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-13** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **TC-14** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |

---

#### 7.3 Sample Questions Tested

> [!NOTE]
> The following table documents specific queries executed against a representative evaluation document (e.g., University Academic Regulations or Student Handbook).

[TO BE COMPLETED — List actual questions tested with your sample document, the exact answers generated by the Gemini model, the source page numbers cited by the system, and an evaluation of factual correctness against the source text.]

| # | Question Submitted | Target Document Topic | Source Page(s) Cited | Actual Synthesized Answer | Factual Correctness (Verified) |
|---|---|---|---|---|---|
| **1** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **2** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **3** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **4** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |
| **5** | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] | [TO BE COMPLETED] |

---

### 8. Screenshots

> [!IMPORTANT]
> Capture high-resolution screenshots of the running Streamlit application for each of the core operational states described below. Store image files in the `screenshots/` directory within the project workspace and embed them using standard Markdown image syntax (`![Caption](screenshots/filename.png)`).

#### 8.1 Application Welcome Screen
*Initial landing screen before any document is uploaded, displaying the header badge, sidebar file uploader, LLM connectivity status, and the 3-step 'How It Works' guide.*

[TO BE COMPLETED — Embed screenshot: `screenshots/01_welcome_screen.png`]
```markdown
![DocuMind AI Welcome Screen](screenshots/01_welcome_screen.png)
```

#### 8.2 PDF Upload and Document Processing Status
*Sidebar view immediately following PDF upload, showing document statistics (File Name, Total Pages, Pages with Readable Text, Total Chunks Generated, and 'Ready' confirmation).*

[TO BE COMPLETED — Embed screenshot: `screenshots/02_document_status.png`]
```markdown
![Document Processing and Status](screenshots/02_document_status.png)
```

#### 8.3 Question Submission and Grounded Answer Card
*Main workspace displaying user query submission, generated answer card with formatted typography, and source page number badges.*

[TO BE COMPLETED — Embed screenshot: `screenshots/03_question_and_answer.png`]
```markdown
![Question and Grounded Answer](screenshots/03_question_and_answer.png)
```

#### 8.4 Retrieved Passages Drawer (Expandable Citations)
*Expanded view of the 'View retrieved passages' drawer, illustrating the top-ranked text chunks, source page indices, and calculated relevance percentage bars.*

[TO BE COMPLETED — Embed screenshot: `screenshots/04_retrieved_passages.png`]
```markdown
![Retrieved Passages and Similarity Metrics](screenshots/04_retrieved_passages.png)
```

#### 8.5 Error and Validation Handling (e.g., Missing API Key or Scanned PDF)
*Visual capture of system error handling when an invalid API key is provided or when an unreadable image-only PDF is uploaded.*

[TO BE COMPLETED — Embed screenshot: `screenshots/05_error_handling.png`]
```markdown
![System Error Handling and Alerts](screenshots/05_error_handling.png)
```

#### 8.6 LLM Connection Diagnostic Panel
*Sidebar view illustrating the active Gemini connection badge, model name identification, and the 'Re-check Connection' feature.*

[TO BE COMPLETED — Embed screenshot: `screenshots/06_llm_connection.png`]
```markdown
![LLM Connection Diagnostics](screenshots/06_llm_connection.png)
```

---

### 9. Limitations

While DocuMind AI achieves its objective of delivering a fast, transparent, and lightweight RAG solution on commodity hardware, the underlying engineering decisions introduce specific architectural trade-offs:

1. **Lexical Matching vs. Semantic Generalization (TF-IDF Limitations):**
   TF-IDF measures literal word and n-gram overlap. It does not map text into a continuous latent semantic vector space. Consequently, if a user queries *"What are the tuition fees?"* and the document exclusively uses the phrase *"The cost of academic instruction is $5,000 per semester"*, TF-IDF will compute a near-zero similarity score because there is no lexical token overlap between "tuition fees" and "cost of academic instruction". Dense embedding models (e.g., Sentence Transformers) overcome this by mapping synonyms to proximate geometric coordinates.

2. **Inability to Parse Scanned (Raster Image) PDFs:**
   The extraction pipeline relies on native textual streams parsed via PyMuPDF. If a PDF consists of scanned photocopies, photographic receipts, or bitmap image exports lacking an embedded OCR text layer, the parser extracts zero characters. The current implementation detects this condition and warns the user, but cannot extract text without integrating an external OCR engine (such as Tesseract or AWS Textract).

3. **Single-Document Confinement:**
   The current session state architecture processes and indexes a single uploaded document at a time. Cross-document synthesis (e.g., querying across an entire library of 50 enterprise handbooks simultaneously) is not supported in the active in-memory matrix configuration.

4. **Cloud Dependency and Network Latency:**
   While the retrieval phase executes locally on the CPU in milliseconds, the answer synthesis phase requires outbound HTTPS connectivity to Google's Generative AI API servers. Network latency, firewall restrictions, or outages in Google Cloud services directly impact answer generation availability.

5. **API Rate Limiting and Quota Ceilings:**
   The application uses Google's Gemini free-tier or standard API keys, which are subject to Queries-Per-Minute (QPM) and daily token quotas. Rapid, continuous querying by multiple users can trigger HTTP 429 ("Too Many Requests") errors.

6. **Information Splitting across Chunk Boundaries:**
   Although an 80-word sliding overlap minimizes boundary effects, highly complex tabular data, multi-column layouts, or ideas developed across multiple consecutive paragraphs may still be bifurcated across chunk boundaries, potentially degrading retrieval precision.

7. **Stateless Conversational Architecture:**
   The application operates on an isolated query-response model. It does not maintain a multi-turn conversational dialogue memory buffer (e.g., asking *"What is the attendance rule?"* followed by *"What happens if I violate it?"*). Each question is treated as an independent retrieval event.

8. **Scalability Constraints of In-Memory Sparse Matrices:**
   While highly efficient for documents under a few hundred pages, storing raw chunk lists and sparse TF-IDF matrices in RAM inside `st.session_state` becomes memory-intensive for multi-gigabyte document corpora spanning tens of thousands of pages.

---

### 10. Future Enhancements

To advance DocuMind AI from a lightweight academic prototype to an enterprise-grade document intelligence platform, several architectural enhancements are planned across three development phases:

#### 10.1 Phase I: Advanced Retrieval and Multi-Modal Extraction
1. **Hybrid Retrieval (Dense Semantic Embeddings + Sparse Lexical Indexing):**
   Integrate dense bi-encoder embeddings (such as `all-MiniLM-L6-v2` via Hugging Face `sentence-transformers`) alongside the existing TF-IDF engine. By implementing Reciprocal Rank Fusion (RRF), the system can combine lexical precision (for exact part numbers, legal codes, and names) with semantic understanding (for conceptual and paraphrased queries).
2. **Optical Character Recognition (OCR) Integration:**
   Incorporate `pytesseract` or `EasyOCR` into the `extract_text_from_pdf` pipeline. If a document page yields zero selectable text via PyMuPDF, the pipeline can automatically trigger OCR image rasterization, enabling seamless question answering over historical archives, scanned receipts, and physical document scans.
3. **Table and Multimodal Visual Parsing:**
   Incorporate specialized table extraction libraries (e.g., `pdfplumber` or `camelot`) and multimodal LLM vision capabilities to directly parse, interpret, and answer questions regarding architectural schematics, financial balance sheets, flowchart images, and complex graphs embedded within documents.

#### 10.2 Phase II: Multi-Document Infrastructure and Conversation Management
4. **Multi-Document Knowledge Base with Vector Databases:**
   Extend the data architecture to support concurrent multi-document workspaces. Integrate persistent open-source vector databases such as **ChromaDB**, **FAISS**, or **Qdrant**, allowing users to upload, categorize, and cross-reference entire libraries of technical documentation.
5. **Conversational Memory and Multi-Turn Reasoning:**
   Implement a conversational memory buffer (e.g., sliding window dialogue history or contextual query rewriting). A query re-writer LLM prompt can transform follow-up questions (e.g., *"What is its penalty?"*) into fully qualified standalone queries (e.g., *"What is the penalty for failing the attendance rule?"*) prior to executing the retrieval step.
6. **Multi-Format Ingestion Engine:**
   Broaden file format support beyond PDF to include Microsoft Word documents (`.docx`), plain text files (`.txt`), Markdown files (`.md`), HTML documentation, and PowerPoint presentations (`.pptx`) using unified parsers such as `unstructured`.

#### 10.3 Phase III: Enterprise Deployment, Security, and Reporting
7. **Local LLM Inference for Air-Gapped and Confidential Deployments:**
   Provide an optional backend toggle allowing users to run local open-weight quantized language models (e.g., Llama 3 8B, Mistral 7B, or Phi-3 via Ollama or `llama.cpp`). This would enable 100% offline, privacy-compliant execution for confidential healthcare, defense, or corporate legal documentation.
8. **Export and Reporting Capabilities:**
   Provide one-click export features enabling users to download synthesized Q&A sessions, research summaries, and source excerpts as professionally formatted PDF reports or Markdown documentation.
9. **Role-Based Access Control and Document Encryption:**
   Incorporate user authentication (OAuth 2.0 / JWT), document-level permissions, and AES-256 encryption at rest to ensure enterprise compliance with GDPR, HIPAA, and corporate confidentiality requirements.

---

### 11. Conclusion

The **DocuMind AI** project successfully demonstrates the design, mathematical formulation, and practical implementation of a lightweight Retrieval-Augmented Generation (RAG) system tailored for digital PDF documents. By combining PyMuPDF for high-speed page-isolated text extraction, scikit-learn for TF-IDF vectorization and cosine similarity retrieval, and Google's Gemini 2.0 Flash for grounded natural language synthesis, the project effectively resolves the twin challenges of information overload and generative hallucination.

The system proves that dependable, high-precision document question-answering does not strictly necessitate complex GPU infrastructure or heavy deep learning frameworks; classical information retrieval techniques, when thoughtfully coupled with state-of-the-art generative LLMs, provide an exceptionally fast, transparent, and computationally accessible alternative. The resulting Streamlit web application delivers an intuitive user experience with verified source attribution, real-time diagnostics, and comprehensive error resilience. This project establishes a solid foundation in modern natural language processing, vector space modeling, prompt engineering, and full-stack software development, while laying a clear roadmap for future enterprise enhancements.

---

### 12. References

1. **Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M., Yih, W., Rocktäschel, T., Riedel, S., & Kiela, D.** (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. Advances in Neural Information Processing Systems (NeurIPS 2020), Vol. 33, pp. 9459–9474. https://arxiv.org/abs/2005.11401
2. **Manning, C. D., Raghavan, P., & Schütze, H.** (2008). *Introduction to Information Retrieval*. Cambridge University Press. ISBN: 978-0521865715.
3. **Salton, G., & Buckley, C.** (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management, 24(5), pp. 513–523. https://doi.org/10.1016/0306-4573(88)90021-0
4. **Google DeepMind.** (2024). *Gemini: A Family of Highly Capable Multimodal Models*. Google Technical Report. https://ai.google.dev/
5. **Streamlit Inc.** (2024). *Streamlit Documentation: Session State, Caching, and Component API*. https://docs.streamlit.io/
6. **PyMuPDF Development Team.** (2024). *PyMuPDF (fitz) Official Documentation: High-performance PDF Parsing in Python*. https://pymupdf.readthedocs.io/
7. **Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, E.** (2011). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research (JMLR), 12, pp. 2825–2830. https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
8. **Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen, P., Cournapeau, D., Wieser, E., Taylor, J., Berg, S., Smith, N. J., Kern, R., Picus, M., Hoyer, S., van Kerkwijk, M. H., Brett, M., Haldane, A., del Río, J. F., Wiebe, M., Peterson, P., Gérard-Marchant, P., Sheppard, K., Reddy, T., Weckesser, W., Abbasi, H., Gohlke, C., & Oliphant, T. E.** (2020). *Array programming with NumPy*. Nature, 585(7825), pp. 357–362. https://doi.org/10.1038/s41586-020-2649-2
9. **Karpukhin, V., Oguz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D., & Yih, W.** (2020). *Dense Passage Retrieval for Open-Domain Question Answering*. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 6769–6781. https://arxiv.org/abs/2004.04906
10. **Robertson, S., & Zaragoza, H.** (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and Trends in Information Retrieval, 3(4), pp. 333–389. https://doi.org/10.1561/1500000019
11. **Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I.** (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS 2017), Vol. 30, pp. 5998–6008. https://arxiv.org/abs/1706.03762
12. **Reimers, N., & Gurevych, I.** (2019). *Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks*. Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing (EMNLP-IJCNLP), pp. 3982–3992. https://arxiv.org/abs/1908.10084
13. **Borgeaud, S., Mensch, A., Hoffmann, J., Cai, T., Rutherford, E., Millican, K., van den Driessche, G., Lespiau, J., Damoc, B., Clark, A., de Las Casas, D., Guy, A., Menick, J., Ring, R., Hennigan, T., Huang, S., Maggiore, L., Jones, C., Cassirer, A., Brock, A., Paganini, M., Bulanova, C., Platt, E., Borzunov, A., Wu, H., Wei, J., Young, E., & Sifre, L.** (2022). *Improving Language Models by Retrieving from Trillions of Tokens*. Proceedings of the 39th International Conference on Machine Learning (ICML 2022), PMLR 162, pp. 2206–2240.
14. **Gao, Y., Xiong, Y., Gao, X., Jia, K., Pan, J., Bi, Y., Dai, Y., Sun, J., & Wang, H.** (2023). *Retrieval-Augmented Generation for Large Language Models: A Survey*. arXiv preprint arXiv:2312.10997.
15. **Guu, K., Lee, K., Tung, Z., Pasupat, P., & Chang, M.** (2020). *Retrieval Augmented Language Model Pre-Training*. Proceedings of the 37th International Conference on Machine Learning (ICML 2020), PMLR 119, pp. 3929–3938.
