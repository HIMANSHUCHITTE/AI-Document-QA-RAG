# DocuMind AI — AI-Powered Document Question Answering (RAG)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57%2B-FF4B4B.svg)](https://streamlit.io/)
[![Google Gemini](https://img.shields.io/badge/LLM-Google%20Gemini-4285F4.svg)](https://aistudio.google.com/)
[![License: Educational](https://img.shields.io/badge/License-Educational-green.svg)](#license)

 

---

## Overview

A Streamlit-based application that lets users upload PDF documents and ask questions. Uses TF-IDF retrieval + Google Gemini LLM for answer generation. Lightweight RAG demonstration.

Instead of passing an entire document into an LLM or relying solely on the LLM's parametric memory, DocuMind AI:
1. Extracts text from uploaded PDF files page-by-page.
2. Segments the document into overlapping passage chunks while preserving page provenance.
3. Indexes passage representations using TF-IDF (Term Frequency-Inverse Document Frequency) and retrieves the most relevant excerpts via cosine similarity ranking.
4. Supplies the retrieved context along with the user's question to the Google Gemini Large Language Model (LLM) under a strict grounding prompt.

This delivers accurate, hallucination-resistant responses with zero need for costly vector databases, heavy embeddings infrastructure, GPU hardware, Docker containers, or external cloud database services.

---

## Features

- **PDF text extraction (PyMuPDF)**: Fast, dependable page-by-page text extraction retaining exact page numbers for citations.
- **TF-IDF + cosine similarity passage retrieval (scikit-learn)**: Deterministic, transparent keyword and n-gram retrieval without heavy neural vector stores.
- **Google Gemini answer generation**: Context-grounded synthesis using Google's Gemini REST API and strict grounding instructions.
- **Professional SaaS-style UI**: Clean, responsive Streamlit user interface styled with custom CSS, real-time LLM status badges, document statistics, and example query buttons.
- **Source page references and expandable passages**: Transparent answers citing source pages alongside expandable cards displaying retrieved passages and relevance scores.
- **Graceful error handling**: Clear, user-friendly diagnostics for missing API keys, scanned image-only PDFs, rate limits, and network errors.
- **No GPU, database, Docker, or cloud required**: Runs 100% locally on standard CPU hardware with Python 3.9+.

---

## Technology Stack

| Component | Technology | Version | Purpose / Role |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.9+` | Core programming language |
| **User Interface** | Streamlit | `>= 1.57.0` | UI application, session state management, and custom CSS styling |
| **PDF Extraction** | PyMuPDF (`fitz`) | `>= 1.23.0` | PDF text extraction and page metadata handling |
| **Passage Retrieval** | scikit-learn | `>= 1.3.0` | TF-IDF vectorization (unigrams & bigrams) and cosine similarity ranking |
| **LLM Answer Generation** | Gemini REST API | — | Google Gemini API for grounded answer generation |
| **Environment Management** | python-dotenv | `>= 1.0.0` | Safe loading of configuration and credentials from `.env` |
| **Numerical Operations** | NumPy | `>= 1.24.0` | Array manipulation and similarity score processing |

---

## Folder Structure

```
AI_Document_QA_RAG/
├── app.py                  # Streamlit UI application
├── rag_pipeline.py         # RAG backend (extraction, chunking, retrieval, LLM)
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── PROJECT_REPORT.md       # Academic project report
├── .env.example            # Environment variable template
├── .gitignore              # Git ignore rules
├── sample_document.txt     # Sample test document (fictional)
└── screenshots/            # Application screenshots
```

---

## Setup Instructions (Windows)

Follow these step-by-step instructions to set up and run DocuMind AI on a Windows operating system.

### Prerequisites

- **Python 3.9 or higher**: Verify installation in Command Prompt or PowerShell:
  ```cmd
  python --version
  ```
- **pip** (Python package manager):
  ```cmd
  pip --version
  ```
-**A Google Gemini API key**: Obtain a free API key from Google AI Studio.

---

### Step 1: Clone or Download

Download or clone the repository to your local machine (e.g., your Desktop):

```cmd
cd C:\Users\YourName\Desktop
```

Open the `AI_Document_QA_RAG` folder:

```cmd
cd AI_Document_QA_RAG
```

---

### Step 2: Create Virtual Environment

Create and activate an isolated Python virtual environment:

```cmd
python -m venv venv
venv\Scripts\activate
```

> [!NOTE]
> Once activated, your terminal prompt will display `(venv)` indicating the virtual environment is active.

---

### Step 3: Install Dependencies

Install all project dependencies using pip:

```cmd
pip install -r requirements.txt
```

---

### Step 4: Configure Gemini API Key

1. Go to [https://aistudio.google.com/apikey](https://aistudio.google.com/apikey)
2. Sign in with your Google account.
3. Click **'Create API key'**.
4. Copy `.env.example` to `.env`:
   ```cmd
   copy .env.example .env
   ```
5. Open `.env` in a text editor (e.g., Notepad) and replace `your_new_api_key_here` with your new API key. Put only the key value after `GEMINI_API_KEY=`, not the `curl` command:
   ```env
   GEMINI_API_KEY=your_new_api_key_here
   GEMINI_MODEL=gemini-flash-latest
   ```
6. (Optional) Change `GEMINI_MODEL` if desired.

If the app is already running, restart it after editing `.env` so the new key is loaded.

> [!IMPORTANT]
> **Never commit `.env` to Git.** It is already listed in `.gitignore` to prevent leaking your API keys.

---

### Step 5: Run the Application

Launch the Streamlit web application:

```cmd
streamlit run app.py
```

The app will open automatically in your browser at:
```
http://localhost:8501
```

If it does not open automatically, open your browser and navigate to `http://localhost:8501`.

---

## How to Use

1. **Upload a PDF document using the sidebar**: Use the file uploader to select a PDF containing selectable text.
2. **Wait for processing to complete**: The system extracts pages, splits text into chunks, and builds the TF-IDF index. The sidebar will display document statistics and a green `✓ Ready` status badge.
3. **Type a question or click an example question**: Enter a query into the text input field or choose one of the pre-configured example question buttons.
4. **Click 'Ask DocuMind' to get an AI-generated answer**: The system retrieves the most relevant passages and invokes the Google Gemini LLM.
5. **View source page references and expand retrieved passages**: Read the synthesized response, view cited page numbers, and expand the retrieved passage cards to inspect text excerpts and relevance percentages.

---

## Testing with the Sample Document

The included `sample_document.txt` contains a fictional student handbook (*Greenfield Institute of Technology Student Handbook 2025-2026*). To test:

1. **Convert `sample_document.txt` to PDF** using Microsoft Word, Google Docs, or an online text-to-PDF converter (e.g., Open in Word -> Save As PDF).
2. **Upload the PDF** to the application via the sidebar uploader.
3. **Try questions like**:
   - *"What is the minimum attendance requirement?"*
   - *"What are the project deadlines?"*
   - *"What student support services are available?"*

---

## How It Works (Architecture)

DocuMind AI operates through a modular 6-stage RAG pipeline:

```
[ PDF Document ]
       │
       ▼
1. PDF Upload (PyMuPDF)
   └── Extracts text page-by-page while preserving page numbers
       │
       ▼
2. Chunking
   └── Text split into overlapping ~300-word chunks (80-word overlap) preserving page numbers
       │
       ▼
3. Indexing
   └── TF-IDF vectorizer builds term-frequency matrix (unigrams + bigrams)
       │
       ▼
4. Retrieval
   └── User question compared via cosine similarity; top chunks selected (top_k=5, threshold=0.05)
       │
       ▼
5. Generation
   └── Question + retrieved passages sent to Gemini with strict grounding prompt
       │
       ▼
6. Display
   └── Answer shown with source page references and expandable passage cards
```

### Detailed Pipeline Stages

1. **PDF Upload** → PyMuPDF extracts text page by page, recording 1-indexed page numbers and checking for text layer availability.
2. **Chunking** → Text is split into overlapping ~300-word chunks preserving page numbers. The 80-word overlap ensures contextual continuity across boundaries.
3. **Indexing** → TF-IDF vectorizer (`scikit-learn`) builds a term-frequency matrix capturing unigrams and bigrams with English stop-word filtering.
4. **Retrieval** → The user question is vectorized and compared via cosine similarity against all chunk vectors; the top relevant chunks exceeding the similarity threshold are selected.
5. **Generation** → The question and retrieved passages are sent to Gemini with a strict grounding prompt directing the LLM to answer solely from the provided passages without hallucination.
6. **Display** → The answer is shown in a structured answer card with source page references and expandable retrieved passages with percentage relevance scores.

---

## Gemini LLM Configuration

DocuMind AI calls Google's Gemini `generateContent` REST endpoint directly.

### Supported Models

| Model | Speed | Best For |
| :--- | :--- | :--- |
| **`gemini-flash-latest`** | Fast | Default; follows the current Gemini Flash release |
| **`gemini-2.5-flash`** | Fast | Explicit model version |

Set the model in `.env` via `GEMINI_MODEL`:
```env
GEMINI_MODEL=gemini-flash-latest
```

### API Key Notes

- **Free Tier Usage Limits**: The free tier of Google AI Studio includes generous rate limits for development and evaluation.
- **Connectivity Check**: The app verifies the selected model's `generateContent` support, then sends a short text-generation request. A successful connection status means that request returned text; it uses a small amount of generation quota.
- **Key Safety**: If an API key was pasted into a chat, screenshot, or public repository, revoke it and create a replacement. Never share the replacement key.
- **Temporary Service Errors**: If Gemini returns HTTP 503 due to high demand, wait briefly and retry. The app distinguishes this outage from an invalid key or model.
- **Model Availability**: If a configured model is unavailable, the app asks the API for models accessible to that key which support `generateContent` and shows the returned names.
- **Quota Exceeded Recovery**: If your rate limit or quota is exceeded, wait a few minutes and retry your query.

---

## Common Errors and Fixes

| Error | Cause | Fix |
| :--- | :--- | :--- |
| **'Gemini API key is not configured'** | Missing `.env` file or key | Copy `.env.example` to `.env` and add your key |
| **'GEMINI_API_KEY contains a curl command or URL'** | The full API test command was pasted as the key | Put only the API key value after `GEMINI_API_KEY=` |
| **'Invalid API key'** | Wrong or expired key | Generate a new key at [aistudio.google.com](https://aistudio.google.com/apikey) |
| **'Model not found'** | Invalid model name | Check `GEMINI_MODEL` in `.env` |
| **'No readable text found'** | Scanned PDF (images only) | Use a PDF with selectable text |
| **'API quota exceeded'** | Too many requests | Wait a few minutes and retry |
| **`ModuleNotFoundError`** | Dependencies not installed | Run `pip install -r requirements.txt` |
| **'streamlit is not recognized'** | Not in virtual environment | Activate venv: `venv\Scripts\activate` |

---

## Known Limitations

- **Lexical vs. Semantic Matching**: Uses TF-IDF keyword matching, not semantic/embedding-based retrieval. Questions using different vocabulary than the document may not retrieve relevant passages.
- **Scanned Documents**: Does not support scanned/image-only PDFs (OCR not included).
- **Chunk Boundaries**: Chunk boundaries may split sentences or multi-sentence paragraphs.
- **Large Document Processing**: Large PDFs (100+ pages) may take longer to process and index.
- **Internet Connectivity**: Requires an active internet connection for Gemini API calls.
- **API Limits**: Free Gemini API tier has rate and quota limits.
- **Document Scope**: Single document at a time; cross-document retrieval is not supported.

---

## Security Notes

- **Environment-based Credentials**: API keys are loaded securely from `.env` using `python-dotenv` and are never hardcoded.
- **Version Control Security**: `.env` is excluded from Git via `.gitignore` to prevent accidental credential leakage.
- **Local Text Processing**: PDF contents are processed locally on your machine; only retrieved passage snippets are sent to Gemini.
- **Ephemeral Storage**: No user data, uploaded documents, or queries are stored persistently.

---

## Credits

- **Program**: Built for the YHills AI Internship Program (Project 2).
- **Author**: Developed by a third-year Computer Engineering student ([Student Details: Name / Roll No. Placeholder]).
- **Open-Source Libraries**: Powered by Streamlit, PyMuPDF (`fitz`), scikit-learn, python-dotenv, and NumPy.
- **LLM**: Answer generation powered by Google's Gemini `generateContent` REST API.

---

## License

This project was created for educational and internship purposes under the YHills AI Internship Program.
