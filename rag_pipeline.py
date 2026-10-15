"""
rag_pipeline.py - Document Processing and RAG Pipeline for DocuMind AI

This module provides the complete backend for the RAG application:
  1. PDF text extraction using PyMuPDF
  2. Text chunking with configurable overlap
  3. TF-IDF based retrieval using scikit-learn
  4. Answer generation using Google Gemini API (mandatory)

Author : YHills AI Internship - Project 2
Version: 1.0
"""

import json
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pymupdf as fitz  # PyMuPDF — PDF text extraction library
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# =============================================================================
# 1. PDF TEXT EXTRACTION
# =============================================================================

def extract_text_from_pdf(pdf_bytes):
    """
    Extract text from a PDF file, one page at a time.

    Args:
        pdf_bytes (bytes): Raw bytes of the uploaded PDF file.

    Returns:
        list[dict]: A list of dicts, each with keys:
            - 'page_number' (int): 1-based page number
            - 'text' (str): Extracted text from that page

    Raises:
        ValueError: If the file is not a valid PDF, is empty, or contains
                    no extractable text (e.g. scanned images without OCR).
    """
    # Validate input
    if not pdf_bytes:
        raise ValueError("The uploaded file is empty.")

    # Try to open the PDF
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(
            f"Could not open the file as a PDF. It may be corrupted or "
            f"not a valid PDF document. (Error: {e})"
        )

    # Check page count
    if doc.page_count == 0:
        doc.close()
        raise ValueError("The PDF has no pages.")

    # Extract text from each page
    pages = []
    for page_num in range(doc.page_count):
        page = doc.load_page(page_num)
        text = page.get_text("text")  # Plain text extraction
        # Clean up whitespace but preserve paragraph structure
        text = text.strip()
        if text:  # Only include pages with actual text
            pages.append({
                "page_number": page_num + 1,  # 1-based page numbering
                "text": text,
            })

    doc.close()

    # Check if any text was extracted
    if not pages:
        raise ValueError(
            "No readable text was found in this PDF. This usually means "
            "the PDF contains scanned images rather than selectable text. "
            "Scanned PDFs require OCR (Optical Character Recognition), "
            "which is not included in this version of DocuMind AI."
        )

    return pages


# =============================================================================
# 2. TEXT CHUNKING
# =============================================================================

def split_into_chunks(pages, chunk_size=300, chunk_overlap=80):
    """
    Split extracted page texts into overlapping chunks for retrieval.

    Each chunk stays within a single page to ensure accurate page attribution.
    Chunks are created by splitting text into words and grouping them.

    Args:
        pages (list[dict]): Output from extract_text_from_pdf().
        chunk_size (int): Approximate number of words per chunk.
        chunk_overlap (int): Number of overlapping words between consecutive
                             chunks from the same page.

    Returns:
        list[dict]: A list of chunk dicts, each with keys:
            - 'text' (str): The chunk text
            - 'page_number' (int): Source page number (1-based)
            - 'chunk_index' (int): Global chunk index (0-based)
    """
    chunks = []
    chunk_index = 0

    for page in pages:
        page_text = page["text"]
        page_num = page["page_number"]

        # Split into words
        words = page_text.split()

        # If the page has fewer words than chunk_size, keep it as one chunk
        if len(words) <= chunk_size:
            chunks.append({
                "text": " ".join(words),
                "page_number": page_num,
                "chunk_index": chunk_index,
            })
            chunk_index += 1
            continue

        # Create overlapping chunks
        start = 0
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)

            # Only add if the chunk has meaningful content (at least 10 words)
            if len(chunk_words) >= 10:
                chunks.append({
                    "text": chunk_text,
                    "page_number": page_num,
                    "chunk_index": chunk_index,
                })
                chunk_index += 1

            # Move the window forward
            start += chunk_size - chunk_overlap

    return chunks


# =============================================================================
# 3. TF-IDF INDEX BUILDING
# =============================================================================

def build_tfidf_index(chunks):
    """
    Build a TF-IDF index from the text chunks.

    This creates a term-frequency / inverse-document-frequency matrix that
    allows fast similarity comparison between a query and all chunks.

    Args:
        chunks (list[dict]): Output from split_into_chunks().

    Returns:
        tuple: (vectorizer, tfidf_matrix)
            - vectorizer (TfidfVectorizer): Fitted vectorizer for transforming
              new queries.
            - tfidf_matrix (sparse matrix): TF-IDF representation of all chunks.

    Raises:
        ValueError: If there are no chunks to index.
    """
    if not chunks:
        raise ValueError("No text chunks available to build the search index.")

    # Extract the text content from each chunk
    texts = [chunk["text"] for chunk in chunks]

    # Configure the TF-IDF vectorizer
    # - ngram_range=(1,2) captures single words and two-word phrases
    # - stop_words='english' filters out common words like "the", "is", etc.
    # - max_df=0.95 ignores terms appearing in more than 95% of chunks
    # - min_df=1 keeps all terms that appear at least once
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        max_df=0.95,
        min_df=1,
        lowercase=True,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    return vectorizer, tfidf_matrix


# =============================================================================
# 4. RETRIEVAL — FINDING RELEVANT CHUNKS
# =============================================================================

def retrieve_relevant_chunks(query, chunks, vectorizer, tfidf_matrix,
                             top_k=5, threshold=0.05):
    """
    Find the most relevant text chunks for a user's question.

    Uses cosine similarity between the query's TF-IDF vector and each
    chunk's TF-IDF vector to rank relevance.

    Args:
        query (str): The user's question.
        chunks (list[dict]): The text chunks with page numbers.
        vectorizer (TfidfVectorizer): Fitted vectorizer from build_tfidf_index().
        tfidf_matrix (sparse matrix): Pre-computed TF-IDF matrix.
        top_k (int): Maximum number of chunks to return.
        threshold (float): Minimum cosine similarity score (0 to 1). Chunks
                           below this score are considered irrelevant.
                           Default 0.05 is intentionally low to avoid missing
                           partially relevant content. This is a TF-IDF
                           limitation — keyword overlap may be low even when
                           the content is relevant.

    Returns:
        list[dict]: Relevant chunks sorted by score (highest first), each with:
            - 'text' (str): Chunk text
            - 'page_number' (int): Source page
            - 'chunk_index' (int): Chunk position
            - 'score' (float): Cosine similarity score (0 to 1)
        Returns an empty list if no chunk meets the threshold.
    """
    if not query or not query.strip():
        return []

    # Transform the query into TF-IDF space using the same vocabulary
    query_vector = vectorizer.transform([query])

    # Compute cosine similarity between query and every chunk
    similarities = cosine_similarity(query_vector, tfidf_matrix).flatten()

    # Get indices sorted by similarity (highest first)
    ranked_indices = np.argsort(similarities)[::-1]

    # Filter by threshold and take top_k
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


# =============================================================================
# 5. GEMINI API — CONNECTION CHECK
# =============================================================================

def _generate_gemini_content(prompt, api_key, model_name, max_output_tokens):
    """Call Gemini's generateContent REST endpoint with a small JSON request."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{model_name}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": max_output_tokens,
        },
    }
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "X-goog-api-key": api_key,
        },
        method="POST",
    )

    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("Gemini returned an invalid JSON response.") from error
    except HTTPError as error:
        try:
            error_body = json.loads(error.read().decode("utf-8"))
            detail = error_body.get("error", {}).get("message", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = ""
        safe_detail = detail.replace(api_key, "[redacted]")
        raise RuntimeError(
            f"Gemini API request failed (HTTP {error.code})"
            + (f": {safe_detail}" if safe_detail else "")
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not reach the Gemini API: {error.reason}"
        ) from error
    except TimeoutError as error:
        raise RuntimeError("The Gemini API request timed out.") from error
    except OSError as error:
        raise RuntimeError(f"Could not reach the Gemini API: {error}") from error

    if not isinstance(result, dict):
        raise RuntimeError("Gemini returned an invalid response.")

    candidates = result.get("candidates", [])
    if not candidates:
        block_reason = (
            result.get("promptFeedback", {}).get("blockReason")
            or "no response candidate was returned"
        )
        raise RuntimeError(f"Gemini returned no answer ({block_reason}).")

    parts = candidates[0].get("content", {}).get("parts", [])
    return "\n".join(
        part["text"] for part in parts if isinstance(part.get("text"), str)
    ).strip()


def _get_gemini_model(model_name, api_key):
    """Fetch model metadata to verify access without using generation quota."""
    url = (
        "https://generativelanguage.googleapis.com/v1beta/models/"
        f"{quote(model_name, safe='')}"
    )
    request = Request(
        url,
        headers={"X-goog-api-key": api_key},
        method="GET",
    )

    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("Gemini returned an invalid model response.") from error
    except HTTPError as error:
        try:
            error_body = json.loads(error.read().decode("utf-8"))
            detail = error_body.get("error", {}).get("message", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = ""
        safe_detail = detail.replace(api_key, "[redacted]")
        raise RuntimeError(
            f"Gemini API request failed (HTTP {error.code})"
            + (f": {safe_detail}" if safe_detail else "")
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not reach the Gemini API: {error.reason}"
        ) from error
    except TimeoutError as error:
        raise RuntimeError("The Gemini API request timed out.") from error
    except OSError as error:
        raise RuntimeError(f"Could not reach the Gemini API: {error}") from error

    if not isinstance(result, dict):
        raise RuntimeError("Gemini returned an invalid model response.")
    return result


def _list_gemini_models(api_key):
    """List models available to this key that support generateContent."""
    request = Request(
        "https://generativelanguage.googleapis.com/v1beta/models?pageSize=100",
        headers={"X-goog-api-key": api_key},
        method="GET",
    )

    try:
        with urlopen(request, timeout=20) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise RuntimeError("Gemini returned an invalid model list.") from error
    except HTTPError as error:
        try:
            error_body = json.loads(error.read().decode("utf-8"))
            detail = error_body.get("error", {}).get("message", "")
        except (json.JSONDecodeError, UnicodeDecodeError):
            detail = ""
        safe_detail = detail.replace(api_key, "[redacted]")
        raise RuntimeError(
            f"Gemini API request failed (HTTP {error.code})"
            + (f": {safe_detail}" if safe_detail else "")
        ) from error
    except URLError as error:
        raise RuntimeError(
            f"Could not reach the Gemini API: {error.reason}"
        ) from error
    except TimeoutError as error:
        raise RuntimeError("The Gemini API request timed out.") from error
    except OSError as error:
        raise RuntimeError(f"Could not reach the Gemini API: {error}") from error

    if not isinstance(result, dict):
        raise RuntimeError("Gemini returned an invalid model list.")

    return [
        model["name"].removeprefix("models/")
        for model in result.get("models", [])
        if isinstance(model, dict)
        and "generateContent" in model.get("supportedGenerationMethods", [])
        and isinstance(model.get("name"), str)
    ]


def check_gemini_connection(api_key, model_name="gemini-flash-latest"):
    """
    Verify that the Gemini API key is valid and the model is accessible.

    Confirms model support and sends a small text-generation request.

    Args:
        api_key (str): Google Gemini API key.
        model_name (str): Gemini model identifier.

    Returns:
        dict: With keys:
            - 'connected' (bool): True if the connection succeeded.
            - 'message' (str): Description of the connection status.
    """
    if not api_key or not api_key.strip():
        return {
            "connected": False,
            "message": "Gemini API key is not configured. "
                       "Please add GEMINI_API_KEY to your .env file.",
        }

    api_key = api_key.strip()
    if api_key.lower() in {
        "your_new_api_key_here",
        "your_gemini_api_key_here",
        "replace_with_your_api_key",
    }:
        return {
            "connected": False,
            "message": "Replace the GEMINI_API_KEY placeholder in .env with "
                       "your private key, then restart the app.",
        }

    if (api_key.lower().startswith(("curl ", "http://", "https://"))
            or "generativelanguage.googleapis.com" in api_key.lower()):
        return {
            "connected": False,
            "message": "GEMINI_API_KEY contains a curl command or URL, not an "
                       "API key. Put only the key value after the equals sign "
                       "in .env, then restart the app.",
        }

    try:
        model = _get_gemini_model(model_name, api_key)
        if "generateContent" not in model.get("supportedGenerationMethods", []):
            return {
                "connected": False,
                "message": f"Model '{model_name}' is available but does not "
                           "support content generation.",
            }

        response_text = _generate_gemini_content(
            "Reply with exactly: OK", api_key, model_name, max_output_tokens=8
        )
        if not response_text:
            return {
                "connected": False,
                "message": "The model is available but returned no text in "
                           "the connection test.",
            }

        return {
            "connected": True,
            "message": f"Connected to Gemini ({model_name})",
        }

    except RuntimeError as error:
        error_msg = str(error)
        error_lower = error_msg.lower()

        if "http 503" in error_lower or "high demand" in error_lower:
            return {
                "connected": False,
                "message": "Gemini is temporarily experiencing high demand. "
                           "The key and model are configured; try again shortly.",
            }
        elif "http 429" in error_lower or "quota" in error_lower or "rate" in error_lower:
            return {
                "connected": False,
                "message": "API quota exceeded or rate limited. "
                           "Please wait and try again later.",
            }
        elif ("api_key" in error_lower or "api key" in error_lower
              or "invalid key" in error_lower or "http 401" in error_lower):
            return {
                "connected": False,
                "message": "Invalid API key. Please check your GEMINI_API_KEY "
                           "in the .env file. Use the key value only, not the "
                           "curl command.",
            }
        elif "http 400" in error_lower:
            return {
                "connected": False,
                "message": f"Gemini rejected the request: {error_msg}",
            }
        elif "http 403" in error_lower or "permission" in error_lower:
            return {
                "connected": False,
                "message": "Gemini denied access. Check that this API key is "
                           "enabled for the Gemini API and its project.",
            }
        elif "http 404" in error_lower or "not found" in error_lower:
            available_models = []
            try:
                available_models = _list_gemini_models(api_key)
                model_hint = (
                    " Available models supporting text generation: "
                    + ", ".join(available_models[:12])
                    + "."
                    if available_models else
                    " No generateContent models were listed for this key."
                )
            except RuntimeError as discovery_error:
                model_hint = (
                    " Could not list available models: "
                    f"{discovery_error}"
                )

            return {
                "connected": False,
                "message": f"Model '{model_name}' not found or not available "
                           f"for your API key. Try a different model name."
                           f"{model_hint}",
            }
        elif "timed out" in error_lower or "could not reach" in error_lower:
            return {
                "connected": False,
                "message": error_msg,
            }
        else:
            return {
                "connected": False,
                "message": f"Could not connect to Gemini: {error_msg}",
            }


# =============================================================================
# 6. GEMINI API — ANSWER GENERATION
# =============================================================================

def generate_answer(query, retrieved_chunks, api_key,
                    model_name="gemini-flash-latest"):
    """
    Generate an answer using Google Gemini based on retrieved document passages.

    Sends the user's question and the retrieved text passages to the Gemini LLM
    with instructions to answer ONLY from the provided context.

    Args:
        query (str): The user's question.
        retrieved_chunks (list[dict]): Relevant chunks from
                                       retrieve_relevant_chunks().
        api_key (str): Google Gemini API key.
        model_name (str): Gemini model identifier.

    Returns:
        dict: With keys:
            - 'answer' (str or None): The generated answer, or None on failure.
            - 'error' (str or None): Error message if generation failed.
    """
    # --- Validate inputs ---
    if not api_key or not api_key.strip():
        return {
            "answer": None,
            "error": "Gemini API key is not configured. "
                     "Please add GEMINI_API_KEY to your .env file.",
        }

    if not retrieved_chunks:
        return {
            "answer": None,
            "error": "No relevant passages were found in the document for "
                     "this question. The LLM was not called because there "
                     "is no supporting context to generate a reliable answer.",
        }

    # --- Build the prompt with retrieved context ---
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        context_parts.append(
            f"[Passage {i} — Page {chunk['page_number']}]\n"
            f"{chunk['text']}"
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

    # --- Call the Gemini API ---
    try:
        answer_text = _generate_gemini_content(
            prompt, api_key, model_name, max_output_tokens=1024
        )

        if not answer_text:
            return {
                "answer": None,
                "error": "Gemini returned an empty response. "
                         "Please try rephrasing your question.",
            }

        return {"answer": answer_text, "error": None}

    except Exception as error:
        error_msg = str(error)
        error_lower = error_msg.lower()

        # Provide user-friendly error messages
        if ("http 503" in error_lower or "high demand" in error_lower
                or "service unavailable" in error_lower):
            return {
                "answer": None,
                "error": "Gemini is temporarily experiencing high demand. "
                         "Your key is configured correctly; please try again shortly.",
            }
        elif "http 429" in error_lower or "quota" in error_lower or "rate" in error_lower:
            return {
                "answer": None,
                "error": "Gemini API quota exceeded or rate limited. "
                         "Please wait a moment and try again.",
            }
        elif ("api_key" in error_lower or "api key" in error_lower
              or "401" in error_lower):
            return {
                "answer": None,
                "error": "Invalid API key. Please check your GEMINI_API_KEY "
                         "in the .env file. Use the key value only, not the "
                         "curl command.",
            }
        elif "http 404" in error_lower or "not found" in error_lower:
            return {
                "answer": None,
                "error": f"Model '{model_name}' is unavailable for this API key. "
                         "Check GEMINI_MODEL in .env.",
            }
        elif "403" in error_lower or "permission" in error_lower:
            return {
                "answer": None,
                "error": "Gemini denied access. Check that this API key is "
                         "enabled for the Gemini API and its project.",
            }
        elif "blocked" in error_lower or "safety" in error_lower:
            return {
                "answer": None,
                "error": "The response was blocked by Gemini's safety filters. "
                         "Please try rephrasing your question.",
            }
        else:
            return {
                "answer": None,
                "error": f"Gemini API error: {error_msg}",
            }
