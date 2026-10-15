"""
test_rag.py - Automated verification tests for DocuMind AI RAG pipeline
"""

import json
from unittest.mock import patch

from rag_pipeline import (
    extract_text_from_pdf,
    split_into_chunks,
    build_tfidf_index,
    retrieve_relevant_chunks,
    generate_answer,
    check_gemini_connection,
    _generate_gemini_content,
)

def run_tests():
    print("=" * 60)
    print("DocuMind AI — Automated Verification Test Suite")
    print("=" * 60)

    # TEST 1: PDF Extraction
    print("\n[TEST 1] PDF Text Extraction")
    with open("sample_document.pdf", "rb") as f:
        pdf_bytes = f.read()

    pages = extract_text_from_pdf(pdf_bytes)
    print(f"  -> Extracted {len(pages)} pages with text.")
    assert len(pages) > 0, "No pages extracted"
    assert pages[1]["page_number"] == 2, "Page numbering mismatch"
    print("  -> PASSED: Pages extracted with correct 1-based numbering.")

    # TEST 2: Text Chunking
    print("\n[TEST 2] Text Chunking")
    chunks = split_into_chunks(pages, chunk_size=300, chunk_overlap=80)
    print(f"  -> Created {len(chunks)} chunks.")
    assert len(chunks) > 0, "No chunks created"
    for c in chunks:
        assert "text" in c and "page_number" in c and "chunk_index" in c
    print("  -> PASSED: Chunks created with page attribution.")

    # TEST 3: TF-IDF Indexing
    print("\n[TEST 3] TF-IDF Indexing")
    vectorizer, tfidf_matrix = build_tfidf_index(chunks)
    print(f"  -> TF-IDF vocabulary size: {len(vectorizer.vocabulary_)}")
    print(f"  -> TF-IDF Matrix shape: {tfidf_matrix.shape}")
    assert tfidf_matrix.shape[0] == len(chunks), "Matrix rows must match chunk count"
    print("  -> PASSED: TF-IDF matrix built successfully.")

    # TEST 4: Retrieval for Known Fact (Attendance)
    print("\n[TEST 4] Retrieval: In-document query (Attendance)")
    query1 = "What is the minimum attendance requirement?"
    results1 = retrieve_relevant_chunks(query1, chunks, vectorizer, tfidf_matrix, top_k=3)
    print(f"  -> Query: '{query1}'")
    print(f"  -> Found {len(results1)} chunks.")
    for r in results1:
        print(f"     Page {r['page_number']} (score: {r['score']}): {r['text'][:70]}...")
    assert len(results1) > 0, "Should retrieve at least 1 chunk"
    # Attendance policy is on Page 2 in sample_document.pdf
    assert any("75%" in r["text"] for r in results1), "Expected '75%' in retrieved chunk"
    print("  -> PASSED: Correct passage retrieved with 75% attendance rule.")

    # TEST 5: Retrieval for Another Section (Project Deadlines)
    print("\n[TEST 5] Retrieval: In-document query (Project Deadlines)")
    query2 = "What are the deadlines for project submission?"
    results2 = retrieve_relevant_chunks(query2, chunks, vectorizer, tfidf_matrix, top_k=3)
    print(f"  -> Query: '{query2}'")
    print(f"  -> Found {len(results2)} chunks.")
    for r in results2:
        print(f"     Page {r['page_number']} (score: {r['score']}): {r['text'][:70]}...")
    assert len(results2) > 0, "Should retrieve at least 1 chunk"
    assert any("April" in r["text"] or "Project" in r["text"] for r in results2)
    print("  -> PASSED: Correct project deadline passages retrieved.")

    # TEST 6: Retrieval for Unrelated Query
    print("\n[TEST 6] Retrieval: Unrelated query rejection / low score")
    query3 = "What is the recipe for chocolate chip cookies?"
    results3 = retrieve_relevant_chunks(query3, chunks, vectorizer, tfidf_matrix, top_k=3, threshold=0.1)
    print(f"  -> Query: '{query3}'")
    print(f"  -> Chunks above threshold 0.1: {len(results3)}")
    assert len(results3) == 0, "Unrelated query should yield 0 chunks above threshold"
    print("  -> PASSED: Unrelated query properly yielded 0 relevant passages.")

    # TEST 7: Empty Query Handling
    print("\n[TEST 7] Empty Query Handling")
    assert retrieve_relevant_chunks("", chunks, vectorizer, tfidf_matrix) == []
    assert retrieve_relevant_chunks("   ", chunks, vectorizer, tfidf_matrix) == []
    print("  -> PASSED: Empty queries return empty results without error.")

    # TEST 8: Empty & Corrupted PDF Handling
    print("\n[TEST 8] Error Handling: Empty & Corrupted PDF")
    try:
        extract_text_from_pdf(b"")
        assert False, "Should raise ValueError on empty bytes"
    except ValueError as e:
        print(f"  -> Empty PDF handled: {e}")

    try:
        extract_text_from_pdf(b"Not a real PDF file")
        assert False, "Should raise ValueError on invalid bytes"
    except ValueError as e:
        print(f"  -> Invalid PDF handled: {e}")
    print("  -> PASSED: PDF validation properly catches bad inputs.")

    # TEST 9: Gemini Connection Check without API Key
    print("\n[TEST 9] Gemini Connection Check without key")
    status = check_gemini_connection("")
    print(f"  -> Result: connected={status['connected']}, message='{status['message']}'")
    assert not status["connected"], "Should not report connected without API key"
    print("  -> PASSED: Connection check correctly detects missing key.")

    # TEST 10: Rejecting a curl command in place of the API key
    print("\n[TEST 10] Gemini Connection Check with curl command")
    status = check_gemini_connection(
        'curl "https://generativelanguage.googleapis.com/v1beta/models/test"'
    )
    assert not status["connected"]
    assert "curl command or URL" in status["message"]
    print("  -> PASSED: A pasted curl command is rejected before any API call.")

    # TEST 11: Connection check verifies model metadata and generated text
    print("\n[TEST 11] Gemini generation connectivity check")
    with patch(
        "rag_pipeline._get_gemini_model",
        return_value={"supportedGenerationMethods": ["generateContent"]},
    ) as model_check, patch(
        "rag_pipeline._generate_gemini_content",
        return_value="OK",
    ) as generation_check:
        status = check_gemini_connection("test-key")
    assert status["connected"]
    model_check.assert_called_once()
    generation_check.assert_called_once()
    print("  -> PASSED: Status is connected only after text generation succeeds.")

    # TEST 12: Unavailable model shows actual supported model names
    print("\n[TEST 12] Gemini available-model discovery")
    with patch(
        "rag_pipeline._get_gemini_model",
        side_effect=RuntimeError("Gemini API request failed (HTTP 404)"),
    ), patch(
        "rag_pipeline._list_gemini_models",
        return_value=["gemini-flash-latest", "gemini-2.5-flash"],
    ):
        status = check_gemini_connection("test-key", "missing-model")
    assert not status["connected"]
    assert "gemini-flash-latest" in status["message"]
    print("  -> PASSED: Unsupported-model errors include models listed by the API.")

    # TEST 13: Simulated generation outage is not reported as a bad model
    print("\n[TEST 13] Gemini temporary outage handling")
    with patch(
        "rag_pipeline._get_gemini_model",
        return_value={"supportedGenerationMethods": ["generateContent"]},
    ), patch(
        "rag_pipeline._generate_gemini_content",
        side_effect=RuntimeError(
            "Gemini API request failed (HTTP 503): high demand"
        ),
    ):
        status = check_gemini_connection("test-key")
    assert not status["connected"]
    assert "temporarily" in status["message"].lower()
    assert "not found" not in status["message"].lower()
    print("  -> PASSED: A generation outage is reported as temporary.")

    # TEST 14: Answer generation reports a temporary service outage
    print("\n[TEST 14] Gemini answer outage handling")
    with patch(
        "rag_pipeline._generate_gemini_content",
        side_effect=RuntimeError(
            "Gemini API request failed (HTTP 503): high demand"
        ),
    ):
        result = generate_answer(
            "What is stated?", [{"page_number": 1, "text": "A test passage."}],
            "test-key",
        )
    assert result["answer"] is None
    assert "temporarily" in result["error"].lower()
    print("  -> PASSED: Generation explains that Gemini is temporarily busy.")

    # TEST 15: Gemini REST request shape
    print("\n[TEST 15] Gemini REST request format")
    response_body = {
        "candidates": [{
            "content": {"parts": [{"text": "OK"}]},
        }],
    }
    with patch("rag_pipeline.urlopen") as mocked_urlopen:
        mocked_urlopen.return_value.__enter__.return_value.read.return_value = (
            json.dumps(response_body).encode("utf-8")
        )
        answer = _generate_gemini_content(
            "Say OK", "test-key", "gemini-flash-latest", 4
        )
    request = mocked_urlopen.call_args.args[0]
    headers = {name.lower(): value for name, value in request.header_items()}
    payload = json.loads(request.data.decode("utf-8"))
    assert answer == "OK"
    assert request.full_url.endswith(
        "/models/gemini-flash-latest:generateContent"
    )
    assert headers["x-goog-api-key"] == "test-key"
    assert payload["contents"][0]["parts"][0]["text"] == "Say OK"
    print("  -> PASSED: The app uses the expected Gemini REST endpoint and payload.")

    # TEST 16: Gemini Generation without API Key
    print("\n[TEST 16] Gemini Generation without key")
    ans_res = generate_answer("test query", results1, "")
    assert ans_res["answer"] is None
    assert "GEMINI_API_KEY" in ans_res["error"]
    print(f"  -> Correct error returned: '{ans_res['error']}'")
    print("  -> PASSED: Generation cleanly stops and notifies user of missing key.")

    # TEST 17: Gemini Generation with No Chunks
    print("\n[TEST 17] Gemini Generation with No Chunks")
    ans_empty = generate_answer("unrelated query", [], "fake-key")
    assert ans_empty["answer"] is None
    assert "No relevant passages" in ans_empty["error"]
    print(f"  -> Correct rejection returned: '{ans_empty['error']}'")
    print("  -> PASSED: Generation refuses to guess without supporting passages.")

    print("\n" + "=" * 60)
    print("ALL 17 UNIT & INTEGRATION TESTS PASSED SUCCESSFULLY! (100%)")
    print("=" * 60)

if __name__ == "__main__":
    run_tests()
