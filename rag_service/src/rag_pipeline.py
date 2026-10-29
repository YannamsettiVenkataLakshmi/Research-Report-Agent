import re
import requests

from src.document_registry import load_registry, REGISTRY_PATH
from src.query_rewriter import create_rewriter, rewrite_query
from src.query_router import create_router, route_query

rewriter = create_rewriter()
router = create_router()

# Words that say nothing about WHICH document the user means.
_STOP = {
    "the", "and", "for", "are", "was", "with", "this", "that", "what", "which",
    "about", "from", "does", "say", "tell", "discussed", "discuss", "been",
    "document", "documents", "paper", "papers", "pdf", "file", "summary",
}


def match_documents_by_name(query, session_id):
    """
    Deterministic scoping (no LLM): if the query mentions a word that appears
    in a document's filename, search only those documents.

    Returns a list of doc_ids, or None (= search all documents in the session).
    """
    registry = load_registry(REGISTRY_PATH)

    query_words = {
        w for w in re.findall(r"[a-z0-9]+", query.lower())
        if len(w) > 2 and w not in _STOP
    }

    matched = []
    for doc_id, doc in registry.get("documents", {}).items():
        if doc.get("session_id") != session_id:
            continue
        name_words = set(re.findall(r"[a-z0-9]+", doc.get("filename", "").lower()))
        if query_words & name_words:
            matched.append(doc_id)

    return matched or None


def process_query(
    conversation_history,
    query,
    session_id,
    document_context,
    doc_ids=None,
):

    # Step 1: Rewrite the query using conversation context.
    rewritten_query = rewrite_query(
        rewriter,
        conversation_history,
        query
    )

    print("\nRewritten query:", rewritten_query)

    # Step 2: Route the query.
    route = route_query(
        router,
        conversation_history,
        rewritten_query,
        document_context,
    )

    print("Route:", route)

    if route == "DIRECT":
        return {
            "route": "DIRECT",
            "query": rewritten_query,
            "selected_doc_ids": [],
            "results": [],
        }

    # Scope: explicit ids > filename match > whole session
    doc_ids = doc_ids or match_documents_by_name(rewritten_query, session_id)
    print("Doc scope:", doc_ids if doc_ids else "ALL documents in session")

    response = requests.post(
        "http://127.0.0.1:8000/retrieve",
        json={
            "query": rewritten_query,
            "session_id": session_id,
            "doc_ids": doc_ids,
        },
        timeout=60,
    )

    response.raise_for_status()
    retrieval_data = response.json()

    return {
        "route": "RAG",
        "query": rewritten_query,
        "selected_doc_ids": doc_ids or [],
        "results": retrieval_data.get("results", []),
    }