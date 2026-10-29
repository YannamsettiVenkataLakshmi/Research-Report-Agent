from itertools import zip_longest
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from pydantic import BaseModel, Field
import os

from src.loader import load_pdf
from src.splitter import split_documents
from src.config import CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME

from src.embeddings import get_embedding_model
from src.vectorstore import load_vectorstore, create_vectorstore, save_vectorstore, save_index_metadata, load_index_metadata, calculate_bytes_hash
from src.retriever import get_retriever

from src.bm25_retriever import load_bm25_index, bm25_search_session, create_bm25_index, save_bm25_index
from src.hybrid_retriever import HybridRetriever
from src.reranker import CrossEncoderReranker

from src.document_registry import add_document, get_document_by_hash_and_session, generate_doc_id, load_registry, save_registry


# FASTAPI Application

app = FastAPI(
    title = "Hybrid RAG API",
    description = 'Retrieval API for the Hybrid RAG system',
    version = "1.0.0"
)

# Request schema

class RetrievalRequest(BaseModel):
    # reject empty queries
    query: str = Field(..., min_length=1)
    session_id: str = Field(..., min_length=1)
    doc_ids: list[str] | None = None

# Response schema

class RetrievalResult(BaseModel):
    rank: int
    content: str
    source: dict

class RetrievalResponse(BaseModel):
    query: str
    results: list[RetrievalResult]

# Index paths

FAISS_PATH = 'indexes/app_faiss'
BM25_PATH = 'indexes/app_bm25.pkl'

# metadata describing current index configuration
METADATA_PATH = 'indexes/metadata.json'

# Registry containing info about individual documents
REGISTRY_PATH = 'indexes/documents.json'

# Data path - uploaded pdfs will be stored here

DATA_DIR = 'data'

# load retrieval components

print('Loading embedding model...')
embedding_model = get_embedding_model()

if os.path.exists(FAISS_PATH) and os.path.exists(BM25_PATH): 

    print('Loading FAISS Vectorstore...')
    vectorstore = load_vectorstore(FAISS_PATH, embedding_model)
    dense_retriever = get_retriever(vectorstore)

    print('Loading BM25 Index...')
    bm25_retriever, chunks = load_bm25_index(BM25_PATH)

    print('Existing indexes loaded.')

else: 

    # no indexes yet - they will be created when FIRST PDF is uploaded

    vectorstore = None
    dense_retriever = None
    bm25_retriever = None
    chunks = []

    print('No existing Indexes found...')
    print('Indexes will be created when the first PDF is uploaded...')


print('Initalizing Hybrid Retriever...')
hybrid_retriever = HybridRetriever() 

print('Loading Cross-Encoder Reranker...')
reranker = CrossEncoderReranker() 

print('Retrieval pipeline is ready...')

# Health check

@app.get('/health')
def health_check(): 
    return {"status": "ok"}

def debug_docs(stage, docs):
    print(f"\n{'=' * 25} {stage} {'=' * 25}")

    for rank, doc in enumerate(docs, start=1):
        print(f"\nRank: {rank}")
        print("Source:", doc.metadata.get("source_filename"))
        print("Page:", doc.metadata.get("page"))
        print("Content:", doc.page_content[:500])


def make_preview(file_path, max_chars=300):
    """
    Short description of a PDF: the first ~300 characters of its first page.
    Computed once, when the document is registered.
    """
    try:
        pages = load_pdf(file_path)
        if not pages:
            return ""
        return " ".join(pages[0].page_content.split())[:max_chars]
    except Exception as exc:
        print(f"Could not create preview for {file_path}: {exc}")
        return ""

# index configuration validation

def index_configuration_is_valid():
    """
    Check whether the existing index was created using the current embedding model and chunking configuration.

    This does NOT check one particular document.

    It checks the configuration of the whole collection.
    """

    metadata = load_index_metadata()

    if metadata is None:
        return False
    
    # check embedding model
    if metadata.get("embedding_model") != EMBEDDING_MODEL_NAME:
        return False

    # check chunk size
    if metadata.get("chunk_size") != CHUNK_SIZE:
        return False

    # check chunk overlap
    if metadata.get("chunk_overlap") != CHUNK_OVERLAP:
        return False

    # FAISS index path must exist
    if not os.path.exists(FAISS_PATH):
        return False

    # BM25 Index must exist
    if not os.path.exists(BM25_PATH):
        return False

    return True


# Add document to existing indexes

def add_document_to_indexes(file_path, doc_id, filename, session_id):
     """
     Add a new document to the existing FAISS and BM25 indexes

     FAISS:
          Append only the new document's chunks.

    BM25: 
        Rebuild using old chunks + new chunks.

     """

     global vectorstore
     global chunks

     # Load and split new document

     documents = load_pdf(file_path)

     new_chunks = split_documents(documents)

     # Add document level info to every chunk 

     for chunk in new_chunks: 
         chunk.metadata['doc_id'] = doc_id
         chunk.metadata['source_filename'] = filename
         chunk.metadata['session_id'] = session_id

     print(f'New chunks created : {len(new_chunks)} ')

     # Add new chunks to FAISS

     if vectorstore is None: 
         vectorstore = create_vectorstore(new_chunks, embedding_model)
         print('FAISS Index created.')

     else:
         vectorstore.add_documents(new_chunks)
         print('New chunks added to FAISS')

     # save faiss index
     
     save_vectorstore(vectorstore, FAISS_PATH)

     # Rebuild BM25
     # existing + new chunks

     all_chunks = chunks + new_chunks

     bm25 = create_bm25_index(all_chunks)

     # save rebuilt bm25 index

     save_bm25_index(bm25, all_chunks, BM25_PATH)

     print('BM25 Built sucessfully.')

     return bm25, all_chunks

def rebuild_all_indexes(new_session_id=None, new_file_path=None, new_doc_id=None, new_filename=None):
    """
    Rebuild the complete shared FAISS + BM25 collection.

    Existing documents keep their original session_id from the registry.

    A new document uses new session_id.
    """

    global vectorstore
    global chunks

    registry = load_registry(REGISTRY_PATH)

    all_documents = []

    # Load all existing registered documents
    for doc_id, document in registry["documents"].items():

        documents = load_pdf(document["filepath"])
        document_session_id = document.get('session_id')

        if document_session_id is None: 

            raise ValueError(
                f"Document '{doc_id}' doesn't have a session_id in the registry."
            )

        # The splitter will carry this metadata to every chunk.
        for doc in documents:
            doc.metadata["doc_id"] = doc_id
            doc.metadata["source_filename"] = document["filename"]
            doc.metadata['session_id'] = document_session_id

        all_documents.extend(documents)

    # If new document is supplied add it to the collection.
    if new_file_path is not None: 

        if new_session_id is None: 

            raise ValueError(
                'new_session_id is required when adding a new document during rebuild.'
            )

        new_documents = load_pdf(new_file_path)

        for doc in new_documents:
            doc.metadata["doc_id"] = new_doc_id
            doc.metadata["source_filename"] = new_filename
            doc.metadata['session_id'] = new_session_id

        all_documents.extend(new_documents)

    print(f"Documents to rebuild: {len(all_documents)}")

    # Split ALL documents using the current configuration.
    all_chunks = split_documents(all_documents)

    print(f"Chunks created during rebuild: {len(all_chunks)}")

    # Create a completely new FAISS index.
    vectorstore = create_vectorstore(
        all_chunks,
        embedding_model
    )

    save_vectorstore(
        vectorstore,
        FAISS_PATH
    )

    # BM25 must also be rebuilt because its corpus has changed.
    bm25 = create_bm25_index(all_chunks)

    save_bm25_index(
        bm25,
        all_chunks,
        BM25_PATH
    )

    chunks = all_chunks

    print("Complete index rebuild successful.")

    return bm25, chunks
     
# save index metadata

def save_current_index_metadata():

    metadata = {
        'embedding_model': EMBEDDING_MODEL_NAME,
        'chunk_size': CHUNK_SIZE,
        'chunk_overlap': CHUNK_OVERLAP
    }

    save_index_metadata(metadata, METADATA_PATH)
   
# Index document 

@app.post("/index")
async def index_document(
    file: UploadFile = File(...),
    session_id: str = Form(...)
    ):

    print("INDEX SESSION ID:", session_id)
    print("INDEX SESSION ID repr:", repr(session_id))

    global vectorstore
    global dense_retriever
    global bm25_retriever
    global chunks

    # validate session_id 

    session_id = session_id.strip() 

    if not session_id: 

        return {
            'message': 'session_id cannot be empty.'
        }

    # create the data directory if it doesn't already exists

    os.makedirs(DATA_DIR, exist_ok=True)

    # Read uploaded PDF.
    pdf_bytes = await file.read()

    # Calculate hash before saving.
    file_hash = calculate_bytes_hash(pdf_bytes)

    print(f"Uploaded file: {file.filename}")
    print(f'Session ID: {session_id}')
    print(f"File hash: {file_hash}")

    
    # check whether this exact document is already registered

    existing_doc_id, existing_document = get_document_by_hash_and_session(file_hash, session_id, REGISTRY_PATH)

    # check whether the current index configuration is valid

    config_valid = index_configuration_is_valid() 

    # Case 1: Document already exists in same session and configuration is valid

    if existing_doc_id is not None and config_valid:
        return {
            'filename': existing_document['filename'],
            'message': 'Document already exists in this session',
            'doc_id': existing_doc_id,
            'file_hash': file_hash,
            'session_id': session_id
        }

    # Case 2: Configurations changed

    if not config_valid and vectorstore is not None:

        print("Index configuration changed.")
        print("Rebuilding complete collection...")

        # If Document is already registered
        if existing_doc_id is not None: 

            doc_id = existing_doc_id

            # filepath is already stored using it's doc_id
            file_path = existing_document['filepath']

            bm25_retriever, chunks = rebuild_all_indexes()

        else:
            # This is new document
            doc_id = generate_doc_id()

            # save the new document using it's internal doc_id
            file_path = os.path.join(DATA_DIR, f'{doc_id}.pdf')

            with open(file_path, "wb") as f:
                f.write(pdf_bytes)

            bm25_retriever, chunks = rebuild_all_indexes(
                new_session_id=session_id,
                new_file_path=file_path,
                new_doc_id=doc_id,
                new_filename=file.filename
            )

            # Only register the new document after indexing succeeds.
            add_document(
                preview=make_preview(file_path),
                filename=file.filename,
                filepath=file_path,
                file_hash=file_hash,
                doc_id=doc_id,
                path=REGISTRY_PATH,
                session_id=session_id
            )

        # Update dense retriever
        dense_retriever = get_retriever(vectorstore)

        # save new configuration
        save_current_index_metadata()

        return {
            "filename": file.filename,
            "message": "Document indexed successfully after rebuilding indexes",
            "doc_id": doc_id,
            "file_hash": file_hash,
            "session_id": session_id,
            "chunks": len([
                c for c in chunks
                if c.metadata.get("doc_id") == doc_id and c.metadata.get("session_id") == session_id
            ]),
            "total_chunks": len(chunks)
            }

    # Case 3 - Valid Configuration + New document

    doc_id = generate_doc_id() 

    # Save pdf 
    file_path = os.path.join(DATA_DIR, f'{doc_id}.pdf')
    
    with open(file_path, "wb") as f:
        f.write(pdf_bytes)

    # Add to existing indexes
    bm25_retriever, chunks = add_document_to_indexes(file_path, doc_id, file.filename, session_id)

    # Update dense retriever
    dense_retriever = get_retriever(vectorstore)

    # Save metadata describing the NEW index configuration.
    save_current_index_metadata()

    # Register the document
    add_document(
        preview=make_preview(file_path),
        filename=file.filename,
        filepath=file_path,
        file_hash=file_hash,
        doc_id=doc_id,
        path=REGISTRY_PATH,
        session_id=session_id
    )

    return {
        "filename": file.filename,
        "message": "Document indexed successfully after rebuilding indexes",
        "doc_id": doc_id,
        "file_hash": file_hash,
        "session_id": session_id,
        "chunks": len([
            c for c in chunks
            if c.metadata.get("doc_id") == doc_id and c.metadata.get("session_id") == session_id
        ]),
        "total_chunks": len(chunks)
    }

    


# List the documents in a session (used by the agent's list_documents tool)

@app.get("/documents")
def list_documents(session_id: str):
    """
    List the documents in a session with a short preview of each, read from
    the registry (the preview is saved once, when the PDF is uploaded).
    """
    session_id = session_id.strip()
    registry = load_registry(REGISTRY_PATH)

    documents = []
    changed = False

    for doc_id, doc in registry.get("documents", {}).items():
        if doc.get("session_id") != session_id:
            continue

        # Documents registered before previews existed: compute once and save.
        if "preview" not in doc:
            doc["preview"] = make_preview(doc.get("filepath", ""))
            changed = True

        documents.append({
            "doc_id": doc_id,
            "filename": doc.get("filename", ""),
            "preview": doc["preview"],
        })

    if changed:
        save_registry(registry, REGISTRY_PATH)

    if not documents:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found for this session."
        )

    return {"session_id": session_id, "documents": documents}


# Retrieval endpoint
@app.post("/retrieve", response_model=RetrievalResponse)
def retrieve(request: RetrievalRequest):
    query = request.query.strip()
    session_id = request.session_id.strip()
    doc_ids = request.doc_ids

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty."
        )

    if vectorstore is None or bm25_retriever is None:
        raise HTTPException(
            status_code=503,
            detail="Retrieval indexes are not initialized."
        )

    # 1. Restrict the search to this session.
    session_chunks = [
        chunk for chunk in chunks
        if chunk.metadata.get("session_id") == session_id
    ]

    if not session_chunks:
        raise HTTPException(
            status_code=404,
            detail="No indexed documents found for this session."
        )

    session_doc_ids = {
        chunk.metadata.get("doc_id")
        for chunk in session_chunks
        if chunk.metadata.get("doc_id") is not None
    }

    # 2. Apply explicit document selection only when supplied.
    if doc_ids:
        requested_doc_ids = set(doc_ids)
        invalid_doc_ids = requested_doc_ids - session_doc_ids

        if invalid_doc_ids:
            raise HTTPException(
                status_code=400,
                detail="One or more doc_ids do not belong to this session."
            )

        allowed_doc_ids = requested_doc_ids
    else:
        # No selector: search all documents in this session.
        allowed_doc_ids = session_doc_ids

    # 3. Dense retrieval: constrain FAISS metadata filtering.
    def search_docs(doc_id_set, top_k):
        """Dense + BM25 -> RRF -> rerank, restricted to doc_id_set."""
        dense_docs = vectorstore.similarity_search(
            query,
            k=10,
            filter=lambda metadata: (
                metadata.get("session_id") == session_id
                and metadata.get("doc_id") in doc_id_set
            ),
            fetch_k=max(len(chunks), 1)
        )

        bm25_docs = bm25_search_session(
            bm25_retriever,
            chunks,
            query,
            session_id,
            top_k=10,
            doc_ids=list(doc_id_set)
        )

        hybrid_docs = hybrid_retriever.reciprocal_rank_fusion(
            bm25_docs,
            dense_docs
        )

        if not hybrid_docs:
            return []

        ranked = reranker.rerank(query, hybrid_docs, top_k=top_k)

        # final guard: nothing out of scope can leave the API
        return [
            doc for doc in ranked
            if (
                doc.metadata.get("session_id") == session_id
                and doc.metadata.get("doc_id") in doc_id_set
            )
        ]

    if doc_ids and len(allowed_doc_ids) > 1:
        # Explicit multi-document request (e.g. a comparison):
        # retrieve PER document so one document cannot crowd out the others,
        # then interleave the results (doc A, doc B, doc A, doc B, ...).
        PER_DOC = 3
        per_doc_results = [
            search_docs({d}, PER_DOC) for d in sorted(allowed_doc_ids)
        ]
        reranked_docs = [
            doc
            for group in zip_longest(*per_doc_results)
            for doc in group
            if doc is not None
        ]
    else:
        reranked_docs = search_docs(allowed_doc_ids, 5)

    print("[retrieve] scope:", sorted(allowed_doc_ids))
    print("[retrieve] returned:", [
        (d.metadata.get("source_filename"), d.metadata.get("page"))
        for d in reranked_docs
    ])

    results = [
        {
            "rank": rank,
            "content": doc.page_content,
            "source": doc.metadata
        }
        for rank, doc in enumerate(reranked_docs, start=1)
    ]

    return {
        "query": request.query,
        "results": results
    }