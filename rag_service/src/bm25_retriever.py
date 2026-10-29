import pickle

from rank_bm25 import BM25Okapi
from src.preprocessing import TechnicalPreprocessor

preprocessor = TechnicalPreprocessor()

def create_bm25_index(chunks):
    """
    Create a bm25 index from document chunks.
    
    chunks:
          List of document objects.
    """

    # convert every chunk's text into tokens

    tokenized_documents = [preprocessor.preprocess(chunk.page_content) for chunk in chunks]

    # build bm25 using all those chunks

    bm25 = BM25Okapi(tokenized_documents)

    return bm25


def bm25_search_session(bm25, chunks, query, session_id, top_k=4, doc_ids=None):
    """
    search the bm25 index 
    
    Return the top-k positive-scoring chunks belonging to the
    requested session and, optionally, the requested document IDs.
    """
    
    session_indices = [i for i, chunk in enumerate(chunks)
        if (
            chunk.metadata.get("session_id") == session_id
            and (
                not doc_ids
                or chunk.metadata.get("doc_id") in doc_ids
            )
        )
    ]

    if not session_indices:
        return []

    tokenized_query = preprocessor.preprocess(query)

    # calculate score for everything

    doc_scores = bm25.get_scores(tokenized_query)

    # Keep only chunks with a positive BM25 score.
    ranked_indices = [
        i for i in session_indices
        if doc_scores[i] > 0
    ]

    ranked_indices.sort(
        key=lambda i: doc_scores[i],
        reverse=True
    )

    return [
        chunks[i] for i in ranked_indices[:top_k]
    ]


def save_bm25_index(bm25, chunks, path):
    """
    Save bm25 index and chunks
    """

    with open(path,'wb') as f:

        pickle.dump({
            'bm25': bm25,
            'chunks': chunks
        }, f)

    print(f'BM25 saved to : {path}')


def load_bm25_index(path):
    """
    Load BM25 and corresponding chunks
    """

    with open(path, 'rb') as f:
        data = pickle.load(f)

    print(f'BM25 index loaded from : {path}')

    return data['bm25'], data['chunks']
