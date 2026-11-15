from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag_service.src.config import CHUNK_SIZE, CHUNK_OVERLAP

def split_documents(documents):
    """
    split documents into smaller chunks

    args:
        documents(list): list of document objects
    returns:
        list: list of document chunks
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        chunk_overlap = CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)

    return chunks