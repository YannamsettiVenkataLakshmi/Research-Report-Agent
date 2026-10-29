import json
import os
import hashlib

from langchain_community.vectorstores import FAISS

def create_vectorstore(chunks, embedding_model):
    """
    create a FAISS vectorstore from document chunks
    args: 
        chunks(list): list of document chunks 
        embedding_model: Hugging Face Embedding model
    returns:
        FAISS: vectorstore containing embedded chunks
    """

    vectorstore = FAISS.from_documents(
        embedding=embedding_model, 
        documents=chunks
    )

    return vectorstore

def save_vectorstore(vectorstore, path):
    """
    Save vectorstore to the path
    """
    os.makedirs(path, exist_ok=True)

    vectorstore.save_local(path)

    print(f'Vectorstore saved to: {path}')

def load_vectorstore(path, embedding_model):
    """
    Load an existing FAISS vectorstore from disk.

    The vectorstore must have been created with the
    same embedding model.
    """

    vectorstore = FAISS.load_local(path, embedding_model, allow_dangerous_deserialization=True)

    print(f'Vectorstore loaded from: {path}')

    return vectorstore


# File Hashing

def calculate_bytes_hash(file_bytes):
    """
    Calculate SHA-256 hash directly from file bytes.

    This is useful when the uploaded file has not been saved to disk yet.
    """
    sha256 = hashlib.sha256()
    
    sha256.update(file_bytes)

    return sha256.hexdigest()

# Metadata
def save_index_metadata(metadata, path='indexes/metadata.json'):
    """
    Save metadata describing the current indexes
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f'Index metadata saved to : {path}')

def load_index_metadata(path="indexes/metadata.json"):
    """
    Load the saved index metadata.

    Returns:
        dict: saved metadata
        None: if metadata file does not exist
    """

    if not os.path.exists(path):
        return None

    with open(path, "r") as f:
        metadata = json.load(f)

    return metadata


   