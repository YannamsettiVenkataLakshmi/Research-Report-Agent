import json 
import os 
import uuid

# loction for information storage about all uploaded documents
REGISTRY_PATH = 'indexes/documents.json'

def load_registry(path=REGISTRY_PATH):
    """
    Load the document registry from the documents.json.
    
    If the file doesn't exist yet, return an empty registry.
    """

    if not os.path.exists(path): 
        return {'documents': {}}

    with open(path, 'r') as f: 
        return json.load(f)

def save_registry(registry, path=REGISTRY_PATH): 
    """
    Save the current document registry to documents.json
    """

    os.makedirs(os.path.dirname(path), exist_ok=True)

    with open(path, 'w') as f: 
        json.dump(registry, f, indent=4)

def generate_doc_id():
    """
    Generate a unique internal ID for a Document.
    """

    return str(uuid.uuid4())

def add_document(filename, filepath, file_hash, session_id, doc_id, path=REGISTRY_PATH, preview=''):
    """
    Add new document to the registry.
    
    Returns: doc_id : internal ID assigned to the document""" 

    # Load existing documents
    registry = load_registry(path)

    # store the document information
    registry['documents'][doc_id] = {
        'filename': filename, 
        'filepath': filepath, 
        'file_hash': file_hash,
        'session_id': session_id,
        'preview': preview
    }

    # save the updated registry
    save_registry(registry, path)

    return doc_id

def get_document(doc_id, path=REGISTRY_PATH):
    """
    Get document using internal doc_id.
    """

    registry = load_registry(path=REGISTRY_PATH) 

    return registry['documents'].get(doc_id)

def get_document_by_hash_and_session(file_hash, session_id, path=REGISTRY_PATH): 
    """
    Find the document only in the current session.

    The same PDF is allowed to exist in multiple sessions.

    Returns: (doc_id, document) if found (None, None) if not found.
    """

    registry = load_registry(path)

    for doc_id, document in registry['documents'].items(): 

        if document.get('file_hash') == file_hash and document.get('session_id') == session_id: 
            return doc_id, document 

    return None, None