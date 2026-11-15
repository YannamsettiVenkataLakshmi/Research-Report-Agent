from langchain_huggingface import HuggingFaceEmbeddings 
from rag_service.src.config import EMBEDDING_MODEL_NAME
import torch

def get_embedding_model(): 
    """
    Load and return the hugging face model
    """

    if torch.backends.mps.is_available(): 
        device = 'mps'
    else: 
        device = 'cpu'

    embedding_model = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL_NAME,
        model_kwargs = {
            'device': device
        },
        encode_kwargs = {
            'normalize_embeddings': True,
            'batch_size': 2
        }
    )

    return embedding_model