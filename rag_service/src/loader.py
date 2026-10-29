from langchain_community.document_loaders import PyMuPDFLoader

def load_pdf(file_path: str):
    """
    Load the PDF and return the list of Langchain Objects.

    args:
        file_path(str) : path to the PDF files

    Returns:
        list: list of Document objects
    """

    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    return documents