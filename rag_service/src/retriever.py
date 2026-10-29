def get_retriever(vectorstore, k=4): 

    """
    create a retriever from FAISS vectorstore.
    args:
        vectorstore: FAISS vectorstore 
        k(int): number of chunks to retrieve 
    returns:
        retriever object
    """

    retriever = vectorstore.as_retriever(
        search_kwargs={'k':k}
    )

    return retriever