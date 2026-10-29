from sentence_transformers import CrossEncoder

class CrossEncoderReranker:
    """
    Reranks retrieved documents using a CrossEncoder model.
    """
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    # def rerank(self, query, retrieved_docs, top_k=4):

    #     pairs = [(query, doc.page_content) for doc in retrieved_docs]

    #     scores = self.model.predict(pairs)

    #     doc_scores = list(zip(retrieved_docs, scores))

    #     doc_scores.sort(key=lambda x: x[1], reverse=True)

    #     reranked_docs = [doc for doc, score in doc_scores[:top_k]]

    #     return reranked_docs


    def rerank(self, query, retrieved_docs, top_k=4):
        pairs = [
            (query, doc.page_content)
            for doc in retrieved_docs
        ]

        scores = self.model.predict(pairs)

        doc_scores = sorted(
            zip(retrieved_docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            doc for doc, score in doc_scores[:top_k]
        ]

        