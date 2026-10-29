from collections import defaultdict


class HybridRetriever:

    def __init__(self, rrf_k=60):
        # k value for RRF
        self.rrf_k = rrf_k

    @staticmethod
    def _key(doc):
        m = doc.metadata
        return (m.get("doc_id"), m.get("page"), doc.page_content)

    def reciprocal_rank_fusion(self, bm25_docs, dense_docs):
        """
        Combine BM25 and dense retrieval results using Reciprocal Rank Fusion.
        BOTH ranked lists contribute: score = sum(1 / (rrf_k + rank)).
        """
        scores = defaultdict(float)
        doc_map = {}

        for ranked_list in (bm25_docs, dense_docs):
            for rank, doc in enumerate(ranked_list, start=1):
                key = self._key(doc)
                doc_map[key] = doc
                scores[key] += 1 / (self.rrf_k + rank)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [doc_map[key] for key, _ in ranked]