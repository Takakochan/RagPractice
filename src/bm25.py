"""A minimal, dependency-free Okapi BM25 index over pre-tokenized documents."""

import math


class BM25Index:
    """An Okapi BM25 index built from pre-tokenized documents.

    Uses an inverted index (term -> {doc_id: term_frequency}) so scoring
    a query only touches documents containing at least one query term,
    rather than scanning the whole corpus.
    """

    def __init__(self, tokenized_docs: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        """Build inverted-index term statistics for a tokenized corpus.

        Args:
            tokenized_docs: One token list per document; doc_id i refers
                to tokenized_docs[i], and top_k() returns these same ids.
            k1: BM25 term-frequency saturation parameter.
            b: BM25 document-length normalization parameter.
        """
        self.k1 = k1
        self.b = b
        self.n_docs = len(tokenized_docs)
        self.doc_lens = [len(doc) for doc in tokenized_docs]
        self.avgdl = sum(self.doc_lens) / self.n_docs if self.n_docs else 0.0

        self.postings: dict[str, dict[int, int]] = {}
        doc_freq: dict[str, int] = {}
        for doc_id, doc in enumerate(tokenized_docs):
            freq: dict[str, int] = {}
            for term in doc:
                freq[term] = freq.get(term, 0) + 1
            for term, count in freq.items():
                self.postings.setdefault(term, {})[doc_id] = count
                doc_freq[term] = doc_freq.get(term, 0) + 1

        self.idf = {
            term: math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1)
            for term, df in doc_freq.items()
        }

    def top_k(self, query_tokens: list[str], k: int) -> list[tuple[int, float]]:
        """Score every matching document against a query and return the top k.

        Args:
            query_tokens: Tokenized query terms.
            k: Maximum number of results to return.

        Returns:
            A list of (doc_id, score) pairs, highest score first. Empty
            for an empty/unmatched query or k <= 0.
        """
        if k <= 0:
            return []
        scores: dict[int, float] = {}
        for term in query_tokens:
            idf = self.idf.get(term)
            if idf is None:
                continue
            for doc_id, freq in self.postings.get(term, {}).items():
                denom = freq + self.k1 * (1 - self.b + self.b * self.doc_lens[doc_id] / self.avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * freq * (self.k1 + 1) / denom
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]
