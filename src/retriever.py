import os
import pickle
from typing import List, Dict, Tuple

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
from tqdm import tqdm


class BM25Retriever:
    def __init__(self, documents: List[str], bm25_index_path: str | None = None, tokenizer=None):
        self.documents = documents
        self.tokenizer = tokenizer
        self.corpus_tokens = [self._tokenize(doc) for doc in documents]
        self.bm25 = BM25Okapi(self.corpus_tokens)
        self.index_path = bm25_index_path

    def _tokenize(self, text: str) -> List[str]:
        if self.tokenizer is None:
            return text.lower().split()
        return self.tokenizer(text)

    def save(self):
        if not self.index_path:
            return
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"documents": self.documents, "corpus_tokens": self.corpus_tokens}, f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(data["documents"])  # tokenizer omitted on load
        obj.corpus_tokens = data["corpus_tokens"]
        obj.bm25 = BM25Okapi(obj.corpus_tokens)
        obj.index_path = path
        return obj

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        idxs = np.argsort(scores)[::-1][:top_k]
        return [(int(i), float(scores[i])) for i in idxs]


class DenseRetriever:
    def __init__(self, model_name: str, index_path: str | None = None, embeddings_path: str | None = None):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.index_path = index_path
        self.embeddings_path = embeddings_path
        self.embeddings = None

    def build(self, documents: List[str], batch_size: int = 64) -> None:
        embeddings = self.model.encode(documents, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=True, normalize_embeddings=True)
        self.embeddings = embeddings.astype("float32")
        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

    def save(self) -> None:
        if self.index_path:
            os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
            faiss.write_index(self.index, self.index_path)
        if self.embeddings_path and self.embeddings is not None:
            np.save(self.embeddings_path, self.embeddings)

    def load(self) -> None:
        if self.index_path and os.path.exists(self.index_path):
            self.index = faiss.read_index(self.index_path)
        if self.embeddings_path and os.path.exists(self.embeddings_path + ".npy"):
            self.embeddings = np.load(self.embeddings_path + ".npy")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        q_emb = self.model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
        scores, idxs = self.index.search(q_emb.astype("float32"), top_k)
        return [(int(i), float(s)) for i, s in zip(idxs[0], scores[0])]


class HybridRetriever:
    def __init__(self, bm25: BM25Retriever, dense: DenseRetriever, bm25_weight: float = 0.5, dense_weight: float = 0.5):
        self.bm25 = bm25
        self.dense = dense
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        bm25_results = self.bm25.search(query, top_k=top_k * 5)  # wider list
        dense_results = self.dense.search(query, top_k=top_k * 5)

        scores: Dict[int, float] = {}
        for i, s in bm25_results:
            scores[i] = scores.get(i, 0.0) + self.bm25_weight * self._normalize(s, [x[1] for x in bm25_results])
        for i, s in dense_results:
            scores[i] = scores.get(i, 0.0) + self.dense_weight * self._normalize(s, [x[1] for x in dense_results])

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return ranked

    @staticmethod
    def _normalize(value: float, arr: List[float]) -> float:
        if not arr:
            return 0.0
        a = float(min(arr))
        b = float(max(arr))
        if b - a < 1e-12:
            return 0.0
        return (value - a) / (b - a)
