"""
Historical resolution retriever for AppleSupport interactions.
Implements a fast hybrid retrieval engine combining BM25 lexical scoring and TF-IDF semantic vector similarity.
"""
import os
import json
import math
import re
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BM25Retriever:
    """Lightweight self-contained BM25 implementation with inverted indexing for high-speed scoring."""
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus_size = 0
        self.avgdl = 0.0
        self.doc_freqs = Counter()
        self.doc_lengths = []
        self.inverted_index = {}  # token -> list of (doc_id, tf)

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_\-\.]+\b", text.lower())

    def fit(self, texts: List[str]):
        self.corpus_size = len(texts)
        self.inverted_index = {}
        self.doc_lengths = []
        self.doc_freqs = Counter()

        for doc_id, text in enumerate(texts):
            tokens = self._tokenize(text)
            self.doc_lengths.append(len(tokens))
            counts = Counter(tokens)
            for token, tf in counts.items():
                self.doc_freqs[token] += 1
                if token not in self.inverted_index:
                    self.inverted_index[token] = []
                self.inverted_index[token].append((doc_id, tf))

        self.avgdl = sum(self.doc_lengths) / max(1, self.corpus_size)

    def get_scores(self, query: str) -> np.ndarray:
        query_tokens = self._tokenize(query)
        scores = np.zeros(self.corpus_size, dtype=np.float32)

        for token in query_tokens:
            if token not in self.inverted_index:
                continue
            df = self.doc_freqs[token]
            idf = math.log((self.corpus_size - df + 0.5) / (df + 0.5) + 1.0)
            if idf <= 0:
                continue

            for doc_id, tf in self.inverted_index[token]:
                num = tf * (self.k1 + 1)
                denom = tf + self.k1 * (1 - self.b + self.b * (self.doc_lengths[doc_id] / self.avgdl))
                scores[doc_id] += idf * (num / denom)

        return scores


class HybridResolutionRetriever:
    """Hybrid Retriever combining BM25 exact match and TF-IDF semantic similarity over historical QA pairs."""
    def __init__(self, pairs_file: str = "data/processed/pairs.jsonl", max_index_size: int = 5000):
        self.pairs_file = pairs_file
        self.max_index_size = max_index_size
        self.pairs: List[Dict[str, Any]] = []
        self.bm25 = BM25Retriever()
        self.tfidf = TfidfVectorizer(max_features=10000, stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = None
        self._is_indexed = False

    def build_index(self):
        if not os.path.exists(self.pairs_file):
            raise FileNotFoundError(f"Historical pairs file not found: {self.pairs_file}")

        print(f"Indexing up to {self.max_index_size} AppleSupport resolution pairs...")
        self.pairs = []
        with open(self.pairs_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= self.max_index_size:
                    break
                self.pairs.append(json.loads(line.strip()))

        corpus_texts = [p["customer_text"] for p in self.pairs]

        # 1. Fit BM25
        self.bm25.fit(corpus_texts)

        # 2. Fit TF-IDF
        self.tfidf_matrix = self.tfidf.fit_transform(corpus_texts)
        self._is_indexed = True
        print(f"Index built successfully with {len(self.pairs)} pairs.")

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        if not self._is_indexed:
            self.build_index()

        # 1. BM25 Scores
        bm25_raw = self.bm25.get_scores(query)
        max_bm25 = bm25_raw.max() if len(bm25_raw) > 0 and bm25_raw.max() > 0 else 1.0
        bm25_norm = bm25_raw / max_bm25

        # 2. TF-IDF Cosine Scores
        query_vec = self.tfidf.transform([query])
        tfidf_scores = cosine_similarity(query_vec, self.tfidf_matrix).flatten()

        # 3. Hybrid fusion
        combined_scores = 0.5 * bm25_norm + 0.5 * tfidf_scores
        top_indices = np.argsort(combined_scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(combined_scores[idx])
            results.append({
                "pair_id": self.pairs[idx]["pair_id"],
                "customer_text": self.pairs[idx]["customer_text"],
                "apple_reply_text": self.pairs[idx]["apple_reply_text"],
                "similarity_score": round(score, 4),
                "bm25_score": round(float(bm25_norm[idx]), 4),
                "tfidf_score": round(float(tfidf_scores[idx]), 4),
            })
        return results
