"""
Simple Baseline for @AppleSupport.
- Intent: Unigram TF-IDF + Multinomial Naive Bayes.
- Reply: Verbatim copy of top-1 BM25 historical tweet (no rewriting, includes raw noise/handles).
- Escalation: Heuristic keyword-list lookup.
"""
import os
import json
import re
from typing import Dict, Any, List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from src.retriever import BM25Retriever


class SimpleBaseline:
    def __init__(self, pairs_file: str = "data/processed/pairs.jsonl", max_samples: int = 4000):
        self.pairs_file = pairs_file
        self.max_samples = max_samples
        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        self.clf = MultinomialNB()
        self.bm25 = BM25Retriever()
        self.historical_replies = []
        self._is_ready = False

        self.esc_keywords = [
            "refund", "unauthorized", "charge", "charged", "locked", "disabled",
            "stolen", "hacked", "lawyer", "sue", "cracked", "broken", "water damage"
        ]

    def fit(self):
        print("Training Simple Baseline (TF-IDF + Naive Bayes & BM25)...")
        from data.builder import analyze_intent_and_escalation

        texts = []
        labels = []
        self.historical_replies = []

        with open(self.pairs_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(texts) >= self.max_samples:
                    break
                row = json.loads(line.strip())
                cust_text = row["customer_text"]
                reply_text = row["apple_reply_text"]
                res = analyze_intent_and_escalation(cust_text, reply_text)
                if res:
                    texts.append(cust_text)
                    labels.append(res["intent"])
                    self.historical_replies.append(reply_text)

        X = self.vectorizer.fit_transform(texts)
        self.clf.fit(X, labels)
        self.bm25.fit(texts)
        self._is_ready = True
        print("Simple Baseline ready.")

    def handle_tweet(self, tweet_text: str) -> Dict[str, Any]:
        if not self._is_ready:
            self.fit()

        # Intent
        X_vec = self.vectorizer.transform([tweet_text])
        pred_intent = self.clf.predict(X_vec)[0]
        probs = self.clf.predict_proba(X_vec)[0]
        conf = float(max(probs))

        # Reply: verbatim top-1 BM25
        scores = self.bm25.get_scores(tweet_text)
        top_idx = int(scores.argmax()) if len(scores) > 0 else 0
        verbatim_reply = self.historical_replies[top_idx] if self.historical_replies else "Please contact Apple Support."

        # Escalation: naive keyword check
        text_lower = tweet_text.lower()
        if any(kw in text_lower for kw in self.esc_keywords):
            decision = "HUMAN_ESCALATION"
            reason = "Triggered by simple keyword match"
        else:
            decision = "AUTO_HANDLE"
            reason = "No sensitive keywords detected"

        return {
            "predicted_intent": pred_intent,
            "intent_confidence": round(conf, 4),
            "escalation_decision": decision,
            "escalation_reason_code": "KEYWORD_MATCH" if decision == "HUMAN_ESCALATION" else "NONE",
            "escalation_reason_text": reason,
            "draft_reply": verbatim_reply,
            "reply_char_count": len(verbatim_reply),
            "is_tweet_compliant": len(verbatim_reply) <= 280,
        }
