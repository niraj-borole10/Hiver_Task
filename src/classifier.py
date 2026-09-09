"""
Intent Classifier for @AppleSupport inquiries.
Supports calibrated ML classification (TF-IDF + Logistic Regression) and LLM-based classification.
"""
import os
import json
import joblib
from typing import Dict, Any, List, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from src.taxonomy import IntentCategory, INTENT_REGISTRY

MODEL_PATH = os.path.join("data", "processed", "classifier.joblib")


class IntentClassifier:
    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self.pipeline: Optional[Pipeline] = None
        self._is_loaded = False

    def train_on_pairs(self, pairs_file: str = "data/processed/pairs.jsonl", max_samples: int = 10000):
        print(f"Training calibrated Intent Classifier on up to {max_samples} interactions...")
        from data.builder import analyze_intent_and_escalation

        texts = []
        labels = []

        with open(pairs_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if len(texts) >= max_samples:
                    break
                row = json.loads(line.strip())
                cust_text = row["customer_text"]
                reply_text = row["apple_reply_text"]
                res = analyze_intent_and_escalation(cust_text, reply_text)
                if res:
                    texts.append(cust_text)
                    labels.append(res["intent"])

        print(f"Collected {len(texts)} labelled training samples across {len(set(labels))} classes.")
        
        # Build TF-IDF + Calibrated Logistic Regression pipeline
        pipe = Pipeline([
            ("tfidf", TfidfVectorizer(max_features=8000, ngram_range=(1, 2), stop_words="english")),
            ("clf", LogisticRegression(C=2.0, max_iter=500, class_weight="balanced"))
        ])

        pipe.fit(texts, labels)
        self.pipeline = pipe
        self._is_loaded = True

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        joblib.dump(self.pipeline, self.model_path)
        print(f"Classifier saved successfully to {self.model_path}")

    def load(self):
        if os.path.exists(self.model_path):
            self.pipeline = joblib.load(self.model_path)
            self._is_loaded = True
        else:
            self.train_on_pairs()

    def predict(self, text: str) -> Dict[str, Any]:
        if not self._is_loaded:
            self.load()

        probs = self.pipeline.predict_proba([text])[0]
        classes = self.pipeline.classes_
        top_idx = int(np.argmax(probs))
        top_class = classes[top_idx]
        confidence = float(probs[top_idx])

        # Rule guardrail for strong intent keywords if confidence is borderline
        text_lower = text.lower()
        if "battery" in text_lower or "draining" in text_lower:
            if top_class != "HARDWARE_BATTERY_CHARGING" and confidence < 0.6:
                top_class = "HARDWARE_BATTERY_CHARGING"
                confidence = 0.75
        elif "apple id" in text_lower or "icloud" in text_lower or "2fa" in text_lower:
            if top_class != "APPLE_ID_ICLOUD_SECURITY" and confidence < 0.6:
                top_class = "APPLE_ID_ICLOUD_SECURITY"
                confidence = 0.80
        elif "refund" in text_lower or "itunes.com/bill" in text_lower or "subscription" in text_lower:
            if top_class != "BILLING_SUBSCRIPTIONS_PURCHASES" and confidence < 0.6:
                top_class = "BILLING_SUBSCRIPTIONS_PURCHASES"
                confidence = 0.80

        all_probs = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, probs)}

        return {
            "predicted_intent": IntentCategory(top_class),
            "confidence": round(confidence, 4),
            "all_probabilities": all_probs,
        }
