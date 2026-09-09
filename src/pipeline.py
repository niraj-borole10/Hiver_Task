"""
Unified AppleSupport AI Agent Pipeline.
Coordinates preprocessing, intent classification, historical resolution retrieval,
calibrated escalation triage, and grounded reply drafting into a single production interface.
"""
from typing import Dict, Any, Optional
from src.preprocessor import TweetPreprocessor
from src.taxonomy import IntentCategory, EscalationDecision
from src.classifier import IntentClassifier
from src.retriever import HybridResolutionRetriever
from src.escalation import EscalationEngine
from src.generator import GroundedReplyGenerator


class AppleSupportAgent:
    def __init__(
        self,
        retriever_max_pairs: int = 5000,
        min_intent_confidence: float = 0.55,
        min_retrieval_similarity: float = 0.12,
    ):
        print("Initializing AppleSupport AI Agent Pipeline...")
        self.preprocessor = TweetPreprocessor()
        self.classifier = IntentClassifier()
        self.retriever = HybridResolutionRetriever(max_index_size=retriever_max_pairs)
        self.escalation_engine = EscalationEngine(
            min_intent_confidence=min_intent_confidence,
            min_retrieval_similarity=min_retrieval_similarity
        )
        self.generator = GroundedReplyGenerator()
        
        # Warm up components
        self.classifier.load()
        self.retriever.build_index()
        print("AppleSupport AI Agent Pipeline is ready.")

    def handle_tweet(self, raw_tweet: str) -> Dict[str, Any]:
        """Processes an incoming customer tweet through the full support pipeline."""
        # 1. Preprocess
        prep = self.preprocessor.process(raw_tweet)
        cleaned_text = prep["cleaned_text"]
        entities = prep["entities"]

        # 2. Classify Intent
        clf_result = self.classifier.predict(cleaned_text)
        predicted_intent = clf_result["predicted_intent"]
        confidence = clf_result["confidence"]

        # 3. Retrieve Historical Resolutions
        retrieved_resolutions = self.retriever.retrieve(cleaned_text, top_k=3)
        top_retrieval_score = retrieved_resolutions[0]["similarity_score"] if retrieved_resolutions else 0.0

        # 4. Evaluate Escalation Triage
        escalation_result = self.escalation_engine.evaluate(
            text=cleaned_text,
            predicted_intent=predicted_intent,
            intent_confidence=confidence,
            top_retrieval_score=top_retrieval_score
        )
        decision = escalation_result["decision"]
        reason_code = escalation_result["reason_code"]
        reason_text = escalation_result["reason_text"]

        # 5. Draft Grounded Response
        generation_result = self.generator.generate(
            customer_text=cleaned_text,
            intent=predicted_intent,
            escalation_decision=decision,
            escalation_reason=reason_code,
            retrieved_resolutions=retrieved_resolutions,
            entities=entities
        )

        return {
            "input_tweet": raw_tweet,
            "cleaned_tweet": cleaned_text,
            "entities": entities,
            "predicted_intent": predicted_intent.value,
            "intent_confidence": confidence,
            "escalation_decision": decision.value,
            "escalation_reason_code": reason_code,
            "escalation_reason_text": reason_text,
            "draft_reply": generation_result["draft_reply"],
            "reply_char_count": generation_result["char_count"],
            "is_tweet_compliant": generation_result["is_within_limit"],
            "retrieved_context": retrieved_resolutions,
        }
