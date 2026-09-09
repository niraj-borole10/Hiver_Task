"""
Escalation and Triage Engine for @AppleSupport.
Decides whether an incoming query can be auto-handled or must be escalated to a human agent,
along with an interpretable, stated rationale.
"""
import re
from typing import Dict, Any, Optional
from src.taxonomy import IntentCategory, EscalationDecision, ESCALATION_REASONS


class EscalationEngine:
    def __init__(
        self,
        min_intent_confidence: float = 0.55,
        min_retrieval_similarity: float = 0.12
    ):
        self.min_intent_confidence = min_intent_confidence
        self.min_retrieval_similarity = min_retrieval_similarity

        # Hard triggers regex patterns
        self.security_patterns = [
            r"\b(locked\s*out|apple\s*id\s*disabled|hacked|compromised|stolen\s*phone|activation\s*lock|forgot(ten)?\s*password|recovery\s*key)\b",
            r"\b(cant\s*log\s*in|cannot\s*access\s*(my\s*)?account)\b",
        ]

        self.financial_patterns = [
            r"\b(refund|unauthorized\s*charge|overcharged|double\s*charge|charged\s*twice|stole\s*my\s*money|scam|fraud|cancel\s*subscription)\b",
            r"\b(itunes\.com/bill|unknown\s*charge|money\s*back)\b",
        ]

        self.hardware_repair_patterns = [
            r"\b(shattered|cracked\s*screen|broken\s*glass|water\s*damage|dropped\s*in\s*water|swollen\s*battery|smoke|spark|burned)\b",
            r"\b(repair\s*cost|replace\s*screen|genius\s*bar\s*repair|bricked|restore\s*error\s*[0-9]+)\b",
        ]

        self.anger_churn_patterns = [
            r"\b(lawyer|lawsuit|sue\s*apple|attorney|better\s*business\s*bureau|unacceptable|furious|disgusted|worst\s*customer\s*service|rip\s*off)\b",
            r"\b(third\s*time\s*asking|ignoring\s*me|never\s*buying\s*apple\s*again)\b",
        ]

        self.auth_required_patterns = [
            r"\b(where\s*is\s*my\s*order|package\s*lost|order\s*never\s*arrived|change\s*delivery\s*address|carrier\s*lock(ed)?)\b",
            r"\b(sim\s*card\s*failure|no\s*service\s*urgent)\b",
        ]

    def evaluate(
        self,
        text: str,
        predicted_intent: IntentCategory,
        intent_confidence: float = 1.0,
        top_retrieval_score: float = 0.5
    ) -> Dict[str, Any]:
        text_lower = text.lower()

        # 1. High Anger & Churn Risk (Safety & Brand Preservation)
        for pat in self.anger_churn_patterns:
            if re.search(pat, text_lower):
                return {
                    "decision": EscalationDecision.HUMAN_ESCALATION,
                    "reason_code": "HIGH_ANGER_CHURN",
                    "reason_text": ESCALATION_REASONS["HIGH_ANGER_CHURN"],
                    "trigger_type": "HARD_POLICY_RULE",
                }

        # 2. Account Takeover / Security Lockout
        for pat in self.security_patterns:
            if re.search(pat, text_lower):
                return {
                    "decision": EscalationDecision.HUMAN_ESCALATION,
                    "reason_code": "SECURITY_LOCKOUT",
                    "reason_text": ESCALATION_REASONS["SECURITY_LOCKOUT"],
                    "trigger_type": "HARD_POLICY_RULE",
                }

        # 3. Financial Disputes & Unauthorized Purchases
        for pat in self.financial_patterns:
            if re.search(pat, text_lower):
                return {
                    "decision": EscalationDecision.HUMAN_ESCALATION,
                    "reason_code": "FINANCIAL_DISPUTE",
                    "reason_text": ESCALATION_REASONS["FINANCIAL_DISPUTE"],
                    "trigger_type": "HARD_POLICY_RULE",
                }

        # 4. Hardware Repair / Severe Physical Damage
        for pat in self.hardware_repair_patterns:
            if re.search(pat, text_lower):
                return {
                    "decision": EscalationDecision.HUMAN_ESCALATION,
                    "reason_code": "HARDWARE_REPAIR_BOOKING",
                    "reason_text": ESCALATION_REASONS["HARDWARE_REPAIR_BOOKING"],
                    "trigger_type": "HARD_POLICY_RULE",
                }

        # 5. Auth / Order Delivery Verification
        for pat in self.auth_required_patterns:
            if re.search(pat, text_lower):
                return {
                    "decision": EscalationDecision.HUMAN_ESCALATION,
                    "reason_code": "AUTH_REQUIRED",
                    "reason_text": ESCALATION_REASONS["AUTH_REQUIRED"],
                    "trigger_type": "HARD_POLICY_RULE",
                }

        # 6. Intent-specific default escalation check
        if predicted_intent == IntentCategory.APPLE_ID_ICLOUD_SECURITY and any(w in text_lower for w in ["locked", "disabled", "cannot log in"]):
            return {
                "decision": EscalationDecision.HUMAN_ESCALATION,
                "reason_code": "SECURITY_LOCKOUT",
                "reason_text": ESCALATION_REASONS["SECURITY_LOCKOUT"],
                "trigger_type": "INTENT_POLICY_RULE",
            }

        # 7. Confidence and retrieval safety guardrails
        if intent_confidence < self.min_intent_confidence:
            return {
                "decision": EscalationDecision.HUMAN_ESCALATION,
                "reason_code": "LOW_CONFIDENCE",
                "reason_text": f"Model intent confidence ({intent_confidence:.2f}) is below safe threshold ({self.min_intent_confidence:.2f}).",
                "trigger_type": "CONFIDENCE_GUARDRAIL",
            }

        if top_retrieval_score < self.min_retrieval_similarity:
            return {
                "decision": EscalationDecision.HUMAN_ESCALATION,
                "reason_code": "LOW_CONFIDENCE",
                "reason_text": f"No verified historical resolution matched with sufficient confidence (score: {top_retrieval_score:.2f} < {self.min_retrieval_similarity:.2f}).",
                "trigger_type": "RETRIEVAL_GUARDRAIL",
            }

        # 8. Safe to Auto-Handle
        return {
            "decision": EscalationDecision.AUTO_HANDLE,
            "reason_code": "NONE",
            "reason_text": ESCALATION_REASONS["NONE"],
            "trigger_type": "STANDARD_AUTOMATION",
        }
