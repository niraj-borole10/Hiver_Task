"""
Trivial Baseline for @AppleSupport.
- Intent: Always predicts the empirical majority class ("SOFTWARE_OS_CRASH").
- Reply: Fixed canned response tweet.
- Escalation: Naive zero-escalation policy (always predicts AUTO_HANDLE).
"""
from typing import Dict, Any


class TrivialBaseline:
    def __init__(self, majority_class: str = "SOFTWARE_OS_CRASH"):
        self.majority_class = majority_class
        self.canned_reply = "Thanks for reaching out! Please send us a DM with your device model and current iOS version so we can look into this."

    def handle_tweet(self, tweet_text: str) -> Dict[str, Any]:
        return {
            "predicted_intent": self.majority_class,
            "intent_confidence": 0.33,
            "escalation_decision": "AUTO_HANDLE",
            "escalation_reason_code": "NONE",
            "escalation_reason_text": "Trivial baseline default auto-handle policy.",
            "draft_reply": self.canned_reply,
            "reply_char_count": len(self.canned_reply),
            "is_tweet_compliant": True,
        }
