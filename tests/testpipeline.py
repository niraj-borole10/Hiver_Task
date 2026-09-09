"""
Unit and integration tests for AppleSupport AI Agent pipeline.
"""
import pytest
from src.preprocessor import TweetPreprocessor
from src.taxonomy import IntentCategory, EscalationDecision
from src.classifier import IntentClassifier
from src.escalation import EscalationEngine
from src.generator import GroundedReplyGenerator
from src.pipeline import AppleSupportAgent


@pytest.fixture(scope="module")
def agent():
    return AppleSupportAgent(retriever_max_pairs=500)


def test_preprocessor():
    prep = TweetPreprocessor()
    res = prep.process("@115712 @AppleSupport My iPhone 11 battery drains quickly on iOS 15.2! https://example.com")
    assert "@customer" in res["cleaned_text"]
    assert "@AppleSupport" in res["cleaned_text"]
    assert "iphone" in res["entities"]["devices"]
    assert len(res["entities"]["ios_versions"]) > 0
    assert res["entities"]["has_link"] is True


def test_escalation_security():
    engine = EscalationEngine()
    res = engine.evaluate(
        text="Someone hacked into my Apple ID and locked me out!",
        predicted_intent=IntentCategory.APPLE_ID_ICLOUD_SECURITY,
        intent_confidence=0.95,
        top_retrieval_score=0.8
    )
    assert res["decision"] == EscalationDecision.HUMAN_ESCALATION
    assert res["reason_code"] == "SECURITY_LOCKOUT"


def test_escalation_financial():
    engine = EscalationEngine()
    res = engine.evaluate(
        text="I was charged twice on my credit card from ITUNES.COM/BILL! Refund me immediately.",
        predicted_intent=IntentCategory.BILLING_SUBSCRIPTIONS_PURCHASES,
        intent_confidence=0.92,
        top_retrieval_score=0.75
    )
    assert res["decision"] == EscalationDecision.HUMAN_ESCALATION
    assert res["reason_code"] == "FINANCIAL_DISPUTE"


def test_escalation_hardware_repair():
    engine = EscalationEngine()
    res = engine.evaluate(
        text="I dropped my phone and the screen is completely shattered with black bleeding spots.",
        predicted_intent=IntentCategory.HARDWARE_BATTERY_CHARGING,
        intent_confidence=0.90,
        top_retrieval_score=0.8
    )
    assert res["decision"] == EscalationDecision.HUMAN_ESCALATION
    assert res["reason_code"] == "HARDWARE_REPAIR_BOOKING"


def test_autohandle_battery():
    engine = EscalationEngine()
    res = engine.evaluate(
        text="My battery is draining a bit faster than usual after updating yesterday.",
        predicted_intent=IntentCategory.HARDWARE_BATTERY_CHARGING,
        intent_confidence=0.88,
        top_retrieval_score=0.65
    )
    assert res["decision"] == EscalationDecision.AUTO_HANDLE
    assert res["reason_code"] == "NONE"


def test_generator_tweet_length_compliance():
    generator = GroundedReplyGenerator()
    out = generator.generate(
        customer_text="My phone freezes when opening camera",
        intent=IntentCategory.SOFTWARE_OS_CRASH,
        escalation_decision=EscalationDecision.AUTO_HANDLE,
        escalation_reason="NONE",
        retrieved_resolutions=[],
        entities={"devices": ["iphone"], "ios_versions": []}
    )
    assert out["is_within_limit"] is True
    assert out["char_count"] <= 280
    assert len(out["draft_reply"]) > 20


def test_agent_end_to_end(agent):
    out = agent.handle_tweet("My AirPods keep disconnecting from my iPhone 12 during calls.")
    assert "predicted_intent" in out
    assert "escalation_decision" in out
    assert "draft_reply" in out
    assert out["is_tweet_compliant"] is True
    assert len(out["draft_reply"]) <= 280
