"""
Intent Taxonomy and Escalation Schema for @AppleSupport.
Derived from real customer inquiries and historical support resolution patterns.
"""
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any


class IntentCategory(str, Enum):
    HARDWARE_BATTERY_CHARGING = "HARDWARE_BATTERY_CHARGING"
    SOFTWARE_OS_CRASH = "SOFTWARE_OS_CRASH"
    APPLE_ID_ICLOUD_SECURITY = "APPLE_ID_ICLOUD_SECURITY"
    BILLING_SUBSCRIPTIONS_PURCHASES = "BILLING_SUBSCRIPTIONS_PURCHASES"
    CONNECTIVITY_BLUETOOTH_WIFI = "CONNECTIVITY_BLUETOOTH_WIFI"
    GENERAL_PRODUCT_INFO = "GENERAL_PRODUCT_INFO"


class EscalationDecision(str, Enum):
    AUTO_HANDLE = "AUTO_HANDLE"
    HUMAN_ESCALATION = "HUMAN_ESCALATION"


@dataclass(frozen=True)
class IntentDefinition:
    name: IntentCategory
    description: str
    keywords: List[str]
    typical_action: str
    default_can_autohandle: bool


INTENT_REGISTRY: Dict[IntentCategory, IntentDefinition] = {
    IntentCategory.HARDWARE_BATTERY_CHARGING: IntentDefinition(
        name=IntentCategory.HARDWARE_BATTERY_CHARGING,
        description="Physical device issues, battery draining quickly, battery health degradation, charging/cable failures, screen flickering, damaged hardware.",
        keywords=["battery", "drain", "charge", "charging", "cable", "screen", "shattered", "cracked", "overheating", "hot", "hardware", "speaker", "microphone"],
        typical_action="Provide battery health check instructions or diagnostic booking links; escalate to human if physical repair/replacement is needed.",
        default_can_autohandle=True,
    ),
    IntentCategory.SOFTWARE_OS_CRASH: IntentDefinition(
        name=IntentCategory.SOFTWARE_OS_CRASH,
        description="iOS/macOS update glitches, apps freezing/crashing, boot loops, black/white screen of death, storage 'System Data' bugs, performance lag.",
        keywords=["ios", "update", "freeze", "crash", "stuck", "boot loop", "apple logo", "lag", "slow", "storage", "system data", "bug", "app store app crash"],
        typical_action="Provide force restart procedures, update recovery steps (DFU/Recovery mode via Finder/iTunes), or cache clearing guide.",
        default_can_autohandle=True,
    ),
    IntentCategory.APPLE_ID_ICLOUD_SECURITY: IntentDefinition(
        name=IntentCategory.APPLE_ID_ICLOUD_SECURITY,
        description="Apple ID locked or disabled for security reasons, two-factor authentication (2FA) verification issues, password recovery, iCloud storage backup errors, compromised account.",
        keywords=["apple id", "icloud", "locked", "disabled", "password", "2fa", "verification code", "hacked", "security questions", "trusted phone", "activation lock"],
        typical_action="Provide iforgot.apple.com self-service link; escalate immediately if account takeover or activation lock dispute is detected.",
        default_can_autohandle=False,  # Security-sensitive: high human escalation probability
    ),
    IntentCategory.BILLING_SUBSCRIPTIONS_PURCHASES: IntentDefinition(
        name=IntentCategory.BILLING_SUBSCRIPTIONS_PURCHASES,
        description="Unrecognized credit card charges from ITUNES.COM/BILL, subscription renewals, cancellation requests, refund disputes, in-app purchase issues.",
        keywords=["charge", "charged", "bill", "billing", "subscription", "refund", "receipt", "itunes.com", "bank", "money", "unauthorized purchase"],
        typical_action="Direct customer to reportaproblem.apple.com; escalate to human if disputing fraud or requesting manual agent refund intervention.",
        default_can_autohandle=False,  # Financial/dispute: frequently requires human agent verification
    ),
    IntentCategory.CONNECTIVITY_BLUETOOTH_WIFI: IntentDefinition(
        name=IntentCategory.CONNECTIVITY_BLUETOOTH_WIFI,
        description="Wi-Fi disconnecting, Bluetooth pairing failures (AirPods, Apple Watch, CarPlay, keyboards), cellular data 'No Service' or dropped calls.",
        keywords=["wifi", "wi-fi", "bluetooth", "pair", "pairing", "airpods", "apple watch", "carplay", "no service", "cellular", "carrier", "disconnect"],
        typical_action="Provide network reset steps, device unpairing/re-pairing guides, and carrier settings update verification.",
        default_can_autohandle=True,
    ),
    IntentCategory.GENERAL_PRODUCT_INFO: IntentDefinition(
        name=IntentCategory.GENERAL_PRODUCT_INFO,
        description="General questions about product release dates, Apple Store appointments, trade-in values, warranty status (AppleCare), feature availability.",
        keywords=["trade in", "apple store", "appointment", "genius bar", "applecare", "warranty", "when will", "release date", "specs", "compatible"],
        typical_action="Provide Apple Store locator, AppleCare coverage check link (checkcoverage.apple.com), or public spec sheet.",
        default_can_autohandle=True,
    ),
}


ESCALATION_REASONS = {
    "AUTH_REQUIRED": "Customer requires private account verification or authentication credentials.",
    "FINANCIAL_DISPUTE": "Customer is disputing charges, requesting a refund, or reporting unauthorized financial transactions.",
    "SECURITY_LOCKOUT": "Customer's Apple ID is disabled, compromised, or blocked by Activation Lock.",
    "HARDWARE_REPAIR_BOOKING": "Physical device failure requires authorized technician diagnostics or serial/IMEI intake.",
    "HIGH_ANGER_CHURN": "Customer displays severe distress, legal threats, or multiple repeat unresolved contacts.",
    "LOW_CONFIDENCE": "Model confidence or retrieval similarity is below safety threshold; requires human review.",
    "NONE": "Resolved automatically with official documentation or troubleshooting guidance."
}
