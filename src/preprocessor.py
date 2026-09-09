"""
Text preprocessing and entity recognition for customer tweets.
"""
import re
from typing import Dict, Any, List


DEVICE_PATTERNS = {
    "iphone": r"\b(iphone\s*(x|xr|xs|1[0-6]|se|[6-8])?(\s*plus|\s*pro|\s*max)?)\b",
    "ipad": r"\b(ipad\s*(pro|air|mini)?)\b",
    "mac": r"\b(macbook\s*(pro|air)?|imac|mac\s*mini|mac\s*pro)\b",
    "watch": r"\b(apple\s*watch(\s*series\s*[1-9])?)\b",
    "airpods": r"\b(airpods?(\s*pro|\s*max)?)\b",
}

IOS_PATTERN = r"\b(ios\s*1[1-8](\.[0-9]+)*)\b"
ERROR_CODE_PATTERN = r"\b(error\s*(code)?\s*[-0-9a-fA-F]+)\b"


class TweetPreprocessor:
    def __init__(self):
        pass

    def clean_text(self, text: str) -> str:
        """Removes extraneous characters, standardizes handles, and normalizes spacing."""
        if not isinstance(text, str):
            return ""
        # Standardize customer mentions
        text = re.sub(r"@\d+", "@customer", text)
        # Standardize AppleSupport mentions
        text = re.sub(r"@AppleSupport", "@AppleSupport", text, flags=re.IGNORECASE)
        # Collapse multiple spaces and strip
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """Extracts key diagnostic entities from customer tweet."""
        text_lower = text.lower()
        devices = []
        for dev_name, pattern in DEVICE_PATTERNS.items():
            if re.search(pattern, text_lower):
                devices.append(dev_name)

        ios_matches = re.findall(IOS_PATTERN, text_lower)
        ios_versions = [m[0] for m in ios_matches] if ios_matches else []

        error_matches = re.findall(ERROR_CODE_PATTERN, text_lower)
        error_codes = [m[0] for m in error_matches] if error_matches else []

        has_link = bool(re.search(r"https?://\S+", text))

        return {
            "devices": devices,
            "ios_versions": ios_versions,
            "error_codes": error_codes,
            "has_link": has_link,
            "char_count": len(text),
            "word_count": len(text.split()),
        }

    def process(self, text: str) -> Dict[str, Any]:
        cleaned = self.clean_text(text)
        entities = self.extract_entities(cleaned)
        return {
            "cleaned_text": cleaned,
            "entities": entities
        }
