"""
agent/intent_detector.py
Rule-based + LLM-assisted intent classification for the Social-to-Lead agent.

Intent categories:
  1. casual_greeting   – Hello, hi, hey, how are you
  2. product_inquiry   – Questions about features, pricing, plans, policies
  3. high_intent_lead  – User signals readiness to sign up / buy
"""

import re
from enum import Enum 


class Intent(str, Enum):
    CASUAL_GREETING = "casual_greeting"
    PRODUCT_INQUIRY = "product_inquiry"
    HIGH_INTENT_LEAD = "high_intent_lead"
    UNKNOWN = "unknown"


# ── Keyword lists ────────────────────────────────────────────────────────────

_GREETING_PATTERNS = [
    r"\b(hi|hey|hello|howdy|greetings|good (morning|afternoon|evening))\b",
    r"^(hi|hey|hello)[\s!?.]*$",
    r"\bhow are you\b",
    r"\bwhat'?s up\b",
]

_PRODUCT_PATTERNS = [
    r"\b(price|pricing|cost|how much|plan|plans|subscription)\b",
    r"\b(feature|features|include|offer|support|resolution|caption)\b",
    r"\b(basic|pro) plan\b",
    r"\brefund\b",
    r"\btrial\b",
    r"\b(cancel|cancellation)\b",
    r"\bteam\b",
    r"\b(youtube|instagram|tiktok|video|edit)\b",
    r"\b(what does|does it|can it|do you|is there)\b",
]

_HIGH_INTENT_PATTERNS = [
    r"\b(sign(?: me)? up|signup|register|get started|start(?: a)? trial|try it|try out)\b",
    r"\b(want to (buy|purchase|subscribe|get|try))\b",
    r"\b(i('?m| am) (interested|ready|in))\b",
    r"\b(go with|choose|pick|select) (the )?(pro|basic) plan\b",
    r"\b(purchase|buy|order|subscribe)\b",
    r"\b(sounds good|looks good|perfect|let'?s do it|let'?s go)\b",
    r"\b(my (youtube|instagram|tiktok|channel|account))\b",
    r"\bhow do i (get|start|sign)\b",
]


def detect_intent_rule_based(message: str) -> Intent:
    """
    Fast rule-based intent detection using regex patterns.
    Used as a first pass before LLM classification.
    """
    text = message.lower().strip()

    # High intent – check first so "I want to sign up, how much?" is captured
    for pattern in _HIGH_INTENT_PATTERNS:
        if re.search(pattern, text):
            return Intent.HIGH_INTENT_LEAD

    # Product inquiry
    for pattern in _PRODUCT_PATTERNS:
        if re.search(pattern, text):
            return Intent.PRODUCT_INQUIRY

    # Greeting
    for pattern in _GREETING_PATTERNS:
        if re.search(pattern, text):
            return Intent.CASUAL_GREETING

    return Intent.UNKNOWN


def build_intent_classification_prompt(message: str, history_summary: str = "") -> str:
    """
    Builds a prompt for the LLM to classify intent when rule-based is UNKNOWN.
    """
    return f"""You are an intent classifier for AutoStream, a SaaS video editing tool.

Classify the user message into exactly one of these intents:
1. casual_greeting     – general chit-chat, hello, how are you
2. product_inquiry     – asking about features, pricing, plans, policies, or how the product works
3. high_intent_lead    – user shows clear readiness to sign up, buy, or is asking HOW to get started

{f'Recent conversation context: {history_summary}' if history_summary else ''}

User message: "{message}"

Respond with ONLY the intent label (casual_greeting / product_inquiry / high_intent_lead).
"""


def map_llm_response_to_intent(llm_response: str) -> Intent:
    """Maps raw LLM text back to an Intent enum value."""
    text = llm_response.strip().lower()
    if "high_intent" in text or "high intent" in text:
        return Intent.HIGH_INTENT_LEAD
    if "product" in text or "inquiry" in text:
        return Intent.PRODUCT_INQUIRY
    if "greeting" in text:
        return Intent.CASUAL_GREETING
    return Intent.UNKNOWN
