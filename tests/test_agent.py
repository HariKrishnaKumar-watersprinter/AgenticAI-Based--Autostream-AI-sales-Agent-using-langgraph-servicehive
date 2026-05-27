"""
tests/test_agent.py
Unit tests for intent detection, RAG pipeline, and lead capture tool.
Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from agent.intent_detector import detect_intent_rule_based, Intent
from agent.rag_pipeline import retrieve_context, get_full_knowledge_base_text
from tools.lead_capture import mock_lead_capture, get_all_leads


# ── Intent Detection Tests ────────────────────────────────────────────────────

class TestIntentDetection:

    def test_greeting_hi(self):
        assert detect_intent_rule_based("Hi there!") == Intent.CASUAL_GREETING

    def test_greeting_hello(self):
        assert detect_intent_rule_based("Hello, how are you?") == Intent.CASUAL_GREETING

    def test_product_pricing(self):
        assert detect_intent_rule_based("What are your pricing plans?") == Intent.PRODUCT_INQUIRY

    def test_product_features(self):
        assert detect_intent_rule_based("Does the Pro plan include AI captions?") == Intent.PRODUCT_INQUIRY

    def test_product_refund(self):
        assert detect_intent_rule_based("What is your refund policy?") == Intent.PRODUCT_INQUIRY

    def test_high_intent_signup(self):
        assert detect_intent_rule_based("I want to sign up for the Pro plan") == Intent.HIGH_INTENT_LEAD

    def test_high_intent_youtube(self):
        intent = detect_intent_rule_based("That sounds great for my YouTube channel, let's go!")
        assert intent == Intent.HIGH_INTENT_LEAD

    def test_high_intent_try(self):
        assert detect_intent_rule_based("I'd like to try it out") == Intent.HIGH_INTENT_LEAD

    def test_high_intent_purchase(self):
        assert detect_intent_rule_based("I want to purchase the Basic plan") == Intent.HIGH_INTENT_LEAD


# ── RAG Pipeline Tests ────────────────────────────────────────────────────────

class TestRAGPipeline:

    def test_retrieve_pricing_context(self):
        ctx = retrieve_context("How much does the Pro plan cost?")
        assert "79" in ctx or "Pro" in ctx

    def test_retrieve_basic_plan(self):
        ctx = retrieve_context("Tell me about the basic plan")
        assert "Basic" in ctx or "29" in ctx

    def test_retrieve_refund_policy(self):
        ctx = retrieve_context("What is the refund policy?")
        assert "7 days" in ctx or "refund" in ctx.lower()

    def test_retrieve_support(self):
        ctx = retrieve_context("Is 24/7 support available?")
        assert "24/7" in ctx or "Pro" in ctx

    def test_full_kb_not_empty(self):
        kb_text = get_full_knowledge_base_text()
        assert len(kb_text) > 100
        assert "AutoStream" in kb_text


# ── Lead Capture Tool Tests ───────────────────────────────────────────────────

class TestLeadCapture:

    def test_successful_capture(self):
        result = mock_lead_capture("Alice Johnson", "alice@example.com", "YouTube", "Pro")
        assert result["status"] == "success"
        assert "lead_id" in result
        assert result["data"]["name"] == "Alice Johnson"

    def test_invalid_email(self):
        result = mock_lead_capture("Bob Smith", "not-an-email", "Instagram", "Basic")
        assert result["status"] == "error"

    def test_missing_field(self):
        result = mock_lead_capture("", "test@test.com", "TikTok", "Pro")
        assert result["status"] == "error"

    def test_lead_stored(self):
        initial_count = len(get_all_leads())
        mock_lead_capture("Carol White", "carol@test.com", "TikTok", "Basic")
        assert len(get_all_leads()) == initial_count + 1

    def test_lead_id_format(self):
        result = mock_lead_capture("Dan Brown", "dan@test.com", "Twitch", "Pro")
        assert result["lead_id"].startswith("LEAD-")
