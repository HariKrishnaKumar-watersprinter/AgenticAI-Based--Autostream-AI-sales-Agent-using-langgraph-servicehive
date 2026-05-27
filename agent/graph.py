"""
agent/graph.py
LangGraph-based agentic workflow for the Social-to-Lead agent.

State machine:
  ┌──────────┐
  │  START   │
  └────┬─────┘
       │
  ┌────▼──────────┐
  │  classify      │  ← detect intent from user message
  └────┬──────────┘
       │
  ┌────▼──────────┐
  │  route_intent  │  ← branch on intent 
  └────┬──────────┘
       │
  ┌────▼────────────────────────────────────────┐
  │  greet │ answer_product │ qualify_lead       │
  └────┬───┴────────────────┴────────────────────┘
       │
  ┌────▼──────────┐
  │ capture_lead   │  ← only when all 4 fields collected
  └────┬──────────┘  ← note: now requires 4 fields (name, email, platform, plan)
       │
     END
"""

from __future__ import annotations

import os
from typing import Annotated, TypedDict, Optional, List
from enum import Enum
from langchain_community.chat_models import ChatZhipuAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
#from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.intent_detector import (
    Intent,
    detect_intent_rule_based,
    build_intent_classification_prompt,
    map_llm_response_to_intent,
)
from agent.rag_pipeline import retrieve_context
from tools.lead_capture import mock_lead_capture


# ── State Schema ─────────────────────────────────────────────────────────────

class LeadInfo(TypedDict, total=False):
    name: Optional[str]
    email: Optional[str]
    platform: Optional[str]
    plan: Optional[str]


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]   # full conversation history
    intent: str                                # current turn intent
    lead_info: LeadInfo                        # progressively collected
    lead_captured: bool                        # flag – tool called?
    awaiting_field: Optional[str]              # which field we're asking for next
    rag_context: str                           # retrieved KB context


# ── LLM Setup ────────────────────────────────────────────────────────────────

def _get_llm():
    api_key2='fe530a38259c4de2bc506cf863512984.Mtfh8Sb2nsy97Sgb'
    #api_key = os.getenv("ANTHROPIC_API_KEY")
    #if not api_key:
        #raise EnvironmentError(
           # "ANTHROPIC_API_KEY is not set. Please add it to your .env file.")
    return ChatZhipuAI(api_key=api_key2, model='GLM-4.5-Flash')


# ── System Prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert sales assistant for AutoStream — an AI-powered video editing SaaS for content creators.

Your goals:
1. Welcome users warmly and professionally
2. Answer product/pricing questions ONLY using the provided knowledge base context
3. Identify when users are ready to sign up (high intent)
4. Collect name, email, creator platform, and subscription plan ONE AT A TIME when they show high intent
5. Never ask for info you already have

Tone: Friendly, concise, helpful. Never be pushy or repetitive.
NEVER make up pricing or features. Strictly follow the Knowledge Base.
AVAILABLE PLANS:
- Basic: $29/mo
- Pro: $79/mo
Note: There is NO Enterprise plan. If asked, inform the user only Basic and Pro are available.
"""


# ── Node Functions ────────────────────────────────────────────────────────────

def classify_node(state: AgentState) -> AgentState:
    """Classifies the latest user message intent."""
    last_msg = state["messages"][-1]
    user_text = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # Rule-based first (fast)
    intent = detect_intent_rule_based(user_text)

    # Fallback to LLM classification if unknown
    if intent == Intent.UNKNOWN:
        llm = _get_llm()
        history_summary = _summarise_history(state["messages"][:-1])
        prompt = build_intent_classification_prompt(user_text, history_summary)
        response = llm.invoke([HumanMessage(content=prompt)])
        intent = map_llm_response_to_intent(response.content)

    # Retrieve relevant RAG context
    rag_ctx = retrieve_context(user_text)

    # Update lead info from current message before routing to ensure state is persistent
    lead_info = dict(state.get("lead_info", {}))
    current_awaiting = state.get("awaiting_field")

    # Determine if we should attempt greedy field extraction based on awaiting_field.
    # We suppress it if the *newly detected intent* is a digression (product inquiry or greeting),
    # but not if the intent is HIGH_INTENT_LEAD, as the user might be directly answering a field request.
    awaiting = current_awaiting
    if current_awaiting and intent in (Intent.PRODUCT_INQUIRY, Intent.CASUAL_GREETING):
        awaiting = None # Suppress greedy extraction for digressions

    updated_lead_info = _extract_lead_fields(user_text, lead_info, awaiting)

    return {
        **state,
        "intent": intent.value,
        "rag_context": rag_ctx,
        "lead_info": updated_lead_info,
    }


def greet_node(state: AgentState) -> AgentState:
    """Handles casual greetings."""
    llm = _get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state["messages"],
    ]
    response = llm.invoke(messages)
    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


def answer_product_node(state: AgentState) -> AgentState:
    """Answers product/pricing questions using RAG context."""
    llm = _get_llm()
    rag_context = state.get("rag_context", "")

    system = (
        SYSTEM_PROMPT
        + f"\n\n📚 Knowledge Base Context (use ONLY this to answer):\n{rag_context}"
    )

    messages = [SystemMessage(content=system), *state["messages"]]
    response = llm.invoke(messages)

    return {
        **state,
        "messages": state["messages"] + [AIMessage(content=response.content)],
    }


def qualify_lead_node(state: AgentState) -> AgentState:
    """
    Progressively collects lead info (name → email → platform → plan).
    Uses conversation context to skip fields the user has already provided.
    """
    # lead_info is already updated in classify_node at the start of the turn
    lead_info = state.get("lead_info", {})
    messages = state["messages"]

    # Determine next missing field
    next_field = _next_missing_field(lead_info)

    if next_field is None:
        # All info collected – delegate to capture node
        return {
            **state,
            "lead_info": lead_info,
            "awaiting_field": None,
        }

    # Ask for the next missing field
    field_questions = {
        "name": "Great! I'd love to get you set up. Could you share your **full name**?",
        "email": "Thanks! What's the best **email address** to reach you at?",
        "platform": "Almost there! Which creator platform are you mainly on? (e.g., YouTube, Instagram, TikTok)",
        "plan": "Excellent! Which plan would you like to subscribe to? We offer a **Basic** plan ($29/mo) and a **Pro** plan ($79/mo).",
    }

    # First-time high-intent message — acknowledge it before asking
    llm = _get_llm()
    if not lead_info and next_field == "name":
        system = (
            SYSTEM_PROMPT
            + "\n\nThe user has shown high intent to sign up. "
            "Acknowledge their interest enthusiastically in 1 sentence, "
            f"then ask: '{field_questions['name']}'"
        )
        response = llm.invoke([SystemMessage(content=system), *messages])
        reply = response.content
    else:
        reply = field_questions.get(next_field, "Could you provide more details?")

    return {
        **state,
        "lead_info": lead_info,
        "awaiting_field": next_field,
        "messages": messages + [AIMessage(content=reply)],
    }


def capture_lead_node(state: AgentState) -> AgentState:
    """Calls mock_lead_capture once all four fields are confirmed."""
    lead_info = state.get("lead_info", {})
    name = lead_info.get("name", "")
    email = lead_info.get("email", "")
    platform = lead_info.get("platform", "")
    plan = lead_info.get("plan", "")

    result = mock_lead_capture(name, email, platform, plan)

    if result["status"] == "success":
        reply = (
            f"🎉 You're all set, **{name}**! "
            f"I've registered your interest in AutoStream. "
            f"Check your inbox at **{email}** — our team will be in touch shortly to get you started on the {plan} plan. "
            f"Welcome aboard! 🚀"
        )
    else:
        reply = (
            f"Hmm, something went wrong while saving your details: {result['message']}. "
            "Could you double-check and try again?"
        )

    return {
        **state,
        "lead_captured": result["status"] == "success",
        "messages": state["messages"] + [AIMessage(content=reply)],
    }


# ── Routing Logic ─────────────────────────────────────────────────────────────

def route_after_classify(state: AgentState) -> str:
    """Decides which node to go to after classification."""
    intent = state.get("intent", Intent.UNKNOWN.value)
    lead_captured = state.get("lead_captured", False)
    awaiting = state.get("awaiting_field")
    lead_info = state.get("lead_info", {})

    # Priority 1: If all required fields are present but the lead hasn't been saved yet,
    # we MUST route to capture_lead. This ensures the mock_lead_capture tool is called
    # and prevents the LLM from "faking" a confirmation in other nodes without saving.
    if not lead_captured and _next_missing_field(lead_info) is None:
        return "capture_lead"

    # 1. Prioritize answering questions, greetings, or repeated intent even after capture.
    if intent == Intent.CASUAL_GREETING.value:
        return "greet"
    # Route inquiries and repeated sign-up attempts (high intent) after capture to the product node.
    if intent == Intent.PRODUCT_INQUIRY.value or (intent == Intent.HIGH_INTENT_LEAD.value and lead_captured):
        return "answer_product"

    # 2. If the lead is already captured, we don't need to re-enter the qualification flow.
    if lead_captured:
        return END

    # 3. If we're mid-collection or the user just started a high-intent flow, qualify the lead.
    if awaiting or (intent == Intent.HIGH_INTENT_LEAD.value):
        return "qualify_lead"

    # Fallback – treat unknown as product inquiry
    return "answer_product"


# ── Helper Utilities ──────────────────────────────────────────────────────────

def _summarise_history(messages: list) -> str:
    """Creates a short text summary of recent messages."""
    parts = []
    for m in messages[-6:]:
        role = "User" if isinstance(m, HumanMessage) else "Agent"
        content = m.content if hasattr(m, "content") else str(m)
        parts.append(f"{role}: {content[:120]}")
    return " | ".join(parts)


def _extract_lead_fields(text: str, existing: dict, awaiting_field: Optional[str]) -> dict:
    """
    Attempts to extract name/email/platform from raw user text.
    Only fills the field currently being awaited.
    """
    import re

    lead = dict(existing)

    # Email – always extract if present
    email_match = re.search(r'[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}', text)
    if email_match and not lead.get("email"):
        lead["email"] = email_match.group()

    # Plan – extract if awaiting the plan, if the user uses a choice verb, or provides a direct plan name.
    # This prevents general questions about a plan from being saved as the user's selection,
    # while still allowing direct responses like "Basic" or "Pro: $79/mo".
    is_choosing = any(v in text.lower() for v in ["want", "choose", "select", "sign up", "go with", "pick", "take", "subscribe"])
    text_clean = text.lower().strip()
    is_direct = text_clean in ["basic", "pro", "basic plan", "pro plan"] or \
                any(text_clean.startswith(p) for p in ["basic:", "pro:", "basic -", "pro -"])

    if not lead.get("plan") and (awaiting_field == "plan" or is_choosing or is_direct):
        if "pro" in text.lower(): lead["plan"] = "Pro"
        elif "basic" in text.lower(): lead["plan"] = "Basic"

    # Platform keywords
    platforms = ["youtube", "instagram", "tiktok", "twitter", "linkedin", "facebook", "twitch"]
    if not lead.get("platform"):
        for p in platforms:
            if p in text.lower():
                lead["platform"] = p.capitalize()
                break

    # Name – greedy capture if we're waiting for it and message doesn't look like a question
    if awaiting_field == "name" and not lead.get("name"):
        is_question = "?" in text or text.lower().startswith(("what", "how", "is", "do", "can", "tell"))
        # Prevent capturing intent-signaling phrases (like "I want to sign up") as the user's name
        intent_indicators = ["sign up", "sign me up", "get started", "want to", "register", "interested"]
        is_intent = any(ind in text.lower() for ind in intent_indicators)

        if not is_question and not is_intent:
            cleaned = re.sub(r'[^a-zA-Z\s]', '', text).strip() # Remove non-alphabetic characters
            # Allow names from 1 to 6 words, and longer than 1 character
            if 1 <= len(cleaned.split()) <= 6 and len(cleaned) > 1:
                lead["name"] = cleaned.title()

    # Platform – greedy capture if we're waiting for it and none matched above
    if awaiting_field == "platform" and not lead.get("platform") and "?" not in text:
        cleaned = re.sub(r'[^a-zA-Z\s]', '', text).strip()
        # Platforms are usually 1-3 words (e.g., "YouTube", "My Twitch Channel")
        if cleaned and len(cleaned.split()) <= 3:
            lead["platform"] = cleaned.title()

    return lead


def _next_missing_field(lead_info: dict) -> Optional[str]:
    """Returns the first missing required field, in order."""
    for field in ("name", "email", "platform", "plan"):
        if not lead_info.get(field):
            return field
    return None


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Assembles and compiles the LangGraph state machine."""

    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("classify", classify_node)
    graph.add_node("greet", greet_node)
    graph.add_node("answer_product", answer_product_node)
    graph.add_node("qualify_lead", qualify_lead_node)
    graph.add_node("capture_lead", capture_lead_node)

    # Entry point
    graph.set_entry_point("classify")

    # Edges from classify → conditional routing
    graph.add_conditional_edges(
        "classify",
        route_after_classify,
        {
            "greet": "greet",
            "answer_product": "answer_product",
            "qualify_lead": "qualify_lead",
            "capture_lead": "capture_lead",
            END: END,
        },
    )

    # All response nodes loop back to END (next turn starts fresh classify)
    for node in ("greet", "answer_product", "qualify_lead"):
        graph.add_edge(node, END)

    graph.add_edge("capture_lead", END)

    return graph.compile()
