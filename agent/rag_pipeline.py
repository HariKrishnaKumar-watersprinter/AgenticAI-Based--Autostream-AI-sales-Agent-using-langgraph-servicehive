"""
agent/rag_pipeline.py
RAG (Retrieval-Augmented Generation) pipeline that loads the AutoStream
knowledge base and retrieves relevant context for a given user query.

Uses simple keyword/semantic matching since we have a small KB.
For production, replace with a vector store (e.g., Chroma, FAISS).
"""

import json
import os
import re
from pathlib import Path
from typing import Optional


KB_PATH = Path(__file__).parent.parent / "knowledge_base" / "autostream_kb.json"


def _load_knowledge_base() -> dict:
    with open(KB_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# This is a type hint indicating that the return value of the function will be a list of dictionaries.

def _flatten_kb_to_chunks(kb: dict) -> list[dict]:
    """
    Converts the structured KB into flat text chunks for retrieval.
    Each chunk has: 'id', 'category', 'content'
    """
    chunks = []

    # Company overview
    company = kb["company"]
    chunks.append({
        "id": "company_overview",
        "category": "company",
        "keywords": ["autostream", "what is", "about", "company", "product"],
        "content": (
            f"{company['name']} – {company['tagline']}.\n"
            f"{company['description']}"
        )
    })

    # Pricing plans
    for plan in kb["pricing"]["plans"]:
        features_text = "\n  - ".join(plan["features"])
        limitations_text = ""
        if plan["limitations"]:
            limitations_text = "\n  Limitations:\n  - " + "\n  - ".join(plan["limitations"])

        chunks.append({
            "id": f"pricing_{plan['name'].lower().replace(' ', '_')}",
            "category": "pricing",
            "keywords": [
                "price", "pricing", "cost", "plan", "how much",
                plan["name"].lower(), "basic", "pro", "monthly",
                "resolution", "video", "caption", "unlimited"
            ],
            "content": (
                f"{plan['name']}: ${plan['price_monthly']}/month\n"
                f"Features:\n  - {features_text}"
                f"{limitations_text}"
            )
        })

    # Policies
    policies = kb["policies"]
    chunks.append({
        "id": "refund_policy",
        "category": "policy",
        "keywords": ["refund", "money back", "cancel", "return"],
        "content": f"Refund Policy: {policies['refund_policy']}"
    })
    chunks.append({
        "id": "support_policy",
        "category": "policy",
        "keywords": ["support", "help", "24/7", "contact", "response time"],
        "content": f"Support Policy: {policies['support_policy']}"
    })
    chunks.append({
        "id": "cancellation_policy",
        "category": "policy",
        "keywords": ["cancel", "subscription", "stop", "end"],
        "content": f"Cancellation Policy: {policies['cancellation_policy']}"
    })

    # FAQs# enumerate() is a built-in Python function that takes an iterable (like a list) and returns an iterator yielding pairs of (index, element). By default, the index starts at 0.
    # \b: Matches a word boundary. This ensures that the match occurs at the beginning or end of a whole word, preventing partial matches inside longer strings (e.g., matching "word" inside "password").
    # \w: Matches any word character. This includes lowercase letters (a-z), uppercase letters (A-Z), digits (0-9), and the underscore (_).
    for i, faq in enumerate(kb["faq"]):
        keywords = re.findall(r'\b\w{4,}\b', faq["question"].lower())
        chunks.append({
            "id": f"faq_{i}",
            "category": "faq",
            "keywords": keywords,
            "content": f"Q: {faq['question']}\nA: {faq['answer']}"
        })

    return chunks


def retrieve_context(query: str, top_k: int = 3) -> str:
    """
    Retrieves the most relevant knowledge base chunks for a given query.

    Args:
        query : User's message / question
        top_k : Number of top chunks to return

    Returns:
        Concatenated context string to inject into the LLM prompt
    """
    kb = _load_knowledge_base()
    chunks = _flatten_kb_to_chunks(kb)
    query_lower = query.lower()
    query_words = set(re.findall(r'\b\w{3,}\b', query_lower))

    scored = []
    for chunk in chunks:
        score = 0
        for kw in chunk["keywords"]:
            if kw in query_lower:
                score += 2  # exact keyword match
        # Also score on word overlap with content
        content_words = set(re.findall(r'\b\w{3,}\b', chunk["content"].lower()))
        overlap = query_words & content_words
        score += len(overlap)

        if score > 0:
            scored.append((score, chunk))

    # Sort by score descending
    scored.sort(key=lambda x: x[0], reverse=True)
    top_chunks = [c for _, c in scored[:top_k]]

    if not top_chunks:
        # Return a general overview if nothing matched
        return chunks[0]["content"]

    return "\n\n---\n\n".join(c["content"] for c in top_chunks)


def get_full_knowledge_base_text() -> str:
    """Returns the complete KB as a formatted string (for system prompt injection)."""
    kb = _load_knowledge_base()
    chunks = _flatten_kb_to_chunks(kb)
    return "\n\n".join(c["content"] for c in chunks)
