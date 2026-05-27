"""
tools/lead_capture.py
Mock lead capture tool that simulates saving a qualified lead to a CRM.
"""

import os
import json
import yaml
import datetime
from pathlib import Path
from typing import Optional

# Path to persistent storage in the project root
LEADS_FILE = Path(__file__).parent.parent / "data" / "leads.json"

def _load_leads_from_file() -> list[dict]:
    """Loads leads from the JSON file."""
    if not LEADS_FILE.exists():
        return []
    try:
        with open(LEADS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, list) else []
    except Exception:
        return []

def _save_leads_to_file(leads: list[dict]):
    """Saves leads to the json file."""
    # Ensure data directory exists
    LEADS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LEADS_FILE, "w", encoding="utf-8") as f:
        json.dump(leads, f, indent=4, ensure_ascii=False)

# In-memory store simulating a CRM database
_captured_leads: list[dict] = []


def mock_lead_capture(name: str, email: str, platform: str, plan: str) -> dict:
    """
    Simulates capturing a qualified lead into the CRM.

    Args:
        name     : Full name of the lead
        email    : Email address of the lead
        platform : Social/creator platform (YouTube, Instagram, TikTok, etc.)
        plan     : Subscription plan (Basic, Pro)

    Returns:
        dict with status and lead_id
    """
    # Basic validation
    if not name or not email or not platform or not plan:
        return {
            "status": "error",
            "message": "All fields (name, email, platform, plan) are required."
        }

    if "@" not in email or "." not in email.split("@")[-1]:
        return {
            "status": "error",
            "message": f"Invalid email address: {email}"
        }

    # Load existing leads to generate the next sequential ID
    all_leads = _load_leads_from_file()

    lead = {
        "lead_id": f"LEAD-{len(all_leads) + 1001}",
        "name": name,
        "email": email,
        "platform": platform,
        "plan": plan,
        "captured_at": datetime.datetime.now().isoformat(),
        "source": "social_chat_agent",
        "status": "new"
    }

    all_leads.append(lead)
    _save_leads_to_file(all_leads)

    # Console output as required by the spec
    print(f"\n{'='*55}")
    print(f"  ✅  Lead captured successfully!")
    print(f"  Name     : {name}")
    print(f"  Email    : {email}")
    print(f"  Platform : {platform}")
    print(f"  Plan     : {plan}")
    print(f"  Lead ID  : {lead['lead_id']}")
    print(f"{'='*55}\n")

    return {
        "status": "success",
        "lead_id": lead["lead_id"],
        "message": f"Lead for {name} captured successfully!",
        "data": lead
    }


def get_all_leads() -> list[dict]:
    """Returns all captured leads (for testing/debugging)."""
    return _load_leads_from_file()
