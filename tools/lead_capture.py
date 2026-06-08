"""
tools/lead_capture.py
Lead capture tool that saves qualified leads to a persistent cloud JSON bin.
"""
import os
import json
import datetime
import requests
from typing import Optional
import streamlit as st

# JSONBin.io credentials (Store these in Streamlit Secrets or Environment Variables)
API_KEY = st.secrets["API_KEY"]
BIN_ID = '6a2697feda38895dfe99062e'
HEADERS = {
    "X-Master-Key": API_KEY,
    "Content-Type": "application/json"
}
BASE_URL = "https://api.jsonbin.io/v3"

def _ensure_bin_exists() -> str:
    """Creates a JSON bin if it doesn't exist and returns the BIN_ID."""
    global BIN_ID
    
    # If BIN_ID already exists, just return it
    if BIN_ID:
        return BIN_ID
        
    if not API_KEY:
        raise ValueError("JSONBIN_API_KEY is missing from environment variables.")

    print("Bin not found. Creating a new JSONBin automatically...")
    create_url = f"{BASE_URL}/b"
    payload = {"leads": [], "created_at": datetime.datetime.now().isoformat()}
    
    try:
        response = requests.post(create_url, json=payload, headers=HEADERS)
        response.raise_for_status()
        new_bin_id = response.json()["metadata"]["id"]
        
        # Save the new BIN_ID to the global variable for the current session
        BIN_ID = new_bin_id
        
        # Persist the BIN_ID to the .env file so it survives restarts locally
        #env_path = Path(__file__).parent.parent / ".env"
        ##with open(env_path, "a", encoding="utf-8") as f:
           #f.write(f"\nJSONBIN_BIN_ID={new_bin_id}")
            
        print(f"Successfully created new bin with ID: {new_bin_id}")
        return new_bin_id
        
    except requests.exceptions.RequestException as e:
        print(f"Error creating JSONBin: {e}")
        raise

def _load_leads_from_file() -> list[dict]:
    """Loads leads from the cloud JSON bin."""
    current_bin_id = _ensure_bin_exists()
    url = f"{BASE_URL}/b/{current_bin_id}"
    
    try:
        response = requests.get(url, headers=HEADERS)
        response.raise_for_status()
        data = response.json()
        # The record contains the JSON payload we stored
        return data.get("record", {}).get("leads", [])
    except Exception as e:
        print(f"Error loading leads from JSONBin: {e}")
        return []

def _save_leads_to_bin(leads: list[dict]):
    """Saves leads to the cloud JSON bin."""
    current_bin_id = _ensure_bin_exists()
    url = f"{BASE_URL}/b/{current_bin_id}"
    
    # We wrap the leads list in an object so we can add metadata later if needed
    payload = {"leads": leads}
    
    try:
        response = requests.put(url, json=payload, headers=HEADERS)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"Error saving to JSONBin: {e}")

def mock_lead_capture(name: str, email: str, platform: str, plan: str) -> dict:
    """
    Captures a qualified lead into the persistent cloud JSON CRM.

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
    _save_leads_to_bin(all_leads)

    return {
        "status": "success",
        "lead_id": lead["lead_id"],
        "message": f"Lead for {name} captured successfully!",
        "data": lead
    }

def get_all_leads() -> list[dict]:
    """Returns all captured leads."""
    return _load_leads_from_file()
