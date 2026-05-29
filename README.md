# AutoStream – Social-to-Lead Agentic Workflow
Deployed link : https://autostream-ai-sales-agent-servicehive.streamlit.app/
> **ServiceHive × Inflx** 
> Built with **LangGraph**  | Python 3.9+

---

## 📹 Demo

Watch the agent in action handling a lead capture flow:
[![Watch the demo video](<video src="demo.mp4" controls>)
## Overview

A production-grade Conversational AI Agent that converts social media conversations into qualified business leads for **AutoStream** — a fictional SaaS video editing platform for content creators.

The agent handles the full funnel:

```
User chats → Intent detected → RAG answers questions → High-intent identified → Lead collected → CRM captured
```

---

## Project Structure

```
social-to-lead-agent/
├── main.py                        # CLI entry point & conversation loop
├── requirements.txt
├── .env.example                   # Rename to .env and add your API key
│
├── knowledge_base/
│   └── autostream_kb.json         # Structured KB: pricing, policies, FAQ
│
├── agent/
│   ├── graph.py                   # LangGraph state machine (core logic)
│   ├── intent_detector.py         # Rule-based + LLM intent classification
│   └── rag_pipeline.py            # Knowledge base retrieval (RAG)
│
├── tools/
│   └── lead_capture.py            # mock_lead_capture() tool + in-memory CRM
│
└── tests/
    └── test_agent.py              # 19 unit tests (pytest)
```

---

## How to Run Locally

### 1. Clone the repository

```bash
git clone https://github.com/your-username/social-to-lead-agent.git
cd social-to-lead-agent
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```
### 5. Run the agent

```bash
python main.py
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Example Conversation

```
You: Hi, tell me about your pricing.
Agent: Hi there! AutoStream offers two plans:
       • Basic ($29/mo) – 10 videos/month, 720p
       • Pro ($79/mo) – Unlimited videos, 4K, AI captions, 24/7 support

You: That sounds great for my YouTube channel. I want to try the Pro plan.
Agent: Awesome! I'd love to get you set up. Could you share your full name?

You: Sarah Connor
Agent: Thanks! What's the best email address to reach you at?

You: sarah@example.com
Agent: Almost there! Which creator platform are you mainly on?

You: YouTube
Agent: 🎉 You're all set, Sarah! Check your inbox — our team will be in touch!

✅ Lead captured: Sarah Connor | sarah@example.com | YouTube | LEAD-1001
```

---

## Architecture Explanation (~200 words)

### Why LangGraph?

LangGraph was chosen over AutoGen because this agent has a clearly defined **stateful, multi-step workflow** that maps naturally to a directed graph: classify intent → branch to the right handler → optionally collect lead fields → trigger the tool. LangGraph's `StateGraph` makes these transitions explicit, debuggable, and easy to extend. AutoGen is better suited for multi-agent collaboration, which is overkill here.

### How State is Managed

A typed `AgentState` dictionary is passed through every node of the graph and updated immutably at each step. It stores:
- `messages` — full conversation history (using LangGraph's `add_messages` reducer for automatic append)
- `intent` — current-turn classification result
- `lead_info` — progressively accumulated `{name, email, platform}`
- `awaiting_field` — tracks which piece of info the agent is currently waiting for
- `lead_captured` — boolean flag preventing duplicate tool calls
- `rag_context` — KB chunks retrieved for the current query

This gives the agent **persistent memory across 5–6+ turns** without any external database — state flows through the graph on every invocation.

### RAG Pipeline

The knowledge base (`autostream_kb.json`) is chunked into labelled segments at startup. Retrieval uses keyword scoring against the user's message to pick the top-3 most relevant chunks, which are injected into the system prompt. For production, this would be replaced with a vector store (Chroma/FAISS + embeddings).

### Intent Detection

A two-layer approach: fast regex rules handle ~90% of cases instantly. For ambiguous inputs, a lightweight LLM call classifies the intent using a focused prompt with conversation history context.

---

## WhatsApp Deployment via Webhooks

To deploy this agent on WhatsApp, the recommended approach uses **WhatsApp Business API (Cloud API)** via Meta + a webhook server:

### Architecture

```
WhatsApp User
      │
      ▼
Meta WhatsApp Cloud API
      │  (webhook POST)
      ▼
FastAPI Webhook Server  ←──── ngrok / public HTTPS URL
      │
      ├─ Verify webhook token (GET)
      └─ Handle incoming message (POST)
              │
              ▼
        Agent graph.invoke(state)
              │
              ▼
        WhatsApp Cloud API
        (send reply via REST)
              │
              ▼
        WhatsApp User (receives response)
```

### Implementation Steps

**1. Set up Meta Business & WhatsApp Cloud API**
- Create a Meta Developer App at developers.facebook.com
- Enable WhatsApp product and get a phone number
- Obtain `WHATSAPP_TOKEN` and `PHONE_NUMBER_ID`

**2. Build a FastAPI webhook server**

```python
from fastapi import FastAPI, Request
import httpx, os

app = FastAPI()
agent_sessions = {}  # phone_number → AgentState

@app.get("/webhook")
async def verify(hub_mode, hub_verify_token, hub_challenge):
    if hub_verify_token == os.getenv("VERIFY_TOKEN"):
        return int(hub_challenge)

@app.post("/webhook")
async def receive_message(request: Request):
    body = await request.json()
    # Parse WhatsApp message payload
    message = body["entry"][0]["changes"][0]["value"]["messages"][0]
    phone   = message["from"]
    text    = message["text"]["body"]

    # Retrieve or init session state
    state = agent_sessions.get(phone, default_state())
    state["messages"].append(HumanMessage(content=text))

    # Run agent
    result = graph.invoke(state)
    agent_sessions[phone] = result

    # Extract reply
    reply = [m for m in result["messages"] if isinstance(m, AIMessage)][-1].content

    # Send reply via WhatsApp Cloud API
    await send_whatsapp_message(phone, reply)
    return {"status": "ok"}
```

**3. Send replies**

```python
async def send_whatsapp_message(to: str, text: str):
    url = f"https://graph.facebook.com/v18.0/{PHONE_NUMBER_ID}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to,
        "type": "text",
        "text": {"body": text}
    }
    async with httpx.AsyncClient() as client:
        await client.post(url, json=payload,
                          headers={"Authorization": f"Bearer {WHATSAPP_TOKEN}"})
```

**4. Expose locally with ngrok**

```bash
ngrok http 8000
# Register the HTTPS URL as your webhook in Meta Developer Console
```

**5. Production deployment** — Replace ngrok with a cloud server (AWS/GCP/Railway) and use Redis for session state persistence across restarts.

---

## Evaluation Criteria Mapping

| Criterion | Implementation |
|---|---|
| Agent reasoning & intent detection | Two-layer: regex rules + LLM fallback |
| Correct use of RAG | KB chunked, keyword-scored retrieval, context injected to prompt |
| Clean state management | Typed `AgentState` TypedDict flowing through LangGraph nodes |
| Proper tool calling | `mock_lead_capture` fires only after all 3 fields confirmed |
| Code clarity & structure | Modular packages: `agent/`, `tools/`, `knowledge_base/`, `tests/` |
| Real-world deployability | WhatsApp webhook architecture documented above |

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Agent Framework | LangGraph 0.2+ |
| LLM | GLM 4.5 Flash |
| State Management | LangGraph `StateGraph` + typed `AgentState` |
| RAG | JSON knowledge base + keyword retrieval |
| Testing | pytest (19 tests) |
| Env Management | python-dotenv |

---

## 🎤 Author

**Hari Krishna Kumar -AI,ML,Data Science & Analytics Enthusiast**

---
## 📬 Contact

For collaboration or queries:

* LinkedIn: *[www.linkedin.com/in/hari-668364112]*
* Email: *[harikrishnakumar368@gmail.com]*

---


