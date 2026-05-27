"""
streamlit_app.py
Streamlit UI for the AutoStream Social-to-Lead Agentic Workflow.

Run:
    streamlit run streamlit_app.py
"""

import os
import sys
import time
import json
import datetime
import re

import streamlit as st
from dotenv import load_dotenv
from tools.lead_capture import _load_leads_from_file

load_dotenv()

# ── Page config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="AutoStream · AI Sales Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)
@st.cache_data
# ── CSS: Dark editorial theme ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Root variables ── */
:root {
    --bg:        #0a0a0f;
    --surface:   #111118;
    --surface2:  #18181f;
    --border:    #2a2a35;
    --accent:    #7c6cfc;
    --accent2:   #fc6c8f;
    --green:     #3dd68c;
    --amber:     #f7b731;
    --text:      #e8e8f0;
    --muted:     #6b6b80;
    --radius:    14px;
}

/* ── Global reset ── */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border) !important;
}

/* ── Streamlit chrome is now fully visible ── */
/* Removed: #MainMenu, footer, header { visibility: hidden; } */
/* Removed: [data-testid="stToolbar"] { display: none; } */
/* ── Top wordmark bar ── */
.wordmark-bar {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 0 28px 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 28px;
}
.wordmark-icon {
    width: 40px; height: 40px;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    border-radius: 10px;
    display: flex; align-items: center; justify-content: center;
    font-size: 20px;
    flex-shrink: 0;
}
.wordmark-text {
    font-family: 'Syne', sans-serif;
    font-size: 20px;
    font-weight: 800;
    letter-spacing: -0.5px;
    color: var(--text);
}
.wordmark-badge {
    font-size: 10px;
    font-weight: 500;
    color: var(--accent);
    background: rgba(124,108,252,0.12);
    border: 1px solid rgba(124,108,252,0.3);
    border-radius: 20px;
    padding: 2px 8px;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Chat container ── */
.chat-wrapper {
    max-width: 760px;
    margin: 0 auto;
    padding: 16px 0;
}

/* ── Message bubbles ── */
.msg-row {
    display: flex;
    gap: 12px;
    margin-bottom: 18px;
    animation: fadeUp 0.3s ease;
}
.msg-row.user { flex-direction: row-reverse; }

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

.avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 16px;
    flex-shrink: 0;
}
.avatar.agent { background: linear-gradient(135deg, var(--accent), var(--accent2)); }
.avatar.user  { background: var(--surface2); border: 1px solid var(--border); }

.bubble {
    max-width: 72%;
    padding: 13px 18px;
    border-radius: var(--radius);
    font-size: 14.5px;
    line-height: 1.6;
    position: relative;
}
.bubble.agent {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-top-left-radius: 4px;
    color: var(--text);
}
.bubble.user {
    background: linear-gradient(135deg, #6a11cb 0%, #2575fc 100%); /* Gradient background */
            color: white;                                  /* White text */
            padding: 12px 18px;                            /* Inner spacing */
            border-radius: 18px 18px 4px 18px;            /* Rounded corners with a tail effect on bottom-right */
            max-width: 100%;                                /* Prevents bubble from taking full width */
            align-self: flex-end;                          /* Pushes the bubble to the right */
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);    /* Subtle shadow for depth */
            font-size: 15px;
            line-height: 1.4;
            word-wrap: break-word;
            position: relative;
            margin-bottom: 10px;
            animation: popIn 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275); /* Smooth pop-in animation */
}

.bubble-meta {
    font-size: 11px;
    
    color: var(--muted);
    margin-top: 5px;
    text-align: right;
}
.msg-row.user .bubble-meta { text-align: left; }

/* intent pill inside agent bubble */
.intent-pill {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    font-size: 10.5px;
    font-weight: 600;
    letter-spacing: 0.4px;
    text-transform: uppercase;
    border-radius: 20px;
    padding: 2px 9px;
    margin-bottom: 8px;
}
.pill-greeting  { background: rgba(61,214,140,0.12); color: var(--green); border: 1px solid rgba(61,214,140,0.3); }
.pill-inquiry   { background: rgba(124,108,252,0.12); color: var(--accent); border: 1px solid rgba(124,108,252,0.3); }
.pill-highintent{ background: rgba(247,183,49,0.12);  color: var(--amber); border: 1px solid rgba(247,183,49,0.3); }
.pill-captured  { background: rgba(61,214,140,0.12);  color: var(--green); border: 1px solid rgba(61,214,140,0.3); }

/* ── Input row ── */
.stTextInput input {
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14.5px !important;
    padding: 12px 18px !important;
    caret-color: var(--accent) !important;
}
.stTextInput input:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(124,108,252,0.15) !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), #5d4fe0) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    font-size: 13.5px !important;
    padding: 10px 22px !important;
    transition: opacity .2s, transform .15s !important;
}
.stButton > button:hover { opacity: 0.88 !important; transform: translateY(-1px) !important; }

/* ── Compact Action Buttons (Edit/Delete/Retry) ── */
.stButton > button[aria-label="✏️"], 
.stButton > button[aria-label="🗑️"] {
    padding: 0px !important;
    width: 32px !important;
    height: 32px !important;
    min-width: 32px !important;
    min-height: 32px !important;
    font-size: 14px !important;
    background: var(--surface2) !important;
    border: 1px solid var(--border) !important;
}
.stButton > button[aria-label="🔄 Retry"] {
    padding: 4px 12px !important;
    font-size: 12px !important;
    width: auto !important;
}

/* ── Sidebar sections ── */
.sidebar-section {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px;
    margin-bottom: 16px;
}
.sidebar-label {
    font-family: 'Syne', sans-serif;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 12px;
}
.lead-card {
    background: var(--bg);
    border: 1px solid var(--border);
    border-left: 3px solid var(--green);
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 10px;
    font-size: 13px;
}
.lead-card .lead-name { font-weight: 600; color: var(--text); margin-bottom: 3px; }
.lead-card .lead-detail { color: var(--muted); font-size: 12px; }
.lead-id { font-size: 10px; color: var(--accent); font-weight: 600; margin-top: 4px; }

/* stat boxes */
.stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.stat-box {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px;
    text-align: center;
}
.stat-val { font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 800; color: var(--text); }
.stat-lbl { font-size: 11px; color: var(--muted); margin-top: 2px; }

/* progress bar for lead collection */
.progress-track {
    background: var(--border);
    border-radius: 99px;
    height: 4px;
    margin-top: 8px;
    overflow: hidden;
}
.progress-fill {
    height: 100%;
    border-radius: 99px;
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    transition: width 0.4s ease;
}

/* typing indicator */
.typing-dot {
    display: inline-block;
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--accent);
    margin: 0 2px;
    animation: blink 1.2s infinite;
}
.typing-dot:nth-child(2) { animation-delay: 0.2s; }
.typing-dot:nth-child(3) { animation-delay: 0.4s; }
@keyframes blink {
    0%, 80%, 100% { opacity: 0.2; transform: scale(0.9); }
    40%           { opacity: 1;   transform: scale(1.1); }
}

/* KB explorer */
.kb-chip {
    display: inline-block;
    background: rgba(124,108,252,0.1);
    border: 1px solid rgba(124,108,252,0.25);
    color: var(--accent);
    border-radius: 20px;
    padding: 3px 10px;
    font-size: 12px;
    margin: 3px 3px 3px 0;
}

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 9px; }

/* hide default stMarkdown padding */
.element-container { margin-bottom: 0 !important; }
</style>
""", unsafe_allow_html=True)

# ── Session State Init ────────────────────────────────────────────────────────
def _init_state():
    defaults = {
        "messages":       [],   # [{role, content, intent, ts}]
        "agent_state":    None, # LangGraph AgentState
        "leads":          _load_leads_from_file(),   # captured leads list from YAML storage
        "total_turns":    0,
        "api_key_ok":     False,
        "graph":          None,
        "awaiting_agent_response": False, # New flag to manage two-phase rendering
        "last_user_message_content": None, # New to store user input for agent processing
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()

# ── Helpers ───────────────────────────────────────────────────────────────────
INTENT_META = {
    "casual_greeting":  ("💬", "Greeting",     "pill-greeting"),
    "product_inquiry":  ("🔍", "Product Q&A",  "pill-inquiry"),
    "high_intent_lead": ("🔥", "High Intent",  "pill-highintent"),
    "captured":         ("✅", "Lead Captured","pill-captured"),
    "unknown":          ("❓", "Processing",   "pill-inquiry"),
    "error":            ("⚠️", "Timeout/Error","pill-highintent"),
}

def _pill(intent: str) -> str:
    icon, label, css = INTENT_META.get(intent, INTENT_META["unknown"])
    return f'<span class="intent-pill {css}">{icon} {label}</span>'

def _ts() -> str:
    return datetime.datetime.now().strftime("%H:%M")

def _lead_fields_collected(state) -> int:
    if state is None:
        return 0
    li = state.get("lead_info", {})
    return sum(1 for f in ("name","email","platform","plan") if li.get(f))

def _init_graph():
    """Load LangGraph graph (cached in session)."""
    if st.session_state["graph"] is None:
        # Use current file directory to find the 'agent' module
        project_root = os.path.dirname(os.path.abspath(__file__))
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
            
        from agent.graph import build_graph, AgentState
        st.session_state["graph"] = build_graph()
        st.session_state["agent_state"] = {
            "messages": [], "intent": "", "lead_info": {},
            "lead_captured": False, "awaiting_field": None, "rag_context": "",
        }

def _run_agent(user_text: str):
    """Invoke the LangGraph agent and return (reply, intent, lead_result)."""
    from langchain_core.messages import HumanMessage, AIMessage

    if st.session_state["graph"] is None or st.session_state["agent_state"] is None:
        _init_graph()

    state = st.session_state["agent_state"]
    if state is None:
        return ["Agent initialization failed. Please check your configuration."], "unknown", None

    # pre_invoke_count is used to identify new messages added by the graph
    pre_invoke_count = len(state["messages"])
    state["messages"] = state["messages"] + [HumanMessage(content=user_text)]
    graph = st.session_state["graph"]

    result = graph.invoke(state)
    if result is None:
        return ["I'm sorry, I encountered an error and couldn't process that."], "unknown", None

    st.session_state["agent_state"] = result
    
    # Capture all new AI messages that appeared after the user's input
    new_messages = result["messages"][pre_invoke_count + 1:]
    replies = [m.content for m in new_messages if isinstance(m, AIMessage)]
    
    if not replies:
        replies = ["..."]

    intent  = result.get("intent", "unknown")

    lead_result = None
    if result.get("lead_captured") and not any(
        m.get("intent") == "captured" for m in st.session_state["messages"]
    ):
        # Sync the session state with the persistent YAML store updated by the agent node
        all_leads = _load_leads_from_file()
        st.session_state["leads"] = all_leads
        lead_result = all_leads[-1] if all_leads else None
        intent = "captured"

    return replies, intent, lead_result


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # Wordmark
    st.markdown("""
    <div class="wordmark-bar">
        <div class="wordmark-icon">🎬</div>
        <div>
            <div class="wordmark-text">AutoStream</div>
        </div>
        <span class="wordmark-badge">Inflx Agent</span>
    </div>
    """, unsafe_allow_html=True)

    # ── API Key ──────────────────────────────────────────────────────────────
    #st.markdown('<div class="sidebar-label">⚙️ Configuration</div>', unsafe_allow_html=True)
    #api_key_input = st.text_input(
       # "Anthropic API Key",
      #  type="password",
     #   value=os.getenv("ANTHROPIC_API_KEY", ""),
     #   placeholder="sk-ant-...",
    #    label_visibility="collapsed", )
    #if api_key_input:
      #  os.environ["ANTHROPIC_API_KEY"] = api_key_input
       # if not st.session_state["api_key_ok"]:
        #    st.session_state["api_key_ok"] = True
       #     _init_graph()
      #  st.markdown('<span style="color:#3dd68c;font-size:13px;">✓ API key set</span>', unsafe_allow_html=True)
    #else:
     #   st.markdown('<span style="color:#fc6c8f;font-size:13px;">Enter your Anthropic API key above</span>', unsafe_allow_html=True)

   # st.markdown("<br>", unsafe_allow_html=True)

    # ── Stats ────────────────────────────────────────────────────────────────
    total_msgs  = len([m for m in st.session_state["messages"] if m["role"] == "user"])
    total_leads = len(st.session_state["leads"])

    fields_done = _lead_fields_collected(st.session_state["agent_state"])
    pct = int((fields_done / 4) * 100) if st.session_state["agent_state"] else 0

    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-label">📊 Session Stats</div>
        <div class="stat-grid">
            <div class="stat-box">
                <div class="stat-val">{total_msgs}</div>
                <div class="stat-lbl">Messages</div>
            </div>
            <div class="stat-box">
                <div class="stat-val" style="color:var(--green)">{total_leads}</div>
                <div class="stat-lbl">Leads</div>
            </div>
        </div>
        <div style="margin-top:14px;">
            <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--muted);margin-bottom:4px;">
                <span>Lead collection progress</span><span>{fields_done}/4 fields</span>
            </div>
            <div class="progress-track">
                <div class="progress-fill" style="width:{pct}%"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Captured Leads ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="sidebar-section">
        <div class="sidebar-label">🏆 Captured Leads ({total_leads})</div>
    """, unsafe_allow_html=True)

    if st.session_state["leads"]:
        for lead in reversed(st.session_state["leads"]):
            # Display captured_at from YAML if available, otherwise fallback
            time_str = lead.get('captured_at', lead.get('ts', ''))[:16].replace('T', ' ')
            st.markdown(f"""
            <div class="lead-card">
                <div class="lead-name">{lead['name']}</div>
                <div class="lead-detail">✉ {lead['email']}</div>
                <div class="lead-detail">📱 {lead['platform']} | 💳 {lead.get('plan', 'N/A')}</div>
                <div class="lead-id">{lead['lead_id']} · {time_str if time_str else lead.get('ts', '')}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="color:var(--muted);font-size:13px;">No leads yet — start chatting!</div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Knowledge Base Explorer ──────────────────────────────────────────────
    with st.expander("📚 Knowledge Base"):
        import json as _json
        # Point to the actual knowledge base file, not the leads file
        kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base", "autostream_kb.json")
        
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                kb = _json.load(f)
            if isinstance(kb, dict) and "pricing" in kb:
                st.markdown('<div class="sidebar-label">Plans</div>', unsafe_allow_html=True)
                for plan in kb["pricing"]["plans"]:
                    chips = "".join(f'<span class="kb-chip">{feat[:30]}</span>' for feat in plan["features"][:4])
                    st.markdown(f"""
                    <div style="margin-bottom:10px;">
                        <div style="font-weight:600;font-size:13px;color:var(--text)">{plan['name']} — ${plan['price_monthly']}/mo</div>
                        <div style="margin-top:6px;">{chips}</div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Knowledge base structure is invalid or empty.")
        else:
            st.warning("Knowledge base file not found.")

    # ── Reset ────────────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🔄 New Conversation"):
        for k in ["messages","leads","total_turns","agent_state","graph","api_key_ok"]:
            del st.session_state[k]
        st.rerun()


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN PANEL
# ═══════════════════════════════════════════════════════════════════════════════

# Header
st.markdown("""
<div style="padding: 1px 0 24px 0; border-bottom: 2px solid var(--border); margin-bottom: 28px;">
    <div style="font-family:'Syne',sans-serif; font-size:25px; font-weight:700; letter-spacing:-0.8px; color:var(--text);">
       🎬 AutoStream · AI Sales Agent
    </div>
    <div style="font-size:13.5px; color:var(--muted); margin-top:4px;">
        AutoStream · Powered by LangGraph · ServiceHive × Inflx
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chat messages ─────────────────────────────────────────────────────────────
chat_area = st.container()

with chat_area:
    if not st.session_state["messages"]:
        # Welcome card
        st.markdown("""
        <div style="
            max-width:680px; margin:40px auto 0; padding:32px;
            background:var(--surface2); border:1px solid var(--border);
            border-radius:18px; text-align:center;">
            <div style="font-size:48px; margin-bottom:16px;">🎬</div>
            <div style="font-family:'Syne',sans-serif;font-size:20px;font-weight:700;margin-bottom:10px;">
                Welcome to AutoStream
            </div>
            <div style="color:var(--muted);font-size:14px;line-height:1.6;max-width:400px;margin:0 auto 24px;">
                I'm your AI sales assistant. Ask me about plans, features, or pricing —
                or tell me you're ready to get started!
            </div>
            <div style="display:flex;gap:10px;flex-wrap:wrap;justify-content:center;">
                <span class="kb-chip">💬 "What plans do you offer?"</span>
                <span class="kb-chip">💬 "Tell me about the Pro plan"</span>
                <span class="kb-chip">💬 "I want to sign up"</span>
                <span class="kb-chip">💬 "What's the refund policy?"</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-wrapper">', unsafe_allow_html=True)
        i = 0
        while i < len(st.session_state["messages"]):
            msg = st.session_state["messages"][i]
            role    = msg["role"]
            content = msg["content"]
            intent  = msg.get("intent", "")
            ts      = msg.get("ts", "")
            is_editing = st.session_state.get("editing_msg_idx") == i

            if role == "user":
                st.markdown(f"""
                <div class="msg-row user">
                    <div class="avatar user">👤</div>
                    <div>
                        <div class="bubble user">{content}</div>
                        <div class="bubble-meta">{ts}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- Edit / Delete Buttons for User Messages ---
                btn_col1, btn_col2, btn_col3 = st.columns([0.9, 0.05, 0.05])
                with btn_col1:
                    if is_editing:
                        # Show text input if editing
                        new_text = st.text_input("Edit message", value=content, key=f"edit_input_{i}", label_visibility="collapsed")
                        if st.button("✅ Save", key=f"save_{i}"):
                            if new_text.strip():
                                # Update the message content
                                st.session_state["messages"][i]["content"] = new_text
                                # Delete all messages AFTER this one (agent responses to old input)
                                st.session_state["messages"] = st.session_state["messages"][:i+1]
                                # Trigger agent to respond to the edited message
                                st.session_state["last_user_message_content"] = new_text
                                st.session_state["awaiting_agent_response"] = True
                                st.session_state["editing_msg_idx"] = None
                                st.rerun()
                        if st.button("❌ Cancel", key=f"cancel_{i}"):
                            st.session_state["editing_msg_idx"] = None
                            st.rerun()
                            
                with btn_col2:
                    if not is_editing:
                        if st.button("✏️", key=f"edit_{i}", help="Edit message"):
                            st.session_state["editing_msg_idx"] = i
                            st.rerun()
                            
                with btn_col3:
                    if not is_editing:
                        if st.button("🗑️", key=f"del_{i}", help="Delete message"):
                            # Delete this message and all subsequent ones
                            st.session_state["messages"] = st.session_state["messages"][:i]
                            st.rerun() # Break and rerun, list size changed

            else:
                pill_html = _pill(intent) if intent else ""
                # Check if this specific message is an error
                is_error = (intent == "error")
                
                # If it's an error, we wrap the bubble and button together so they appear side-by-side
                if is_error:
                    st.markdown(f"""
                    <div class="msg-row agent">
                        <div class="avatar agent">🎬</div>
                        <div style="display:flex; align-items:flex-start; gap:10px;">
                            <div>
                                <div class="bubble agent">
                                    {pill_html}
                                    <div>{content}</div>
                                </div>
                                <div class="bubble-meta">{ts}</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Add the retry button directly underneath the error bubble
                    if st.button("🔄 Retry", key=f"retry_{i}"):
                        # Remove the error message from the chat history
                        st.session_state["messages"].pop(i)
                        
                        # Find the user message that triggered this error (it should be right before the error)
                        if i > 0 and st.session_state["messages"][i-1]["role"] == "user":
                            st.session_state["last_user_message_content"] = st.session_state["messages"][i-1]["content"]
                        
                        # Trigger the agent processing phase
                        st.session_state["awaiting_agent_response"] = True
                        st.rerun()
                else:
                    # Normal agent message (no retry button)
                    st.markdown(f"""
                    <div class="msg-row agent">
                        <div class="avatar agent">🎬</div>
                        <div>
                            <div class="bubble agent">
                                {pill_html}
                                <div>{content}</div>
                            </div>
                            <div class="bubble-meta">{ts}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            i += 1 # Increment while loop index

        st.markdown('</div>', unsafe_allow_html=True)


# ── Input area ────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)

# Quick-reply chips
st.markdown("""
<div style="margin-top:12px;display:flex;gap:8px;flex-wrap:wrap;">
    <span style="color:var(--muted);font-size:12px;align-self:center;">Quick:</span>
</div>
""", unsafe_allow_html=True)

qr_cols = st.columns(4)
quick_replies = [
    "What are your pricing plans?",
    "Tell me about the Pro plan",
    "I want to sign up now",
    "What's your refund policy?",
]
for i, (col, qr) in enumerate(zip(qr_cols, quick_replies)):
    with col:
        if st.button(qr, key=f"qr_{i}"):
            st.session_state["pending_qr"] = qr

# Main chat input
chat_val = st.chat_input("Ask about pricing, features, or say you're ready to sign up…")

# Determine if we have a new message from either the text box or a quick reply
user_input = chat_val or st.session_state.get("pending_qr")

# ── Process message (Two-phase rendering for responsiveness) ─────────────────

# Phase 1: User input received, display it immediately, then trigger agent processing
if user_input and not st.session_state.get("awaiting_agent_response", False):
    # Clear the pending QR so it doesn't trigger again on next run
    if "pending_qr" in st.session_state:
        del st.session_state["pending_qr"]

    # Append user message to display history
    st.session_state["messages"].append({
        "role": "user", "content": user_input, "ts": _ts()
    })
    st.session_state["total_turns"] += 1
    
    # Store user input for the agent to process in the next rerun
    st.session_state["last_user_message_content"] = user_input
    st.session_state["awaiting_agent_response"] = True
    st.rerun() # Rerun to display the user's message immediately

# Phase 2: Agent processing (triggered by awaiting_agent_response flag)
if st.session_state.get("awaiting_agent_response", False) and st.session_state.get("last_user_message_content"):
    user_text_for_agent = st.session_state["last_user_message_content"]
    
    # Show typing indicator briefly then get response
    with st.spinner(""):
        try:
            # _run_agent expects the user_text to be added to the agent_state's messages
            # It will add HumanMessage(content=user_text_for_agent) internally
            replies, intent, lead_result = _run_agent(user_text_for_agent)
        except Exception as e:
            replies = [f"⚠️ Agent error: {str(e)}"]
            intent = "error"
            lead_result = None

    # Append each agent reply in the sequence to display history
    for reply_text in replies:
        st.session_state["messages"].append({
            "role": "agent", "content": reply_text, "intent": intent, "ts": _ts()
        })

    if lead_result:
        st.balloons()
    
    # Reset flags and clear stored user input
    st.session_state["awaiting_agent_response"] = False
    # Only clear the user input if the agent actually succeeded
    if intent != "error":
        st.session_state["last_user_message_content"] = None
    st.rerun() # Rerun to display the agent's messages
