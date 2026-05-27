"""
main.py
Entry point for the AutoStream Social-to-Lead Conversational Agent.

Run:
    python main.py

Uses LangGraph compiled graph with persistent state across turns.
"""

import os
import sys
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

# Load .env (ANTHROPIC_API_KEY)
load_dotenv()

# Verify API key
#if not os.getenv("ANTHROPIC_API_KEY"):
    #print("\n❌  Error: ANTHROPIC_API_KEY not found in environment.")
    #print("   Create a .env file with: ANTHROPIC_API_KEY=your-key-here\n")
    #sys.exit(1)
#
from agent.graph import build_graph, AgentState


# ── Banner ────────────────────────────────────────────────────────────────────

BANNER = """
╔══════════════════════════════════════════════════════╗
║        AutoStream – AI Sales Assistant               ║
║        Social-to-Lead Agentic Workflow               ║
║        Powered by LangGraph           ║
╚══════════════════════════════════════════════════════╝
  Type your message and press Enter.
  Type 'exit' or 'quit' to end the conversation.
  Type 'leads' to view all captured leads.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""


def run_agent():
    """Main conversation loop."""
    print(BANNER)

    # Build the compiled LangGraph
    graph = build_graph()

    # Persistent state across conversation turns
    state: AgentState = {
        "messages": [],
        "intent": "",
        "lead_info": {},
        "lead_captured": False,
        "awaiting_field": None,
        "rag_context": "",
    }

    from tools.lead_capture import get_all_leads

    while True:
        try:
            user_input = input("You: ").strip()
        

            if not user_input:
               continue

            if user_input.lower() in ("exit", "quit", "bye"):
                 print("\nAgent: Thanks for chatting! Have a great day! 👋\n")
                 break

            if user_input.lower() == "leads":
                 leads = get_all_leads()
                 if leads:
                     print(f"\n📋 Captured Leads ({len(leads)}):")
                     for lead in leads:
                         print(f"  • {lead['name']} | {lead['email']} | {lead['platform']} | {lead['lead_id']}")
                         print()
                 else:
                    print("\n  No leads captured yet.\n")
                    continue

            # Track message count before adding user input
            pre_invoke_count = len(state["messages"])
            state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run the graph
            try:
              result = graph.invoke(state)
              if result is None:
                 raise ValueError("Graph invocation returned None.")
            except Exception as e:
                print(f"\n⚠️  Agent error: {e}\n")
                continue
        
            # Update persistent state
            state = {**state, **result}

            # Print all new agent messages in sequence
            new_messages = state["messages"][pre_invoke_count + 1:]
            for msg in new_messages:
                if isinstance(msg, AIMessage):
                    print(f"\nAgent: {msg.content}")
            print()

        # Check if lead was just captured
            if state.get("lead_captured"):
               print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
               print("  Lead successfully saved to CRM. Continuing conversation...")
               print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\nError: {type(e).__name__}: {str(e)}")
            print("Continuing...\n")

if __name__ == "__main__":
    run_agent()
