"""
Quick test runner — verifies the orchestrator compiles and runs a simple task.
Run from backend/ directory:
  python test_run.py "Write a short poem about AI"
"""
import sys
import os
from pathlib import Path

# Ensure backend root in path
BACKEND_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(BACKEND_ROOT))

from dotenv import load_dotenv
load_dotenv()

def test_imports():
    print("Testing imports...")
    from shared.llm import get_gemini_flash
    from shared.tools import web_search
    from orchestrator.state import OrchestratorState
    from orchestrator.ceo_agent import ceo_router_node
    from departments.research.graph import research_department_node
    from departments.content.graph import content_department_node
    from departments.code.graph import code_department_node
    from orchestrator.graph import compiled_graph
    print("✅ All imports successful")
    return compiled_graph


def run_test(user_request: str):
    print(f"\n🚀 Running test: '{user_request}'\n{'='*60}")
    compiled_graph = test_imports()

    initial_state = {
        "user_request": user_request,
        "conversation_id": "test-001",
        "task_plan": None,
        "active_departments": [],
        "completed_departments": [],
        "department_outputs": {},
        "agent_events": [],
        "final_output": "",
        "clarification_needed": False,
        "clarification_question": None,
        "error": None,
    }

    print("Invoking orchestrator...\n")
    final_state = compiled_graph.invoke(initial_state)

    print("\n" + "="*60)
    print("📊 EVENTS:")
    for e in final_state.get("agent_events", []):
        dept = f"[{e.get('department', 'CEO')}]" if e.get('department') else "[CEO]"
        agent = e.get('agent', '')
        data = e.get('data', e.get('event', ''))
        print(f"  {dept} {agent}: {data}")

    print("\n" + "="*60)
    print("📋 FINAL OUTPUT:")
    print(final_state.get("final_output", "No output"))


if __name__ == "__main__":
    request = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "Write a short blog post about the benefits of Python"
    run_test(request)
