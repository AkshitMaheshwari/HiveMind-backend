from pathlib import Path
import sys
from typing import Dict, Any, List, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from state import NexusState
from utils.doc_utils import Chunk, search_local

class ResearchState(TypedDict):
	question: str
	local_chunks: List[Chunk]
	route: Literal["local","web"]
	evidence: List[Dict[str, str]]
	draft_answer: str
	verifier_verdict: str
	verifier_gaps: List[str]
	final_answer: str


class TriageDecision(BaseModel):
	route: Literal["local","web"] = Field(description = "Choose 'local' if docs are provided, else 'web'. ")
	confidence: float = Field(description = "Confidence score from 0 to 1")
	rationale: str = Field(description = "Explain why you chose this route")

class DraftResponse(BaseModel):
	evidence: List[Dict[str,str]] = Field(description = "List of dicts with 'source' and 'summary' keys")
	draft_answer: str = Field(description = "the drafted answer based on the evidence ")

class VerificationDecision(BaseModel):
	verdict: Literal["sufficient", "insufficient"]
	gaps: List[str] = Field(description = "List of missing information")


llm = ChatOpenAI(model="gpt-4o", temperature=0.2)

def triage_node(state: ResearchState):
    """Replaces triage_agent"""
    # Force the LLM to output the TriageDecision JSON structure
    structured_llm = llm.with_structured_output(TriageDecision)
    
    doc_summary = "No local documents provided."
    if state.get("local_chunks"):
        doc_summary = f"Total local chunks available: {len(state['local_chunks'])}"
        
    prompt = f"Question: {state['question']}\nContext: {doc_summary}\nDecide if we should use 'local' or 'web' research."
    
    decision = structured_llm.invoke([SystemMessage(content="You are a triage agent."), HumanMessage(content=prompt)])
    
    # Fallback to web if local is chosen but no chunks exist
    final_route = "web" if (decision.route == "local" and not state.get("local_chunks")) else decision.route
    
    return {"route": final_route}

def local_research_node(state: ResearchState):
    """Replaces local_research_agent"""
    structured_llm = llm.with_structured_output(DraftResponse)
    
    hits = search_local(state["question"], state.get("local_chunks", []), top_k=5)
    formatted_hits = "\n".join([f"- {c.doc_name}: {c.text[:300]}" for c in hits])
    
    prompt = f"Question: {state['question']}\nExcerpts:\n{formatted_hits}"
    draft = structured_llm.invoke([
        SystemMessage(content="You are a local research agent. Use ONLY provided excerpts."), 
        HumanMessage(content=prompt)
    ])
    
    return {"evidence": draft.evidence, "draft_answer": draft.draft_answer}

def web_research_node(state: ResearchState):
    """Replaces web_research_agent (Swapped SearXNG for DuckDuckGo for easier local execution)"""
    structured_llm = llm.with_structured_output(DraftResponse)
    search_tool = DuckDuckGoSearchRun()
    
    # Run the web search
    search_results = search_tool.invoke(state["question"])
    
    prompt = f"Question: {state['question']}\nWeb Results:\n{search_results}"
    draft = structured_llm.invoke([
        SystemMessage(content="You are a web research agent. Use ONLY provided web results."), 
        HumanMessage(content=prompt)
    ])
    
    return {"evidence": draft.evidence, "draft_answer": draft.draft_answer}

def verifier_node(state: ResearchState):
    """Replaces verifier_agent"""
    structured_llm = llm.with_structured_output(VerificationDecision)
    
    prompt = f"Draft: {state['draft_answer']}\nEvidence: {state['evidence']}\nDoes the evidence fully answer the prompt?"
    verification = structured_llm.invoke([
        SystemMessage(content="You are a verifier. Check evidence sufficiency."),
        HumanMessage(content=prompt)
    ])
    
    return {"verifier_verdict": verification.verdict, "verifier_gaps": verification.gaps}

def synthesizer_node(state: ResearchState):
    """Replaces synthesizer_agent"""
    prompt = f"Draft: {state['draft_answer']}\nGaps identified: {state['verifier_gaps']}\nWrite final answer with citations."
    response = llm.invoke([
        SystemMessage(content="You are the final synthesizer. Produce a clear, cited answer."),
        HumanMessage(content=prompt)
    ])
    
    return {"final_answer": response.content}


def route_research(state: ResearchState):
    return f"{state['route']}_research_node" 

research_graph = StateGraph(ResearchState)

research_graph.add_node("triage_node", triage_node)
research_graph.add_node("local_research_node", local_research_node)
research_graph.add_node("web_research_node", web_research_node)
research_graph.add_node("verifier_node", verifier_node)
research_graph.add_node("synthesizer_node", synthesizer_node)

research_graph.add_edge(START, "triage_node")
research_graph.add_conditional_edges("triage_node", route_research)
research_graph.add_edge("local_research_node", "verifier_node")
research_graph.add_edge("web_research_node", "verifier_node")
research_graph.add_edge("verifier_node", "synthesizer_node")
research_graph.add_edge("synthesizer_node", END)

adaptive_research_team = research_graph.compile()


def adaptive_research_node(state: NexusState) -> Dict[str, Any]:
    """
    This is the outer node that plugs into your main application. 
    It triggers the internal Sub-Graph we just compiled.
    """
    
    initial_research_state = {
        "question": state["user_prompt"],
        "local_chunks": state.get("uploaded_documents", [])
    }
    
  
    final_research_state = adaptive_research_team.invoke(initial_research_state)
    
  
    return {
        "domain_insights": {
            "adaptive_research_intel": final_research_state["final_answer"]
        }
    }