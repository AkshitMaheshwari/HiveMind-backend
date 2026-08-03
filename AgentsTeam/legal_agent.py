import os
from typing import Dict, Any, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools.retriever import create_retriever_tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from state import NexusState

class LegalState(TypedDict):
    query: str
    analysis_type: str
    research_notes: str
    contract_analysis: str
    final_strategy: str


# Qdrant client setup

def get_legal_retriever_tool():
    qdrant_url = os.getenv("QDRANT_URL", "http://localhost:6333")
    qdrant_api_key = os.getenv("QDRANT_API_KEY","")

    client = QdrantClient(url = qdrant_url,api_key = qdrant_api_key)
    embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")

    vector_store = QdrantVectorStore(
        client = client,
        collection_name = "legal_docs",
        embedding = embeddings,
    )

    retriever = vector_store.as_retriever(search_kwargs={"k":5})
    return create_retriever_tool(
        retriever,
        "search_legal_docs",
        "search the uploaded legal document for specific clauses, terms and obligations."
    )

llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens = 800, api_key = os.getenv('OPENAI_API_KEY'), base_url = "https://models.inference.ai.azure.com") 
web_tool = DuckDuckGoSearchRun()


def legal_researcher_node(state: LegalState):
    retriever_tool = get_legal_retriever_tool()
    researcher_agent = create_agent(
        model = llm,
        tools = [web_tool, retriever_tool],
        system_prompt = (
            "You are a Legal Researcher. Find and cite relevant legal cases and precedents. "
            "Always search the legal document for context."
        )
    )

    prompt = f"Task: {state['analysis_type']}\nQuery: {state['query']}\nPlease research relevant precedents and extract facts."
    response = researcher_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return {"research_notes": response["messages"][-1].content}

def contract_analyst_node(state: LegalState):
    """Focuses strictly on the contract clauses and terms."""
    retriever_tool = get_legal_retriever_tool()
    
    analyst_agent = create_agent(
        model=llm,
        tools=[retriever_tool],
        state_modifier=(
            "You are a Contract Analyst. Review contracts thoroughly. "
            "Identify key terms, obligations, and potential issues. Reference specific clauses."
        )
    )
    
    prompt = (
        f"Task: {state['analysis_type']}\nQuery: {state['query']}\n"
        f"Context from Legal Researcher:\n{state['research_notes']}\n"
        "Analyze the contract terms based on this information."
    )
    response = analyst_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return {"contract_analysis": response["messages"][-1].content}

def legal_strategist_node(state: LegalState):
    """Synthesizes the findings into actionable recommendations."""
    prompt = (
        f"Task: {state['analysis_type']}\nQuery: {state['query']}\n\n"
        f"--- RESEARCH NOTES ---\n{state['research_notes']}\n\n"
        f"--- CONTRACT ANALYSIS ---\n{state['contract_analysis']}\n\n"
        "You are a Legal Strategist. Based on the research and contract analysis above, "
        "develop comprehensive legal strategies, consider risks/opportunities, and provide actionable recommendations."
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    return {"final_strategy": response.content}

# Build and compile grpah

legal_graph = StateGraph(LegalState)

legal_graph.add_node("legal_researcher", legal_researcher_node)
legal_graph.add_node("contract_analyst", contract_analyst_node)
legal_graph.add_node("legal_strategist", legal_strategist_node)

# Sequential flow mirroring 

legal_graph.add_edge(START, "legal_researcher")
legal_graph.add_edge("legal_researcher", "contract_analyst")
legal_graph.add_edge("contract_analyst", "legal_strategist")
legal_graph.add_edge("legal_strategist", END)

legal_team = legal_graph.compile()


def legal_domain_node(state: NexusState) -> Dict[str, Any]:
    """Outer node that plugs into the main Nexus application."""
    
  
    initial_legal_state = {
        "query": state["user_prompt"],
        "analysis_type": "Comprehensive Legal Analysis", # Can be extracted dynamically if needed
    }
    
 
    final_legal_state = legal_team.invoke(initial_legal_state)
    
    
    consolidated_report = (
        "### 1. Legal Research & Precedents\n" + final_legal_state["research_notes"] + "\n\n"
        "### 2. Contract Analysis\n" + final_legal_state["contract_analysis"] + "\n\n"
        "### 3. Strategy & Recommendations\n" + final_legal_state["final_strategy"]
    )
    
    return {
        "domain_insights": {
            "legal_intel": consolidated_report
        }
    }

