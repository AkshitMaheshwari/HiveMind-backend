import os
import base64
from typing import Dict, Any, List
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

from state import NexusState


class DesignState(TypedDict):
    context: str
    focus_areas: List[str]
    design_image_paths: List[str]
    competitor_image_paths: List[str]
    
    # Agent Outputs
    visual_analysis: str
    ux_analysis: str
    market_analysis: str
    final_report: str

def prepare_multimodal_message(prompt: str, image_paths: List[str]) -> HumanMessage:
    """Converts local image paths into base64 format for LangChain's Gemini integration."""
    content = [{"type": "text", "text": prompt}]
    
    for path in image_paths:
        if os.path.exists(path):
            with open(path, "rb") as img_file:
                encoded_img = base64.b64encode(img_file.read()).decode("utf-8")
                # Assuming JPEG for simplicity; can be dynamically mapped
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded_img}"}
                })
                
    return HumanMessage(content=content)


def get_gemini_model():
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        raise ValueError("GEMINI_API_KEY is not set.")
    return ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=gemini_key)

def visual_design_node(state: DesignState):
    """Analyzes design elements, UI, typography, and color schemes."""
    llm = get_gemini_model()
    
    sys_msg = SystemMessage(content=(
        "You are a visual analysis expert. Identify design elements, patterns, and visual hierarchy. "
        "Analyze color schemes, typography, and layouts. Be specific and technical."
    ))
    
    prompt = f"Analyze these designs. Focus areas: {', '.join(state['focus_areas'])}.\nContext: {state['context']}"
    human_msg = prepare_multimodal_message(prompt, state['design_image_paths'])
    
    response = llm.invoke([sys_msg, human_msg])
    return {"visual_analysis": response.content}

def ux_design_node(state: DesignState):
    """Evaluates user flows, interaction patterns, and accessibility."""
    llm = get_gemini_model()
    
    sys_msg = SystemMessage(content=(
        "You are a UX analysis expert. Evaluate user flows, interaction patterns, and accessibility. "
        "Suggest practical UX improvements based on best practices."
    ))
    
    prompt = f"Evaluate the UX of these designs. Focus areas: {', '.join(state['focus_areas'])}.\nContext: {state['context']}"
    human_msg = prepare_multimodal_message(prompt, state['design_image_paths'])
    
    response = llm.invoke([sys_msg, human_msg])
    return {"ux_analysis": response.content}

def market_analysis_node(state: DesignState):
    """Compares the design against competitors and searches for market trends."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # Using GPT-4o for robust tool calling
    search_tool = DuckDuckGoSearchRun()
    
    # We use a standard ReAct agent here because it needs to search the web
    market_agent = create_agent(
        model=llm,
        tools=[search_tool],
        state_modifier=(
            "You are a market research expert. Identify market trends, analyze similar products, "
            "and suggest positioning. Use the web search tool to find current industry standards."
        )
    )
    
    prompt = (
        f"Context: {state['context']}\nFocus Areas: {', '.join(state['focus_areas'])}\n"
        "Search the web for current UI/UX market trends related to this context and provide a positioning strategy."
    )
    
    # If competitor images were provided, we could pass them to Gemini first to extract descriptions, 
    # but for simplicity, we focus on the web search aspect here.
    response = market_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return {"market_analysis": response["messages"][-1].content}

def synthesize_design_node(state: DesignState):
    """Compiles the parallel analyses into a cohesive final document."""
    report = f"## Multimodal Design & UX Analysis\n\n"
    report += f"### 🎨 Visual Design Insights\n{state['visual_analysis']}\n\n"
    report += f"### 🔄 User Experience (UX) Audit\n{state['ux_analysis']}\n\n"
    report += f"### 📊 Market Positioning\n{state['market_analysis']}\n"
    
    return {"final_report": report}


design_graph = StateGraph(DesignState)

design_graph.add_node("visual_design", visual_design_node)
design_graph.add_node("ux_design", ux_design_node)
design_graph.add_node("market_analysis", market_analysis_node)
design_graph.add_node("synthesize", synthesize_design_node)

# Fan-out: Start all three analysis nodes at the same time
design_graph.add_edge(START, "visual_design")
design_graph.add_edge(START, "ux_design")
design_graph.add_edge(START, "market_analysis")

# Fan-in: Wait for all three to finish before synthesizing
design_graph.add_edge("visual_design", "synthesize")
design_graph.add_edge("ux_design", "synthesize")
design_graph.add_edge("market_analysis", "synthesize")
design_graph.add_edge("synthesize", END)

multimodal_design_team = design_graph.compile()


def multimodal_design_domain_node(state: NexusState) -> Dict[str, Any]:
    """Outer node that plugs into the main Nexus application."""
    
    # In a full implementation, these paths would be extracted from the user's uploaded files in the UI
    # For now, we map the generic state to the specific design state
    initial_design_state = {
        "context": state["user_prompt"],
        "focus_areas": ["Layout", "Interactions", "Color Scheme", "Market Fit"],
        "design_image_paths": state.get("uploaded_images", []), 
        "competitor_image_paths": []
    }
    
    final_design_state = multimodal_design_team.invoke(initial_design_state)
    
    return {
        "domain_insights": {
            "design_and_ux_intel": final_design_state["final_report"]
        }
    }