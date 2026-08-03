import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

from google import genai
from google.genai import types

from state import NexusState

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)


class SalesIntellState(TypedDict):
    original_prompt: str
    competitor_name: str
    your_product_name: str
    
    # Pipeline Outputs
    competitor_profile: str
    feature_analysis: str
    positioning_intel: str
    swot_analysis: str
    objection_scripts: str
    
    # Final Artifacts
    battle_card_html_path: str
    comparison_chart_path: str
    final_report: str

class SalesExtraction(BaseModel):
    competitor_name: str = Field(description="The competitor company to analyze")
    your_product_name: str = Field(description="The user's product being sold against the competitor")
    is_valid: bool = Field(description="True if both competitor and product are identified")



llm = ChatOpenAI(model="gpt-4o", temperature=0.2)
search_tool = DuckDuckGoSearchRun()

def extract_entities_node(state: SalesIntellState):
    """Extracts the competitor and user's product from the prompt."""
    structured_llm = llm.with_structured_output(SalesExtraction)
    prompt = f"Extract the competitor and the user's product from this request: '{state['original_prompt']}'"
    
    extraction = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return {
        "competitor_name": extraction.competitor_name if extraction.is_valid else "Unknown Competitor",
        "your_product_name": extraction.your_product_name if extraction.is_valid else "Our Product"
    }

def competitor_research_node(state: SalesIntellState):
    """Researches competitor company information using web search."""
    research_agent = create_agent(
        model=llm,
        tools=[search_tool],
        state_modifier="You are a competitive intelligence analyst. Use search to gather overview, market, pricing, and news."
    )
    
    prompt = f"Research the competitor: {state['competitor_name']}. We are selling {state['your_product_name']} against them. Provide a comprehensive company profile."
    response = research_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return {"competitor_profile": response["messages"][-1].content}

def product_feature_node(state: SalesIntellState):
    """Analyzes competitor product features and capabilities."""
    feature_agent = create_agent(
        model=llm,
        tools=[search_tool],
        state_modifier="You are a product analyst comparing competitor features using search."
    )
    
    prompt = (
        f"Competitor Profile: {state['competitor_profile']}\n\n"
        f"Analyze the core features, integrations, architecture, and limitations of {state['competitor_name']}."
    )
    response = feature_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return {"feature_analysis": response["messages"][-1].content}

def positioning_analyzer_node(state: SalesIntellState):
    """Analyzes competitor positioning and messaging."""
    positioning_agent = create_agent(
        model=llm,
        tools=[search_tool],
        state_modifier="You are a marketing strategist analyzing competitor positioning."
    )
    
    prompt = (
        f"Competitor: {state['competitor_name']}\nFeatures: {state['feature_analysis']}\n\n"
        f"Find their messaging, target personas, social proof, and how they position against {state['your_product_name']}."
    )
    response = positioning_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return {"positioning_intel": response["messages"][-1].content}

def swot_node(state: SalesIntellState):
    """Synthesizes SWOT analysis from research."""
    prompt = (
        f"Competitor: {state['competitor_profile']}\nFeatures: {state['feature_analysis']}\n"
        f"Positioning: {state['positioning_intel']}\n\n"
        f"Create a brutally honest SWOT analysis comparing {state['competitor_name']} to {state['your_product_name']}."
        "Include Their Strengths, Their Weaknesses, Our Advantages, and Competitive Landmines."
    )
    
    response = llm.invoke([
        SystemMessage(content="You are a competitive strategist. Synthesize a SWOT analysis. Do not use search."),
        HumanMessage(content=prompt)
    ])
    
    return {"swot_analysis": response.content}

def objection_handler_node(state: SalesIntellState):
    """Creates objection handling scripts."""
    prompt = (
        f"SWOT: {state['swot_analysis']}\n\n"
        f"Create objection handling scripts for sales reps selling {state['your_product_name']} against {state['competitor_name']}. "
        "Include killer questions and trap-setting phrases."
    )
    
    response = llm.invoke([
        SystemMessage(content="You are a sales enablement expert creating objection handling scripts."),
        HumanMessage(content=prompt)
    ])
    
    return {"objection_scripts": response.content}

def battle_card_generator_node(state: SalesIntellState):
    """Generates professional HTML battle card and saves it."""
    current_date = datetime.now().strftime("%B %d, %Y")
    
    compiled_data = (
        f"Competitor: {state['competitor_profile']}\n"
        f"Features: {state['feature_analysis']}\n"
        f"SWOT: {state['swot_analysis']}\n"
        f"Objections: {state['objection_scripts']}"
    )
    
    prompt = f"""Generate a professional sales battle card in HTML format for {state['your_product_name']} vs {state['competitor_name']}.
    DATE: {current_date}
    Style it for SALES TEAMS with clean design, color coding (GREEN for us, RED for them), and collapsible sections.
    DATA: {compiled_data}
    Output ONLY valid HTML with embedded CSS/JS. Do not include markdown formatting ticks."""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    html_content = response.content.replace("```html", "").replace("```", "").strip()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_name = f"battle_card_{state['competitor_name'].replace(' ', '_')}_{timestamp}.html"
    filepath = OUTPUTS_DIR / artifact_name
    filepath.write_text(html_content, encoding='utf-8')
    
    return {"battle_card_html_path": str(filepath)}

def comparison_chart_node(state: SalesIntellState):
    """Creates visual comparison infographic using Google GenAI Image generation."""
    google_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not google_key:
        return {"comparison_chart_path": "Error: GEMINI_API_KEY missing."}
        
    client = genai.Client(api_key=google_key)
    
    comparison_data = f"Features: {state['feature_analysis']}\nSWOT: {state['swot_analysis']}"
    prompt = f"""Create a professional competitive comparison infographic.
    COMPARISON: {state['your_product_name']} vs {state['competitor_name']}
    Style: Clean, modern, sales-ready infographic. Green for {state['your_product_name']}, Red for {state['competitor_name']}.
    DATA: {comparison_data[:1000]}""" # Truncated to avoid prompt limits for image gen
    
    try:
        response = client.models.generate_content(
            model="gemini-3-pro-image-preview", # Note: Adjust to stable model name if preview expires
            contents=prompt,
        )
        
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.mime_type.startswith("image/"):
                image_bytes = part.inline_data.data
                ext = "png" if "png" in part.inline_data.mime_type else "jpg"
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                artifact_name = f"comparison_chart_{timestamp}.{ext}"
                filepath = OUTPUTS_DIR / artifact_name
                filepath.write_bytes(image_bytes)
                return {"comparison_chart_path": str(filepath)}
                
        return {"comparison_chart_path": "No image data returned from model."}
    except Exception as e:
        return {"comparison_chart_path": f"Image generation failed: {str(e)}"}

def synthesize_sales_node(state: SalesIntellState):
    """Compiles the final report linking to the generated assets."""
    report = f"## 🎯 Sales Intelligence: {state['your_product_name']} vs {state['competitor_name']}\n\n"
    report += f"**✅ Battle Card Generated:** `{state.get('battle_card_html_path', 'Not generated')}`\n"
    report += f"**📊 Comparison Chart Generated:** `{state.get('comparison_chart_path', 'Not generated')}`\n\n"
    report += "### SWOT Overview\n"
    report += f"{state['swot_analysis']}\n"
    
    return {"final_report": report}


sales_graph = StateGraph(SalesIntellState)

sales_graph.add_node("extract_entities", extract_entities_node)
sales_graph.add_node("competitor_research", competitor_research_node)
sales_graph.add_node("product_feature", product_feature_node)
sales_graph.add_node("positioning_analyzer", positioning_analyzer_node)
sales_graph.add_node("swot", swot_node)
sales_graph.add_node("objection_handler", objection_handler_node)
sales_graph.add_node("battle_card_generator", battle_card_generator_node)
sales_graph.add_node("comparison_chart", comparison_chart_node)
sales_graph.add_node("synthesize", synthesize_sales_node)

# Sequential Pipeline Flow
sales_graph.add_edge(START, "extract_entities")
sales_graph.add_edge("extract_entities", "competitor_research")
sales_graph.add_edge("competitor_research", "product_feature")
sales_graph.add_edge("product_feature", "positioning_analyzer")
sales_graph.add_edge("positioning_analyzer", "swot")
sales_graph.add_edge("swot", "objection_handler")

# Fan-out to generate artifacts in parallel
sales_graph.add_edge("objection_handler", "battle_card_generator")
sales_graph.add_edge("objection_handler", "comparison_chart")

# Fan-in to synthesize
sales_graph.add_edge("battle_card_generator", "synthesize")
sales_graph.add_edge("comparison_chart", "synthesize")
sales_graph.add_edge("synthesize", END)

sales_intelligence_team = sales_graph.compile()



def sales_intelligence_domain_node(state: NexusState) -> Dict[str, Any]:
    """Outer node that plugs into the main Nexus application."""
    initial_state = {
        "original_prompt": state["user_prompt"]
    }
    
    final_state = sales_intelligence_team.invoke(initial_state)
    
    return {
        "domain_insights": {
            "sales_intel": final_state["final_report"]
        }
    }