import os
import base64
from typing import Dict, Any, Literal
from typing_extensions import TypedDict

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from state import NexusState


from google import genai
from google.genai import types

class UIUXState(TypedDict):
    request_type: Literal["analysis", "edit", "info"]
    user_prompt: str
    original_image_path: str
    current_artifact_filename: str # Tracks the latest generated image
    
    # Pipeline Outputs
    ui_analysis: str
    design_strategy: str
    generation_report: str

class RouterDecision(BaseModel):
    route: Literal["analysis", "edit", "info"]



def get_gemini_vision_model():
    # Gemini 3.7 Flash handles both text and multimodal vision
    return ChatGoogleGenerativeAI(model="gemini-3.7-flash", temperature=0)

def uiux_router_node(state: UIUXState):
    """Replaces the root_agent Coordinator. Decides if this is a new analysis, an edit, or info."""
    llm = get_gemini_vision_model().with_structured_output(RouterDecision)
    
    prompt = (
        f"Analyze the request: '{state['user_prompt']}'. "
        "If there is an image path provided, route to 'analysis'. "
        "If the user is asking to modify an existing design, route to 'edit'. "
        "Otherwise, route to 'info'."
    )
    
    decision = llm.invoke([HumanMessage(content=prompt)])
    return {"request_type": decision.route}

def info_node(state: UIUXState):
    """Replaces info_agent."""
    return {"generation_report": "Hi! I'm the AI UI/UX Feedback Team. Upload a landing page screenshot, and I'll analyze its layout, colors, and CTAs, then generate an improved version for you!"}

def ui_critic_node(state: UIUXState):
    """Replaces UICritic agent. Analyzes the landing page image."""
    llm = get_gemini_vision_model()
    
    with open(state["original_image_path"], "rb") as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        
    prompt = (
        "You are a Senior UI/UX Designer. Analyze this landing page image. "
        "Focus on Layout & Hierarchy, Typography, Color & Contrast, CTA, and Whitespace. "
        "Provide scores, critical issues, and top 3 impact priorities."
    )
    
    message = HumanMessage(
        content=[
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
        ]
    )
    
    response = llm.invoke([message])
    return {"ui_analysis": response.content}

def design_strategist_node(state: UIUXState):
    """Replaces design_strategist agent. Creates an improvement plan."""
    llm = get_gemini_vision_model()
    
    prompt = f"Based on the UI Critic's analysis, create a specific improvement plan:\n\n{state['ui_analysis']}"
    
    response = llm.invoke([
        SystemMessage(content="You are a Design Strategist. Provide ultra-specific colors (hex codes), sizes (px), and placements."),
        HumanMessage(content=prompt)
    ])
    return {"design_strategy": response.content}

def visual_implementer_node(state: UIUXState):
    """Replaces visual_implementer agent and the generate_improved_landing_page tool."""
    client = genai.Client()
    
    # 1. Synthesize the generation prompt
    prompt = f"Create a professional landing page incorporating these improvements:\n{state['design_strategy']}\nMake it a desktop web design screenshot, 16:9 aspect ratio."
    
    # 2. Call Gemini 2.5 Flash Image generation
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=prompt,
        )
        
        # Save the generated image
        new_filename = "landing_page_improved_v1.png"
        if response.candidates and response.candidates[0].content.parts[0].inline_data:
            image_bytes = response.candidates[0].content.parts[0].inline_data.data
            with open(new_filename, "wb") as f:
                f.write(image_bytes)
                
            report = f"✅ **Improved Landing Page Generated!**\nSaved as {new_filename}.\n\nThis design addresses the critical issues identified in the analysis."
            return {"current_artifact_filename": new_filename, "generation_report": report}
            
    except Exception as e:
        return {"generation_report": f"Error generating improved landing page: {str(e)}"}

def design_editor_node(state: UIUXState):
    """Replaces design_editor agent and the edit_landing_page_image tool."""
    client = genai.Client()
    
    try:
        # Load the previous image to edit
        with open(state["current_artifact_filename"], "rb") as f:
            image_bytes = f.read()
            
        image_part = types.Part.from_bytes(data=image_bytes, mime_type="image/png")
        prompt_part = types.Part.from_text(text=f"{state['user_prompt']}\nApply these UI/UX best practices while editing.")
        
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[image_part, prompt_part],
        )
        
        # Save edited image
        new_filename = state["current_artifact_filename"].replace(".png", "_edited.png")
        if response.candidates and response.candidates[0].content.parts[0].inline_data:
            new_image_bytes = response.candidates[0].content.parts[0].inline_data.data
            with open(new_filename, "wb") as f:
                f.write(new_image_bytes)
                
            return {"current_artifact_filename": new_filename, "generation_report": f"✅ Design edited successfully! Saved as {new_filename}."}
            
    except Exception as e:
        return {"generation_report": f"Error editing landing page: {str(e)}"}



def route_coordinator(state: UIUXState) -> str:
    if state["request_type"] == "analysis":
        return "ui_critic_node"
    elif state["request_type"] == "edit":
        return "design_editor_node"
    return "info_node"

uiux_graph = StateGraph(UIUXState)

uiux_graph.add_node("uiux_router_node", uiux_router_node)
uiux_graph.add_node("info_node", info_node)
uiux_graph.add_node("ui_critic_node", ui_critic_node)
uiux_graph.add_node("design_strategist_node", design_strategist_node)
uiux_graph.add_node("visual_implementer_node", visual_implementer_node)
uiux_graph.add_node("design_editor_node", design_editor_node)

# Flow definitions
uiux_graph.add_edge(START, "uiux_router_node")
uiux_graph.add_conditional_edges("uiux_router_node", route_coordinator)

# Analysis Pipeline
uiux_graph.add_edge("ui_critic_node", "design_strategist_node")
uiux_graph.add_edge("design_strategist_node", "visual_implementer_node")
uiux_graph.add_edge("visual_implementer_node", END)

# Edit and Info Pipelines
uiux_graph.add_edge("design_editor_node", END)
uiux_graph.add_edge("info_node", END)

uiux_feedback_team = uiux_graph.compile()



def uiux_feedback_domain_node(state: NexusState) -> Dict[str, Any]:
    """Outer node that plugs into the main Nexus application."""
    
    is_image = state["user_prompt"].lower().endswith(('.png', '.jpg', '.jpeg'))
    
    initial_state = {
        "user_prompt": state["user_prompt"],
        "original_image_path": state["user_prompt"] if is_image else None,
        "current_artifact_filename": state.get("latest_generated_image", "")
    }
    
    final_state = uiux_feedback_team.invoke(initial_state)
    
    # Consolidate outputs for the Nexus State
    final_output = final_state.get("generation_report", "")
    if final_state.get("ui_analysis"):
        final_output = f"### UI/UX Audit\n{final_state['ui_analysis']}\n\n### Implementation\n{final_output}"
    
    return {
        "domain_insights": {
            "uiux_intel": final_output
        },
        # Pass the filename back to global state so subsequent edits can access it
        "latest_generated_image": final_state.get("current_artifact_filename", "")
    }