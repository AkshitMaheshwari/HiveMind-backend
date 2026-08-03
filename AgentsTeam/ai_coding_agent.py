import os
import re
import base64
from typing import Dict,Any,Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from e2b_code_interpreter import Sandbox

from state import NexusState


class CodingState(TypedDict):
    input_type: str # "text" or "image"
    problem_text: str
    image_path: Optional[str]
    generated_code: str
    execution_logs: str
    execution_error: str
    execution_analysis: str
    final_report: str

def vision_node(state: CodingState):
    gemini_key = os.gentenv("GOOGLE_API_KEY")
    if not gemini_key:
        return {"problem_text": "Error: GEMINI_API_KEY missing."}
        
    vision_llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=gemini_key)
    
    try:
        with open(state["image_path"], "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
            
        prompt = (
            "Analyze this image and extract any coding problem or code snippet shown. "
            "Describe it in clear natural language, including the problem statement, "
            "input/output examples, and constraints."
        )
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_image}"}}
            ]
        )
        
        response = vision_llm.invoke([message])
        return {"problem_text": response.content}
    except Exception as e:
        return {"problem_text": f"Error processing image: {str(e)}"}

def coding_node(state: CodingState):
    """Uses an advanced reasoning model (like o3-mini or gpt-4o) to write the code."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0) # Replace with o3-mini if available in your OpenAI tier
    
    system_prompt = (
        "You are an expert Python programmer. Analyze the problem carefully and optimally. "
        "Write clean, efficient Python code to solve it. Include proper documentation. "
        "Ensure your code is complete and enclosed within standard ```python code blocks."
    )
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Problem Statement:\n{state['problem_text']}")
    ])
    
    # Extract just the code from the markdown block
    content = response.content
    code_blocks = content.split("```python")
    if len(code_blocks) > 1:
        code = code_blocks[1].split("```")[0].strip()
    else:
        code = content.replace("```", "").strip()
        
    return {"generated_code": code}

def sandbox_execution_node(state: CodingState):
    """Executes the generated code inside the E2B secure sandbox."""
    e2b_key = os.getenv("E2B_API_KEY")
    if not e2b_key:
        return {"execution_error": "Error: E2B_API_KEY missing."}
        
    os.environ['E2B_API_KEY'] = e2b_key
    
    try:
        with Sandbox(timeout=30) as sandbox:
            execution = sandbox.run_code(state["generated_code"])
            
            error_msg = ""
            if execution.error:
                if "TimeoutException" in str(execution.error):
                    error_msg = "Execution Timeout: The code took longer than 30 seconds."
                else:
                    error_msg = str(execution.error)
            
            return {
                "execution_logs": "\n".join([str(log) for log in execution.logs]) if execution.logs else "No standard output.",
                "execution_error": error_msg
            }
    except Exception as e:
        return {"execution_error": f"Sandbox Exception: {str(e)}"}

def execution_analysis_node(state: CodingState):
    """Analyzes the sandbox output and generates the final explanation."""
    llm = ChatOpenAI(model="gpt-4o", temperature=0)
    
    system_prompt = (
        "You are an expert at executing Python code and explaining results. "
        "Review the execution logs or errors and explain what happened clearly."
    )
    
    if state.get("execution_error"):
        content = f"The code resulted in an error:\n{state['execution_error']}\n\nCode:\n{state['generated_code']}\nExplain what went wrong."
    else:
        content = f"The code executed successfully.\nLogs:\n{state['execution_logs']}\n\nExplain the results."
        
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=content)
    ])
    
    # Format the final unified report
    report = f"### 1. Problem Statement\n{state['problem_text']}\n\n"
    report += f"### 2. Generated Code\n```python\n{state['generated_code']}\n```\n\n"
    report += f"### 3. Execution Logs\n```text\n{state.get('execution_error') or state.get('execution_logs')}\n```\n\n"
    report += f"### 4. Analysis\n{response.content}"
    
    return {"execution_analysis": response.content, "final_report": report}


def route_initial_input(state: CodingState) -> str:
    """Routes to vision node if an image is provided, otherwise goes straight to coding."""
    if state["input_type"] == "image":
        return "vision_node"
    return "coding_node"

coding_graph = StateGraph(CodingState)

coding_graph.add_node("vision_node", vision_node)
coding_graph.add_node("coding_node", coding_node)
coding_graph.add_node("sandbox_execution_node", sandbox_execution_node)
coding_graph.add_node("execution_analysis_node", execution_analysis_node)

coding_graph.add_conditional_edges(START, route_initial_input)
coding_graph.add_edge("vision_node", "coding_node")
coding_graph.add_edge("coding_node", "sandbox_execution_node")
coding_graph.add_edge("sandbox_execution_node", "execution_analysis_node")
coding_graph.add_edge("execution_analysis_node", END)

coding_team = coding_graph.compile()


def multimodal_coding_domain_node(state: NexusState) -> Dict[str, Any]:
    """Outer node that plugs into the main Nexus application."""
    
    # Check if the user prompt is an image path (basic heuristic, can be refined)
    is_image = state["user_prompt"].lower().endswith(('.png', '.jpg', '.jpeg'))
    
    initial_coding_state = {
        "input_type": "image" if is_image else "text",
        "problem_text": "" if is_image else state["user_prompt"],
        "image_path": state["user_prompt"] if is_image else None
    }
    
    final_coding_state = coding_team.invoke(initial_coding_state)
    
    return {
        "domain_insights": {
            "coding_intel": final_coding_state["final_report"],
            "generated_code": final_coding_state["generated_code"]
        }
    }