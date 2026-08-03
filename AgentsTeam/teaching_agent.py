import os
import re
from typing import Dict, Any, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import Tool
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

from composio_langchain import ComposioToolSet, Action

from state import NexusState


class TeachingState(TypedDict):
    topic: str
    
    # Agent Outputs
    knowledge_base: str
    roadmap: str
    resources: str
    practice_materials: str
    
    # Final Compilation
    final_report: str


def get_gdocs_tool():
    """Initializes the Composio Google Docs tool for LangChain."""
    composio_key = os.getenv("COMPOSIO_API_KEY")
    if not composio_key:
        raise ValueError("COMPOSIO_API_KEY is not set.")
    
    toolset = ComposioToolSet(api_key=composio_key)
    # Get the specific tool for creating a document
    tools = toolset.get_tools(actions=[Action.GOOGLEDOCS_CREATE_DOCUMENT])
    return tools[0]

def get_search_tool():
    """Initializes the SerpAPI search tool."""
    serpapi_key = os.getenv("SERPAPI_API_KEY")
    if not serpapi_key:
        raise ValueError("SERPAPI_API_KEY is not set.")
    
    search = SerpAPIWrapper(serpapi_api_key=serpapi_key)
    return Tool(
        name="Search",
        func=search.run,
        description="Useful for when you need to answer questions about current events or find learning resources."
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

def professor_node(state: TeachingState):
    """Researches the topic and creates a detailed knowledge base."""
    gdocs_tool = get_gdocs_tool()
    
    professor_agent = create_agent(
        model=llm,
        tools=[gdocs_tool],
        state_modifier=(
            "You are a Research and Knowledge Specialist. Explain topics from first principles. "
            "Include key terminology, core principles, and practical applications. "
            "IMPORTANT: You MUST use your tool to create a new Google Doc with your response, and include the Doc Link in your final text."
        )
    )
    
    prompt = f"Create a comprehensive knowledge base for the topic: {state['topic']}."
    response = professor_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return {"knowledge_base": response["messages"][-1].content}

def academic_advisor_node(state: TeachingState):
    """Designs a structured learning roadmap based on the Professor's knowledge base."""
    gdocs_tool = get_gdocs_tool()
    
    advisor_agent = create_agent(
        model=llm,
        tools=[gdocs_tool],
        state_modifier=(
            "You are a Learning Path Designer. Create detailed learning roadmaps. "
            "Break down topics logically, include time commitments. "
            "IMPORTANT: You MUST use your tool to create a new Google Doc with your response, and include the Doc Link in your final text."
        )
    )
    
    prompt = (
        f"Topic: {state['topic']}\n"
        f"Knowledge Base Context:\n{state['knowledge_base']}\n\n"
        "Create a detailed learning roadmap based on this knowledge base."
    )
    
    response = advisor_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    return {"roadmap": response["messages"][-1].content}

def research_librarian_node(state: TeachingState):
    """Curates high-quality learning resources using web search."""
    gdocs_tool = get_gdocs_tool()
    search_tool = get_search_tool()
    
    librarian_agent = create_agent(
        model=llm,
        tools=[gdocs_tool, search_tool],
        state_modifier=(
            "You are a Learning Resource Specialist. Use the Search tool to find technical blogs, "
            "GitHub repos, documentation, and courses. Curate a list with descriptions. "
            "IMPORTANT: You MUST use your tool to create a new Google Doc with your response, and include the Doc Link in your final text."
        )
    )
    
    prompt = f"Find and curate a list of high-quality learning resources for the topic: {state['topic']}."
    response = librarian_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return {"resources": response["messages"][-1].content}

def teaching_assistant_node(state: TeachingState):
    """Creates practice materials and exercises."""
    gdocs_tool = get_gdocs_tool()
    search_tool = get_search_tool()
    
    assistant_agent = create_agent(
        model=llm,
        tools=[gdocs_tool, search_tool],
        state_modifier=(
            "You are an Exercise Creator. Use the Search tool to find example problems. "
            "Create progressive exercises, quizzes, and hands-on projects with detailed solutions. "
            "IMPORTANT: You MUST use your tool to create a new Google Doc with your response, and include the Doc Link in your final text."
        )
    )
    
    # We pass the roadmap to the assistant so exercises align with the learning path
    prompt = (
        f"Topic: {state['topic']}\n"
        f"Learning Roadmap Context:\n{state['roadmap']}\n\n"
        "Create comprehensive practice materials that align with this roadmap."
    )
    response = assistant_agent.invoke({"messages": [HumanMessage(content=prompt)]})
    
    return {"practice_materials": response["messages"][-1].content}

def synthesize_teaching_node(state: TeachingState):
    """Compiles all outputs and extracts the Google Doc links for a clean final report."""
    
    def extract_link(text: str) -> str:
        match = re.search(r'(https://docs\.google\.com/[^\s\)]+)', text)
        return match.group(1) if match else "Link not generated."

    report = f"## 👨‍🏫 AI Teaching Hub: {state['topic']}\n\n"
    
    report += "### 📚 Generated Course Documents\n"
    report += f"- **Knowledge Base (Professor):** [View Document]({extract_link(state.get('knowledge_base', ''))})\n"
    report += f"- **Learning Roadmap (Advisor):** [View Document]({extract_link(state.get('roadmap', ''))})\n"
    report += f"- **Curated Resources (Librarian):** [View Document]({extract_link(state.get('resources', ''))})\n"
    report += f"- **Practice Materials (Assistant):** [View Document]({extract_link(state.get('practice_materials', ''))})\n\n"
    
    report += "---\n### 📖 Course Overview\n"
    # Provide a brief summary of the knowledge base in the main UI, the rest is in the GDocs
    report += state['knowledge_base'][:1500] + "...\n*(See full document for more)*"
    
    return {"final_report": report}


teaching_graph = StateGraph(TeachingState)

teaching_graph.add_node("professor", professor_node)
teaching_graph.add_node("advisor", academic_advisor_node)
teaching_graph.add_node("librarian", research_librarian_node)
teaching_graph.add_node("assistant", teaching_assistant_node)
teaching_graph.add_node("synthesize", synthesize_teaching_node)

# Flow: Professor -> Advisor -> Fan out to Librarian & Assistant -> Synthesize
teaching_graph.add_edge(START, "professor")
teaching_graph.add_edge("professor", "advisor")

# Parallel Execution
teaching_graph.add_edge("advisor", "librarian")
teaching_graph.add_edge("advisor", "assistant")

# Fan-in
teaching_graph.add_edge("librarian", "synthesize")
teaching_graph.add_edge("assistant", "synthesize")
teaching_graph.add_edge("synthesize", END)

teaching_team = teaching_graph.compile()


def teaching_domain_node(state: NexusState) -> Dict[str, Any]:
    """Outer node that plugs into the main Nexus application."""
    initial_state = {
        "topic": state["user_prompt"]
    }
    
    final_state = teaching_team.invoke(initial_state)
    
    return {
        "domain_insights": {
            "teaching_intel": final_state["final_report"]
        }
    }