import os
import re
from typing import Dict, Any, List, Optional
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from firecrawl import FirecrawlApp
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END

from state import NexusState


class HeadingItem(BaseModel):
    tag: str = Field(..., description="Heading tag such as h1, h2, h3.")
    text: str = Field(..., description="Text content of the heading.")

class LinkCounts(BaseModel):
    internal: Optional[int] = Field(None, description="Number of internal links on the page.")
    external: Optional[int] = Field(None, description="Number of external links on the page.")
    broken: Optional[int] = Field(None, description="Number of broken links detected.")
    notes: Optional[str] = Field(None, description="Additional qualitative observations about linking.")

class AuditResults(BaseModel):
    title_tag: str = Field(..., description="Full title tag text.")
    meta_description: str = Field(..., description="Meta description text.")
    primary_heading: str = Field(..., description="Primary H1 heading on the page.")
    secondary_headings: List[HeadingItem] = Field(default_factory=list, description="Secondary headings (H2-H4) in reading order.")
    word_count: Optional[int] = Field(None, description="Approximate number of words in the main content.")
    content_summary: str = Field(..., description="Summary of the main topics and structure of the content.")
    link_counts: LinkCounts = Field(..., description="Quantitative snapshot of internal/external/broken links.")
    technical_findings: List[str] = Field(default_factory=list, description="List of notable technical SEO issues.")
    content_opportunities: List[str] = Field(default_factory=list, description="Observed content gaps or opportunities.")

class TargetKeywords(BaseModel):
    primary_keyword: str = Field(..., description="Most likely primary keyword target.")
    secondary_keywords: List[str] = Field(default_factory=list, description="Related secondary or supporting keywords.")
    search_intent: str = Field(..., description="Dominant search intent inferred from the page.")
    supporting_topics: List[str] = Field(default_factory=list, description="Cluster of supporting topics or entities.")

class PageAuditOutput(BaseModel):
    audit_results: AuditResults = Field(..., description="Structured on-page audit findings.")
    target_keywords: TargetKeywords = Field(..., description="Keyword focus derived from page content.")

class SerpResult(BaseModel):
    rank: int = Field(..., description="Organic ranking position.")
    title: str = Field(..., description="Title of the search result.")
    url: str = Field(..., description="Landing page URL.")
    snippet: str = Field(..., description="SERP snippet or summary.")
    content_type: str = Field(..., description="Content format (blog post, landing page, etc.).")

class SerpAnalysis(BaseModel):
    primary_keyword: str = Field(..., description="Keyword used for SERP research.")
    top_10_results: List[SerpResult] = Field(..., description="Top organic competitors for the keyword.")
    title_patterns: List[str] = Field(default_factory=list, description="Common patterns used in competitor titles.")
    content_formats: List[str] = Field(default_factory=list, description="Typical content formats found.")
    people_also_ask: List[str] = Field(default_factory=list, description="Representative questions surfaced.")
    key_themes: List[str] = Field(default_factory=list, description="Notable recurring themes.")
    differentiation_opportunities: List[str] = Field(default_factory=list, description="Opportunities to stand out.")


class SeoAuditState(TypedDict):
    original_prompt: str
    target_url: str
    page_audit: Optional[dict]
    serp_analysis: Optional[dict]
    final_report: str


llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

def extract_url_node(state: SeoAuditState):
    """Extracts the target URL from the user prompt."""
    url_match = re.search(r'(https?://[^\s]+)', state["original_prompt"])
    target_url = url_match.group(1) if url_match else ""
    return {"target_url": target_url}

def page_auditor_node(state: SeoAuditState):
    """Scrapes the target URL and extracts the structural audit + keyword focus."""
    if not state["target_url"]:
        return {"page_audit": None, "final_report": "Error: No valid URL found in the request."}
        
    firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_key:
        return {"page_audit": None, "final_report": "Error: FIRECRAWL_API_KEY is not set."}

    # 1. Scrape the URL
    app = FirecrawlApp(api_key=firecrawl_key)
    try:
        scrape_result = app.scrape_url(
            state["target_url"], 
            params={'formats': ['markdown', 'html']}
        )
        markdown_content = scrape_result.get('markdown', '')
    except Exception as e:
        return {"page_audit": None, "final_report": f"Firecrawl scraping failed: {str(e)}"}

    # 2. Analyze with Structured Output
    structured_llm = llm.with_structured_output(PageAuditOutput)
    prompt = f"Perform a structural on-page SEO audit on the following scraped content:\n\n{markdown_content[:20000]}"
    
    try:
        audit_output = structured_llm.invoke([HumanMessage(content=prompt)])
        return {"page_audit": audit_output.dict()}
    except Exception as e:
        return {"page_audit": None, "final_report": f"Audit parsing failed: {str(e)}"}

def serp_analyst_node(state: SeoAuditState):
    """Performs competitive analysis using the discovered primary keyword."""
    if not state.get("page_audit"):
        return {"serp_analysis": None}

    primary_keyword = state["page_audit"]["target_keywords"]["primary_keyword"]
    search_tool = DuckDuckGoSearchRun()
    
    # 1. Gather raw SERP data
    raw_search_results = search_tool.invoke(f"{primary_keyword}")

    # 2. Analyze with Structured Output
    structured_llm = llm.with_structured_output(SerpAnalysis)
    prompt = (
        f"Keyword: {primary_keyword}\n\n"
        f"Raw Search Results:\n{raw_search_results}\n\n"
        "Analyze these results and extract the competitive landscape, title patterns, and differentiation opportunities."
    )
    
    try:
        serp_output = structured_llm.invoke([
            SystemMessage(content="You are an expert SERP Analyst. Return structured JSON."),
            HumanMessage(content=prompt)
        ])
        return {"serp_analysis": serp_output.dict()}
    except Exception as e:
        return {"serp_analysis": None}

def optimization_advisor_node(state: SeoAuditState):
    """Synthesizes the audit and SERP insights into a prioritized optimization report."""
    if state.get("final_report"): # If an error occurred earlier
        return {"final_report": state["final_report"]}
        
    prompt = (
        f"Page Audit Data: {state['page_audit']}\n\n"
        f"SERP Analysis Data: {state['serp_analysis']}\n\n"
        "Write a highly professional, prioritized SEO Optimization Report in Markdown. "
        "Include an Executive Summary, Technical Findings, Keyword Analysis, SERP Analysis, and Prioritized Recommendations (P0/P1/P2)."
    )

    response = llm.invoke([
        SystemMessage(content="You are a Senior SEO Advisor. Output ONLY clean Markdown starting with '# SEO Audit Report'."),
        HumanMessage(content=prompt)
    ])

    return {"final_report": response.content}


def route_after_audit(state: SeoAuditState) -> str:
    if not state.get("page_audit"):
        return "optimization_advisor" # Skip SERP if audit failed
    return "serp_analyst"

seo_graph = StateGraph(SeoAuditState)

seo_graph.add_node("extract_url", extract_url_node)
seo_graph.add_node("page_auditor", page_auditor_node)
seo_graph.add_node("serp_analyst", serp_analyst_node)
seo_graph.add_node("optimization_advisor", optimization_advisor_node)

seo_graph.add_edge(START, "extract_url")
seo_graph.add_edge("extract_url", "page_auditor")
seo_graph.add_conditional_edges("page_auditor", route_after_audit)
seo_graph.add_edge("serp_analyst", "optimization_advisor")
seo_graph.add_edge("optimization_advisor", END)

seo_audit_team = seo_graph.compile()


def seo_audit_domain_node(state: NexusState) -> Dict[str, Any]:
    """Outer node that plugs into the main Nexus application."""
    initial_state = {
        "original_prompt": state["user_prompt"]
    }
    
    final_state = seo_audit_team.invoke(initial_state)
    
    return {
        "domain_insights": {
            "seo_intel": final_state["final_report"]
        }
    }