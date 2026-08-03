import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from state import NexusState
from firecrawl import FirecrawlApp
from typing import Dict, Any, List, Optional
import json
from typing_extensions import TypedDict
from pydantic import BaseModel, Field



class RealEstateState(TypedDict):
    original_prompt: str
    search_criteria: dict
    raw_properties: List[dict]
    market_analysis: str
    property_valuations: str
    final_report: str

class SearchCriteria(BaseModel):
    city: str = Field(description="The city to search in")
    state: str = Field(description="The state abbreviation (e.g., CA, NY)")
    budget_range: str = Field(description="The budget range mentioned, or 'Any'")
    property_type: str = Field(description="House, Condo, Townhouse, Apartment, or 'Any'")
    bedrooms: str = Field(description="Number of bedrooms, or 'Any'")
    bathrooms: str = Field(description="Number of bathrooms, or 'Any'")
    special_features: str = Field(description="Any specific features requested")
    websites: List[str] = Field(description="List of sites to search. Choose from: Zillow, Realtor.com, Trulia, Homes.com")

class PropertyDetails(BaseModel):
    address: str
    price: Optional[str]
    bedrooms: Optional[str]
    bathrooms: Optional[str]
    square_feet: Optional[str]
    property_type: Optional[str]
    description: Optional[str]
    listing_url: Optional[str]

class PropertyListing(BaseModel):
    properties: List[PropertyDetails]
    total_count: int


llm = ChatOpenAI(model="gpt-4o", temperature=0.2, max_tokens = 800, api_key = os.getenv('OPENAI_API_KEY'), base_url = "https://models.inference.ai.azure.com") 

def extract_criteria_node(state: RealEstateState):
    """Extracts structured search parameters from the user's free-form prompt."""
    structured_llm = llm.with_structured_output(SearchCriteria)
    
    prompt = f"Extract real estate search criteria from this request: {state['original_prompt']}\nIf websites aren't specified, default to ['Zillow', 'Realtor.com']."
    criteria = structured_llm.invoke([HumanMessage(content=prompt)])
    
    return {"search_criteria": criteria.dict()}

def firecrawl_search_node(state: RealEstateState):
    """Uses Firecrawl to scrape real estate websites based on the extracted criteria."""
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY", "")
    if not firecrawl_api_key:
        return {"raw_properties": [], "final_report": "Error: FIRECRAWL_API_KEY is not set."}
        
    app = FirecrawlApp(api_key=firecrawl_api_key)
    crit = state["search_criteria"]
    
    # Format URL parameters
    city_fmt = crit['city'].replace(' ', '-').lower()
    state_up = crit['state'].upper()
    state_low = crit['state'].lower()
    city_tru = crit['city'].replace(' ', '_')
    
    search_urls = {
        "Zillow": f"https://www.zillow.com/homes/for_sale/{city_fmt}-{state_up}/",
        "Realtor.com": f"https://www.realtor.com/realestateandhomes-search/{city_fmt}_{state_up}/pg-1",
        "Trulia": f"https://www.trulia.com/{state_up}/{city_tru}/",
        "Homes.com": f"https://www.homes.com/homes-for-sale/{city_fmt}-{state_low}/"
    }
    
    urls_to_search = [url for site, url in search_urls.items() if site in crit['websites']]
    if not urls_to_search:
        urls_to_search = [search_urls["Zillow"]] # Fallback
        
    prompt = f"""Extract ALL property listings. Budget: {crit['budget_range']}, Beds: {crit['bedrooms']}, Type: {crit['property_type']}"""
    
    try:
        raw_response = app.extract(
            urls_to_search,
            prompt=prompt,
            schema=PropertyListing.model_json_schema()
        )
        
        # Parse response based on Firecrawl's return structure
        if hasattr(raw_response, 'success') and raw_response.success:
            properties = raw_response.data.get('properties', [])
        elif isinstance(raw_response, dict) and raw_response.get('success'):
            properties = raw_response['data'].get('properties', [])
        else:
            properties = []
            
        return {"raw_properties": properties}
        
    except Exception as e:
        print(f"Firecrawl Error: {e}")
        return {"raw_properties": []}

def market_analysis_node(state: RealEstateState):
    """Generates a brief market analysis of the scraped properties."""
    if not state.get("raw_properties"):
        return {"market_analysis": "No properties found to analyze."}
        
    prompt = (
        f"Provide CONCISE market analysis for these {len(state['raw_properties'])} properties in "
        f"{state['search_criteria']['city']}. Focus on market conditions, neighborhoods, and investment outlook."
    )
    
    response = llm.invoke([
        SystemMessage(content="You are a Market Analysis Agent. Keep it under 100 words per section."),
        HumanMessage(content=prompt)
    ])
    
    return {"market_analysis": response.content}

def valuation_node(state: RealEstateState):
    """Evaluates the investment potential of each scraped property."""
    if not state.get("raw_properties"):
        return {"property_valuations": "No valuations available."}
        
    properties_json = json.dumps(state['raw_properties'][:5], indent=2) # Limit to top 5 to save tokens
    
    prompt = (
        f"Evaluate these properties based on a budget of {state['search_criteria']['budget_range']}:\n{properties_json}\n"
        "Format as:\n**Property [Address]**\n- Value: [Fair/Over/Under]\n- Potential: [High/Medium/Low]\n- Recommendation: [Insight]"
    )
    
    response = llm.invoke([
        SystemMessage(content="You are a Property Valuation Agent. Be extremely concise."),
        HumanMessage(content=prompt)
    ])
    
    return {"property_valuations": response.content}

def synthesize_node(state: RealEstateState):
    """Compiles the search results, market analysis, and valuations into a final Markdown report."""
    if not state.get("raw_properties"):
        return {"final_report": "No properties found matching your criteria."}
        
    report = f"### Real Estate Intel for {state['search_criteria']['city']}, {state['search_criteria']['state']}\n\n"
    report += f"**Market Analysis:**\n{state['market_analysis']}\n\n"
    report += f"**Valuations (Top Properties):**\n{state['property_valuations']}\n\n"
    report += "**Raw Listings Summary:**\n"
    
    for prop in state['raw_properties'][:5]:
        report += f"- {prop.get('address')} | {prop.get('price')} | {prop.get('bedrooms')} Bed, {prop.get('bathrooms')} Bath\n"
        if prop.get('listing_url'):
            report += f"  [Link]({prop.get('listing_url')})\n"
            
    return {"final_report": report}

re_graph = StateGraph(RealEstateState)

re_graph.add_node("extract_criteria", extract_criteria_node)
re_graph.add_node("firecrawl_search", firecrawl_search_node)
re_graph.add_node("market_analysis", market_analysis_node)
re_graph.add_node("valuation", valuation_node)
re_graph.add_node("synthesize", synthesize_node)

re_graph.add_edge(START, "extract_criteria")
re_graph.add_edge("extract_criteria", "firecrawl_search")
re_graph.add_edge("firecrawl_search", "market_analysis")
re_graph.add_edge("market_analysis", "valuation")
re_graph.add_edge("valuation", "synthesize")
re_graph.add_edge("synthesize", END)

real_estate_team = re_graph.compile()


def real_estate_domain_node(state: NexusState) -> Dict[str, Any]:
    """Outer node that plugs into the main Nexus application."""
    initial_re_state = {
        "original_prompt": state["user_prompt"]
    }
    
    final_re_state = real_estate_team.invoke(initial_re_state)
    
    return {
        "domain_insights": {
            "real_estate_intel": final_re_state["final_report"]
        }
    }