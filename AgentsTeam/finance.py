import yfinance as yf
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool, tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
# from langgraph.prebuilt import create_react_agent

from state import NexusState
import os
from dotenv import load_dotenv

load_dotenv()

web_search_tool = DuckDuckGoSearchRun()

@tool
def get_current_stock_price(ticker: str) -> str:
    """ Gets the current stock price from Yahoo Finanace. """
    stock = yf.Ticker(ticker)
    price = stock.history(period ="1d")['Close'].iloc[-1]
    return f"The current price of {ticker} is ${price:.2f}"

@tool
def get_company_info(ticker: str) -> str:
    """ Gets general company info and business summary. """
    stock = yf.Ticker(ticker)
    return stock.info.get("longBusinessSummary", "Info not found.")

@tool
def get_company_news(ticker: str)->str:
    """ Gets the latest financial news for a ticker. """
    stock = yf.Ticker(ticker)
    news = stock.news[:3]
    return "\n".join([f"-{n['title']}: {n['link']}" for n in news])

finance_tools = [get_current_stock_price, get_company_info, get_company_news]

llm = ChatOpenAI(model_name = "gpt-4.1",temperature=0.1, max_tokens=500,api_key = os.getenv("OPENAI_API_KEY"),base_url = "https://models.inference.ai.azure.com")

web_agent = create_agent(
    model = llm,
    tools = [web_search_tool],
    system_prompt = "You are a web agent search the web for information and answer questions based on the search results.",
)

finance_agent = create_agent(
    model = llm,
    tools = finance_tools,
    system_prompt = "You are a financial agent that can provide information about stocks, companies, and financial news. Always use tables to display the data",
)


def finance_domain_node(state:NexusState) -> Dict[str,Any]:
    """
    This node acts as the 'Team' wrapper. It recieves the user prompt from the
    NexusState and decides whether to use the Web Agent, Finance Agent or both.
    """

    prompt = state['user_prompt']

    web_response = web_agent.invoke({"messages": [HumanMessage(content=f"find recent market trends related to: {prompt}")]})
    web_context = web_response["messages"][-1].content

    finance_response = finance_agent.invoke({"messages": [
        SystemMessage(content=f"context from web: {web_context}"),
        HumanMessage(content = f"Execute financial analysis for: {prompt}")
    ]})
    finance_data = finance_response["messages"][-1].content

    return {
        "domain_insights":{
            "finance_and_web_intel": f"WEB RESEARCH:\n {web_context}\n\nFINANCIAL DATA: \n{finance_data}"
        }
    }

    
