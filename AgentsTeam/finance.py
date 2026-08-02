import yfinance as yf
from typing import Dict, Any
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools import Tool, tool
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.agents import create_agent
from state import NexusState

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

llm = ChatOpenAI(model_name = "gpt-4.1",temperature=0.1, max_tokens=500)

