"""
Financial Department — Worker Agents
- FinancialRouterAgent: Classifies the intent and identifies which agents to run.
- MarketDataAgent: Fetches current price, OHLCV, market cap, and trading volume.
- FundamentalAnalysisAgent: Analyzes company financial health.
- TechnicalAnalysisAgent: Computes technical indicators via code execution.
- NewsSentimentAgent: Pulls recent news and summarizes tone.
- PortfolioAnalystAgent: Analyzes user's stated holdings via code execution.
- ComparativeAnalysisAgent: Compares 2+ tickers using Fundamental and Market Data.
- SynthesizerAgent: Combines results, formats charts JSON, and adds disclaimer.
"""
import asyncio
import json
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from shared.base_agent import ProductionAgent, AgentOutput
from shared.tools import web_search, execute_code

# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class FinancialRoute(BaseModel):
    required_agents: List[str] = Field(description="List of agents to run. Options: 'market_data', 'fundamental', 'technical', 'news', 'portfolio', 'comparative'.")
    tickers: List[str] = Field(description="List of stock tickers mentioned in the query.")
    reasoning: str = Field(description="Why these agents and tickers were selected.")

class MarketData(BaseModel):
    current_price: float = Field(description="Current stock price")
    historical_data: List[Dict[str, Any]] = Field(description="List of OHLCV data points (date, open, high, low, close, volume)")
    market_cap: float = Field(description="Market capitalization")
    volume: float = Field(description="Trading volume")

class FundamentalData(BaseModel):
    revenue: float = Field(description="Total revenue")
    net_income: float = Field(description="Net income")
    pe_ratio: float = Field(description="Price to Earnings ratio")
    debt_to_equity: float = Field(description="Debt to Equity ratio")
    summary: str = Field(description="Summary of financial health")

class TechnicalData(BaseModel):
    indicators: Dict[str, Any] = Field(description="Technical indicators (SMA, RSI, MACD)")
    trend_analysis: str = Field(description="Analysis of the trend and momentum")

class NewsSentiment(BaseModel):
    articles: List[Dict[str, str]] = Field(description="List of news articles with source, title, and summary")
    overall_sentiment: str = Field(description="Overall sentiment (positive, negative, mixed)")
    summary: str = Field(description="Summary of the news tone")

class PortfolioAnalysis(BaseModel):
    diversification_score: float = Field(description="Diversification score (0-100)")
    risk_metrics: Dict[str, Any] = Field(description="Volatility, Beta, etc.")
    summary: str = Field(description="Summary of portfolio health")

class ComparisonData(BaseModel):
    comparison_table: Dict[str, Any] = Field(description="Side-by-side comparison metrics")
    winner: str = Field(description="Which ticker appears stronger and why")

class SynthesizedReport(BaseModel):
    report_text: str = Field(description="The complete written investment report (markdown)")
    charts_json: List[Dict[str, Any]] = Field(description="JSON data for the frontend to render interactive charts (e.g. historical price data)")
    disclaimer: str = Field(description="Mandatory disclaimer stating this is not financial advice")


# ─── FinancialRouterAgent ──────────────────────────────────────────────────────

class FinancialRouterAgent(ProductionAgent):
    name = "FinancialRouterAgent"
    department = "financial"
    system_prompt = """You are the Financial Routing Director.
Analyze the user's query to determine which financial analysis agents are needed and extract any stock tickers.

Agent capabilities:
- 'market_data': Raw prices, historical data, volume.
- 'fundamental': Financial health, revenue, P/E ratio, debt.
- 'technical': Charts, momentum, RSI, MACD, overbought/oversold.
- 'news': Market sentiment, latest news.
- 'portfolio': Analysis of user's portfolio/holdings diversification.
- 'comparative': Comparing two or more tickers.

Always extract stock tickers explicitly (e.g., AAPL, TSLA, MSFT). If the query implies comparison between tickers, include 'comparative'.
If the query mentions holdings or diversification, include 'portfolio'.
"""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            result: FinancialRoute = await self._ainvoke_structured(
                f"Analyze this query and route it:\n\n{task}",
                FinancialRoute
            )
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.reasoning,
                metadata={
                    "required_agents": result.required_agents,
                    "tickers": result.tickers,
                    "reasoning": result.reasoning
                }
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content=f"Routing failed: {e}",
                metadata={"required_agents": ["market_data"], "tickers": []},
                error=str(e)
            )


# ─── MarketDataAgent ──────────────────────────────────────────────────────────

class MarketDataAgent(ProductionAgent):
    name = "MarketDataAgent"
    department = "financial"
    system_prompt = """You are a Market Data Analyst. Write a python script using yfinance to fetch raw price data (OHLCV), market cap, and volume for the given tickers. Output the requested structured data."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        tickers = context.get("tickers", []) if context else []
        if not tickers:
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content="No tickers provided.", metadata={"data": {}})
        
        # In a real MVP, we could just run yfinance natively here instead of LLM generating code, 
        # but let's do it natively using yfinance directly.
        import yfinance as yf
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="1mo")
                info = stock.info
                
                historical_data = []
                for date, row in hist.iterrows():
                    historical_data.append({
                        "date": date.strftime("%Y-%m-%d"),
                        "open": float(row["Open"]),
                        "high": float(row["High"]),
                        "low": float(row["Low"]),
                        "close": float(row["Close"]),
                        "volume": float(row["Volume"])
                    })
                    
                data[ticker] = {
                    "current_price": float(info.get("currentPrice", hist["Close"].iloc[-1] if not hist.empty else 0)),
                    "historical_data": historical_data,
                    "market_cap": float(info.get("marketCap", 0)),
                    "volume": float(info.get("volume", 0))
                }
            except Exception as e:
                data[ticker] = {"error": str(e)}

        return AgentOutput(
            agent_name=self.name,
            department=self.department,
            success=True,
            content="Market data fetched successfully.",
            metadata={"data": data}
        )


# ─── FundamentalAnalysisAgent ─────────────────────────────────────────────────

class FundamentalAnalysisAgent(ProductionAgent):
    name = "FundamentalAnalysisAgent"
    department = "financial"
    system_prompt = """You are a Fundamental Analyst. Analyze the financial health of the given companies."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        tickers = context.get("tickers", []) if context else []
        if not tickers:
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content="No tickers provided.", metadata={"data": {}})
        
        import yfinance as yf
        data = {}
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Fetch basic fundamentals
                fundamentals = {
                    "revenue": float(info.get("totalRevenue", 0)),
                    "net_income": float(info.get("netIncomeToCommon", 0)),
                    "pe_ratio": float(info.get("trailingPE", 0)),
                    "debt_to_equity": float(info.get("debtToEquity", 0)),
                    "profit_margin": float(info.get("profitMargins", 0)),
                    "summary": f"Analyzed {ticker} fundamentals."
                }
                data[ticker] = fundamentals
            except Exception as e:
                data[ticker] = {"error": str(e)}

        return AgentOutput(
            agent_name=self.name,
            department=self.department,
            success=True,
            content="Fundamental data fetched successfully.",
            metadata={"data": data}
        )


# ─── TechnicalAnalysisAgent ───────────────────────────────────────────────────

class TechnicalAnalysisAgent(ProductionAgent):
    name = "TechnicalAnalysisAgent"
    department = "financial"
    system_prompt = """You are a Technical Analyst. Analyze market trends, momentum, and indicators.
Use the execute_code tool to compute SMA, RSI, etc."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        tickers = context.get("tickers", []) if context else []
        if not tickers:
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content="No tickers provided.", metadata={"data": {}})

        code = f"""
import yfinance as yf
import pandas as pd
import json

def calculate_ta(tickers):
    results = {{}}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period="3mo")
            if df.empty:
                continue
            
            # Simple Moving Averages
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # RSI calculation
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            latest = df.iloc[-1]
            results[ticker] = {{
                "SMA_20": float(latest['SMA_20']) if not pd.isna(latest['SMA_20']) else None,
                "SMA_50": float(latest['SMA_50']) if not pd.isna(latest['SMA_50']) else None,
                "RSI": float(latest['RSI']) if not pd.isna(latest['RSI']) else None,
            }}
        except Exception as e:
            results[ticker] = {{"error": str(e)}}
    
    print(json.dumps(results))

calculate_ta({tickers})
"""
        try:
            output = await asyncio.to_thread(execute_code, code)
            try:
                # find json in output
                data_str = output[output.find('{'):output.rfind('}')+1]
                data = json.loads(data_str)
            except:
                data = {"raw_output": output}
                
            prompt = f"Analyze these technical indicators for {tickers}:\n{data}\nProvide a brief trend analysis."
            result: TechnicalData = await self._ainvoke_structured(prompt, TechnicalData)
            
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.trend_analysis,
                metadata={"data": data, "trend_analysis": result.trend_analysis}
            )
        except Exception as e:
             return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e), metadata={"data": {}}
            )


# ─── NewsSentimentAgent ───────────────────────────────────────────────────────

class NewsSentimentAgent(ProductionAgent):
    name = "NewsSentimentAgent"
    department = "financial"
    system_prompt = """You are a Financial News Analyst. Use web search to find recent news on the tickers and summarize the sentiment (positive, negative, mixed)."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        tickers = context.get("tickers", []) if context else []
        query = f"Latest financial news for {' '.join(tickers)}" if tickers else task
        
        try:
            raw_news = await asyncio.to_thread(web_search, query, max_results=5)
            prompt = f"Analyze the sentiment of this news for {tickers}:\n\n{raw_news}"
            result: NewsSentiment = await self._ainvoke_structured(prompt, NewsSentiment)
            
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.summary,
                metadata={
                    "articles": result.articles,
                    "overall_sentiment": result.overall_sentiment,
                    "summary": result.summary
                }
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e), metadata={"data": {}}
            )


# ─── PortfolioAnalystAgent ────────────────────────────────────────────────────

class PortfolioAnalystAgent(ProductionAgent):
    name = "PortfolioAnalystAgent"
    department = "financial"
    system_prompt = """You are a Portfolio Analyst. Analyze the user's holdings to calculate diversification and risk metrics.
You will write and execute python code to compute these metrics."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        prompt = f"Extract the user's stated portfolio holdings from this query: {task}\n\nWrite a python script to calculate portfolio diversification score (0-100) and risk metrics (e.g. historical volatility). Output JSON."
        try:
            # We will just ask the LLM to provide the analysis directly, relying on its internal reasoning, or explicitly executing code.
            # For simplicity, we just use LLM structured output for MVP, assuming holdings are simple.
            result: PortfolioAnalysis = await self._ainvoke_structured(
                f"Analyze the portfolio in this query and provide risk metrics: {task}", PortfolioAnalysis
            )
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.summary,
                metadata={"data": result.model_dump()}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e), metadata={"data": {}}
            )


# ─── ComparativeAnalysisAgent ─────────────────────────────────────────────────

class ComparativeAnalysisAgent(ProductionAgent):
    name = "ComparativeAnalysisAgent"
    department = "financial"
    system_prompt = """You are a Comparative Analyst. You receive fundamental and market data for multiple tickers and provide a side-by-side comparison."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        market_data = context.get("market_data", {}) if context else {}
        fundamental_data = context.get("fundamental_data", {}) if context else {}
        
        prompt = f"Compare these companies based on the following data:\nMarket Data: {market_data}\nFundamental Data: {fundamental_data}\n\nTask: {task}"
        
        try:
            result: ComparisonData = await self._ainvoke_structured(prompt, ComparisonData)
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.winner,
                metadata={"comparison_data": result.model_dump()}
            )
        except Exception as e:
             return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e), metadata={"data": {}}
            )


# ─── SynthesizerAgent ─────────────────────────────────────────────────────────

class SynthesizerAgent(ProductionAgent):
    name = "SynthesizerAgent"
    department = "financial"
    system_prompt = """You are the Lead Investment Strategist.
Synthesize the reports from the specialist agents into a final, coherent investment report in Markdown.
Mandatory: You must explicitly include a disclaimer stating "This is not financial advice."

Also, format numerical price/volume data into a `charts_json` array for the frontend to render.
The charts_json should contain objects like: 
{"type": "line", "title": "AAPL Price", "data": [{"date": "2023-01-01", "value": 150}, ...]}
"""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        ctx_dump = json.dumps({
            "market_data": context.get("market_data", {}),
            "fundamental_data": context.get("fundamental_data", {}),
            "technical_data": context.get("technical_data", {}),
            "news_sentiment": context.get("news_sentiment", {}),
            "portfolio_analysis": context.get("portfolio_analysis", {}),
            "comparison_data": context.get("comparison_data", {})
        }, default=str)
        
        prompt = f"Original Query: {task}\n\nAgent Outputs:\n{ctx_dump[:8000]}\n\nSynthesize this into a final report and extract chart JSON data."
        
        try:
            result: SynthesizedReport = await self._ainvoke_structured(prompt, SynthesizedReport)
            
            # Enforce disclaimer
            disclaimer = result.disclaimer if result.disclaimer else "**Disclaimer: This is not financial advice.**"
            if "not financial advice" not in result.report_text.lower():
                final_text = f"{result.report_text}\n\n{disclaimer}"
            else:
                final_text = result.report_text
                
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=final_text,
                metadata={"charts_json": result.charts_json, "disclaimer": disclaimer}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content=f"Synthesis failed: {e}\n\n**Disclaimer: This is not financial advice.**", 
                error=str(e), metadata={"charts_json": []}
            )
