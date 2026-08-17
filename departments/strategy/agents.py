"""
Business Strategy Department — Worker Agents
- StrategyRouterAgent:      Classifies the request and selects needed agents
- MarketAnalystAgent:       Calls Research dept directly (inter-dept pattern)
- CompetitorAnalystAgent:   Web search for competitive landscape
- FinancialModelerAgent:    Uses execute_code for projections (shared Code sandbox)
- SWOTAgent:                Structured SWOT from all gathered inputs
- BusinessPlanAgent:        Executive summary, GTM, operations narrative
- PitchDeckAgent:           Slide-by-slide investor pitch deck content
- StrategySynthesizerAgent: Final combined strategy report
"""
import asyncio
import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from shared.base_agent import ProductionAgent, AgentOutput
from shared.tools import web_search, execute_code


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class StrategyRoute(BaseModel):
    required_agents: List[str] = Field(
        description="Agents to run. Options: 'market', 'competitor', 'financial_model', 'swot', 'business_plan', 'pitch_deck'."
    )
    reasoning: str = Field(description="Why these agents were selected.")


class SWOTOutput(BaseModel):
    strengths: List[str] = Field(description="Internal strengths")
    weaknesses: List[str] = Field(description="Internal weaknesses")
    opportunities: List[str] = Field(description="External opportunities")
    threats: List[str] = Field(description="External threats")
    summary: str = Field(description="SWOT narrative summary")


class PitchDeckSlide(BaseModel):
    slide_number: int
    title: str
    content: str
    speaker_notes: str


class PitchDeckOutput(BaseModel):
    slides: List[PitchDeckSlide] = Field(description="All slides in order")
    pitch_summary: str = Field(description="One-paragraph pitch summary")


# ─── StrategyRouterAgent ─────────────────────────────────────────────────────

class StrategyRouterAgent(ProductionAgent):
    name = "StrategyRouterAgent"
    department = "strategy"
    system_prompt = """You are the Strategy Director at a top consulting firm.
Analyze the user's strategy request and determine which specialist agents to deploy.

Agent options:
- 'market': Market size, trends, TAM/SAM/SOM analysis
- 'competitor': Competitive landscape, positioning, market share
- 'financial_model': Revenue projections, unit economics, break-even
- 'swot': Strengths, weaknesses, opportunities, threats analysis
- 'business_plan': Executive summary, GTM strategy, operations
- 'pitch_deck': Investor pitch deck (include only if explicitly requested)

Select the minimum set needed to answer the question well."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            result: StrategyRoute = await self._ainvoke_structured(
                f"Strategy request: {task}\n\nSelect agents to deploy.", StrategyRoute
            )
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=result.reasoning,
                metadata={"required_agents": result.required_agents, "reasoning": result.reasoning}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e),
                metadata={"required_agents": ["market", "competitor", "swot", "business_plan"]}
            )


# ─── MarketAnalystAgent ───────────────────────────────────────────────────────

class MarketAnalystAgent(ProductionAgent):
    """
    HIVE MIND PATTERN: This agent calls the Research Department directly
    via inter_dept.call_research_dept() — bypassing the CEO entirely.
    Strategy dept gets full Research dept capability without duplicating any tools.
    """
    name = "MarketAnalystAgent"
    department = "strategy"
    system_prompt = """You are a Market Research Analyst at a strategy consulting firm.
Analyze market size, growth trends, TAM/SAM/SOM, customer segments, and market dynamics.
Use the research data provided to quantify the market opportunity."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            # INTER-DEPT CALL: Strategy calls Research directly
            from shared.inter_dept import call_research_dept
            api_keys = (context or {}).get("api_keys")
            selected_model = (context or {}).get("selected_model")
            user_id = (context or {}).get("user_id")

            research_task = (
                f"Market research for: {task}. "
                f"Find: total addressable market size, growth rate (CAGR), "
                f"key market segments, major trends, and customer demographics."
            )

            raw_research = await call_research_dept(
                task=research_task,
                api_keys=api_keys,
                selected_model=selected_model,
                user_id=user_id,
            )

            # Synthesize the research into market analysis
            prompt = f"""Strategy task: {task}

Raw market research (from Research Department):
{raw_research[:4000]}

Synthesize this into a structured market analysis covering:
1. Market size and TAM/SAM/SOM estimates
2. Growth rate and trajectory (CAGR)
3. Key customer segments
4. Critical market trends
5. Market timing and entry window"""

            analysis = await self._ainvoke(prompt)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=analysis, metadata={"raw_research": raw_research[:500]}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content=f"Market analysis unavailable: {e}", error=str(e), metadata={}
            )


# ─── CompetitorAnalystAgent ───────────────────────────────────────────────────

class CompetitorAnalystAgent(ProductionAgent):
    name = "CompetitorAnalystAgent"
    department = "strategy"
    system_prompt = """You are a Competitive Intelligence Analyst.
Map the competitive landscape: who are the key players, how do they position themselves,
what are their strengths/weaknesses, pricing strategies, and market share?
Identify white space opportunities and differentiation levers."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            query = f"Competitors and competitive landscape for: {task}"
            raw_results = await asyncio.to_thread(web_search, query, max_results=6)

            prompt = f"""Strategy task: {task}

Competitive intelligence gathered:
{raw_results[:4000]}

Provide a competitive analysis covering:
1. Top 3-5 competitors with their positioning
2. Competitive advantages of each
3. Pricing landscape
4. Market gaps and underserved segments
5. Recommended differentiation strategy"""

            analysis = await self._ainvoke(prompt)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=analysis, metadata={}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e), metadata={}
            )


# ─── FinancialModelerAgent ────────────────────────────────────────────────────

class FinancialModelerAgent(ProductionAgent):
    """
    Uses the shared Code sandbox (execute_code) for financial projections.
    This is another key hive-mind reuse: Code dept's sandbox serves Strategy dept.
    """
    name = "FinancialModelerAgent"
    department = "strategy"
    system_prompt = """You are a Financial Modeling expert (ex-Goldman Sachs, McKinsey).
Build financial projections using Python: 3-year revenue model, unit economics,
CAC/LTV ratio, gross margin, and break-even analysis.
Use realistic assumptions based on the market context provided."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        market_data = context.get("market_research", "")

        code = f"""
import json

# Simple 3-year SaaS financial model
assumptions = {{
    "year1_customers": 50,
    "monthly_growth_rate": 0.15,  # 15% MoM growth
    "arpu_monthly": 200,          # Average Revenue Per User
    "churn_monthly": 0.05,        # 5% monthly churn
    "cac": 500,                   # Customer Acquisition Cost
    "gross_margin": 0.75,         # 75% gross margin typical SaaS
    "operating_costs_monthly": 50000  # Fixed costs
}}

results = []
customers = assumptions["year1_customers"]
for month in range(1, 37):  # 36 months = 3 years
    revenue = customers * assumptions["arpu_monthly"]
    gross_profit = revenue * assumptions["gross_margin"]
    operating_loss = gross_profit - assumptions["operating_costs_monthly"]
    new_customers = int(customers * assumptions["monthly_growth_rate"])
    churned = int(customers * assumptions["churn_monthly"])
    customers = customers + new_customers - churned
    
    if month in [12, 24, 36]:
        results.append({{
            "period": f"Year {{month // 12}}",
            "monthly_revenue": round(revenue, 0),
            "annual_revenue": round(revenue * 12, 0),
            "customers": customers,
            "gross_margin_pct": assumptions["gross_margin"] * 100,
            "monthly_operating_pnl": round(operating_loss, 0)
        }})

# Unit economics
ltv = (assumptions["arpu_monthly"] * assumptions["gross_margin"]) / assumptions["churn_monthly"]
cac = assumptions["cac"]
ltv_cac_ratio = ltv / cac

break_even_customers = assumptions["operating_costs_monthly"] / (
    assumptions["arpu_monthly"] * assumptions["gross_margin"]
)

output = {{
    "projections": results,
    "unit_economics": {{
        "LTV": round(ltv, 0),
        "CAC": cac,
        "LTV_CAC_ratio": round(ltv_cac_ratio, 2),
        "payback_months": round(cac / (assumptions["arpu_monthly"] * assumptions["gross_margin"]), 1)
    }},
    "break_even_customers": round(break_even_customers, 0),
    "assumptions": assumptions
}}
print(json.dumps(output))
"""
        try:
            raw_output = await asyncio.to_thread(execute_code, code)
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            model_data = {}
            if json_start >= 0 and json_end > json_start:
                model_data = json.loads(raw_output[json_start:json_end])

            prompt = f"""Strategy context: {task}

Financial model results:
{json.dumps(model_data, indent=2, default=str)[:3000]}

Market context:
{market_data[:1000]}

Interpret these financial projections in the context of the strategy. 
Highlight: Year 3 revenue potential, LTV:CAC health, break-even timeline, and key financial risks."""

            narrative = await self._ainvoke(prompt)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=narrative, metadata={"model_data": model_data}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="Financial modeling unavailable.", error=str(e), metadata={}
            )


# ─── SWOTAgent ────────────────────────────────────────────────────────────────

class SWOTAgent(ProductionAgent):
    name = "SWOTAgent"
    department = "strategy"
    system_prompt = """You are a McKinsey-trained Strategy Consultant.
Conduct a thorough SWOT analysis based on market research, competitive intelligence, and financial data.
Be specific and evidence-based — avoid generic SWOT points."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        market = context.get("market_research", "")
        competitor = context.get("competitor_data", "")
        financial = context.get("financial_model", "")

        prompt = f"""Task: {task}

Market Research Summary:
{market[:1500]}

Competitive Landscape:
{competitor[:1500]}

Financial Model Insights:
{financial[:1000]}

Conduct a detailed SWOT analysis. Be specific and evidence-based."""

        try:
            result: SWOTOutput = await self._ainvoke_structured(prompt, SWOTOutput)
            swot_md = f"""## SWOT Analysis

**Strengths:**
{chr(10).join(f'- {s}' for s in result.strengths)}

**Weaknesses:**
{chr(10).join(f'- {w}' for w in result.weaknesses)}

**Opportunities:**
{chr(10).join(f'- {o}' for o in result.opportunities)}

**Threats:**
{chr(10).join(f'- {t}' for t in result.threats)}

{result.summary}"""

            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=swot_md,
                metadata={"swot": result.model_dump()}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e), metadata={"swot": {}}
            )


# ─── BusinessPlanAgent ────────────────────────────────────────────────────────

class BusinessPlanAgent(ProductionAgent):
    name = "BusinessPlanAgent"
    department = "strategy"
    system_prompt = """You are a senior business strategist writing an executive business plan.
Structure it clearly: Executive Summary, Market Opportunity, Product/Service, 
Go-to-Market Strategy, Operations, Financial Overview, and Risk Mitigation.
Use Markdown with clear headers. Be concrete — include specific numbers and timelines."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        prompt = f"""Strategy task: {task}

Market Research:
{context.get('market_research', 'Not available')[:2000]}

Competitive Analysis:
{context.get('competitor_data', 'Not available')[:1500]}

Financial Model:
{context.get('financial_model', 'Not available')[:1000]}

SWOT Summary:
{str(context.get('swot_analysis', {}))[:800]}

Write a comprehensive business plan covering all key sections."""

        try:
            plan = await self._ainvoke(prompt)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=plan, metadata={}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e), metadata={}
            )


# ─── PitchDeckAgent ──────────────────────────────────────────────────────────

class PitchDeckAgent(ProductionAgent):
    name = "PitchDeckAgent"
    department = "strategy"
    system_prompt = """You are a pitch deck expert who has helped startups raise Series A-C rounds.
Create a compelling investor pitch deck following the proven structure:
1. Problem, 2. Solution, 3. Market Size (TAM/SAM/SOM), 4. Product, 
5. Business Model, 6. Traction, 7. Competition, 8. Team, 9. Financial Ask & Use of Funds.

For each slide: write the headline, key bullets (max 3-4), and speaker notes.
Be specific about numbers, timelines, and the unique insight."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        prompt = f"""Strategy task: {task}

Market Research:
{context.get('market_research', '')[:2000]}

Financial Model:
{context.get('financial_model', '')[:1000]}

Competitor Data:
{context.get('competitor_data', '')[:1000]}

Create a complete 9-slide investor pitch deck with slide content and speaker notes."""

        try:
            result: PitchDeckOutput = await self._ainvoke_structured(prompt, PitchDeckOutput)
            slides_md = "\n\n---\n\n".join([
                f"## Slide {s.slide_number}: {s.title}\n\n{s.content}\n\n*Speaker notes: {s.speaker_notes}*"
                for s in result.slides
            ])
            full_deck = f"# Investor Pitch Deck\n\n**Pitch Summary:** {result.pitch_summary}\n\n---\n\n{slides_md}"
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=full_deck, metadata={"slide_count": len(result.slides)}
            )
        except Exception as e:
            # Fallback: unstructured pitch deck
            try:
                fallback = await self._ainvoke(
                    f"Create a 9-slide investor pitch deck for: {task}\n\nContext: {str(context)[:2000]}"
                )
                return AgentOutput(
                    agent_name=self.name, department=self.department, success=True,
                    content=fallback, metadata={}, error=str(e)
                )
            except Exception as e2:
                return AgentOutput(
                    agent_name=self.name, department=self.department, success=False,
                    content="", error=str(e2), metadata={}
                )


# ─── StrategySynthesizerAgent ─────────────────────────────────────────────────

class StrategySynthesizerAgent(ProductionAgent):
    name = "StrategySynthesizerAgent"
    department = "strategy"
    system_prompt = """You are the Lead Strategy Partner synthesizing a complete strategy package.
Combine all research, analysis, and recommendations into a cohesive, executive-ready strategy document.
Structure: Executive Summary → Key Findings → Strategic Recommendations → Implementation Roadmap → Financial Outlook.
Be decisive — give clear recommendations, not wishy-washy "it depends" answers.
Use Markdown with clear headers and a professional consulting tone."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        ctx_summary = {
            "market_research": (context.get("market_research") or "")[:1500],
            "competitor_data": (context.get("competitor_data") or "")[:1500],
            "financial_model": (context.get("financial_model") or "")[:1000],
            "swot_analysis": context.get("swot_analysis", {}),
            "business_plan": (context.get("business_plan") or "")[:1000],
            "pitch_deck": (context.get("pitch_deck") or "")[:500],
        }

        prompt = f"""Strategy request: {task}

All specialist agent outputs:
{json.dumps(ctx_summary, indent=2, default=str)[:6000]}

Synthesize everything into a final, executive-ready strategy document with clear recommendations."""

        try:
            synthesis = await self._ainvoke(prompt)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=synthesis, metadata={}
            )
        except Exception as e:
            # Fallback: concatenate what we have
            parts = [f"# Strategy Report: {task}\n"]
            for key, val in ctx_summary.items():
                if val:
                    parts.append(f"## {key.replace('_', ' ').title()}\n{val}\n")
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="\n".join(parts), error=str(e), metadata={}
            )
