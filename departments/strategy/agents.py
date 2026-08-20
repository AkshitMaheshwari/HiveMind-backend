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

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        # Deploy all 4 specialist analysts in parallel for comprehensive strategy requests
        required = ["market", "competitor", "financial_model", "swot", "business_plan", "pitch_deck"]
        return AgentOutput(
            agent_name=self.name, department=self.department, success=True,
            content="Deploying parallel strategy specialists (Market, Competitors, Financial Sandbox, SWOT, Pitch Deck).",
            metadata={"required_agents": required, "reasoning": "Full strategic analysis requested."}
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
            query = f"Market size TAM SAM SOM statistics trends: {task[:200]}"
            raw_research = await asyncio.to_thread(web_search, query, max_results=4)

            prompt = f"""Strategy task: {task}

Raw market data gathered:
{raw_research[:3500]}

Synthesize this into a concise, structured market analysis covering:
1. TAM, SAM, and SOM estimates (distinguish verified data vs assumptions)
2. Growth rate and trajectory (CAGR)
3. Key customer segments & demographics
4. Critical industry tailwinds and trends"""

            analysis = await self._ainvoke(prompt)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=analysis, metadata={"raw_research": raw_research[:300]}
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
    Financial parameters are derived from the actual task context via LLM
    to prevent domain drift (e.g., Indian EdTech B2C getting hardcoded
    enterprise SaaS defaults that led to the 'Project Horizon' hallucination).
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

        # ── Step 1: Extract domain-aware parameters from context via LLM ─────────
        # This prevents the hardcoded enterprise SaaS defaults from causing
        # domain drift when the actual task is B2C EdTech, HealthTech, etc.
        params_prompt = f"""You are a financial analyst. Given the business context below, extract realistic financial model parameters.

Business task: {task[:500]}

Market context: {market_data[:800] if market_data else "Not provided"}

Return a JSON object with ONLY these keys (no extra text, just valid JSON):
{{
  "year1_customers": <integer, realistic starting user/customer count>,
  "monthly_growth_rate": <float 0.05-0.30, monthly growth rate>,
  "arpu_monthly": <float, average revenue per user per month in USD>,
  "churn_monthly": <float 0.01-0.10, monthly churn rate>,
  "cac": <float, customer acquisition cost in USD>,
  "gross_margin": <float 0.50-0.90, gross margin ratio>,
  "operating_costs_monthly": <float, monthly fixed operating costs in USD>,
  "currency_label": <string, e.g. "INR" or "USD" or "GBP">
}}

IMPORTANT: Match the scale to the business type. 
- Indian B2C EdTech (students): arpu_monthly ~3-10, year1_customers 5000-20000, cac 5-15
- Indian B2B SaaS: arpu_monthly 50-200, year1_customers 20-100, cac 200-800
- Global Enterprise SaaS: arpu_monthly 500-5000, year1_customers 10-50, cac 5000-20000
- Consumer App (free with premium): arpu_monthly 2-8, year1_customers 10000-100000"""

        try:
            params_raw = await self._ainvoke(params_prompt)
            import json as _json
            # Extract JSON from response
            json_start = params_raw.find('{')
            json_end = params_raw.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                params = _json.loads(params_raw[json_start:json_end])
            else:
                raise ValueError("No JSON found in params response")
        except Exception as e:
            # Fallback to generic consumer/SMB defaults if LLM extraction fails
            params = {
                "year1_customers": 500,
                "monthly_growth_rate": 0.12,
                "arpu_monthly": 15,
                "churn_monthly": 0.05,
                "cac": 50,
                "gross_margin": 0.75,
                "operating_costs_monthly": 15000,
                "currency_label": "USD",
            }

        # ── Step 2: Run financial model with domain-extracted parameters ──────────
        code = f"""
import json

assumptions = {{
    "year1_customers": {int(params.get("year1_customers", 500))},
    "monthly_growth_rate": {float(params.get("monthly_growth_rate", 0.12))},
    "arpu_monthly": {float(params.get("arpu_monthly", 15))},
    "churn_monthly": {float(params.get("churn_monthly", 0.05))},
    "cac": {float(params.get("cac", 50))},
    "gross_margin": {float(params.get("gross_margin", 0.75))},
    "operating_costs_monthly": {float(params.get("operating_costs_monthly", 15000))},
    "currency_label": "{params.get('currency_label', 'USD')}"
}}

results = []
customers = assumptions["year1_customers"]
for month in range(1, 37):  # 36 months = 3 years
    revenue = customers * assumptions["arpu_monthly"]
    gross_profit = revenue * assumptions["gross_margin"]
    operating_pnl = gross_profit - assumptions["operating_costs_monthly"]
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
            "monthly_operating_pnl": round(operating_pnl, 0),
        }})

ltv = (assumptions["arpu_monthly"] * assumptions["gross_margin"]) / assumptions["churn_monthly"]
ltv_cac = ltv / assumptions["cac"]
payback = assumptions["cac"] / (assumptions["arpu_monthly"] * assumptions["gross_margin"])
break_even = assumptions["operating_costs_monthly"] / (assumptions["arpu_monthly"] * assumptions["gross_margin"])

output = {{
    "projections": results,
    "unit_economics": {{
        "LTV": round(ltv, 0),
        "CAC": assumptions["cac"],
        "LTV_CAC_ratio": round(ltv_cac, 2),
        "payback_months": round(payback, 1),
        "currency": assumptions["currency_label"],
    }},
    "break_even_customers": round(break_even, 0),
    "assumptions": assumptions,
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

Financial model results (domain-calibrated parameters):
{json.dumps(model_data, indent=2, default=str)[:3000]}

Market context:
{market_data[:1000]}

Interpret these financial projections in the context of the actual business described.
Highlight: Year 3 revenue potential, LTV:CAC health, break-even timeline, and key financial risks.
Use the correct currency ({params.get('currency_label', 'USD')}) throughout."""

            narrative = await self._ainvoke(prompt)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=narrative, metadata={"model_data": model_data, "params_used": params}
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
Conduct a thorough, evidence-based SWOT analysis.
Structure with:
### Strengths (Internal)
### Weaknesses (Internal)
### Opportunities (External)
### Threats (External)
### Strategic Synthesis & Risk Takeaways"""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        prompt = f"""Strategy Task: {task}

Conduct a crisp, evidence-based SWOT analysis for this venture."""
        try:
            swot_md = await self._ainvoke(prompt)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=swot_md,
                metadata={"swot": {"summary": swot_md}}
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
    system_prompt = """You are the Lead Strategy Partner synthesizing a complete, investor-grade strategy package.
Combine all research, analysis, financial projections, unit economics, SWOT, and strategic dimensions into a cohesive, executive-ready strategy document.
Ensure all requested dimensions are fully detailed:
- TAM, SAM, and SOM (distinguishing verified data vs assumptions)
- Competitive landscape & whitespace
- Evidence-based SWOT analysis
- 3-Year financial projections & profitability
- Unit economics (CAC, LTV, Gross Margin, Payback Period)
- Go-to-market (GTM) & product differentiation strategy
- Key risks & mitigation playbook
- 3-Year strategic execution roadmap
- 9-Slide investor pitch deck (with headlines, bullets, and speaker notes if requested)

Format with clean Markdown, tables, and consulting-grade executive structure."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        ctx_summary = {
            "market_research": (context.get("market_research") or "")[:2500],
            "competitor_data": (context.get("competitor_data") or "")[:2500],
            "financial_model": (context.get("financial_model") or "")[:2500],
            "swot_analysis": context.get("swot_analysis", {}),
            "business_plan": (context.get("business_plan") or "")[:1500],
            "pitch_deck": (context.get("pitch_deck") or "")[:1000],
        }

        prompt = f"""Strategy request: {task}

All specialist agent research & model outputs:
{json.dumps(ctx_summary, indent=2, default=str)[:8000]}

Synthesize everything into a comprehensive, executive-ready master strategy report covering all requested strategic pillars, projections, unit economics, GTM, and the pitch deck."""

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
