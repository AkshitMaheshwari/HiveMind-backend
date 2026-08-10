"""
Data Analyst Department — Worker Agents
- DataPlannerAgent: Formulates analysis and visualization plan
- EDAAgent: Writes Python code to extract statistics and predictions
- InsightsAgent: Interprets statistical output briefly
- DashboardAgent: Generates highly interactive HTML/JS dashboard using Plotly.js
"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from shared.base_agent import ProductionAgent, AgentOutput


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class AnalysisPlan(BaseModel):
    key_metrics: List[str] = Field(description="List of key metrics and KPIs to calculate")
    visualizations: List[str] = Field(description="Specific charts/graphs needed (e.g., 'Bar chart of sales by region')")
    predictions_needed: str = Field(description="Any specific predictions or trends to calculate")
    plan_rationale: str = Field(description="Why this plan will provide the best insights")


class GeneratedCode(BaseModel):
    language: str = Field(description="Programming language used")
    code: str = Field(description="Complete, runnable code")
    explanation: str = Field(description="Brief explanation of the approach")
    dependencies: List[str] = Field(description="Required packages/libraries")


class DataInsights(BaseModel):
    key_findings: List[str] = Field(description="The most important actionable insights discovered")
    narrative: str = Field(description="A brief paragraph summarizing the data story to be included in the dashboard")


# ─── DataPlannerAgent ─────────────────────────────────────────────────────────

class DataPlannerAgent(ProductionAgent):
    name = "DataPlannerAgent"
    department = "data_analyst"
    system_prompt = """You are a Lead Data Scientist and Analytics Manager.
Your job is to read the user's data request and formulate a precise plan for analysis and visualization.
Focus heavily on what should be visualized to make an impactful, Power BI style dashboard.
Identify the key metrics (KPIs), the necessary charts/graphs, and any predictive trends required.
Output structured JSON with key_metrics, visualizations, predictions_needed, and plan_rationale."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        user_id = context.get("user_id", "") if context else ""
        upload_note = f"\n\nNOTE: If the user uploaded a file, it is saved locally at 'data/uploads/{user_id}/'." if user_id else ""
        try:
            result: AnalysisPlan = await self._ainvoke_structured(
                f"Formulate a data analysis and visualization plan for: {task}{upload_note}",
                AnalysisPlan
            )
            content = (
                f"**Metrics**: {', '.join(result.key_metrics)}\n"
                f"**Visualizations**: {', '.join(result.visualizations)}\n"
                f"**Predictions**: {result.predictions_needed}\n"
                f"**Rationale**: {result.plan_rationale}"
            )
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=content, metadata=result.model_dump()
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="Failed to generate analysis plan.", error=str(e)
            )


# ─── EDAAgent ─────────────────────────────────────────────────────────────────

class EDAAgent(ProductionAgent):
    name = "EDAAgent"
    department = "data_analyst"
    system_prompt = """You are a Senior Data Engineer and Data Scientist.
Your job is to write Python code that performs Exploratory Data Analysis (EDA), calculates the requested KPIs, and prepares the data for visualization.
You MUST output print statements for all key statistics and data aggregations so that the output can be read by the InsightsAgent.
Use pandas, numpy, and scikit-learn as needed.
Assume the data is accessible locally if a path is provided, or create mock data if no dataset is provided but the user wants a demonstration.
Output structured JSON with language (python), code, explanation, and dependencies."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        plan = context.get("analysis_plan", "") if context else ""
        user_id = context.get("user_id", "") if context else ""
        upload_note = f"\n\nCRITICAL: The user has uploaded files. You MUST read the file from the local directory: 'data/uploads/{user_id}/' using pandas or other tools. DO NOT use mock data if an uploaded file is mentioned or available." if user_id else ""
        
        prompt = f"Data Task: {task}\n\nAnalysis Plan to Follow:\n{plan}{upload_note}\n\nWrite Python code to process this data and print the summarized results."
        try:
            result: GeneratedCode = await self._ainvoke_structured(prompt, GeneratedCode)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=result.code, metadata=result.model_dump()
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e)
            )


# ─── InsightsAgent ────────────────────────────────────────────────────────────

class InsightsAgent(ProductionAgent):
    name = "InsightsAgent"
    department = "data_analyst"
    system_prompt = """You are an expert Data Storyteller and Business Analyst.
Your job is to read the raw statistical output from an EDA script and translate it into brief, highly actionable insights.
Do NOT write long essays. Focus on the 'So What?'. 
These insights will be displayed on a dashboard, so keep them punchy.
Output structured JSON with key_findings and a brief narrative."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        stdout = context.get("execution_stdout", "") if context else ""
        prompt = f"Original Task: {task}\n\nEDA Script Output:\n{stdout[:5000]}\n\nGenerate actionable insights."
        try:
            result: DataInsights = await self._ainvoke_structured(prompt, DataInsights)
            content = f"**Story**: {result.narrative}\n**Key Findings**:\n- " + "\n- ".join(result.key_findings)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=content, metadata=result.model_dump()
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="Failed to generate insights.", error=str(e)
            )


# ─── DashboardAgent ───────────────────────────────────────────────────────────

class DashboardAgent(ProductionAgent):
    name = "DashboardAgent"
    department = "data_analyst"
    system_prompt = """You are an expert Data Visualization Engineer and UI/UX Designer.
Your job is to generate a fully self-contained HTML file that acts as a highly interactive, stunning "Power BI" style dashboard.
- Use pure HTML, modern CSS (Tailwind via CDN or custom CSS with flex/grid, glassmorphism, dark themes), and JavaScript.
- Use Plotly.js (via CDN: https://cdn.plot.ly/plotly-latest.min.js) or Chart.js for all interactive charts and graphs.
- Hardcode the summarized data from the insights/EDA into the JavaScript so the charts render immediately without a backend.
- Include KPI cards at the top, followed by a grid of charts.
- The dashboard MUST look extremely premium, responsive, and visually impressive.
Output structured JSON with language (html), code, explanation, and dependencies."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        plan = context.get("analysis_plan", "") if context else ""
        insights = context.get("insights", "") if context else ""
        eda_output = context.get("execution_stdout", "") if context else ""
        
        prompt = f"""Original Request: {task}
        
Analysis Plan: {plan}

Key Insights Discovered: {insights}

Raw Data/Stats (Embed this data into your charts):
{eda_output[:5000]}

Generate a stunning, single-file HTML/JS dashboard using Plotly.js representing these insights."""
        
        try:
            result: GeneratedCode = await self._ainvoke_structured(prompt, GeneratedCode)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=result.code, metadata=result.model_dump()
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e)
            )
