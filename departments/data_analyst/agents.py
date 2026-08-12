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

class AnalysisTask(BaseModel):
    step: int
    agent: str
    action: str
    target_columns: List[str]
    title: str
    reason: str
    additional_notes: str

class AnalysisPlan(BaseModel):
    dataset_type: str
    priority: str
    summary_of_approach: str
    tasks: List[AnalysisTask]


class GeneratedCode(BaseModel):
    language: str = Field(description="Programming language used")
    code: str = Field(description="Complete, runnable code")
    explanation: str = Field(description="Brief explanation of the approach")
    dependencies: List[str] = Field(description="Required packages/libraries")


class DashboardCode(BaseModel):
    language: str = Field(description="Programming language used")
    code: str = Field(description="Complete, runnable code")
    explanation: str = Field(description="Brief explanation of the approach")
    dependencies: List[str] = Field(description="Required packages/libraries")
    charts_json: List[Dict[str, Any]] = Field(
        default=[],
        description="A list of structured JSON objects for frontend Recharts rendering. Each object MUST have a 'title', a 'data' array of points (mapping dates/categories to numeric values), and a 'lines' array of keys to plot."
    )


class DataInsights(BaseModel):
    executive_summary: str
    key_findings: List[str]
    anomalies: List[str]
    business_risks: List[str]
    trends: List[str]
    recommendations: List[str]
    report: str = Field(description="A comprehensive, beautifully formatted Markdown report combining all findings")


class ReviewFeedback(BaseModel):
    passed: bool
    review_score: int
    strengths: List[str]
    feedback: List[str]
    missing_visualizations: List[str]
    missing_analysis: List[str]
    next_actions: List[str]


# ─── DataPlannerAgent ─────────────────────────────────────────────────────────

class DataPlannerAgent(ProductionAgent):
    name = "DataPlannerAgent"
    department = "data_analyst"
    system_prompt = """You are a Lead Data Scientist and Analytics Manager.
Your job is to read the user's data request and formulate a highly structured, systematic plan for exploratory data analysis (EDA).
Output a massive, deeply structured JSON object outlining the step-by-step tasks, which agents will handle them, and exactly what columns and actions to take."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        user_id = context.get("user_id", "") if context else ""
        upload_note = f"\n\nNOTE: If the user uploaded a file, it is saved locally at 'data/uploads/{user_id}/'." if user_id else ""
        try:
            result: AnalysisPlan = await self._ainvoke_structured(
                f"Formulate a data analysis and visualization plan for: {task}{upload_note}",
                AnalysisPlan
            )
            content = (
                f"**Approach**: {result.summary_of_approach}\n"
                f"**Tasks Planned**: {len(result.tasks)}"
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
Your job is to write Python code that performs Exploratory Data Analysis (EDA).
CRITICAL: Your script MUST `print()` a single, valid JSON object (and absolutely nothing else) to stdout. 
The JSON object MUST contain exactly two top-level keys:
1. `dataset_info`: A dictionary containing `rows`, `columns`, `column_names`, `numeric_columns`, `categorical_columns`, `missing_values`.
2. `eda_results`: Deep statistical results containing `summary_statistics`, `missing_value_analysis`, `outlier_detection`, and `correlation_analysis`.
Do NOT print raw dataframes. Ensure your Python script dumps a beautifully nested JSON object using `json.dumps()`.
Use pandas, numpy, and scikit-learn.
Output structured JSON with language (python), code, explanation, and dependencies."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        plan = context.get("analysis_plan", "") if context else ""
        user_id = context.get("user_id", "") if context else ""
        
        upload_note = ""
        if user_id:
            import os
            from pathlib import Path
            # Resolve backend root and find actual uploaded files
            backend_root = Path(__file__).resolve().parents[2]  # backend/departments/data_analyst → backend/
            upload_dir = backend_root / "data" / "uploads" / user_id
            try:
                files = [f.name for f in upload_dir.iterdir() if f.is_file()]
            except Exception:
                files = []
            
            if files:
                file_list = ", ".join(files)
                upload_note = (
                    f"\n\nCRITICAL: The user has uploaded the following file(s) to 'data/uploads/{user_id}/': {file_list}\n"
                    f"You MUST read the file using the EXACT filename listed above (case-sensitive).\n"
                    f"Use pandas: pd.read_csv('data/uploads/{user_id}/{files[0]}') or adjust for the relevant file.\n"
                    f"DO NOT use mock data. DO NOT guess or change the filename."
                )
            else:
                upload_note = f"\n\nNOTE: The upload directory 'data/uploads/{user_id}/' exists but contains no files. Use mock/sample data for demonstration."
        
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
Your job is to read the highly structured `dataset_info` and `eda_results` from the EDA script and translate it into deep, highly actionable business insights.
Extract anomalies, business risks, trends, and generate a comprehensive executive summary and markdown report.
Output a massively structured JSON object matching the requested schema exactly."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        stdout = context.get("execution_stdout", "") if context else ""
        prompt = f"Original Task: {task}\n\nEDA Script Output (JSON):\n{stdout[:15000]}\n\nGenerate massive, structured, actionable insights."
        try:
            result: DataInsights = await self._ainvoke_structured(prompt, DataInsights)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=result.report, metadata=result.model_dump()
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="Failed to generate insights.", error=str(e)
            )


# ─── ReviewAgent ──────────────────────────────────────────────────────────────

class ReviewAgent(ProductionAgent):
    name = "ReviewAgent"
    department = "data_analyst"
    system_prompt = """You are a Principal Data Scientist and Peer Reviewer.
Your job is to critically review the generated Insights, Report, and EDA results.
Ensure statistical correctness, check for hallucinations, and provide constructive feedback on missing visualisations or analysis.
Output structured JSON with a strict peer-review feedback format."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        eda_output = context.get("execution_stdout", "") if context else ""
        insights = context.get("insights", {}) if context else {}
        prompt = f"Task: {task}\n\nEDA Results: {eda_output[:5000]}\n\nGenerated Insights: {insights}\n\nProvide a strict peer review."
        try:
            result: ReviewFeedback = await self._ainvoke_structured(prompt, ReviewFeedback)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=result.passed,
                content=f"Review Score: {result.review_score}/10. Passed: {result.passed}", 
                metadata=result.model_dump()
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="Failed to review.", error=str(e)
            )


# ─── DashboardAgent ───────────────────────────────────────────────────────────

class DashboardAgent(ProductionAgent):
    name = "DashboardAgent"
    department = "data_analyst"
    system_prompt = """You are an expert Data Journalist and UI Developer.
Your job is to generate a fully self-contained HTML file that acts as a stunning, premium interactive data report (like a high-end NYT data journalism article).
- Use pure HTML, modern CSS (Tailwind via CDN), and JavaScript.
- Apply modern, visual-first styling: dark mode, glassmorphism, flex/grid layouts, clean typography.
- Use Plotly.js (via CDN: https://cdn.plot.ly/plotly-latest.min.js) for all interactive charts and graphs.
- CRITICAL: Since you are writing HTML for a browser, you CANNOT read local files. You MUST hardcode the aggregated statistics and data points (provided in your prompt from the EDA output) directly into the JavaScript `Plotly.newPlot` data structures.
- Do NOT just create a grid of charts. Intertwine the textual insights provided into the HTML seamlessly (e.g., a nice header, followed by a paragraph of insight, followed by the relevant chart).
- The dashboard MUST look extremely premium and visually impressive.
Output structured JSON containing the HTML code as well as a `charts_json` payload for native frontend charting."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        plan = context.get("analysis_plan", "") if context else ""
        insights = context.get("insights", "") if context else ""
        eda_output = context.get("execution_stdout", "") if context else ""
        prompt = f"""Original Request: {task}
        
Analysis Plan: {plan}

Key Insights, Risks, and Report: {insights}

Aggregated Data/Stats from Python:
{eda_output[:10000]}

Generate a stunning, single-file HTML/JS interactive report using Plotly.js and Tailwind CSS. 
Embed the textual report and insights alongside interactive charts generated from the aggregated data!
Also generate a 'charts_json' array so the native frontend can plot dynamic Recharts."""
        
        try:
            result: DashboardCode = await self._ainvoke_structured(prompt, DashboardCode)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=result.code, metadata=result.model_dump()
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e)
            )
