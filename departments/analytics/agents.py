"""
Analytics Department — Worker Agents
- DataProfilerAgent:   Profiles the dataset (shape, dtypes, nulls, value counts)
- DataCleanerAgent:    Detects and handles outliers, fixes dtypes
- StatisticsAgent:     Descriptive stats, correlation, outlier detection, KPIs
- ChartGeneratorAgent: Produces charts_json for frontend rendering
- InsightNarratorAgent: Plain-English explanation of all findings
"""
import asyncio
import json
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from shared.base_agent import ProductionAgent, AgentOutput
from shared.tools import execute_code


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class DataProfile(BaseModel):
    row_count: int = Field(description="Number of rows")
    column_count: int = Field(description="Number of columns")
    columns: List[str] = Field(description="Column names")
    dtypes: Dict[str, str] = Field(description="Column name to dtype mapping")
    null_counts: Dict[str, int] = Field(description="Null/missing value counts per column")
    value_counts_summary: Dict[str, Any] = Field(description="Top value counts for categorical columns")
    numeric_summary: str = Field(description="Brief summary of numeric ranges")


class CleanedData(BaseModel):
    cleaned_csv: str = Field(description="The cleaned data as CSV string")
    changes_made: List[str] = Field(description="List of cleaning steps applied")
    outliers_removed: int = Field(description="Number of outlier rows removed or capped")


class Statistics(BaseModel):
    descriptive_stats: Dict[str, Any] = Field(description="Mean, median, std, min, max per numeric column")
    correlation_summary: str = Field(description="Key correlations found")
    outlier_details: List[str] = Field(description="Outlier findings")
    kpis: Dict[str, Any] = Field(description="Computed KPIs relevant to the task")
    trend: str = Field(description="Overall trend description")


class ChartSpec(BaseModel):
    type: str = Field(description="Chart type: 'line', 'bar', 'scatter', 'histogram', 'pie'")
    title: str = Field(description="Chart title")
    x_label: str = Field(description="X-axis label")
    y_label: str = Field(description="Y-axis label")
    data: List[Dict[str, Any]] = Field(description="Array of {x, y} or {label, value} data points")


class ChartsOutput(BaseModel):
    charts: List[ChartSpec] = Field(description="List of chart specifications")


class Insights(BaseModel):
    key_findings: List[str] = Field(description="Top 3-5 key insights from the data")
    narrative: str = Field(description="Plain-English explanation of the full analysis")
    recommendations: List[str] = Field(description="Actionable recommendations based on the data")


# ─── DataProfilerAgent ────────────────────────────────────────────────────────

class DataProfilerAgent(ProductionAgent):
    name = "DataProfilerAgent"
    department = "analytics"
    system_prompt = """You are a Data Scientist specializing in exploratory data analysis.
You receive raw data (CSV or JSON) and profile it: shape, column types, missing values, and distributions.
Be precise and complete."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        data = context.get("data_source", "")

        if not data or not data.strip():
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="No data provided to profile.", metadata={"profile": {}}
            )

        code = f"""
import pandas as pd
import json
import io

data_str = '''{data[:50000]}'''

try:
    # Try CSV first
    df = pd.read_csv(io.StringIO(data_str))
except Exception:
    try:
        df = pd.read_json(io.StringIO(data_str))
    except Exception as e:
        print(json.dumps({{"error": str(e)}}))
        exit()

profile = {{
    "row_count": len(df),
    "column_count": len(df.columns),
    "columns": list(df.columns),
    "dtypes": {{col: str(dtype) for col, dtype in df.dtypes.items()}},
    "null_counts": df.isnull().sum().to_dict(),
    "value_counts_summary": {{
        col: df[col].value_counts().head(5).to_dict()
        for col in df.select_dtypes(include=['object']).columns[:5]
    }},
    "numeric_describe": df.describe().to_dict()
}}
print(json.dumps(profile, default=str))
"""
        try:
            raw_output = await asyncio.to_thread(execute_code, code)
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                profile_data = json.loads(raw_output[json_start:json_end])
            else:
                profile_data = {"raw": raw_output[:500]}

            summary = (
                f"Dataset: {profile_data.get('row_count', '?')} rows × "
                f"{profile_data.get('column_count', '?')} columns. "
                f"Columns: {', '.join(profile_data.get('columns', [])[:8])}."
            )
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=summary, metadata={"profile": profile_data}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e), metadata={"profile": {}}
            )


# ─── DataCleanerAgent ─────────────────────────────────────────────────────────

class DataCleanerAgent(ProductionAgent):
    name = "DataCleanerAgent"
    department = "analytics"
    system_prompt = """You are a Data Quality Engineer.
You receive a dataset and profile info, then clean it: remove duplicates,
handle missing values (median imputation for numeric, mode for categorical),
cap outliers using IQR method. Report what was changed."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        data = context.get("data_source", "")

        if not data or not data.strip():
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content="No data to clean.", metadata={"cleaned_data": data, "changes": []}
            )

        code = f"""
import pandas as pd
import json
import io

data_str = '''{data[:50000]}'''
changes = []

try:
    df = pd.read_csv(io.StringIO(data_str))
except Exception:
    try:
        df = pd.read_json(io.StringIO(data_str))
    except Exception as e:
        print(json.dumps({{"error": str(e), "cleaned_csv": data_str, "changes": []}}))
        exit()

# Remove duplicates
dupes = df.duplicated().sum()
if dupes > 0:
    df = df.drop_duplicates()
    changes.append(f"Removed {{dupes}} duplicate rows")

# Fill missing values
for col in df.select_dtypes(include=['number']).columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        df[col] = df[col].fillna(df[col].median())
        changes.append(f"Imputed {{null_count}} missing values in '{{col}}' with median")

for col in df.select_dtypes(include=['object']).columns:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
        changes.append(f"Imputed {{null_count}} missing values in '{{col}}' with mode")

# Cap outliers using IQR
outliers_capped = 0
for col in df.select_dtypes(include=['number']).columns:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    out_count = ((df[col] < lower) | (df[col] > upper)).sum()
    if out_count > 0:
        df[col] = df[col].clip(lower, upper)
        outliers_capped += out_count

if outliers_capped > 0:
    changes.append(f"Capped {{outliers_capped}} outlier values using IQR method")

result = {{
    "cleaned_csv": df.to_csv(index=False),
    "changes": changes,
    "outliers_capped": outliers_capped,
    "final_shape": [len(df), len(df.columns)]
}}
print(json.dumps(result, default=str))
"""
        try:
            raw_output = await asyncio.to_thread(execute_code, code)
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            if json_start >= 0 and json_end > json_start:
                result = json.loads(raw_output[json_start:json_end])
            else:
                result = {"cleaned_csv": data, "changes": [], "outliers_capped": 0}

            cleaned_data = result.get("cleaned_csv", data)
            changes = result.get("changes", [])
            summary = f"Cleaned dataset: {'; '.join(changes)}" if changes else "Data was already clean."
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=summary, metadata={"cleaned_data": cleaned_data, "changes": changes}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content="Cleaning skipped due to error.", error=str(e),
                metadata={"cleaned_data": data, "changes": []}
            )


# ─── StatisticsAgent ──────────────────────────────────────────────────────────

class StatisticsAgent(ProductionAgent):
    name = "StatisticsAgent"
    department = "analytics"
    system_prompt = """You are a Quantitative Analyst.
Compute descriptive statistics, correlations, outlier analysis, and KPIs from the given data.
Always relate stats back to the user's specific question."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        data = context.get("cleaned_data", context.get("data_source", ""))

        if not data or not data.strip():
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content="No data for statistical analysis.", metadata={"statistics": {}}
            )

        code = f"""
import pandas as pd
import json
import io

data_str = '''{data[:50000]}'''
try:
    df = pd.read_csv(io.StringIO(data_str))
except Exception:
    try:
        df = pd.read_json(io.StringIO(data_str))
    except Exception as e:
        print(json.dumps({{"error": str(e)}}))
        exit()

numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
result = {{}}

# Descriptive stats
if numeric_cols:
    desc = df[numeric_cols].describe().round(2).to_dict()
    result["descriptive_stats"] = desc

# Correlation (only if 2+ numeric cols)
if len(numeric_cols) >= 2:
    corr = df[numeric_cols].corr().round(3)
    high_corr = []
    for i in range(len(corr.columns)):
        for j in range(i+1, len(corr.columns)):
            val = corr.iloc[i,j]
            if abs(val) > 0.6:
                high_corr.append(f"{{corr.columns[i]}} vs {{corr.columns[j]}}: {{val:.3f}}")
    result["high_correlations"] = high_corr

# Outlier detection
outlier_info = []
for col in numeric_cols[:5]:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    out_count = ((df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)).sum()
    if out_count > 0:
        outlier_info.append(f"{{col}}: {{out_count}} outliers detected")
result["outliers"] = outlier_info

# Growth trend (if there's a numeric sequence)
if len(df) >= 3 and numeric_cols:
    first_col = numeric_cols[0]
    first_val = df[first_col].iloc[0]
    last_val = df[first_col].iloc[-1]
    pct_change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
    result["trend"] = f"{{first_col}} changed by {{pct_change:.1f}}% from start to end"

print(json.dumps(result, default=str))
"""
        try:
            raw_output = await asyncio.to_thread(execute_code, code)
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            stats_data = {}
            if json_start >= 0 and json_end > json_start:
                stats_data = json.loads(raw_output[json_start:json_end])

            # Ask LLM to interpret
            prompt = f"""User's question: {task}

Statistical results computed:
{json.dumps(stats_data, indent=2, default=str)[:3000]}

Provide a concise interpretation of these statistics that answers the user's question.
Focus on what the numbers mean in plain English."""
            interpretation = await self._ainvoke(prompt)

            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=interpretation, metadata={"statistics": stats_data}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="", error=str(e), metadata={"statistics": {}}
            )


# ─── ChartGeneratorAgent ──────────────────────────────────────────────────────

class ChartGeneratorAgent(ProductionAgent):
    name = "ChartGeneratorAgent"
    department = "analytics"
    system_prompt = """You are a Data Visualization Specialist.
Generate chart specifications (JSON) from datasets. Choose appropriate chart types:
- Line charts for trends over time
- Bar charts for category comparisons
- Scatter plots for correlations
- Histograms for distributions
- Pie charts for proportions (only when there are <7 categories)

Output structured chart JSON that the frontend can render directly."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        data = context.get("cleaned_data", context.get("data_source", ""))
        stats = context.get("statistics", {})

        if not data or not data.strip():
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content="No data for chart generation.", metadata={"charts_json": []}
            )

        code = f"""
import pandas as pd
import json
import io

data_str = '''{data[:30000]}'''
try:
    df = pd.read_csv(io.StringIO(data_str))
except Exception:
    try:
        df = pd.read_json(io.StringIO(data_str))
    except Exception as e:
        print(json.dumps({{"charts": []}}))
        exit()

charts = []
numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
cat_cols = df.select_dtypes(include=['object']).columns.tolist()

# Time series / line chart if data has rows and one numeric column
if numeric_cols and len(df) > 1:
    col = numeric_cols[0]
    charts.append({{
        "type": "line",
        "title": f"{{col}} Distribution",
        "x_label": "Index",
        "y_label": col,
        "data": [{{"x": int(i), "y": round(float(v), 4)}} for i, v in enumerate(df[col].head(100).tolist())]
    }})

# Bar chart for categorical columns
if cat_cols and numeric_cols:
    cat_col = cat_cols[0]
    num_col = numeric_cols[0]
    grouped = df.groupby(cat_col)[num_col].mean().head(10)
    charts.append({{
        "type": "bar",
        "title": f"Average {{num_col}} by {{cat_col}}",
        "x_label": cat_col,
        "y_label": f"Avg {{num_col}}",
        "data": [{{"label": str(k), "value": round(float(v), 4)}} for k, v in grouped.items()]
    }})

# Histogram for first numeric column
if numeric_cols:
    col = numeric_cols[0]
    hist_vals, bin_edges = pd.cut(df[col].dropna(), bins=10, retbins=True)
    hist_counts = pd.cut(df[col].dropna(), bins=10).value_counts().sort_index()
    charts.append({{
        "type": "histogram",
        "title": f"{{col}} Distribution",
        "x_label": col,
        "y_label": "Count",
        "data": [{{"label": str(interval), "value": int(count)}} for interval, count in hist_counts.items()]
    }})

print(json.dumps({{"charts": charts}}, default=str))
"""
        try:
            raw_output = await asyncio.to_thread(execute_code, code)
            json_start = raw_output.find('{')
            json_end = raw_output.rfind('}') + 1
            charts_data = {"charts": []}
            if json_start >= 0 and json_end > json_start:
                charts_data = json.loads(raw_output[json_start:json_end])

            charts = charts_data.get("charts", [])
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=f"Generated {len(charts)} chart(s).", metadata={"charts_json": charts}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content="Chart generation failed; continuing without charts.", error=str(e),
                metadata={"charts_json": []}
            )


# ─── InsightNarratorAgent ─────────────────────────────────────────────────────

class InsightNarratorAgent(ProductionAgent):
    name = "InsightNarratorAgent"
    department = "analytics"
    system_prompt = """You are a Senior Data Analyst explaining findings to a business stakeholder.

Your job is to take raw statistical analysis and translate it into a clear, actionable narrative.

Guidelines:
- Lead with the 3-5 most important findings
- Explain what the numbers mean in business terms (not math terms)
- Connect insights to the user's original question
- End with 2-3 concrete, actionable recommendations
- Use Markdown: ## headings, **bold** key metrics, bullet points for findings
- Be direct and specific — no filler phrases like "it's worth noting that"
"""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        profile = context.get("profile_data", {})
        stats = context.get("statistics", {})
        charts = context.get("charts_json", [])

        prompt = f"""User's question: {task}

Dataset Profile:
{json.dumps(profile, indent=2, default=str)[:1500]}

Statistical Analysis:
{json.dumps(stats, indent=2, default=str)[:2000]}

Charts Generated: {len(charts)} chart(s) — {', '.join(c.get('title', '') for c in charts[:5])}

Write a comprehensive analytical narrative that:
1. Answers the user's specific question
2. Highlights the key data patterns found
3. Explains any outliers or anomalies
4. Provides actionable recommendations"""

        try:
            narrative = await self._ainvoke(prompt)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=narrative, metadata={}
            )
        except Exception as e:
            fallback = f"Analysis complete. Dataset has {profile.get('row_count', '?')} rows and {profile.get('column_count', '?')} columns."
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content=fallback, error=str(e), metadata={}
            )
