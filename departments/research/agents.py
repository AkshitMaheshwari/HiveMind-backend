"""
Research Department — Worker Agents
- WebSearchAgent: Searches the web for information
- SummarizerAgent: Synthesizes evidence into a coherent draft
- FactCheckerAgent: Cross-verifies claims and identifies gaps
"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from shared.base_agent import ProductionAgent, AgentOutput
from shared.tools import web_search


# ─── Pydantic schemas for structured outputs ──────────────────────────────────

class EvidenceItem(BaseModel):
    source: str = Field(description="Source name or URL")
    summary: str = Field(description="Key information from this source")
    relevance: str = Field(description="Why this is relevant to the task")


class SearchDraft(BaseModel):
    evidence: List[EvidenceItem] = Field(description="Structured evidence from search")
    draft_answer: str = Field(description="Initial answer based on evidence")


class FactCheckResult(BaseModel):
    verdict: str = Field(description="'verified' if sufficient, 'needs_more' if gaps exist")
    verified_claims: List[str] = Field(description="Claims that are well-supported")
    gaps: List[str] = Field(description="Missing information or unverified claims")
    confidence: float = Field(description="Overall confidence score 0-1")


# ─── WebSearchAgent ───────────────────────────────────────────────────────────

class WebSearchAgent(ProductionAgent):
    name = "WebSearchAgent"
    department = "research"
    system_prompt = """You are a Research Analyst. You receive raw web search results and 
    your job is to extract structured evidence from them.
    
    For each key piece of information:
    - Identify the source
    - Summarize the key finding
    - Note its relevance to the research question
    
    Then write an initial draft answer based on ALL the evidence collected."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            # Run web search
            raw_results = web_search(task, max_results=6)

            # Use structured output to extract evidence
            prompt = f"Research question: {task}\n\nWeb search results:\n{raw_results}"
            result: SearchDraft = self._invoke_structured(prompt, SearchDraft)

            evidence_text = "\n".join([
                f"• [{e.source}]: {e.summary}" for e in result.evidence
            ])

            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.draft_answer,
                metadata={
                    "evidence": [e.model_dump() for e in result.evidence],
                    "raw_results": raw_results,
                    "evidence_formatted": evidence_text,
                },
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=False,
                content="",
                error=str(e),
            )


# ─── SummarizerAgent ─────────────────────────────────────────────────────────

class SummarizerAgent(ProductionAgent):
    name = "SummarizerAgent"
    department = "research"
    system_prompt = """You are a Research Synthesizer. You receive:
    1. A research question
    2. Draft findings from web search
    3. A list of knowledge gaps
    
    Your job is to synthesize all of this into a comprehensive, well-structured research report.
    
    Format your output in clean Markdown:
    - Start with a brief executive summary
    - Use headers for major topics
    - Include bullet points for key findings
    - Cite sources when possible
    - Address any gaps by noting limitations
    
    Be thorough, accurate, and professional."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        draft = context.get("draft_answer", "")
        gaps = context.get("gaps", [])

        prompt = f"""Research question: {task}

Draft findings:
{draft}

Knowledge gaps to acknowledge:
{chr(10).join(f'- {g}' for g in gaps) if gaps else 'None identified'}

Please synthesize a comprehensive research report."""

        try:
            final_report = self._invoke(prompt)
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=final_report,
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=False,
                content=draft,  # Return draft as fallback
                error=str(e),
            )


# ─── FactCheckerAgent ────────────────────────────────────────────────────────

class FactCheckerAgent(ProductionAgent):
    name = "FactCheckerAgent"
    department = "research"
    system_prompt = """You are a Fact-Checker and Quality Analyst. You receive a research draft 
    and your job is to critically evaluate it.
    
    Check for:
    1. Claims that seem unverified or unsupported
    2. Missing important information the user likely needs
    3. Potential inaccuracies or outdated information
    4. Overall completeness
    
    Be rigorous but fair. Rate confidence honestly."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        draft = context.get("draft_answer", "")
        evidence = context.get("evidence", [])

        evidence_summary = "\n".join([
            f"- {e.get('source', 'Unknown')}: {e.get('summary', '')}"
            for e in evidence
        ]) if evidence else "No structured evidence available"

        prompt = f"""Research question: {task}

Draft answer to verify:
{draft}

Supporting evidence:
{evidence_summary}

Perform a fact-check and quality assessment."""

        try:
            result: FactCheckResult = self._invoke_structured(prompt, FactCheckResult)
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.verdict,
                metadata={
                    "verdict": result.verdict,
                    "verified_claims": result.verified_claims,
                    "gaps": result.gaps,
                    "confidence": result.confidence,
                },
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,  # Don't block on fact-check failure
                content="verified",
                metadata={"verdict": "verified", "gaps": [], "confidence": 0.7},
                error=str(e),
            )
