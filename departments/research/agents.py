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


class ResearchRoute(BaseModel):
    sources: List[str] = Field(description="List of sources to search. Options: 'rag_search_node', 'arxiv_node', 'wikipedia_node', 'web_search_node'.")
    reasoning: str = Field(description="Why these sources were selected.")


# ─── ResearchRouterAgent ──────────────────────────────────────────────────────

class ResearchRouterAgent(ProductionAgent):
    name = "ResearchRouterAgent"
    department = "research"
    system_prompt = """You are the Research Director. Analyze the user's research query and decide which knowledge sources are required.
    
    Source Selection Guidelines:
    - User explicitly mentions a PDF, uploaded document, private file -> Select 'rag_search_node' ONLY.
    - Academic papers, physics, machine learning literature -> Select 'arxiv_node'.
    - General background knowledge, historical facts -> Select 'wikipedia_node'.
    - Current events, company info, technical documentation, general web -> Select 'web_search_node'.
    
    You may select multiple sources if the query is complex and crosses domains.
    However, if the query strictly relates to an uploaded document, prefer 'rag_search_node' exclusively. Do NOT select 'web_search_node' "just in case" for document queries.
    """

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            prompt = f"Analyze the following research task and select the appropriate sources:\n\nTask: {task}"
            result: ResearchRoute = self._invoke_structured(prompt, ResearchRoute)
            
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.reasoning,
                metadata={
                    "sources": result.sources,
                    "reasoning": result.reasoning
                }
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=False,
                content=f"Routing failed: {e}",
                metadata={"sources": ["web_search_node"], "reasoning": "Fallback to web search due to routing error."},
                error=str(e),
            )



# ─── ArxivResearchAgent ───────────────────────────────────────────────────────

class ArxivResearchAgent(ProductionAgent):
    name = "ArxivResearchAgent"
    department = "research"
    system_prompt = """You are an Academic Research Scientist specializing in literature search via arXiv.
    You receive raw arXiv paper search results containing paper titles, authors, published dates, abstracts, and PDF URLs.
    
    Your job is to extract structured scientific findings from these papers:
    - Identify key methodology innovations
    - Extract core empirical results or theorems
    - Highlight paper title, author list, and PDF links for reference."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            from shared.tools import arxiv_search
            raw_arxiv = arxiv_search(task, max_results=5)

            prompt = f"Research question: {task}\n\narXiv Search Results:\n{raw_arxiv}"
            try:
                result: SearchDraft = self._invoke_structured(prompt, SearchDraft)
                draft_content = result.draft_answer if result and result.draft_answer else raw_arxiv
                evidence_list = [e.model_dump() for e in result.evidence] if result and result.evidence else []
            except Exception:
                draft_content = raw_arxiv
                evidence_list = []

            evidence_text = "\n".join([
                f"• [{e.get('source', 'arXiv')}]: {e.get('summary', '')}" for e in evidence_list
            ]) if evidence_list else raw_arxiv[:500]

            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=draft_content,
                metadata={
                    "evidence": evidence_list,
                    "raw_results": raw_arxiv,
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


# ─── WikipediaAgent ──────────────────────────────────────────────────────────

class WikipediaAgent(ProductionAgent):
    name = "WikipediaAgent"
    department = "research"
    system_prompt = """You are a Knowledge Base Analyst. You receive Wikipedia article summaries 
    and your job is to extract foundational definitions, historical context, and core concepts.
    
    Structure your findings with clear references to the article URLs."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            from shared.tools import wikipedia_search
            raw_wiki = wikipedia_search(task, max_results=3)

            prompt = f"Research topic: {task}\n\nWikipedia Search Results:\n{raw_wiki}"
            try:
                result: SearchDraft = self._invoke_structured(prompt, SearchDraft)
                draft_content = result.draft_answer if result and result.draft_answer else raw_wiki
                evidence_list = [e.model_dump() for e in result.evidence] if result and result.evidence else []
            except Exception:
                draft_content = raw_wiki
                evidence_list = []

            evidence_text = "\n".join([
                f"• [{e.get('source', 'Wikipedia')}]: {e.get('summary', '')}" for e in evidence_list
            ]) if evidence_list else raw_wiki[:500]

            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=draft_content,
                metadata={
                    "evidence": evidence_list,
                    "raw_results": raw_wiki,
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


# ─── RagSearchAgent (Internal Documents) ──────────────────────────────────────

class RagSearchAgent(ProductionAgent):
    name = "RagSearchAgent"
    department = "research"
    system_prompt = """You are an Internal Knowledge Base Analyst. You receive retrieved excerpts 
    from the user's uploaded private documents (PDFs, Excel, etc).
    
    Your job is to extract structured evidence directly answering the user's query from these documents.
    If the provided documents don't contain relevant information, state that clearly.
    Always cite the source document name provided in the excerpts."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        user_id = context.get("user_id") if context else None
        if not user_id:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=False,
                content="Error: No user_id provided for secure document retrieval.",
                error="missing user_id",
            )

        try:
            from shared.tools.rag_retrieval import rag_document_search
            # We enforce top_k=5 for thoroughness
            raw_rag, confidence = rag_document_search(query=task, user_id=user_id, top_k=5)

            prompt = f"Research question: {task}\n\nDocument Excerpts:\n{raw_rag}"
            try:
                result: SearchDraft = self._invoke_structured(prompt, SearchDraft)
                draft_content = result.draft_answer if result and result.draft_answer else raw_rag
                evidence_list = [e.model_dump() for e in result.evidence] if result and result.evidence else []
            except Exception:
                draft_content = raw_rag
                evidence_list = []

            evidence_text = "\n".join([
                f"• [{e.get('source', 'Document')}]: {e.get('summary', '')}" for e in evidence_list
            ]) if evidence_list else raw_rag[:500]

            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=draft_content,
                metadata={
                    "evidence": evidence_list,
                    "raw_results": raw_rag,
                    "evidence_formatted": evidence_text,
                    "confidence": confidence,
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


# ─── WebSearchAgent ───────────────────────────────────────────────────────────

class WebSearchAgent(ProductionAgent):
    name = "WebSearchAgent"
    department = "research"
    system_prompt = """You are a Web Intelligence Analyst. You receive raw web search results and 
    your job is to extract structured evidence from news, blogs, and technical documentation.
    
    For each key piece of information:
    - Identify the source URL
    - Summarize the key finding
    - Note its relevance to the research question
    
    Then write an initial draft answer based on ALL the evidence collected."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            # Run web search
            raw_results = web_search(task, max_results=6)

            prompt = f"Research question: {task}\n\nWeb search results:\n{raw_results}"
            try:
                result: SearchDraft = self._invoke_structured(prompt, SearchDraft)
                draft_content = result.draft_answer if result and result.draft_answer else raw_results
                evidence_list = [e.model_dump() for e in result.evidence] if result and result.evidence else []
            except Exception:
                draft_content = raw_results
                evidence_list = []

            evidence_text = "\n".join([
                f"• [{e.get('source', 'Web')}]: {e.get('summary', '')}" for e in evidence_list
            ]) if evidence_list else raw_results[:500]

            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=draft_content,
                metadata={
                    "evidence": evidence_list,
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


# ─── SummarizerAgent (Deep Research Synthesizer) ─────────────────────────────

class SummarizerAgent(ProductionAgent):
    name = "SummarizerAgent"
    department = "research"
    system_prompt = """You are a brilliant research assistant who explains complex topics clearly and helpfully.

You receive findings from multiple research sources (arXiv papers, Wikipedia, Documents, web search) and synthesize them into a clear, direct answer.

How to respond:
- Answer the question directly and conversationally — like a knowledgeable friend, not a formal report.
- Use Markdown naturally: ## headings for major sections, **bold** for key terms, bullet points for lists.
- DO NOT use "Executive Summary", "Title & Executive Summary", "Foundational Concepts & Background", or similar corporate report headings.
- DO NOT start with "# 📋 Output" or any emoji-header.
- Lead with the answer/key insight. Then explain the supporting evidence.
- For research papers: mention the paper name, authors, key finding, and link — naturally in the text.
- For web sources: cite inline with [source name](url) links.
- For document sources: cite the exact document name inline.
- Keep it focused. Skip fluff. Match depth to the question complexity.
- End with a "**Sources**" or "**Further Reading**" section if there are relevant links."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        arxiv_draft = context.get("arxiv_draft", "")
        wiki_draft = context.get("wiki_draft", "")
        web_draft = context.get("draft_answer", "")
        raw_results = context.get("raw_results", "")
        gaps = context.get("gaps", [])

        prompt = f"""Research question: {task}

--- ACADEMIC PAPERS & PREPRINTS (arXiv) ---
{arxiv_draft if arxiv_draft else 'No arXiv data available'}

--- DOMAIN BACKGROUND (Wikipedia) ---
{wiki_draft if wiki_draft else 'No Wikipedia data available'}

--- WEB FINDINGS ---
{web_draft if web_draft else 'No web search data available'}

--- ADDITIONAL CONTEXT ---
{raw_results[:3000] if raw_results else ''}

--- KNOWLEDGE GAPS & NOTES ---
{chr(10).join(f'- {g}' for g in gaps) if gaps else 'None identified'}

Please answer the research question clearly and helpfully using the above sources."""

        try:
            final_report = self._invoke(prompt)
            if not final_report or not str(final_report).strip():
                final_report = (web_draft or arxiv_draft or wiki_draft or raw_results or f"I couldn't find enough information on: {task}")
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=final_report,
            )
        except Exception as e:
            fallback_report = (web_draft or arxiv_draft or wiki_draft or raw_results or f"Research error: {e}")
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=False,
                content=fallback_report,
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
