"""
Content Department — Worker Agents
- CopywriterAgent: Creates engaging draft content
- SEOOptimizerAgent: Optimizes for search engines
- EditorAgent: Polishes and proofreads the final content
"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from shared.base_agent import ProductionAgent, AgentOutput


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class ContentDraft(BaseModel):
    title: str = Field(description="Compelling title for the content")
    content: str = Field(description="Full content body in Markdown")
    tone: str = Field(description="Tone used: professional/casual/technical/conversational")
    word_count: int = Field(description="Approximate word count")


class SEOEnhancements(BaseModel):
    primary_keyword: str = Field(description="Primary SEO keyword")
    secondary_keywords: List[str] = Field(description="Supporting keywords")
    meta_description: str = Field(description="SEO meta description (150-160 chars)")
    optimized_content: str = Field(description="Content with natural keyword integration")
    seo_score_estimate: int = Field(description="Estimated SEO score 0-100")


class EditedContent(BaseModel):
    final_content: str = Field(description="Polished, publication-ready content in Markdown")
    changes_made: List[str] = Field(description="List of edits made")
    quality_score: int = Field(description="Quality score 0-100")


# ─── CopywriterAgent ─────────────────────────────────────────────────────────

class CopywriterAgent(ProductionAgent):
    name = "CopywriterAgent"
    department = "content"
    system_prompt = """You are a world-class Copywriter and Content Strategist.

Your job is to create engaging, high-quality content based on:
1. The content task given
2. Any research context provided

Guidelines:
- Write compelling, well-structured content in Markdown
- Use appropriate headers, bullet points, and formatting
- Match the tone to the request (professional for reports, conversational for blogs)
- Include relevant examples, data points, or stories where appropriate
- Aim for depth and value — not filler content
- Length should match the task: blog posts 800-1500 words, reports 500-1000 words, social posts 50-200 words

Always output structured JSON with title, content, tone, and word_count."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        research = context.get("research_context", "")

        research_section = f"\n\nResearch context available:\n{research[:3000]}" if research else ""

        prompt = f"""Content task: {task}{research_section}

Create engaging, well-structured content for this task."""

        try:
            result: ContentDraft = self._invoke_structured(prompt, ContentDraft)
            formatted = f"# {result.title}\n\n{result.content}"
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=formatted,
                metadata={
                    "title": result.title,
                    "tone": result.tone,
                    "word_count": result.word_count,
                },
            )
        except Exception as e:
            # Fallback to unstructured
            try:
                fallback = self._invoke(prompt)
                return AgentOutput(
                    agent_name=self.name,
                    department=self.department,
                    success=True,
                    content=fallback,
                )
            except Exception as e2:
                return AgentOutput(
                    agent_name=self.name,
                    department=self.department,
                    success=False,
                    content="",
                    error=str(e2),
                )


# ─── SEOOptimizerAgent ────────────────────────────────────────────────────────

class SEOOptimizerAgent(ProductionAgent):
    name = "SEOOptimizerAgent"
    department = "content"
    system_prompt = """You are an expert SEO Specialist and Content Optimizer.

Your job is to take existing content and optimize it for search engines WITHOUT sacrificing readability.

SEO optimization checklist:
1. Identify the primary keyword and 3-5 secondary keywords from context
2. Naturally integrate keywords (density 1-2%, no stuffing)
3. Optimize headings (H1, H2, H3) to include keywords
4. Add/improve meta description (150-160 characters)
5. Ensure content has proper structure for featured snippets
6. Add internal link suggestions where appropriate

The content must read naturally — SEO is secondary to user value.
Output structured JSON with keyword data and optimized content."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        draft = context.get("draft_content", "")

        prompt = f"""Content task: {task}

Draft content to optimize:
{draft[:4000]}

Perform SEO optimization and return structured output."""

        try:
            result: SEOEnhancements = self._invoke_structured(prompt, SEOEnhancements)
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.optimized_content,
                metadata={
                    "primary_keyword": result.primary_keyword,
                    "secondary_keywords": result.secondary_keywords,
                    "meta_description": result.meta_description,
                    "seo_score": result.seo_score_estimate,
                },
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=draft,  # Return original draft as fallback
                metadata={"primary_keyword": "", "secondary_keywords": [], "meta_description": ""},
                error=str(e),
            )


# ─── EditorAgent ─────────────────────────────────────────────────────────────

class EditorAgent(ProductionAgent):
    name = "EditorAgent"
    department = "content"
    system_prompt = """You are a Senior Editor and Proofreader at a top-tier content agency.

Your job is to take content through final review and polish it for publication.

Editorial checklist:
1. Fix grammar, spelling, and punctuation errors
2. Improve sentence flow and readability
3. Ensure consistent tone throughout
4. Check logical structure and transitions between sections
5. Trim redundant or filler content
6. Strengthen the opening hook and closing CTA
7. Verify formatting consistency (Markdown headers, lists, bold text)

Output publication-ready content with a list of changes made and a quality score."""

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        content = context.get("seo_content", context.get("draft_content", ""))

        prompt = f"""Editorial task: {task}

Content for review and editing:
{content[:5000]}

Perform final editorial review and output polished content."""

        try:
            result: EditedContent = self._invoke_structured(prompt, EditedContent)
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.final_content,
                metadata={
                    "changes_made": result.changes_made,
                    "quality_score": result.quality_score,
                },
            )
        except Exception as e:
            # Fallback to unstructured editing
            try:
                fallback = self._invoke(
                    f"Edit and polish this content:\n\n{content[:4000]}",
                    system_override="You are a senior editor. Fix grammar, improve flow, and return polished content."
                )
                return AgentOutput(
                    agent_name=self.name,
                    department=self.department,
                    success=True,
                    content=fallback,
                )
            except Exception as e2:
                return AgentOutput(
                    agent_name=self.name,
                    department=self.department,
                    success=True,
                    content=content,
                    error=str(e2),
                )
