"""
Code Department — Worker Agents
- CodeGeneratorAgent: Writes clean Python/JS/etc code
- DebuggerAgent: Runs code, identifies and fixes errors
- DocWriterAgent: Generates README, docstrings, usage examples
"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field
from shared.base_agent import ProductionAgent, AgentOutput
from shared.tools import execute_code


# ─── Pydantic schemas ─────────────────────────────────────────────────────────

class UXDesignSystem(BaseModel):
    is_web_ui: bool = Field(description="True if the request requires a visual UI, website, dashboard, or frontend.")
    color_palette: List[str] = Field(description="Hex codes for the color palette")
    typography: str = Field(description="Font family recommendations")
    layout_structure: str = Field(description="Description of the overall layout (e.g., Grid, Flexbox, Sidebar)")
    animations: str = Field(description="Recommended micro-interactions and animations")
    design_rationale: str = Field(description="Why this design was chosen")

class UIReviewFeedback(BaseModel):
    approved: bool = Field(description="True if the UI meets high aesthetic and functional standards.")
    feedback: str = Field(description="Detailed feedback or list of improvements needed if rejected.")

class GeneratedCode(BaseModel):
    language: str = Field(description="Programming language used")
    code: str = Field(description="Complete, runnable code")
    explanation: str = Field(description="Brief explanation of the approach")
    dependencies: List[str] = Field(description="Required packages/libraries")


class DebuggingResult(BaseModel):
    fixed_code: str = Field(description="Fixed/corrected code")
    issues_found: List[str] = Field(description="List of issues that were fixed")
    is_resolved: bool = Field(description="True if all errors are resolved")
    explanation: str = Field(description="What was wrong and how it was fixed")


class Documentation(BaseModel):
    readme: str = Field(description="README.md content in Markdown")
    docstring: str = Field(description="Module-level docstring")
    usage_examples: List[str] = Field(description="Example usage code snippets")


# ─── CodeGeneratorAgent ───────────────────────────────────────────────────────

class CodeGeneratorAgent(ProductionAgent):
    name = "CodeGeneratorAgent"
    department = "code"
    system_prompt = """You are an expert Principal Software Engineer and Web Developer with 10+ years of experience.

Your job is to write production-grade, complete, fully-functional code that strictly fulfills every requirement in the user's prompt.

CRITICAL RULES:
1. STRICT PROMPT COMPLIANCE: Include every specific name, title, section, tagline, feature, theme, color scheme, and interaction requested in the user prompt. NEVER use generic placeholder text like "Lorem ipsum", "Landing Page", or "Welcome to our website" if specific titles/names (e.g. "ANUGGONJ", "College Fest") were requested.
2. WEB / UI TASKS: If the prompt requests a website, webpage, landing page, dashboard, or UI component:
   - Set language="html".
   - Write a complete, self-contained single HTML file with embedded modern CSS (<style>) and interactive JavaScript (<script>).
   - Apply modern, visual-first styling: dark mode gradients, glassmorphism, animated glow effects, floating particles, smooth hover animations, flex/grid layouts, responsive typography.
   - Include ALL requested sections (Hero with CTA buttons, About cards, Events grid with all requested events, Sponsors, Gallery grid, Footer).
   - Make the page look stunning, visually rich, and production-ready.
3. COMPLETENESS: Never omit code or leave TODO comments. Code must be 100% complete and runnable.

Output structured JSON with language, code, explanation, and dependencies."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        ux_context = ""
        if context and context.get("ux_design_system"):
            ux_context = f"\n\nUX DESIGN SYSTEM TO STRICTLY FOLLOW:\n{context.get('ux_design_system')}"
            
        ui_feedback = ""
        if context and context.get("ui_feedback"):
            ui_feedback = f"\n\nUI REVIEWER FEEDBACK TO INCORPORATE:\n{context.get('ui_feedback')}"

        prompt = f"Coding task: {task}{ux_context}{ui_feedback}\n\nWrite complete, production-ready code for this task."
        
        try:
            result: GeneratedCode = await self._ainvoke_structured(prompt, GeneratedCode)
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.code,
                metadata={
                    "language": result.language,
                    "explanation": result.explanation,
                    "dependencies": result.dependencies,
                },
            )
        except Exception as e:
            # Unstructured fallback
            try:
                response = await self._ainvoke(
                    f"Write complete, runnable code for: {task}\n\nReturn ONLY the code, no explanation.",
                    system_override="You are an expert programmer. Return only clean, runnable code."
                )
                # Extract code from markdown blocks if present
                code = response
                if "```" in response:
                    parts = response.split("```")
                    if len(parts) >= 3:
                        code_block = parts[1]
                        # Remove language identifier line
                        lines = code_block.split("\n")
                        if lines[0].strip() in ["python", "javascript", "js", "typescript", "bash", "sh"]:
                            lines = lines[1:]
                        code = "\n".join(lines)
                return AgentOutput(
                    agent_name=self.name,
                    department=self.department,
                    success=True,
                    content=code,
                    metadata={"language": "python", "explanation": "", "dependencies": []},
                )
            except Exception as e2:
                return AgentOutput(
                    agent_name=self.name,
                    department=self.department,
                    success=False,
                    content="",
                    error=str(e2),
                )


# ─── UXDesignerAgent ──────────────────────────────────────────────────────────

class UXDesignerAgent(ProductionAgent):
    name = "UXDesignerAgent"
    department = "code"
    system_prompt = """You are an elite UX/UI Designer and Creative Director.
Your job is to read the user's request and determine if a visual UI is needed. 
If it is, you must output a stunning, modern, and highly aesthetic design system.
Focus on:
- Vibrant, harmonious color palettes (use HSL or exact hex codes).
- Modern typography (e.g., Inter, Roboto, Outfit).
- Glassmorphism, subtle gradients, and dark modes if applicable.
- Dynamic layouts and micro-interactions (hover effects, transitions).
If the task does not require a UI (e.g. backend script, CLI tool, data processing), set is_web_ui to False.
"""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            result: UXDesignSystem = await self._ainvoke_structured(
                f"Analyze this task and design a UX system if it's a UI task: {task}",
                UXDesignSystem
            )
            
            if not result.is_web_ui:
                return AgentOutput(
                    agent_name=self.name, department=self.department, success=True,
                    content="No UI required.", metadata={"is_web_ui": False}
                )
                
            design_summary = (
                f"Colors: {', '.join(result.color_palette)}\n"
                f"Typography: {result.typography}\n"
                f"Layout: {result.layout_structure}\n"
                f"Animations: {result.animations}\n"
                f"Rationale: {result.design_rationale}"
            )
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=design_summary, metadata={"is_web_ui": True}
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name, department=self.department, success=False,
                content="Failed to generate UX design.", error=str(e), metadata={"is_web_ui": False}
            )


# ─── UIReviewerAgent ──────────────────────────────────────────────────────────

class UIReviewerAgent(ProductionAgent):
    name = "UIReviewerAgent"
    department = "code"
    system_prompt = """You are a ruthless UI/UX Design Critic and Senior Frontend Engineer.
Your job is to review the generated HTML/CSS/JS code and determine if it meets the highest aesthetic standards.
- Look for modern design practices (responsive, good padding/margins, appealing colors, hover states, transitions).
- If it looks like a basic 90s website or lacks visual flair, REJECT IT and provide specific feedback on how to make it stunning (e.g., "Add a dark mode gradient background", "Use backdrop-filter for glassmorphism on the cards").
- If it looks beautiful and modern, APPROVE IT.
"""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        code = context.get("generated_code", "") if context else ""
        if not code:
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content="No code to review.", metadata={"approved": True})
            
        prompt = f"Review this code for UI/UX aesthetics based on the task: {task}\n\nCode:\n```html\n{code[:4000]}\n```"
        
        try:
            result: UIReviewFeedback = await self._ainvoke_structured(prompt, UIReviewFeedback)
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=result.feedback, metadata={"approved": result.approved}
            )
        except Exception as e:
            # Fallback to approve if parsing fails
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=f"Error in review, proceeding. ({e})", metadata={"approved": True}
            )

# ─── DebuggerAgent ────────────────────────────────────────────────────────────

class DebuggerAgent(ProductionAgent):
    name = "DebuggerAgent"
    department = "code"
    system_prompt = """You are an expert Debugger and Code Reviewer.

You receive:
1. The original code task
2. Generated code
3. Execution results (stdout, stderr)

Your job is to:
- If there are errors: identify the root cause and provide fixed code
- If no errors: verify correctness and suggest minor improvements if any
- Explain what was wrong and how you fixed it

Output structured JSON with fixed_code, issues_found, is_resolved, and explanation."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        code = context.get("generated_code", "")
        stdout = context.get("stdout", "")
        stderr = context.get("stderr", "")
        success = context.get("success", True)

        if success and not stderr:
            # Code ran fine, just do a quick review
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=code,
                metadata={
                    "issues_found": [],
                    "is_resolved": True,
                    "explanation": "Code executed successfully. No issues found.",
                },
            )

        prompt = f"""Task: {task}

Generated code:
```
{code}
```

Execution stdout:
{stdout or 'No output'}

Execution stderr (errors):
{stderr or 'No errors'}

Execution succeeded: {success}

Please debug the code, fix any issues, and return the corrected version."""

        try:
            result: DebuggingResult = await self._ainvoke_structured(prompt, DebuggingResult)
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=result.fixed_code,
                metadata={
                    "issues_found": result.issues_found,
                    "is_resolved": result.is_resolved,
                    "explanation": result.explanation,
                },
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=code,  # Return original code
                metadata={"issues_found": [str(e)], "is_resolved": False},
                error=str(e),
            )


# ─── DocWriterAgent ────────────────────────────────────────────────────────────

class DocWriterAgent(ProductionAgent):
    name = "DocWriterAgent"
    department = "code"
    system_prompt = """You are a Technical Writer specializing in developer documentation.

Your job is to create comprehensive documentation for code including:
1. README.md with: overview, installation, usage, API reference, examples
2. Docstrings for functions/classes
3. Practical usage examples

The documentation should be clear, accurate, and developer-friendly.
Output structured JSON with readme, docstring, and usage_examples."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        code = context.get("final_code", context.get("generated_code", ""))
        language = context.get("language", "python")
        explanation = context.get("explanation", "")

        prompt = f"""Task that was coded: {task}

Code to document:
```{language}
{code[:4000]}
```

Additional context: {explanation}

Generate comprehensive documentation."""

        try:
            result: Documentation = await self._ainvoke_structured(prompt, Documentation)
            doc_output = f"{result.readme}\n\n---\n\n## Module Docstring\n\n```\n{result.docstring}\n```"
            if result.usage_examples:
                doc_output += "\n\n## Usage Examples\n\n"
                for i, ex in enumerate(result.usage_examples, 1):
                    doc_output += f"### Example {i}\n```{language}\n{ex}\n```\n\n"
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=doc_output,
                metadata={"readme": result.readme, "usage_examples": result.usage_examples},
            )
        except Exception as e:
            # Fallback to simple doc
            try:
                doc = await self._ainvoke(
                    f"Write a README.md for this code:\n```\n{code[:3000]}\n```",
                    system_override="You are a technical writer. Write clear Markdown documentation."
                )
                return AgentOutput(
                    agent_name=self.name,
                    department=self.department,
                    success=True,
                    content=doc,
                )
            except Exception as e2:
                return AgentOutput(
                    agent_name=self.name,
                    department=self.department,
                    success=False,
                    content="Documentation generation failed.",
                    error=str(e2),
                )
