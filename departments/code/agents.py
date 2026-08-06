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

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        try:
            result: GeneratedCode = self._invoke_structured(
                f"Coding task: {task}\n\nWrite complete, production-ready code for this task.",
                GeneratedCode,
            )
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
                response = self._invoke(
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

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
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
            result: DebuggingResult = self._invoke_structured(prompt, DebuggingResult)
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

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
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
            result: Documentation = self._invoke_structured(prompt, Documentation)
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
                doc = self._invoke(
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
