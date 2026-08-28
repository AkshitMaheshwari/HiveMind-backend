"""
Code Department — Worker Agents
- CodeGeneratorAgent: Writes clean Python/JS/etc code
- DebuggerAgent: Runs code, identifies and fixes errors
- DocWriterAgent: Generates README, docstrings, usage examples
- UXDesignerAgent: Plans design systems for frontend/web requests
- UIReviewerAgent: Reviews UI quality
- GitHubOpsAgent: Live GitHub repository inspection, tree visualization, branching, code commits, and PR generation
"""
import logging
import re
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field
from shared.base_agent import ProductionAgent, AgentOutput
from shared.tools import execute_code

logger = logging.getLogger(__name__)


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


class GitHubTaskPlan(BaseModel):
    is_github_task: bool = Field(description="True if the request mentions or implies GitHub repository operations.")
    repo: Optional[str] = Field(None, description="GitHub repository in 'owner/repo' format (e.g. 'octocat/Hello-World' or 'AkshitMaheshwari/portfolio')")
    target_files: List[str] = Field(default_factory=list, description="List of file paths mentioned or implied in the task")
    action_type: str = Field("project_structure", description="One of: 'project_structure', 'read_code', 'edit_code', 'create_pr', 'list_repos'")
    branch_name: Optional[str] = Field(None, description="Feature branch name for code changes (e.g. 'feature/add-auth', 'fix/bug-123')")
    pr_title: Optional[str] = Field(None, description="Pull Request title if creating PR")
    pr_description: Optional[str] = Field(None, description="Detailed Pull Request summary")


# ─── CodeGeneratorAgent ───────────────────────────────────────────────────────

class CodeGeneratorAgent(ProductionAgent):
    name = "CodeGeneratorAgent"
    department = "code"
    system_prompt = """You are an expert Principal Software Engineer and Web Developer with 10+ years of experience.

Your job is to write production-grade, complete, fully-functional code that strictly fulfills every requirement in the user's prompt.

CRITICAL RULES:
1. STRICT PROMPT COMPLIANCE: Include every specific name, title, section, tagline, feature, theme, color scheme, and interaction requested in the user prompt.
2. WEB / UI TASKS: If the prompt requests a website, webpage, landing page, dashboard, or UI component:
   - Set language="html".
   - Write a complete, self-contained single HTML file with embedded modern CSS (<style>) and interactive JavaScript (<script>).
   - Apply modern styling: dark mode gradients, glassmorphism, floating particles, smooth hover animations.
   - Include ALL requested sections (Hero, About, Events/Features grid, Gallery, Footer).
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
            try:
                response = await self._ainvoke(
                    f"Write complete, runnable code for: {task}\n\nReturn ONLY the code, no explanation.",
                    system_override="You are an expert programmer. Return only clean, runnable code."
                )
                code = response
                if "```" in response:
                    parts = response.split("```")
                    if len(parts) >= 3:
                        code_block = parts[1]
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
If the task does not require a UI (e.g. backend script, CLI tool, data processing), set is_web_ui to False."""

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
    system_prompt = """You are a UI/UX Design Critic and Senior Frontend Engineer.
Your job is to review the generated HTML/CSS/JS code and determine if it meets the highest aesthetic standards."""

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
            return AgentOutput(
                agent_name=self.name, department=self.department, success=True,
                content=f"Error in review, proceeding. ({e})", metadata={"approved": True}
            )


# ─── DebuggerAgent ────────────────────────────────────────────────────────────

class DebuggerAgent(ProductionAgent):
    name = "DebuggerAgent"
    department = "code"
    system_prompt = """You are an expert Debugger and Code Reviewer.
Identify root causes of execution errors and provide fixed code."""

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        context = context or {}
        code = context.get("generated_code", "")
        stdout = context.get("stdout", "")
        stderr = context.get("stderr", "")
        success = context.get("success", True)

        if success and not stderr:
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
Generated code:\n```\n{code}\n```
Execution stdout: {stdout or 'No output'}
Execution stderr (errors): {stderr or 'No errors'}
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
                content=code,
                metadata={"issues_found": [str(e)], "is_resolved": False},
                error=str(e),
            )


# ─── DocWriterAgent ────────────────────────────────────────────────────────────

class DocWriterAgent(ProductionAgent):
    name = "DocWriterAgent"
    department = "code"
    system_prompt = """You are a Technical Writer specializing in developer documentation."""

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


# ─── GitHub Operations Schema & Agent ────────────────────────────────────────

def _build_ascii_tree(files: List[str]) -> str:
    """Helper to convert flat list of file paths into an ASCII tree diagram."""
    tree: Dict[str, Any] = {}
    for f in sorted(files):
        parts = f.split("/")
        curr = tree
        for part in parts:
            if part not in curr:
                curr[part] = {}
            curr = curr[part]

    lines = []

    def _render(node: Dict[str, Any], prefix: str = ""):
        items = list(node.items())
        for i, (name, child) in enumerate(items):
            is_last = (i == len(items) - 1)
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{name}")
            if child:
                sub_prefix = prefix + ("    " if is_last else "│   ")
                _render(child, sub_prefix)

    _render(tree)
    return "\n".join(lines)


class GitHubOpsAgent(ProductionAgent):
    """
    Agent responsible for end-to-end GitHub operations:
    - Live repository inspection & ASCII tree visualization
    - Reading repository source files
    - Modifying/generating code for specific files
    - Creating isolated feature branches
    - Committing code changes
    - Opening clean Pull Requests with markdown summaries
    """
    name = "GitHubOpsAgent"
    department = "code"
    system_prompt = (
        "You are an expert GitHub Operations and DevOps AI Agent. "
        "You inspect live GitHub repositories, display file trees and architectural summaries, "
        "read source code, write clean code modifications, manage git branches, and construct Pull Requests."
    )

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    async def execute(self, task: str, context: Optional[Dict[str, Any]] = None) -> AgentOutput:
        context = context or {}
        api_keys = context.get("api_keys") or {}
        github_token = api_keys.get("github_token") or api_keys.get("github")

        from shared.tools.github_tools import (
            github_read_file,
            github_list_files,
            github_create_branch,
            github_create_or_update_file,
            github_create_pull_request,
            github_list_user_repos,
        )

        # 1. Parse user intent into structured plan
        plan_prompt = (
            f"Analyze this task and formulate a structured GitHub execution plan:\n"
            f"Task: {task}\n\n"
            f"Extract the repository (owner/repo), target files, action type (project_structure, read_code, edit_code, create_pr, list_repos), branch name, PR title and summary."
        )

        try:
            plan: GitHubTaskPlan = await self._ainvoke_structured(plan_prompt, GitHubTaskPlan)
        except Exception:
            plan = GitHubTaskPlan(
                is_github_task=True,
                repo=None,
                action_type="project_structure",
            )

        repo = plan.repo or context.get("repo")

        # Extract repo if regex matches owner/repo in task string
        if not repo:
            match = re.search(r"([a-zA-Z0-9_-]+/[a-zA-Z0-9_\.-]+)", task)
            if match:
                repo = match.group(1).rstrip(".")

        # Smart Auto-Discovery: If user said "my portfolio repository" and has token
        if not repo and github_token:
            try:
                user_repos = await github_list_user_repos(token=github_token)
                if user_repos and isinstance(user_repos, list) and "error" not in user_repos[0]:
                    task_keywords = [
                        w.lower() for w in re.findall(r"\w+", task)
                        if len(w) > 2 and w.lower() not in ["show", "the", "project", "structure", "repository", "repo", "please", "and", "for"]
                    ]
                    for r in user_repos:
                        r_name = (r.get("full_name") or "").lower()
                        if any(kw in r_name for kw in task_keywords):
                            repo = r.get("full_name")
                            break
                    if not repo and len(user_repos) == 1:
                        repo = user_repos[0].get("full_name")
            except Exception as e:
                logger.warning("Could not auto-list user repos: %s", e)

        if not repo:
            # If user has a token, list their repos so they can pick
            user_repo_list_md = ""
            if github_token:
                try:
                    user_repos = await github_list_user_repos(token=github_token)
                    if user_repos and isinstance(user_repos, list) and "error" not in user_repos[0]:
                        user_repo_list_md = "\n\n**Your Accessible Repositories:**\n" + "\n".join([f"- `{r.get('full_name')}`" for r in user_repos[:10]])
                except Exception:
                    pass

            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=(
                    "### 🐙 GitHub Agent Ready\n\n"
                    "Please specify the target GitHub repository (e.g. `owner/repository`) and the task you'd like to perform:\n"
                    "- 📂 **Project Structure:** `Show me the project structure of AkshitMaheshwari/portfolio`\n"
                    "- 🔍 **Read & Analyze:** `Analyze the auth flow in facebook/react`\n"
                    "- 🛠️ **Edit & PR:** `Fix the typo in README.md in my-org/my-project and open a PR`"
                    + user_repo_list_md
                ),
            )

        # 2. Fetch live file tree from GitHub API
        tree = await github_list_files(repo=repo, token=github_token)

        if isinstance(tree, list) and tree and "error" in tree[0]:
            err_msg = tree[0]["error"]
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=False,
                content=(
                    f"### ❌ Could not access GitHub repository [`{repo}`](https://github.com/{repo})\n\n"
                    f"**Error:** {err_msg}\n\n"
                    f"*If this is a private repository, please ensure your GitHub Personal Access Token is configured in **Settings & Integrations**.*"
                ),
                error=err_msg,
            )

        all_file_paths = [item["path"] for item in tree if item.get("type") == "blob"]

        # If user is asking for project structure or listing files
        is_structure_query = (
            plan.action_type == "project_structure"
            or any(kw in task.lower() for kw in ["structure", "tree", "files", "show", "list", "directory", "architecture", "overview"])
        )

        if is_structure_query and not plan.target_files and plan.action_type not in ["edit_code", "create_pr"]:
            ascii_tree = _build_ascii_tree(all_file_paths)
            
            # Read package.json or README.md for extra architectural context
            readme_summary = ""
            for doc_file in ["README.md", "package.json", "requirements.txt", "pyproject.toml"]:
                if doc_file in all_file_paths:
                    file_info = await github_read_file(repo=repo, path=doc_file, token=github_token)
                    if "content" in file_info and file_info["content"]:
                        readme_summary = file_info["content"][:1500]
                        break

            arch_prompt = (
                f"Repository: {repo}\n"
                f"File list ({len(all_file_paths)} files total):\n{', '.join(all_file_paths[:60])}\n\n"
                f"Sample config/readme content:\n{readme_summary}\n\n"
                f"Provide a concise, high-level architectural summary of what this project does and how its folders are organized."
            )
            analysis = await self._ainvoke(arch_prompt)

            content = (
                f"### 📂 Project Structure for [`{repo}`](https://github.com/{repo})\n\n"
                f"**Total Files:** {len(all_file_paths)} files indexed\n\n"
                f"```text\n{repo}/\n{ascii_tree}\n```\n\n"
                f"### 🏗️ Architecture & Component Overview\n{analysis}\n\n"
                f"---\n💡 *Ask me to inspect any specific file, modify code, or open a Pull Request!*"
            )

            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=content,
                metadata={"repo": repo, "file_count": len(all_file_paths)},
            )

        # 3. Reading specific files
        file_contents = {}
        target_files = plan.target_files or []

        for fpath in target_files[:5]:
            read_res = await github_read_file(repo=repo, path=fpath, token=github_token)
            if "content" in read_res:
                file_contents[fpath] = read_res["content"]

        # If read-only request:
        if plan.action_type in ["read_code", "read_or_structure"] or not github_token:
            code_context = "\n\n".join([f"### File: `{p}`\n```\n{c[:3000]}\n```" for p, c in file_contents.items()])
            analysis_prompt = (
                f"User Task: {task}\n\n"
                f"Repository: {repo}\n"
                f"Code Context:\n{code_context}\n\n"
                f"Provide a comprehensive technical analysis answering the user's request."
            )
            analysis = await self._ainvoke(analysis_prompt)
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=f"### 🔍 GitHub Code Analysis ([`{repo}`](https://github.com/{repo}))\n\n{analysis}",
                metadata={"repo": repo, "files_read": list(file_contents.keys())},
            )

        # 4. Write & PR flow
        branch_name = plan.branch_name or f"agent-patch-{int(time.time())}"
        pr_title = plan.pr_title or "refactor: automated updates by AI agent"

        # Create branch
        branch_res = await github_create_branch(
            repo=repo,
            branch_name=branch_name,
            from_branch="main",
            token=github_token,
        )

        if "error" in branch_res and branch_res.get("status") != "already_exists":
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=False,
                content=f"❌ Failed to create branch `{branch_name}`: {branch_res.get('error')}",
                error=branch_res.get("error"),
            )

        # Generate modifications
        committed_files = []
        for fpath, old_content in file_contents.items():
            mod_prompt = (
                f"You are modifying the file `{fpath}` in repository `{repo}`.\n"
                f"User Task: {task}\n\n"
                f"Original File Content:\n```\n{old_content}\n```\n\n"
                f"Return ONLY the complete updated file content without markdown code block backticks if possible."
            )
            new_content = await self._ainvoke(mod_prompt)
            if new_content.startswith("```") and new_content.endswith("```"):
                lines = new_content.split("\n")[1:-1]
                new_content = "\n".join(lines)

            commit_res = await github_create_or_update_file(
                repo=repo,
                path=fpath,
                content=new_content,
                message=f"feat({fpath}): updates by AI multi-agent system",
                branch=branch_name,
                token=github_token,
            )
            if commit_res.get("status") == "success":
                committed_files.append(fpath)

        # Create Pull Request
        pr_res = await github_create_pull_request(
            repo=repo,
            title=pr_title,
            head=branch_name,
            base="main",
            body=(
                f"## 🤖 AI Multi-Agent Pull Request\n\n"
                f"**Task:** {task}\n\n"
                f"### Modified Files:\n"
                + "\n".join([f"- `{f}`" for f in committed_files])
                + f"\n\n**Summary:**\n{plan.pr_description or 'Automated code changes and enhancements.'}"
            ),
            token=github_token,
        )

        if "html_url" in pr_res:
            pr_url = pr_res["html_url"]
            pr_num = pr_res.get("number")
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=(
                    f"### 🚀 Pull Request Created Successfully!\n\n"
                    f"**Repository:** [`{repo}`](https://github.com/{repo})\n"
                    f"**Branch:** `{branch_name}`\n"
                    f"**PR #{pr_num}:** [{pr_title}]({pr_url})\n\n"
                    f"#### Modified Files:\n"
                    + "\n".join([f"- `{f}`" for f in committed_files])
                    + f"\n\n[👉 View and Merge on GitHub]({pr_url})"
                ),
                metadata={"pr_url": pr_url, "pr_number": pr_num, "branch": branch_name},
            )
        else:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=(
                    f"### 🌿 Changes Committed to Branch `{branch_name}`\n\n"
                    f"**Repository:** [`{repo}`](https://github.com/{repo})\n"
                    f"Files updated: {', '.join(committed_files)}\n\n"
                    f"*(Pull request note: {pr_res.get('error', 'Could not open automatic PR')})*"
                ),
                metadata={"branch": branch_name, "files": committed_files},
            )
