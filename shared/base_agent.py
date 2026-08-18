"""
Base agent class for all production worker agents.
Every department worker inherits from ProductionAgent.
"""
import ast
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage


def normalize_llm_content(content: Any) -> str:
    """Extract clean plain-text/markdown from any LLM response structure."""
    if content is None:
        return ""
    if isinstance(content, str):
        trimmed = content.strip()
        if ((trimmed.startswith("[{'type': 'text'") or trimmed.startswith('[{"type": "text"') or
             trimmed.startswith("{'type': 'text'") or trimmed.startswith('{"type": "text"')) and
            ("'text':" in trimmed or '"text":' in trimmed)):
            try:
                val = ast.literal_eval(trimmed)
                return normalize_llm_content(val)
            except Exception:
                pass
        return content
    elif isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                if "text" in item:
                    parts.append(str(item["text"]))
                elif "content" in item:
                    parts.append(normalize_llm_content(item["content"]))
            elif isinstance(item, str):
                parts.append(normalize_llm_content(item))
            else:
                parts.append(str(item))
        return "".join(parts).strip()
    elif isinstance(content, dict):
        if "text" in content:
            return str(content["text"]).strip()
        if "content" in content:
            return normalize_llm_content(content["content"])
        return str(content).strip()
    return str(content).strip()


@dataclass
class AgentOutput:
    agent_name: str
    department: str
    success: bool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ProductionAgent:
    """
    Universal base class for all worker agents in the system.
    
    Subclasses override:
      - system_prompt: str
      - execute(task, context) -> AgentOutput
    """
    name: str = "BaseAgent"
    department: str = "unknown"
    system_prompt: str = "You are a helpful AI assistant."

    def __init__(self, llm=None, api_keys: Optional[Dict[str, str]] = None, selected_model: Optional[str] = None):
        if llm is None:
            from shared.llm import worker_llm
            self._llm = worker_llm(api_keys, selected_model=selected_model)
        else:
            self._llm = llm

    async def _ainvoke(self, user_content: str, system_override: Optional[str] = None) -> str:
        """Invoke the LLM asynchronously with system + human messages and normalized string output."""
        system = system_override or self.system_prompt
        response = await self._llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ])
        return normalize_llm_content(response.content)

    async def _ainvoke_structured(self, user_content: str, schema, system_override: Optional[str] = None):
        """Invoke asynchronously with structured output (Pydantic model)."""
        system = system_override or self.system_prompt
        structured_llm = self._llm.with_structured_output(schema)
        return await structured_llm.ainvoke([
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ])

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        """Override in subclass."""
        raise NotImplementedError(f"{self.name} must implement execute()")
