"""
Base agent class for all production worker agents.
Every department worker inherits from ProductionAgent.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from langchain_core.messages import SystemMessage, HumanMessage


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

    def __init__(self, llm=None):
        if llm is None:
            from shared.llm import worker_llm
            self._llm = worker_llm()
        else:
            self._llm = llm

    def _invoke(self, user_content: str, system_override: Optional[str] = None) -> str:
        """Invoke the LLM with system + human messages."""
        system = system_override or self.system_prompt
        response = self._llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ])
        return response.content

    def _invoke_structured(self, user_content: str, schema, system_override: Optional[str] = None):
        """Invoke with structured output (Pydantic model)."""
        system = system_override or self.system_prompt
        structured_llm = self._llm.with_structured_output(schema)
        return structured_llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=user_content),
        ])

    def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        """Override in subclass."""
        raise NotImplementedError(f"{self.name} must implement execute()")
