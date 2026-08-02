from typing import Any, Dict

from typing_extensions import TypedDict


class NexusState(TypedDict):
    user_prompt: str
    domain_insights: Dict[str, Any]
