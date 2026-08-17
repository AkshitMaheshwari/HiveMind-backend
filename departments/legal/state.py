"""Legal Department State."""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

class LegalDeptState(TypedDict):
    task: str
    required_agents: List[str]
    user_id: Optional[str]
    api_keys: Optional[Dict[str, str]]
    selected_model: Optional[str]
    contract_review: str
    tos_draft: str
    compliance_checklist: str
    final_legal_output: str
    events: List[Dict[str, Any]]

