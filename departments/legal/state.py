"""Legal Department State."""
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from shared.state_utils import merge_events

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
    events: Annotated[List[Dict[str, Any]], merge_events]
