"""Sales/Outreach Department state."""
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from shared.state_utils import merge_events

class SalesDeptState(TypedDict):
    task: str
    required_agents: List[str]
    user_id: Optional[str]
    api_keys: Optional[Dict[str, str]]
    selected_model: Optional[str]
    lead_research: str
    cold_emails: str
    follow_up_sequence: str
    final_outreach_kit: str
    events: Annotated[List[Dict[str, Any]], merge_events]
