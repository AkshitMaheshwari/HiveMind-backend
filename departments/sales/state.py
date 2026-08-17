"""Sales/Outreach Department state."""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

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
    events: List[Dict[str, Any]]
