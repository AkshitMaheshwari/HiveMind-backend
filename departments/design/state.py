"""Design Department state."""
from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict

class DesignDeptState(TypedDict):
    task: str
    required_agents: List[str]
    user_id: Optional[str]
    api_keys: Optional[Dict[str, str]]
    selected_model: Optional[str]
    branding_guide: str
    logo_concepts: List[str]
    visual_assets: List[str]
    final_design_package: str
    events: List[Dict[str, Any]]
