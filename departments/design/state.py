"""Design Department state."""
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict
from shared.state_utils import merge_events, merge_lists

class DesignDeptState(TypedDict):
    task: str
    required_agents: List[str]
    user_id: Optional[str]
    api_keys: Optional[Dict[str, str]]
    selected_model: Optional[str]
    branding_guide: str
    logo_concepts: Annotated[List[str], merge_lists]
    visual_assets: Annotated[List[str], merge_lists]
    pitch_visuals: str
    final_design_package: str
    events: Annotated[List[Dict[str, Any]], merge_events]
