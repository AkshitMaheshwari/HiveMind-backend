from typing import Annotated, Any, Dict, List
from typing_extensions import TypedDict
from shared.state_utils import merge_events


class DocumentDeptState(TypedDict):
    """
    State for the Document Department subgraph.
    """
    task: str
    original_request: str
    user_id: str
    api_keys: Dict[str, str]
    selected_model: str
    events: Annotated[List[Dict[str, Any]], merge_events]
    
    # Outputs
    final_answer: str
