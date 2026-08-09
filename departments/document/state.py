from typing import Any, Dict, List
from typing_extensions import TypedDict


class DocumentDeptState(TypedDict):
    """
    State for the Document Department subgraph.
    """
    task: str
    original_request: str
    user_id: str
    api_keys: Dict[str, str]
    selected_model: str
    events: List[Dict[str, Any]]
    
    # Outputs
    final_answer: str
