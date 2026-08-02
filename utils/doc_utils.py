import re
from dataclasses import dataclass
from typing import List

@dataclass
class Chunk:
    doc_name: str
    chunk_id: int
    text: str

def _clean_text(text: str) -> str:
    """
    Cleans the text by removing extra whitespace and newlines.
    """
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def search_local(query: str, index : List[Chunk], top_k: int = 5) -> List[Chunk]:
    """ Your exact local token-overlap search implementation """
    query_tokens = set(_tokenize(query))
    scored = []
    for chunk in index:
        chunk_tokens = set(_tokenize(chunk.text))
        overlap = len(query_tokens & chunk_tokens)
        if overlap == 0:
            continue
        scored.append((overlap,chunk))
    scored.sort(key = lambda x:x[0], reverse = True)
    return [x[1] for x in scored[:top_k]]




