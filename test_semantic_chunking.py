import os
import sys
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).parent.resolve()
sys.path.insert(0, str(backend_dir))

from rag.chunker import chunk_text

test_text = """
Semantic chunking is an innovative approach to splitting text. It ensures that sentences with similar meanings remain in the same chunk.
This differs from traditional methods. Traditional methods often split text arbitrarily based on character count.
Arbitrary splitting can sever context and meaning. This is bad for RAG applications.
By preserving semantic boundaries, the language model can retrieve more relevant and cohesive information.
The implementation uses a sophisticated embedding model. This model calculates the cosine distance between adjacent sentences.
If the distance exceeds a certain threshold, a new chunk is formed.
"""

print("Chunking text...")
try:
    chunks = chunk_text(test_text, 100, 10)
    for i, chunk in enumerate(chunks):
        print(f"\n--- Chunk {i+1} ---")
        print(chunk)
except Exception as e:
    print(f"Error: {e}")
