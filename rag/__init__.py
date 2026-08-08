"""
rag — Retrieval-Augmented Generation pipeline.

Provides:
- Document ingestion: chunking → embedding → vector store upsert
- Document retrieval: query embedding → filtered vector search → ranked chunks

All operations are scoped to a user_id. The user_id filter is enforced at the
Qdrant query level, not in application code after fetching results.
"""
