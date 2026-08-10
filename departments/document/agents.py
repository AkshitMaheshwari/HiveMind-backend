"""
Document Department — Worker Agents
- DocumentQAAgent: Retrieves document chunks and answers the user query directly.
"""
import asyncio
from typing import Any, Dict

from shared.base_agent import ProductionAgent, AgentOutput
from shared.tools.rag_retrieval import rag_document_search


class DocumentQAAgent(ProductionAgent):
    name = "DocumentQAAgent"
    department = "document"
    system_prompt = """You are a Document Intelligence Agent. Your job is to answer questions strictly based on the provided document excerpts.
    
    Guidelines:
    - Answer the question directly using ONLY the provided excerpts.
    - Do NOT hallucinate or add general knowledge outside of the document context.
    - If the provided document excerpts do not contain the answer, simply state: "I could not find the answer to this in your uploaded documents."
    - Do NOT write general tutorials or fallback to web-like answers.
    - Cite the document source if applicable.
    """

    async def execute(self, task: str, context: Dict[str, Any] = None) -> AgentOutput:
        user_id = context.get("user_id") if context else None
        if not user_id:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=False,
                content="Error: No user_id provided for secure document retrieval.",
                error="missing user_id",
            )

        try:
            # 1. Retrieve the document chunks
            raw_rag, confidence = await asyncio.to_thread(rag_document_search, query=task, user_id=user_id, top_k=5)

            # 2. Answer the question using the retrieved chunks
            prompt = f"Question: {task}\n\nDocument Excerpts:\n{raw_rag}\n\nAnswer the question directly based only on the excerpts."
            
            final_answer = await self._ainvoke(prompt)

            # If the LLM generates empty string, handle it gracefully
            if not final_answer or not str(final_answer).strip():
                final_answer = "I could not find the answer to this in your uploaded documents."

            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=True,
                content=str(final_answer),
                metadata={
                    "raw_results": raw_rag,
                    "confidence": confidence,
                },
            )
        except Exception as e:
            return AgentOutput(
                agent_name=self.name,
                department=self.department,
                success=False,
                content=f"Error reading documents: {e}",
                error=str(e),
            )
