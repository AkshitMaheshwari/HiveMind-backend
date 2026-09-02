"""
Sales/Outreach Department - Worker Agents.
LeadResearchAgent calls Research dept directly via inter_dept (hive-mind pattern).
"""
import asyncio
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from shared.base_agent import ProductionAgent, AgentOutput
from shared.tools import web_search

class SalesRoute(BaseModel):
    required_agents: List[str] = Field(description="Options: lead_research, cold_email, follow_up_sequence")
    reasoning: str

class SalesRouterAgent(ProductionAgent):
    name = "SalesRouterAgent"
    department = "sales"
    system_prompt = "Classify sales request. Options: lead_research, cold_email, follow_up_sequence."
    async def execute(self, task, context=None):
        try:
            result = await self._ainvoke_structured(f"Sales request: {task}", SalesRoute)
            return AgentOutput(agent_name=self.name, department=self.department, success=True,
                               content=result.reasoning, metadata={"required_agents": result.required_agents})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=False,
                               content="", error=str(e), metadata={"required_agents": ["cold_email"]})

class LeadResearchAgent(ProductionAgent):
    """HIVE MIND: calls Research dept directly via inter_dept."""
    name = "LeadResearchAgent"
    department = "sales"
    system_prompt = "You are a Sales Intelligence Analyst. Research prospects thoroughly using provided research data."
    async def execute(self, task, context=None):
        try:
            from shared.inter_dept import call_research_dept
            research = await call_research_dept(
                task=f"Research sales prospects and company info for: {task}. Find decision makers, pain points, recent news.",
                api_keys=(context or {}).get("api_keys"),
                selected_model=(context or {}).get("selected_model"),
            )
            prompt = f"Sales task: {task}\n\nResearch data:\n{research[:3000]}\n\nCreate a prospect profile with: company overview, decision makers, pain points, conversation starters."
            result = await self._ainvoke(prompt)
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content=result, metadata={})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=False, content="", error=str(e), metadata={})

class ColdEmailAgent(ProductionAgent):
    name = "ColdEmailAgent"
    department = "sales"
    system_prompt = """You are an expert B2B sales copywriter. Write personalized, high-converting cold emails.
Format: Subject line, opening hook (personalized), value proposition (1-2 sentences), social proof, clear CTA.
Keep under 150 words. No fluff. Write 2 subject line variants."""
    async def execute(self, task, context=None):
        context = context or {}
        prompt = f"Write a cold email for: {task}\n\nProspect info:\n{context.get('lead_research','')[:2000]}"
        try:
            email = await self._ainvoke(prompt)
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content=email, metadata={})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=False, content="", error=str(e), metadata={})

class FollowUpSequencerAgent(ProductionAgent):
    name = "FollowUpSequencerAgent"
    department = "sales"
    system_prompt = "Write a 4-email follow-up sequence. Day 3: add value, Day 7: case study, Day 14: FOMO angle, Day 21: breakup email. Each under 100 words."
    async def execute(self, task, context=None):
        context = context or {}
        prompt = f"Create 4-email follow-up sequence for: {task}\n\nInitial email:\n{context.get('cold_emails','')[:1000]}"
        try:
            sequence = await self._ainvoke(prompt)
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content=sequence, metadata={})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=False, content="", error=str(e), metadata={})

class SalesSynthesizerAgent(ProductionAgent):
    name = "SalesSynthesizerAgent"
    department = "sales"
    system_prompt = "Package all sales materials into a ready-to-use outreach kit with clear sections."
    async def execute(self, task, context=None):
        context = context or {}
        parts = ["# Sales Outreach Kit\n"]
        if context.get("lead_research"): parts.append(f"## Lead Research\n{context['lead_research']}\n")
        if context.get("cold_emails"): parts.append(f"## Cold Email\n{context['cold_emails']}\n")
        if context.get("follow_up_sequence"): parts.append(f"## Follow-Up Sequence\n{context['follow_up_sequence']}\n")
        kit = "\n".join(parts)
        try:
            final = await self._ainvoke(f"Polish this sales kit into a professional package:\n\n{kit[:5000]}")
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content=final, metadata={})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content=kit, error=str(e), metadata={})
