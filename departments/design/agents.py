"""
Design Department - Worker Agents.
Uses DALL-E 3 image generation tool (new tool category, only dept that needs it).
"""
import asyncio
from typing import Any, Dict, List
from pydantic import BaseModel, Field
from shared.base_agent import ProductionAgent, AgentOutput

class DesignRoute(BaseModel):
    required_agents: List[str] = Field(description="Options: branding, logo_concept, pitch_visuals")
    reasoning: str

class DesignRouterAgent(ProductionAgent):
    name = "DesignRouterAgent"
    department = "design"
    system_prompt = "Classify design request. Options: branding (colors/typography/voice), logo_concept (logo ideas + image), pitch_visuals (pitch deck slides). Select all relevant."
    async def execute(self, task, context=None):
        try:
            result = await self._ainvoke_structured(f"Design request: {task}", DesignRoute)
            return AgentOutput(agent_name=self.name, department=self.department, success=True,
                               content=result.reasoning, metadata={"required_agents": result.required_agents})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=False,
                               content="", error=str(e), metadata={"required_agents": ["branding","logo_concept"]})

class BrandingAgent(ProductionAgent):
    name = "BrandingAgent"
    department = "design"
    system_prompt = """You are a Senior Brand Strategist.
Create a comprehensive brand identity guide: color palette (hex codes + rationale),
typography (primary + secondary fonts), brand voice (3 adjectives + dos/donts),
logo design principles, and usage guidelines. Format in Markdown."""
    async def execute(self, task, context=None):
        prompt = f"Create a brand identity guide for: {task}\n\nInclude: color palette, typography, brand voice, logo principles."
        try:
            guide = await self._ainvoke(prompt)
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content=guide, metadata={})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=False, content="", error=str(e), metadata={})

class LogoConceptAgent(ProductionAgent):
    name = "LogoConceptAgent"
    department = "design"
    system_prompt = "You are a Creative Director. Generate logo concepts and create visual mockups using DALL-E 3."
    async def execute(self, task, context=None):
        context = context or {}
        branding = context.get("branding_guide", "")
        concepts_prompt = f"""Create 3 logo concepts for: {task}

Branding context:
{branding[:1000]}

For each concept describe: style, symbolism, color application, and typography usage."""
        try:
            concepts_text = await self._ainvoke(concepts_prompt)
            # Generate one logo image via DALL-E 3
            image_urls = []
            try:
                from shared.tools.image_generation import generate_image
                dalle_prompt = (
                    f"Minimalist professional logo for '{task}'. "
                    f"Clean vector style, white background, suitable for a modern tech company. "
                    f"No text, icon only."
                )
                url = await asyncio.to_thread(generate_image, dalle_prompt, "1024x1024")
                if url and url.startswith("http"):
                    image_urls.append(url)
            except Exception:
                pass
            
            full_output = concepts_text
            if image_urls:
                full_output += f"\n\n**Generated Logo Mockup:**\n![]({image_urls[0]})"
            return AgentOutput(agent_name=self.name, department=self.department, success=True,
                               content=full_output, metadata={"image_urls": image_urls})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=False, content="", error=str(e), metadata={})

class PitchVisualsAgent(ProductionAgent):
    name = "PitchVisualsAgent"
    department = "design"
    system_prompt = "You are a Pitch Deck Designer. Provide visual direction, slide layout recommendations, and color/typography application for each slide."
    async def execute(self, task, context=None):
        context = context or {}
        prompt = f"Create visual design directions for pitch deck: {task}\n\nBranding: {context.get('branding_guide','')[:500]}"
        try:
            visuals = await self._ainvoke(prompt)
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content=visuals, metadata={})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=False, content="", error=str(e), metadata={})

class DesignSynthesizerAgent(ProductionAgent):
    name = "DesignSynthesizerAgent"
    department = "design"
    system_prompt = "Package all design assets into a cohesive design system document with clear implementation guidance."
    async def execute(self, task, context=None):
        context = context or {}
        parts = [f"# Design Package: {task}\n"]
        if context.get("branding_guide"): parts.append(f"## Brand Identity\n{context['branding_guide']}\n")
        if context.get("logo_concepts"): parts.append(f"## Logo Concepts\n{chr(10).join(context['logo_concepts'])}\n")
        if context.get("visual_assets"): parts.append(f"## Visual Assets\n{chr(10).join(context['visual_assets'])}\n")
        package = "\n".join(parts)
        try:
            final = await self._ainvoke(f"Polish this design package:\n\n{package[:5000]}")
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content=final, metadata={})
        except Exception as e:
            return AgentOutput(agent_name=self.name, department=self.department, success=True, content=package, error=str(e), metadata={})
