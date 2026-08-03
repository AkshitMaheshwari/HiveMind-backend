import os
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import SystemMessage, HumanMessage
from state import NexusState
from firecrawl import FirecrawlApp
from typing import Dict, Any, List, Optional
import json
from typing_extensions import TypedDict
from pydantic import BaseModel, Field



class NexusStateModel(BaseModel):
    user_prompt: str = Field(description="The user's input prompt.")
    context: Optional[str] = Field(default=None, description="Optional context for the prompt.")
    additional_info: Optional[Dict[str, Any]] = Field(default=None, description="Any additional information.")
