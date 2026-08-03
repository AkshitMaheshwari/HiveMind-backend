import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Literal
from typing_extensions import TypedDict

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun
from langgraph.graph import StateGraph, START, END
from langchain.agents import create_agent

from google import genai
from google.genai import types

from state import NexusState

OUTPUTS_DIR = Path("outputs")
OUTPUTS_DIR.mkdir(exist_ok=True)