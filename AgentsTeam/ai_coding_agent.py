import os
import re
import base64
from typing import Dict,Any,Optional
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, START, END
from e2b_code_interpreter import Sandbox

from state import NexusState