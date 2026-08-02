import os
from typing import Dict, Any, List
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.tools.retriever import create_retriever_tool
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage

from state import NexusState

