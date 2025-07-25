import os
import uuid
import mimetypes
from typing import Optional, Dict, Any, List, Literal

from pydantic import BaseModel, Field, validator

# langchain langraph imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain.graph import StateGraph, MessageState, End, Start
from langraph.checkpoint.memory import MemorySaver

# Google Gemini imports (LLM Model)
from langchain_google_gemini import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

# Mimetypes initialization
try:
    import magic
except ImportError:
    print("python-magic not installed, using default mimetypes")
    magic = None

