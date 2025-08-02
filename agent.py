# agent.py

import os
import uuid
import mimetypes
from typing import Optional, Dict, Any, List, Literal
import json
from pydantic import BaseModel, Field

# langchain langraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import MemorySaver

# Google Gemini imports (LLM Model)
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.generativeai.types import GenerationConfig
from dotenv import load_dotenv

# Mimetypes initialization
try:
    import magic
except ImportError:
    print("python-magic not installed, using default mimetypes")
    magic = None

# llm setup
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)


# resume parser logic
def _read_file_bytes(file_path: str) -> bytes:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Required example file not found: {file_path}. Ensure it's in the same directory.")
    with open(file_path, "rb") as f:
        return f.read()

def get_mime_type(file_path: str) -> str:
    if magic:
        mime_type = magic.from_file(file_path, mime=True)
    else:
        mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type is None:
        raise ValueError(f"Could not determine MIME type for file: {file_path}")
    return mime_type

EXAMPLE_PDF_PATH = "Ritik Dutta Resume-June.pdf"
EXAMPLE_IMAGE_PATH = "Ritik Dutta Resume-June.jpg"
FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {"role": "user", "parts": [{"mime_type": "application/pdf", "data": _read_file_bytes(EXAMPLE_PDF_PATH)}]},
    {"role": "model", "parts": [{"text": "### **Personal Details**\n* **Name:** Ritik Dutta\n* **Title:** Data Scientist & AI Engineer...\n---"}]},
    {"role": "user", "parts": [{"mime_type": "image/jpeg", "data": _read_file_bytes(EXAMPLE_IMAGE_PATH)}]},
    {"role": "model", "parts": [{"text": "### **Personal Details**\n* **Name:** Ritik Dutta\n* **Title:** Data Scientist & AI Engineer...\n---"}]},
]

def extract_resume_info(file_path: str) -> str:
    system_instruction = "You are an expert resume parser. Extract all content from the provided resume (PDF or image) and format it clearly using Markdown."
    # using a fast model for parsing
    model = genai.GenerativeModel("gemini-2.5-flash", system_instruction=system_instruction)
    mime_type = get_mime_type(file_path)
    file_data = _read_file_bytes(file_path)
    user_prompt = {"role": "user", "parts": [{"mime_type": mime_type, "data": file_data}]}
    contents = FEW_SHOT_EXAMPLES + [user_prompt]
    response = model.generate_content(
        contents=contents,
        generation_config=GenerationConfig(response_mime_type="text/plain"),
        request_options={"timeout": 600}
    )
    return response.text

# memory management and user profile management
class InterviewProfile(BaseModel):
    user_name: Optional[str] = Field(default=None, description="The user's full name.")
    domain: Optional[str] = Field(default=None, description="The job domain.")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    technical_skills: List[str] = Field(default_factory=list)
    project_experience: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)

memory_store: Dict[str, Any] = {}
def get_profile(user_id: str) -> InterviewProfile:
    return InterviewProfile(**memory_store.get(user_id, {}))
def save_profile(user_id: str, profile: InterviewProfile):
    memory_store[user_id] = profile.model_dump()


# tools
@tool
def resume_extraction_tool(file_path: str) -> str:
    """Reads a resume file (PDF or image) and extracts its text. Use this when a user provides a file path."""
    if not os.path.exists(file_path):
        return f"Error: File not found at path: {file_path}."
    try:
        return extract_resume_info(file_path)
    except Exception as e:
        return f"An error occurred during resume extraction: {e}"
@tool
def update_interview_profile(
    user_name: Optional[str] = None, domain: Optional[str] = None, strengths: Optional[List[str]] = None,
    weaknesses: Optional[List[str]] = None, technical_skills: Optional[List[str]] = None,
    project_experience: Optional[List[str]] = None, personality_traits: Optional[List[str]] = None
) -> str:
    """Updates the user's interview profile. Use this to save details from a conversation or after analyzing resume text."""
    return "Profile update request received. The system will process it."

# system prompt for the agent
AGENT_SYSTEM_PROMPT = """**Your Persona:** You are a friendly, professional, and highly skilled AI Interview Coach.

**Your Primary Goal:** Your main job is to engage the user in a realistic, conversational mock interview. **You must use the information you gather to ask progressively deeper, more relevant, and more challenging questions.**

**Your Secondary, Silent Task: Profile Management**
You have tools to remember information about the user. This is a background task.
- As the user speaks, you MUST analyze their responses for key information (name, skills, experience, strengths, weaknesses).
- When you identify new information, silently call the `update_interview_profile` tool to save it.
- **CRITICAL RULE:** NEVER mention your tools or the fact that you are "updating a profile." This breaks the immersion. Act like a human interviewer who just naturally remembers things.

**Conversation Flow:**
1.  **Greeting & Setup:** Start with a warm, natural greeting. Ask the user what job domain they are preparing for.
2.  **Optional Resume Intake:** Once you know their domain, offer them the *option* to provide a resume. Phrase it casually, like: "Great! To help me tailor my questions, you can optionally provide a file path to your resume. Or, we can just jump right in. What works for you?"
3.  **The Mock Interview:**
    - Acknowledge any resume processing and then START the interview. A great first question is always, "So, to get started, could you tell me a little bit about yourself?".
    - Continue asking relevant questions one by one, keeping the conversation flowing by responding to their answers naturally before moving on.

    **--- Question Strategy & The Art of Contextual Questioning ---**
    **This is your most important instruction. What separates a good coach from a great one is the ability to ask deeply contextual questions.**

    *   **Weave in the Details:** You MUST look at the user's profile before asking a question. Then, **weave specific details from their profile directly into the question itself.** This demonstrates that you have understood their background and makes the interview far more realistic and effective.

    *   **Compare these two styles:**

        *   **Bad Question (Generic):** "Could you tell me about a challenging project?"

        *   **Excellent Question (Contextual & Specific):** "Looking at your profile, I see you were a **Computer Vision Engineer Intern at Ineuron** and have experience **deploying models on AWS/GCP**. Could you tell me about a time you faced a significant challenge during the **deployment phase** of a machine learning model? What was the issue, and how did you troubleshoot and resolve it to ensure reliability?"

    *   **Mix Question Types:** Use this contextual approach across a variety of question types to effectively probe the user's abilities:
        - **Technical:** "I see you've listed 'SQL' as a skill. Can you explain the difference between a `LEFT JOIN` and an `INNER JOIN` using a practical example?"
        - **Behavioral:** "Your resume mentions you 'led a team project'. Tell me about a time you had to manage a conflict within that team."
        - **Situational / Creative:** "Given your background in e-commerce, imagine our website's recommendation engine suddenly starts suggesting completely irrelevant products. What are your first three steps to diagnose the problem?"
        - **Office / Client-Facing:** "You have experience with both technical and business stakeholders. How would you explain a complex topic like 'model overfitting' to a non-technical sales manager?"

**Your Memory (Current User Profile):**
You have access to the user's current profile. Use this information to ask more relevant questions and avoid asking for the same information twice.
<interview_profile>
{interview_profile}
</interview_profile>
"""

# using a powerful model for the main agent's reasoning
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
tools = [resume_extraction_tool, update_interview_profile]
model_with_tools = model.bind_tools(tools)

# graph and conversation flow
def agent_node(state: MessagesState, config: dict):
    user_id = config["configurable"]["user_id"]
    profile = get_profile(user_id)
    system_prompt = AGENT_SYSTEM_PROMPT.format(interview_profile=profile.model_dump_json(indent=2))
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}

def tool_node(state: MessagesState, config: dict):
    user_id = config["configurable"]["user_id"]
    tool_calls = state["messages"][-1].tool_calls
    tool_messages = []
    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        if tool_name == resume_extraction_tool.name:
            raw_resume_text = resume_extraction_tool.invoke(tool_call["args"])
            parser_prompt = f"You are a data extraction specialist. Analyze the following resume text and call the `update_interview_profile` tool with all the relevant information you can find.\n\nResume Text:\n---\n{raw_resume_text}\n---"
            parser_model = model.bind_tools([update_interview_profile])
            parser_response = parser_model.invoke(parser_prompt)
            if parser_response.tool_calls:
                update_tool_call = parser_response.tool_calls[0]
                profile = get_profile(user_id)
                profile_data = profile.model_dump()
                new_data = update_tool_call["args"]
                for key, value in new_data.items():
                    if value is not None:
                        if isinstance(profile_data.get(key), list) and isinstance(value, list):
                            profile_data[key].extend(v for v in value if v not in profile_data[key])
                        else:
                            profile_data[key] = value
                save_profile(user_id, InterviewProfile(**profile_data))
                tool_messages.append(ToolMessage(content="Successfully processed the resume and updated the profile.", tool_call_id=tool_call["id"]))
            else:
                tool_messages.append(ToolMessage(content="Could not automatically parse the resume.", tool_call_id=tool_call["id"]))
        elif tool_name == update_interview_profile.name:
            profile = get_profile(user_id)
            profile_data = profile.model_dump()
            new_data = tool_call["args"]
            for key, value in new_data.items():
                if value is not None and isinstance(profile_data.get(key), list) and isinstance(value, list):
                    profile_data[key].extend(v for v in value if v not in profile_data[key])
                elif value is not None:
                    profile_data[key] = value
            save_profile(user_id, InterviewProfile(**profile_data))
            tool_messages.append(ToolMessage(content="Profile updated based on conversation.", tool_call_id=tool_call["id"]))
    return {"messages": tool_messages}

def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END

builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")
memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# main streaming function for the flask app
def stream_agent_events(user_id: str, thread_id: str, user_message: str):
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    messages = [HumanMessage(content=user_message)]
    
    try:
        events = graph.stream({"messages": messages}, config, stream_mode="values")
        
        for event in events:
            last_message = event["messages"][-1]

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                tool_name = last_message.tool_calls[0]['name']
                payload = {"type": "status", "content": f"Calling tool: `{tool_name}`..."}
                yield json.dumps(payload)

            elif isinstance(last_message, ToolMessage):
                status_content = "Tool execution finished. Analyzing results..."
                payload = {"type": "status", "content": status_content}
                yield json.dumps(payload)

            elif isinstance(last_message, AIMessage):
                payload = {"type": "final_response", "content": last_message.content}
                yield json.dumps(payload)

    except Exception as e:
        print(f"An error occurred during agent streaming: {e}")
        error_payload = {"type": "error", "content": "An error occurred in the agent. Check server logs."}
        yield json.dumps(error_payload)