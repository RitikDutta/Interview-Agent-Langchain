# Updated Agent.py with Vector Database Integration
import os
import uuid
import mimetypes
from typing import Optional, Dict, Any, List, Literal
import json
from pydantic import BaseModel, Field

# LangChain LangGraph imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.checkpoint.memory import MemorySaver

# Google Gemini imports (LLM Model)
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from google.generativeai.types import GenerationConfig

# Vector database integration
from vector_store import VectorStore
from dotenv import load_dotenv

# Mimetypes initialization
try:
    import magic
except ImportError:
    print("python-magic not installed, using default mimetypes")
    magic = None

# LLM setup
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY not found in environment variables. Please set it in your .env file.")

genai.configure(api_key=GOOGLE_API_KEY)

# Initialize vector store
vector_store = VectorStore(api_key=PINECONE_API_KEY)

# Resume parser logic (enhanced with vector storage)
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

# Example files for few-shot learning
EXAMPLE_PDF_PATH = "Ritik Dutta Resume-June.pdf"
EXAMPLE_IMAGE_PATH = "Ritik Dutta Resume-June.jpg"

FEW_SHOT_EXAMPLES: List[Dict[str, Any]] = [
    {"role": "user", "parts": [{"mime_type": "application/pdf", "data": _read_file_bytes(EXAMPLE_PDF_PATH)}]},
    {"role": "model", "parts": [{"text": "### Personal Details\n* Name: Ritik Dutta\n* Title: Data Scientist & AI Engineer...\n---"}]},
    {"role": "user", "parts": [{"mime_type": "image/jpeg", "data": _read_file_bytes(EXAMPLE_IMAGE_PATH)}]},
    {"role": "model", "parts": [{"text": "### Personal Details\n* Name: Ritik Dutta\n* Title: Data Scientist & AI Engineer...\n---"}]},
]

def extract_resume_info(file_path: str) -> str:
    """Extract resume information and return structured text"""
    system_instruction = """You are an expert resume parser. Extract all content from the provided resume (PDF or image) and format it clearly using Markdown.
    
    Structure the output with clear sections:
    - Personal Details (Name, Contact, Location)
    - Professional Summary
    - Technical Skills
    - Work Experience (with specific projects and achievements)
    - Education
    - Certifications
    - Projects
    
    Be thorough and extract all relevant information."""
    
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

# Enhanced user profile with vector database integration
class InterviewProfile(BaseModel):
    user_name: Optional[str] = Field(default=None, description="The user's full name.")
    domain: Optional[str] = Field(default=None, description="The job domain.")
    experience_level: Optional[str] = Field(default=None, description="Experience level (junior, mid, senior)")
    strengths: List[str] = Field(default_factory=list)
    weaknesses: List[str] = Field(default_factory=list)
    technical_skills: List[str] = Field(default_factory=list)
    project_experience: List[str] = Field(default_factory=list)
    personality_traits: List[str] = Field(default_factory=list)
    interview_history: List[Dict[str, Any]] = Field(default_factory=list, description="Previous interview sessions")

# Memory management with vector database
memory_store: Dict[str, Any] = {}

def get_profile(user_id: str) -> InterviewProfile:
    """Get user profile from memory store"""
    return InterviewProfile(**memory_store.get(user_id, {}))

def save_profile(user_id: str, profile: InterviewProfile):
    """Save user profile to memory store"""
    memory_store[user_id] = profile.model_dump()

# Enhanced tools with vector database integration
@tool
def resume_extraction_tool(file_path: str) -> str:
    """Reads a resume file (PDF or image) and extracts its text. Also stores relevant chunks in vector database."""
    if not os.path.exists(file_path):
        return f"Error: File not found at path: {file_path}."
    
    try:
        resume_text = extract_resume_info(file_path)
        return resume_text
    except Exception as e:
        return f"An error occurred during resume extraction: {e}"

@tool
def update_interview_profile(
    user_name: Optional[str] = None,
    domain: Optional[str] = None,
    experience_level: Optional[str] = None,
    strengths: Optional[List[str]] = None,
    weaknesses: Optional[List[str]] = None,
    technical_skills: Optional[List[str]] = None,
    project_experience: Optional[List[str]] = None,
    personality_traits: Optional[List[str]] = None
) -> str:
    """Updates the user's interview profile. Use this to save details from a conversation or after analyzing resume text."""
    return "Profile update request received. The system will process it."

@tool
def search_relevant_questions(query: str, user_id: str, top_k: int = 5) -> str:
    """Search for relevant interview questions based on user profile and query using vector database."""
    try:
        # Get user profile for context
        profile = get_profile(user_id)
        
        # Enhance query with user context
        enhanced_query = f"""
        Domain: {profile.domain or 'general'}
        Experience Level: {profile.experience_level or 'unknown'}
        Technical Skills: {', '.join(profile.technical_skills) if profile.technical_skills else 'none specified'}
        Query: {query}
        """
        
        # Search vector database
        results = vector_store.search_questions(
            query=enhanced_query,
            top_k=top_k,
            filter_metadata={
                "domain": profile.domain.lower() if profile.domain else None,
                "experience_level": profile.experience_level.lower() if profile.experience_level else None
            }
        )
        
        if results:
            questions_text = "\n\n".join([
                f"Question {i+1}: {result['text']}\nDifficulty: {result.get('metadata', {}).get('difficulty', 'Unknown')}\nCategory: {result.get('metadata', {}).get('category', 'General')}"
                for i, result in enumerate(results)
            ])
            return f"Found {len(results)} relevant questions:\n\n{questions_text}"
        else:
            return "No relevant questions found in the database."
            
    except Exception as e:
        return f"Error searching for questions: {str(e)}"

@tool
def store_user_response(user_id: str, question: str, response: str, feedback: str = "") -> str:
    """Store user's interview response in vector database for future analysis and personalization."""
    try:
        # Get user profile for metadata
        profile = get_profile(user_id)
        
        metadata = {
            "user_id": user_id,
            "content_type": "user_response",
            "domain": profile.domain.lower() if profile.domain else "general",
            "experience_level": profile.experience_level.lower() if profile.experience_level else "unknown",
            "timestamp": str(uuid.uuid4()),
            "question": question,
            "feedback": feedback
        }
        
        # Store response in vector database
        vector_store.store_user_response(
            user_id=user_id,
            text=response,
            metadata=metadata
        )
        
        return "Response stored successfully for future personalization."
    except Exception as e:
        return f"Error storing response: {str(e)}"

# Enhanced system prompt with vector database context
AGENT_SYSTEM_PROMPT = """Your Persona: You are a friendly, professional, and highly skilled AI Interview Coach with access to a comprehensive question database and user response patterns.

Your Primary Goal: Your main job is to engage the user in a realistic, conversational mock interview. You now have access to:
1. A comprehensive question database with thousands of categorized interview questions
2. User response patterns and feedback history
3. Personalized question recommendations based on user profile

Your Enhanced Capabilities:
- Search for relevant questions using the search_relevant_questions tool
- Store user responses for continuous learning using store_user_response tool
- Access past performance patterns to provide better coaching

Your Secondary, Silent Task: Profile Management
You have tools to remember information about the user. This is a background task.
As the user speaks, you MUST analyze their responses for key information (name, skills, experience, strengths, weaknesses).
When you identify new information, silently call the update_interview_profile tool to save it.
CRITICAL RULE: NEVER mention your tools or the fact that you are "updating a profile." This breaks the immersion. Act like a human interviewer who just naturally remembers things.

Conversation Flow:
1. Greeting & Setup: Start with a warm, natural greeting. Ask the user what job domain they are preparing for.

2. Optional Resume Intake: Once you know their domain, offer them the option to provide a resume. Phrase it casually, like: "Great! To help me tailor my questions, you can optionally provide a file path to your resume. Or, we can just jump right in. What works for you?"

3. The Enhanced Mock Interview:
   - Acknowledge any resume processing and then START the interview
   - Use search_relevant_questions to find appropriate questions based on their profile
   - A great first question is always: "So, to get started, could you tell me a little bit about yourself?"
   - After each response, use store_user_response to save their answer for learning
   - Continue asking relevant questions using the vector database recommendations

--- Enhanced Question Strategy & Contextual Intelligence ---
Your questioning is now supercharged with database intelligence:

1. Database-Driven Questions: Before asking questions, search the vector database for relevant ones using their profile
2. Contextual Weaving: Take search results and weave specific details into questions
3. Progressive Difficulty: Start with easier questions and progressively increase difficulty based on responses
4. Adaptive Learning: Use stored responses to identify patterns and adjust questioning approach

Compare these approaches:
Bad Approach (Old): Generic questions without context
Excellent Approach (New): "I found some relevant questions for a {domain} professional with {experience_level} experience. Let me ask you about {specific_topic} - [tailored question from database]"

Question Categories to Leverage:
- Technical: Domain-specific technical questions from database
- Behavioral: Situation-based questions matched to their experience
- Situational: Problem-solving scenarios relevant to their field
- Company/Role-specific: Questions tailored to their target positions

Your Memory (Current User Profile):
You have access to the user's current profile and can search for relevant questions and patterns.

<interview_profile>
{interview_profile}
</interview_profile>

Enhanced Interview Flow:
1. Profile Assessment → Question Database Search → Personalized Question Selection
2. User Response → Response Storage → Feedback Analysis → Next Question Recommendation
3. Continuous Learning Loop for increasingly personalized interviews

Remember: Use the vector database to make every question count. Every interaction should feel tailored to their specific background and goals.
"""

# Using a powerful model for the main agent's reasoning
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GOOGLE_API_KEY)
tools = [resume_extraction_tool, update_interview_profile, search_relevant_questions, store_user_response]
model_with_tools = model.bind_tools(tools)

# Enhanced graph and conversation flow
def agent_node(state: MessagesState, config: dict):
    user_id = config["configurable"]["user_id"]
    profile = get_profile(user_id)
    
    # Format system prompt with current profile
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
            # Enhanced resume processing with vector storage
            raw_resume_text = resume_extraction_tool.invoke(tool_call["args"])
            
            if not raw_resume_text.startswith("Error:"):
                # Store resume chunks in vector database
                try:
                    profile = get_profile(user_id)
                    vector_store.store_resume_chunks(
                        user_id=user_id,
                        resume_text=raw_resume_text,
                        user_domain=profile.domain
                    )
                except Exception as e:
                    print(f"Warning: Could not store resume in vector database: {e}")
            
            # Parse resume and update profile
            parser_prompt = f"""You are a data extraction specialist. Analyze the following resume text and call the update_interview_profile tool with all the relevant information you can find.

Resume Text:
---
{raw_resume_text}
---"""
            
            parser_model = model.bind_tools([update_interview_profile])
            parser_response = parser_model.invoke(parser_prompt)
            
            if parser_response.tool_calls:
                update_tool_call = parser_response.tool_calls[0]
                profile = get_profile(user_id)
                profile_data = profile.model_dump()
                new_data = update_tool_call["args"]
                
                # Merge new data with existing profile
                for key, value in new_data.items():
                    if value is not None:
                        if isinstance(profile_data.get(key), list) and isinstance(value, list):
                            profile_data[key].extend(v for v in value if v not in profile_data[key])
                        else:
                            profile_data[key] = value
                
                save_profile(user_id, InterviewProfile(**profile_data))
                tool_messages.append(ToolMessage(
                    content="Successfully processed the resume, updated the profile, and stored resume chunks in vector database for enhanced question matching.",
                    tool_call_id=tool_call["id"]
                ))
            else:
                tool_messages.append(ToolMessage(
                    content="Could not automatically parse the resume.",
                    tool_call_id=tool_call["id"]
                ))
                
        elif tool_name == update_interview_profile.name:
            # Update profile as before
            profile = get_profile(user_id)
            profile_data = profile.model_dump()
            new_data = tool_call["args"]
            
            for key, value in new_data.items():
                if value is not None and isinstance(profile_data.get(key), list) and isinstance(value, list):
                    profile_data[key].extend(v for v in value if v not in profile_data[key])
                elif value is not None:
                    profile_data[key] = value
            
            save_profile(user_id, InterviewProfile(**profile_data))
            tool_messages.append(ToolMessage(
                content="Profile updated based on conversation.",
                tool_call_id=tool_call["id"]
            ))
            
        elif tool_name == search_relevant_questions.name:
            # Search for relevant questions
            result = search_relevant_questions.invoke(tool_call["args"])
            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            ))
            
        elif tool_name == store_user_response.name:
            # Store user response for learning
            result = store_user_response.invoke(tool_call["args"])
            tool_messages.append(ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            ))
    
    return {"messages": tool_messages}

def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    else:
        return END

# Build the enhanced graph
builder = StateGraph(MessagesState)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
builder.add_edge("tools", "agent")

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)

# Enhanced streaming function for the flask app
def stream_agent_events(user_id: str, thread_id: str, user_message: str):
    """Stream agent events with enhanced vector database integration"""
    config = {"configurable": {"thread_id": thread_id, "user_id": user_id}}
    messages = [HumanMessage(content=user_message)]
    
    try:
        events = graph.stream({"messages": messages}, config, stream_mode="values")
        
        for event in events:
            last_message = event["messages"][-1]

            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                tool_name = last_message.tool_calls[0]['name']
                
                # Provide more descriptive status messages
                status_messages = {
                    "resume_extraction_tool": "Processing your resume and storing it in our knowledge base...",
                    "search_relevant_questions": "Finding the most relevant questions for your profile...",
                    "store_user_response": "Learning from your response to personalize future questions...",
                    "update_interview_profile": "Updating your profile with new insights..."
                }
                
                status_content = status_messages.get(tool_name, f"Calling tool: {tool_name}...")
                payload = {"type": "status", "content": status_content}
                yield json.dumps(payload)

            elif isinstance(last_message, ToolMessage):
                status_content = "Analysis complete. Preparing personalized response..."
                payload = {"type": "status", "content": status_content}
                yield json.dumps(payload)

            elif isinstance(last_message, AIMessage):
                payload = {"type": "final_response", "content": last_message.content}
                yield json.dumps(payload)

    except Exception as e:
        print(f"An error occurred during agent streaming: {e}")
        error_payload = {"type": "error", "content": "An error occurred in the agent. Check server logs."}
        yield json.dumps(error_payload)

# Additional utility functions for vector database management
def initialize_user_session(user_id: str, domain: str = None, experience_level: str = None) -> str:
    """Initialize a new user session with optional domain and experience level"""
    profile = InterviewProfile(
        domain=domain,
        experience_level=experience_level
    )
    save_profile(user_id, profile)
    return f"User session initialized for {user_id}"

def get_user_progress(user_id: str) -> Dict[str, Any]:
    """Get user's interview progress and statistics"""
    profile = get_profile(user_id)
    
    # Get response patterns from vector database
    try:
        response_stats = vector_store.get_user_response_stats(user_id)
        return {
            "profile": profile.model_dump(),
            "response_stats": response_stats,
            "total_sessions": len(profile.interview_history)
        }
    except Exception as e:
        return {
            "profile": profile.model_dump(),
            "response_stats": {"error": str(e)},
            "total_sessions": len(profile.interview_history)
        }

def reset_user_profile(user_id: str) -> str:
    """Reset user profile and clear vector database entries"""
    try:
        # Clear from memory
        if user_id in memory_store:
            del memory_store[user_id]
        
        # Clear from vector database
        vector_store.delete_user_data(user_id)
        
        return f"Profile reset successfully for user {user_id}"
    except Exception as e:
        return f"Error resetting profile: {str(e)}"