import os
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain/LangGraph specific imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SQLiteSaver # The key import

# --- Load environment variables ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# --- 1. Define AgentState ---
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list)
    current_member_id: Optional[str] = Field(None)

# --- 2. Define Mock Tools (for demonstration) ---
class MockApiClient:
    def get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        print(f"MOCK API CALL: GET {endpoint} with {params}")
        if endpoint == "/id-list" and params.get("member_id") == "12345":
            return {"ids": ["MED-ID-12345-A", "MED-ID-12345-B"], "primary_id": "MED-ID-12345-A"}
        return {"result": "Mock data for " + endpoint}

    def post(self, endpoint: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"MOCK API CALL: POST {endpoint} with {json_data}")
        return {"status": "Mock success for " + endpoint}

_mock_api_client = MockApiClient()

@tool
def get_id_list(member_id: str) -> str:
    """Retrieves a list of all active Medicare ID cards."""
    return str(_mock_api_client.get("/id-list", {"member_id": member_id}))

@tool
def get_dental_coverage_status(member_id: str) -> str:
    """Checks the dental coverage status for a specific member."""
    return f"Member {member_id} has active dental coverage."

all_tools = [get_id_list, get_dental_coverage_status]

# --- 3. Define System Prompt ---
SYSTEM_PROMPT = """You are a helpful Medicare Assistant. Use the tools to answer questions.
If you need to make multiple API calls to answer a question, chain them together as needed.
Synthesize information from tool outputs into clear, user-friendly responses.
"""

# --- 4. Define Agent Node ---
def call_agent(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, List[BaseMessage]]:
    llm = config["llm"]
    tools_to_bind = config["tools"]
    llm_with_tools = llm.bind_tools(tools_to_bind)
    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + state.messages
    result = llm_with_tools.invoke(messages_for_llm)
    return {"messages": [result]}

# --- 5. Define Tool Executor Node ---
tool_executor = ToolNode(all_tools)

# --- 6. Define Router Function ---
def should_continue(state: AgentState) -> Literal["call_tool", "respond"]:
    last_message = state.messages[-1]
    if last_message.tool_calls:
        return "call_tool"
    else:
        return "respond"

# --- 7. Build the LangGraph Workflow ---
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_agent)
workflow.add_node("tool_executor", tool_executor)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {"call_tool": "tool_executor", "respond": END})
workflow.add_edge("tool_executor", "agent")

# --- 8. Configure State Persistence with SQLiteSaver ---
memory = SQLiteSaver.from_conn_string("sqlite:///conversations.db")

# --- 9. Compile the graph with the checkpointer ---
app_graph = workflow.compile(checkpointer=memory)

# --- Runnable Example ---
if __name__ == "__main__":
    llm_model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

    # Example 1: New conversation or resuming existing one
    conv_id_1 = "user_session_alpha"
    print(f"\n===== Conversation ID: {conv_id_1} =====")
    config_1 = {"configurable": {"thread_id": conv_id_1}, "llm": llm_model, "tools": all_tools}

    # Turn 1
    print("\nUser: What is my ID card status? My member ID is 12345.")
    result_1 = app_graph.invoke(
        {"messages": [HumanMessage(content="What is my ID card status? My member ID is 12345.")]},
        config=config_1
    )
    print(f"AI: {result_1['messages'][-1].content}")

    # Turn 2 (same conversation)
    print("\nUser: Can you also tell me about my dental coverage?")
    result_2 = app_graph.invoke(
        {"messages": [HumanMessage(content="Can you also tell me about my dental coverage?")]},
        config=config_1
    )
    print(f"AI: {result_2['messages'][-1].content}")

    # Example 2: A different conversation
    conv_id_2 = "user_session_beta"
    print(f"\n\n===== Conversation ID: {conv_id_2} =====")
    config_2 = {"configurable": {"thread_id": conv_id_2}, "llm": llm_model, "tools": all_tools}

    print("\nUser: Hi there!")
    result_3 = app_graph.invoke(
        {"messages": [HumanMessage(content="Hi there!")]},
        config=config_2
    )
    print(f"AI: {result_3['messages'][-1].content}")

    print(f"\n\n--- Check 'conversations.db' file in this directory for saved states. ---")
