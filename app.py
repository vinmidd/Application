# app.py (Core LangGraph Workflow Definition)

import os
from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# LangChain/LangGraph specific imports
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool # Import the tool decorator for API wrappers
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.sqlite import SQLiteSaver

# --- Load environment variables (for OpenAI API Key) ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")

# --- 1. Define your AgentState (GraphState) ---
# This defines the structure of the state that flows through your graph
class AgentState(BaseModel):
    messages: List[BaseMessage] = Field(default_factory=list, description="List of all messages in the conversation, including user input, AI responses, and tool outputs.")
    # Optional: You can add other useful state variables here if needed globally across turns
    # For example, to store the current member ID after it's identified:
    current_member_id: Optional[str] = Field(None, description="The member ID currently being discussed.")
    # You could add a flag for tracking if a critical piece of info is missing
    # waiting_for_member_id: bool = False

# --- 2. Define your Tools (API Wrappers) ---
# In a real project, these would be in agent/tools/*.py and imported.
# For this example, we define them here to make the graph self-contained.

# Placeholder for shared API client logic (actual implementation in agent/tools/api_client.py)
class MockApiClient:
    def get(self, endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
        print(f"MOCK API CALL: GET {endpoint} with {params}")
        if endpoint == "/id-list" and params.get("member_id") == "12345":
            return {"ids": ["MED-ID-12345-A", "MED-ID-12345-B"], "primary_id": "MED-ID-12345-A"}
        if endpoint == "/id-status" and params.get("id") == "MED-ID-12345-A":
            return {"status": "Shipped", "tracking_number": "TRK789", "estimated_delivery": "2025-07-08"}
        if endpoint == "/comets-data" and params.get("id") == "MED-ID-12345-A":
             return {"comets_status": "Active", "last_update": "2025-06-30", "details": "No issues detected."}
        # Add more mock responses for other endpoints/tools
        return {"error": "Not Found", "message": f"No mock data for {endpoint} with {params}"}

    def post(self, endpoint: str, json_data: Dict[str, Any]) -> Dict[str, Any]:
        print(f"MOCK API CALL: POST {endpoint} with {json_data}")
        if endpoint == "/new-id-card-request":
            return {"request_id": "REQ9876", "status": "Pending", "message": "New ID card request submitted."}
        # Add more mock responses for other endpoints/tools
        return {"error": "Not Implemented", "message": f"No mock data for POST {endpoint} with {json_data}"}

_mock_api_client = MockApiClient() # Instantiate the mock client

@tool
def get_id_list(member_id: str) -> str:
    """
    Retrieves a list of all active Medicare ID cards associated with a given member ID.
    Use this tool when the user asks to 'list my ID cards', 'what IDs do I have', or as a first step to find an ID before checking its status.
    Args:
        member_id (str): The unique identifier for the member.
    """
    response = _mock_api_client.get("/id-list", {"member_id": member_id})
    return str(response) # Tools typically return strings

@tool
def get_id_card_status(id: str) -> str:
    """
    Retrieves the current shipping or activation status of a specific Medicare ID card.
    Use this tool when the user provides a specific ID card number and asks about its 'status', 'where is it', or 'is it active'.
    Args:
        id (str): The specific ID card number (e.g., MED-ID-12345-A).
    """
    response = _mock_api_client.get("/id-status", {"id": id})
    return str(response)

@tool
def get_comets_data(id: str) -> str:
    """
    Fetches detailed COmets system data related to a specific Medicare ID card, often containing additional
    status, activation, or historical information not found in basic ID card status.
    Use this tool as a follow-up to get_id_card_status for more granular details, especially for activation or deeper troubleshooting.
    Args:
        id (str): The specific ID card number (e.g., MED-ID-12345-A).
    """
    response = _mock_api_client.get("/comets-data", {"id": id})
    return str(response)

@tool
def request_new_id_card(member_id: str, reason: str) -> str:
    """
    Submits a request to issue and mail a new Medicare ID card to the member's registered address.
    Use this tool when the user explicitly asks to 'request a new ID card', 'send me a replacement card', or 'order a new card'.
    Args:
        member_id (str): The unique identifier for the member.
        reason (str): The reason for the new card request (e.g., 'lost', 'stolen', 'damaged').
    """
    response = _mock_api_client.post("/new-id-card-request", {"member_id": member_id, "reason": reason})
    return str(response)

@tool
def get_member_benefits(member_id: str, plan_type: str) -> str:
    """
    Retrieves a detailed list of benefits associated with a member's specific plan type.
    Use this tool when the user asks about 'benefits', 'coverage details', or 'what's included'
    for a given member and plan.
    Args:
        member_id (str): The unique identifier for the member.
        plan_type (str): The type of plan (e.g., 'HMO', 'PPO', 'Medicare Advantage').
    """
    # This is a placeholder; would call _mock_api_client.get() or similar
    print(f"MOCK TOOL CALL: get_member_benefits for {member_id}, plan {plan_type}")
    if member_id == "67890" and plan_type.lower() == "hmo":
        return "Member 67890 under HMO plan has dental, vision, and prescription benefits."
    return "Benefits not found for provided details."

@tool
def get_dental_coverage_status(member_id: str) -> str:
    """
    Checks the dental coverage status for a specific member.
    Use this tool when the user asks specifically about 'dental coverage', 'is my dental covered'.
    Args:
        member_id (str): The unique identifier for the member.
    """
    print(f"MOCK TOOL CALL: get_dental_coverage_status for {member_id}")
    if member_id == "12345":
        return "Member 12345 has active dental coverage with 80% co-insurance for preventative care."
    return "Dental coverage status not found for provided member ID."

@tool
def get_member_status(member_id: str) -> str:
    """
    Retrieves the general enrollment status of a member (e.g., active, inactive, pending).
    Use this tool when the user asks about their 'membership status', 'are they active', or general enrollment queries.
    Args:
        member_id (str): The unique identifier for the member.
    """
    print(f"MOCK TOOL CALL: get_member_status for {member_id}")
    if member_id == "12345":
        return "Member 12345 is currently in active enrollment status."
    return "Member status not found for provided ID."


# Collect all your tools here (in actual project, this would be imported from agent/__init__.py)
all_tools = [
    get_id_list,
    get_id_card_status,
    get_comets_data,
    request_new_id_card,
    get_member_benefits,
    get_dental_coverage_status,
    get_member_status
]

# --- 3. Define the LLM's System Prompt ---
SYSTEM_PROMPT = """You are a helpful and polite Medicare Assistant. Your primary goal is to assist users with their Medicare ID card and member-related inquiries by leveraging the tools at your disposal.

Here's how you should operate:

1.  **Understand the User's Request:** Carefully analyze the user's message and the conversation history to understand their core need.

2.  **Utilize Tools (APIs) When Necessary:**
    * You have access to a suite of specialized tools (functions that interact with Medicare backend systems).
    * **Always consider if a tool is needed to fulfill the user's request.** If information is required from the Medicare system or an action needs to be performed (like requesting a new ID card), you **must** use the appropriate tool.
    * **Prioritize using the most specific tool available** for the user's query.
    * **Extract all necessary parameters** for the chosen tool from the conversation context. If you cannot find a required parameter, ask the user for it clearly.
    * **Be precise with tool arguments.** Ensure the arguments you pass to the tool exactly match its schema and requirements.

3.  **Process Tool Outputs:**
    * After a tool executes, you will receive its output as a ToolMessage. Analyze this output carefully.
    * Use the tool's output to formulate a helpful and accurate response to the user.
    * **If one API call's result indicates the need for another API call to fully answer the query (e.g., getting an ID list before checking an ID's status), make the subsequent tool call.** You are expected to chain tools together as needed.

4.  **Handle Missing Information/Clarification:**
    * If the user's query is ambiguous or if you lack a critical piece of information (e.g., a member ID) required by a tool, politely ask the user for clarification. State exactly what information you need.

5.  **Handle Out-of-Scope Queries:**
    * If a user's request is completely outside the scope of Medicare ID cards or member benefits (i.e., you don't have a tool or internal knowledge to answer it), politely state that you cannot assist with that specific request and offer to help with Medicare-related inquiries.

6.  **Generate User-Friendly Responses:**
    * Keep your responses clear, concise, and easy for the user to understand.
    * Maintain a polite and professional tone.
    * Do not share internal tool names or raw API responses directly with the user. Translate all information from tool outputs into natural, user-friendly language.
    * If multiple API calls are made, synthesize the information from all relevant tool outputs into a coherent, single answer for the user.

7.  **Think Step-by-Step (Internal Monologue - for advanced LLMs):**
    * For complex queries, it can be helpful for you to internally "think" about your reasoning process before deciding on an action or response. (Some LLM models support a `reasoning` or `thought` field in their output when configured, which can be useful for debugging and fine-tuning.)

**Examples of when to use tools (and associated queries):**
* **`get_id_list`**: "List my ID cards.", "What Medicare IDs do I have?"
* **`get_id_card_status`**: "What's the status of my ID card?", "Where is my new Medicare card?"
* **`request_new_id_card`**: "I need a new ID card.", "Can you send me a replacement card?"
* **`get_member_benefits`**: "What are the benefits of my plan?", "Does my plan cover X?"
* **`get_dental_coverage_status`**: "Is dental included in my Medicare plan?", "Do I have dental coverage?"
* **`get_member_status`**: "Am I an active member?", "What's my membership status?"

**Begin the conversation. I will provide the user's messages and any tool outputs.**
"""

# --- 4. Define the Agent Node (`call_agent`) ---
# This is where the LLM makes decisions
def call_agent(state: AgentState, config: Optional[RunnableConfig] = None) -> Dict[str, List[BaseMessage]]:
    # Retrieve LLM and tools from the config provided during graph invocation
    llm = config["llm"]
    tools_to_bind = config["tools"]

    # Bind the tools to the LLM. This is how the LLM knows what tools are available.
    llm_with_tools = llm.bind_tools(tools_to_bind)

    # Construct the full message list for the LLM
    # The SystemMessage ensures the LLM always remembers its core instructions
    messages_for_llm = [SystemMessage(content=SYSTEM_PROMPT)] + state.messages

    print(f"\n--- Agent Node: LLM Reasoning (Messages for LLM) ---")
    for msg in messages_for_llm:
        print(f"  {type(msg).__name__}: {msg.content if not msg.tool_calls else 'Tool Calls: ' + str(msg.tool_calls)}")

    # Invoke the LLM with the current state's messages
    result = llm_with_tools.invoke(messages_for_llm)

    print(f"--- Agent Node: LLM Output ---")
    print(f"  Type: {type(result).__name__}")
    if result.tool_calls:
        print(f"  Tool Calls: {result.tool_calls}")
    else:
        print(f"  Content: {result.content}")

    # Return the LLM's response to update the graph state
    return {"messages": [result]}

# --- 5. Define the Tool Executor Node (`tool_executor`) ---
# This node automatically executes any tool calls made by the LLM
tool_executor = ToolNode(all_tools) # Initialize with all your defined tools

# --- 6. Define the Router Function (`should_continue`) ---
# This function determines the next step in the graph based on the LLM's output
def should_continue(state: AgentState) -> Literal["call_tool", "respond"]:
    last_message = state.messages[-1]
    # If the last message contains tool calls, it means the LLM wants to use a tool
    if last_message.tool_calls:
        print(f"\n--- Router: LLM wants to call a tool. Routing to 'tool_executor'. ---")
        return "call_tool"
    else:
        # Otherwise, the LLM has generated a final response (or clarification)
        print(f"\n--- Router: LLM wants to respond. Routing to 'END'. ---")
        return "respond"

# --- 7. Build the LangGraph Workflow ---
# Create an instance of StateGraph with our AgentState
workflow = StateGraph(AgentState)

# Add the nodes to the workflow
workflow.add_node("agent", call_agent)
workflow.add_node("tool_executor", tool_executor) # ToolNode is already defined above

# Set the entry point for the graph
workflow.set_entry_point("agent")

# Define the conditional edges from the 'agent' node
# This dictates where the flow goes after the LLM makes a decision
workflow.add_conditional_edges(
    "agent",          # From the 'agent' node
    should_continue,  # Use the 'should_continue' function to determine the next node
    {
        "call_tool": "tool_executor", # If LLM wants to call a tool, go to 'tool_executor'
        "respond": END,               # If LLM wants to respond, end the graph execution
    },
)

# Define the edge from the 'tool_executor' node
# After a tool is executed, the flow always returns to the 'agent' for further reasoning
workflow.add_edge("tool_executor", "agent")

# --- 8. Configure State Persistence with SQLiteSaver ---
# 'sqlite:///conversations.db' tells SQLiteSaver to use a file named conversations.db
# This file will be created/updated in your project's root directory.
memory = SQLiteSaver.from_conn_string("sqlite:///conversations.db")

# --- 9. Compile the graph ---
# Compiling the workflow prepares it for execution and attaches the checkpointer
app_graph = workflow.compile(checkpointer=memory)

# --- Runnable Example: How to Interact with the Graph ---
if __name__ == "__main__":
    # Initialize your LLM
    # Ensure OPENAI_API_KEY is set in your .env file or environment
    llm_model = ChatOpenAI(model="gpt-4o", temperature=0, openai_api_key=OPENAI_API_KEY)

    # --- Conversation 1: Multi-step ID Card Status Inquiry ---
    conversation_id_1 = "user_session_medicare_001"
    print(f"\n===== Conversation ID: {conversation_id_1} =====")

    # Turn 1: User asks about ID card status
    print("\n--- User Query (Turn 1): What's the status of my ID card? My member ID is 12345.")
    config_1_turn1 = {
        "configurable": {"thread_id": conversation_id_1},
        "llm": llm_model,
        "tools": all_tools # Pass all tools so the LLM knows what's available
    }
    result_1_turn1 = app_graph.invoke(
        {"messages": [HumanMessage(content="What's the status of my ID card? My member ID is 12345.")]},
        config=config_1_turn1
    )
    # The final message in the state will be the LLM's response
    print(f"\nAI Response (Turn 1): {result_1_turn1['messages'][-1].content}")


    # Turn 2: Follow-up question in the same conversation
    print("\n--- User Query (Turn 2): Can you tell me about my dental coverage?")
    config_1_turn2 = {
        "configurable": {"thread_id": conversation_id_1},
        "llm": llm_model,
        "tools": all_tools
    }
    result_1_turn2 = app_graph.invoke(
        {"messages": [HumanMessage(content="Can you tell me about my dental coverage?")]},
        config=config_1_turn2
    )
    print(f"\nAI Response (Turn 2): {result_1_turn2['messages'][-1].content}")

    # --- Conversation 2: New request, different conversation ID ---
    conversation_id_2 = "user_session_medicare_002"
    print(f"\n\n===== Conversation ID: {conversation_id_2} =====")

    print("\n--- User Query (New Conversation): I need to request a new ID card. My member ID is 67890. I lost it.")
    config_2_turn1 = {
        "configurable": {"thread_id": conversation_id_2},
        "llm": llm_model,
        "tools": all_tools
    }
    result_2_turn1 = app_graph.invoke(
        {"messages": [HumanMessage(content="I need to request a new ID card. My member ID is 67890. I lost it.")]},
        config=config_2_turn1
    )
    print(f"\nAI Response (New Conv): {result_2_turn1['messages'][-1].content}")


    print(f"\n\n--- Done. Check 'conversations.db' file in this directory for saved states. ---")
