# agent/state.py

from typing import List, Optional
from pydantic import BaseModel, Field
from langchain_core.messages import BaseMessage # Crucial for messages list


class AgentState(BaseModel):
    """
    Represents the state of the conversational agent at any point in the graph.
    This state is passed between nodes and updated by them.
    """
    messages: List[BaseMessage] = Field(
        default_factory=list,
        description="List of all messages in the conversation history. "
                    "This includes HumanMessages (user input), AIMessages (AI responses "
                    "and tool calls), and ToolMessages (outputs from tool executions)."
    )
    
    # --- Optional Fields (Consider adding only if truly necessary for direct access by nodes) ---
    current_member_id: Optional[str] = Field(
        None,
        description="The confirmed member ID being discussed in the current conversation thread. "
                    "This can be populated by the agent node if a member ID is clearly identified "
                    "and needs to be persistent across multiple tool calls without re-parsing messages."
    )
    # Example of another optional field if you later need to track a specific claim
    # current_claim_id: Optional[str] = Field(
    #     None,
    #     description="The claim ID currently being processed or discussed."
    # )

    # Example of a flag for a specific state
    # requires_clarification_on_plan: bool = Field(
    #     False,
    #     description="Flag indicating if the agent needs to ask for plan type clarification."
    # )

    # You can add more fields here if you find a specific piece of information
    # is frequently needed by multiple nodes and is difficult to parse directly
    # from the messages list in every step.
