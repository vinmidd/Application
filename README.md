graph TD
    %% Graph Title
    title: APES Conversational Agent Flow (LangGraph)

    %% Nodes Definition
    start([User Input])
    agent{Agent Node: LLM Reasoning &amp; Decision}
    tool_executor[Tool Executor Node: Execute API Calls]
    final_response((End: Provide Final Response))

    %% Edges (Flow)
    start --&gt; agent

    %% Conditional Edges from Agent Node
    agent -- &quot;LLM wants to call tool(s)&quot; --&gt; tool_executor
    agent -- &quot;LLM ready to respond / No tool calls&quot; --&gt; final_response

    %% Edge from Tool Executor Node
    tool_executor --&gt; agent

    %% Styling (Optional - enhances readability)
    classDef llmNode fill:#ADD8E6,stroke:#333,stroke-width:2px,color:#000;
    classDef toolNode fill:#90EE90,stroke:#333,stroke-width:2px,color:#000;
    classDef endNode fill:#FFB6C1,stroke:#333,stroke-width:2px,color:#000;
    class agent llmNode;
    class tool_executor toolNode;
    class final_response endNode;
