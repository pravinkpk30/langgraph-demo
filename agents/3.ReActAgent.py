# Import necessary modules and classes
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage  # Base class for all message types
from langchain_core.messages import SystemMessage  # For system instructions
from langchain.chat_models import init_chat_model  # OpenAI chat model
from langchain_core.tools import tool  # Decorator for creating tools
from langgraph.graph.message import add_messages  # For message handling
from langgraph.graph import StateGraph, END  # For building the graph
from langgraph.prebuilt import ToolNode  # Prebuilt node for tool execution

# Load environment variables (like API keys)
load_dotenv()

# Define the state structure for our agent
# Uses Annotated to specify message sequence type with add_messages functionality
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# Define custom tools using the @tool decorator
@tool
def add(a: int, b: int):
    """Adds two numbers together"""
    return a + b 

@tool
def subtract(a: int, b: int):
    """Subtracts the second number from the first"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiplies two numbers together"""
    return a * b

# List of available tools
tools = [add, subtract, multiply]

# Initialize the OpenAI model with tool support
model = init_chat_model("google_genai:gemini-2.0-flash").bind_tools(tools)

def model_call(state: AgentState) -> AgentState:
    """
    Processes the current state through the language model.
    Adds a system prompt and gets a response.
    """
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    # Invoke model with system prompt and current messages
    response = model.invoke([system_prompt] + state["messages"])
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    """
    Determines if the agent should continue processing or end.
    Checks if the last message contains any tool calls.
    """
    messages = state["messages"]
    last_message = messages[-1]
    # If no tool calls, end the process
    if not last_message.tool_calls: 
        return "end"
    # Otherwise, continue to tool execution
    return "continue"

# Create the state graph
graph = StateGraph(AgentState)

# Add the main agent node
graph.add_node("agent", model_call)

# Add the tools node that will execute the tools
tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

# Set the entry point of the graph
graph.set_entry_point("agent")

# Add conditional edges based on whether we need to continue processing
graph.add_conditional_edges(
    "agent",  # Source node
    should_continue,  # Function to determine next step
    {
        "continue": "tools",  # If continuing, go to tools node
        "end": END,  # If done, end the graph
    },
)

# Add edge from tools back to agent for processing tool responses
graph.add_edge("tools", "agent")

# Compile the graph
app = graph.compile()

# Generate and save the graph visualization
graph_image = app.get_graph().draw_mermaid_png()
with open("agents/3_react_agent.png", "wb") as f:
    f.write(graph_image)
print("Graph visualization saved as 'agents/3_react_agent.png'")

def print_stream(stream):
    """Helper function to print the streaming output"""
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

# Example input that will trigger tool usage
inputs = {
    "messages": [
        ("user", "Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")
    ]
}

# Execute the graph with the input and print the streaming output
print_stream(app.stream(inputs, stream_mode="values"))