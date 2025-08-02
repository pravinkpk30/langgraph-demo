# Import necessary modules and classes
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Load environment variables (for API keys, etc.)
load_dotenv()

# Global variable to store the document content in memory
document_content = ""

# Define the state structure for our agent
# Uses Annotated to specify message sequence type with add_messages functionality
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

@tool
def update(content: str) -> str:
    """
    Updates the document with the provided content.
    
    Args:
        content: The new content to update the document with.
    Returns:
        str: Confirmation message with the updated content.
    """
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"

@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.
    
    Args:
        filename: Name for the text file (with or without .txt extension)
    Returns:
        str: Success or error message
    """
    global document_content

    # Ensure the filename has a .txt extension
    if not filename.endswith('.txt'):
        filename = f"agents/{filename}.txt"

    try:
        # Write the document content to the specified file
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    
    except Exception as e:
        return f"Error saving document: {str(e)}"

# List of available tools for the agent
tools = [update, save]

# Initialize the language model with tool support
model = init_chat_model("google_genai:gemini-2.0-flash").bind_tools(tools)

def agent(state: AgentState) -> AgentState:
    """
    Main agent function that processes user input and generates responses.
    
    Args:
        state: Current state containing message history
    Returns:
        Updated state with new messages
    """
    # System prompt with instructions for the AI
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    # Handle initial message or get new user input
    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    # Combine system prompt, message history, and new user message
    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    # Get response from the language model
    response = model.invoke(all_messages)

    # Print the AI's response and any tool usage
    print(f"\n🤖 AI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"🔧 USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": list(state["messages"]) + [user_message, response]}

def should_continue(state: AgentState) -> str:
    """
    Determines if the agent should continue processing or end.
    
    Args:
        state: Current state containing message history
    Returns:
        str: "continue" to keep going, "end" to stop
    """
    messages = state["messages"]
    
    if not messages:
        return "continue"
    
    # Check for a save confirmation message in the recent messages
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"  # End if document was saved
        
    return "continue"  # Continue processing otherwise

def print_messages(messages):
    """Helper function to print the last few messages in a readable format"""
    if not messages:
        return
    
    # Print only the last 3 messages to avoid cluttering the console
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

# Create and configure the state graph
graph = StateGraph(AgentState)

# Add nodes to the graph
graph.add_node("agent", agent)  # Main agent node
graph.add_node("tools", ToolNode(tools))  # Tools execution node

# Set the entry point
graph.set_entry_point("agent")

# Define the graph edges
graph.add_edge("agent", "tools")

# Add conditional edges based on whether to continue or end
graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue": "agent",  # Go back to agent for next input
        "end": END,  # End the graph execution
    },
)

# Compile the graph
app = graph.compile()

# Generate and save the graph visualization
graph_image = app.get_graph().draw_mermaid_png()
with open("agents/4_drafter_agent.png", "wb") as f:
    f.write(graph_image)
print("Graph visualization saved as 'agents/4_drafter_agent.png'")

def run_document_agent():
    """Main function to run the document agent"""
    print("\n ===== DRAFTER =====")
    
    # Initialize the state
    state = {"messages": []}
    
    # Run the graph with the initial state
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    
    print("\n ===== DRAFTER FINISHED =====")

# Entry point for the script
if __name__ == "__main__":
    run_document_agent()