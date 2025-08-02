# Import necessary modules and classes
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Define the state structure for our agent
# This TypedDict defines that our state will have a 'messages' key
# containing a list of either HumanMessage or AIMessage objects
class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

# Initialize the language model using Google's Gemini 2.0 Flash
llm = init_chat_model("google_genai:gemini-2.0-flash")

def chatbot_node(state: AgentState) -> AgentState:
    """
    Process node that handles the conversation flow.
    Takes the current state, generates a response, and updates the state.
    """
    # Generate a response using the language model with current messages
    response = llm.invoke(state["messages"])

    # Add the AI's response to the message history
    state["messages"].append(AIMessage(content=response.content)) 
    
    # Print the AI's response and current state for debugging
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])

    return state

# Create a new graph with our defined state
graph = StateGraph(AgentState)

# Add the process node to the graph
graph.add_node("chatbot", chatbot_node)

# Define the edges of the graph
graph.add_edge(START, "chatbot")  # From start to process
graph.add_edge("chatbot", END)    # From process to end

# Compile the graph to create an executable agent
agent = graph.compile()

# Save the graph visualization as an image
graph_image = agent.get_graph().draw_mermaid_png()
with open("agents/2_memory_agent_bot.png", "wb") as f:
    f.write(graph_image)
print("Graph visualization saved as 'agents/2_memory_agent_bot.png'")

# Initialize an empty list to store conversation history
conversation_history = []

# Start the conversation loop
user_input = input("Enter: ")
while user_input != "exit":
    # Add user's message to conversation history
    conversation_history.append(HumanMessage(content=user_input))
    
    # Invoke the agent with current conversation history
    result = agent.invoke({"messages": conversation_history})
    
    # Update conversation history with the agent's response
    conversation_history = result["messages"]
    
    # Get next user input
    user_input = input("Enter: ")

# Save the conversation to a file when the loop exits
with open("agents/chatbot-conversation.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    # Write each message to file with appropriate formatting
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    
    file.write("End of Conversation")

print("Conversation saved to agents/chatbot-conversation.txt")