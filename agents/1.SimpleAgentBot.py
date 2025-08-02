from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

load_dotenv()

# Initialize the language model
llm = init_chat_model("google_genai:gemini-2.0-flash")

# Define the state of our application
class State(TypedDict):
    messages: List[HumanMessage]

# Define the chatbot node that processes messages
def chatbot_node(state: State):
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

# Define the graph
builder = StateGraph(State)
builder.add_node("chatbot", chatbot_node)
builder.add_edge(START, "chatbot")
builder.add_edge("chatbot", END)

# Compile the graph
graph = builder.compile()

# Save the graph visualization as an image
graph_image = graph.get_graph().draw_mermaid_png()
with open("agents/1_simple_agent_bot.png", "wb") as f:
    f.write(graph_image)
print("Graph visualization saved as 'agents/1_simple_agent_bot.png'")

# Run the graph
graph.invoke({"messages":[HumanMessage(content="Hello")]})

user_input = input("Enter: ")
while user_input != "exit":
    graph.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")

    
