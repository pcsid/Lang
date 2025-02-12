from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph


load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")
'''
#basic hardcoded memory
response1 = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)

#print(response1)
'''
# Define a new graph
workflow = StateGraph(state_schema=MessagesState)


# Define the function that calls the model
def call_model(state: MessagesState):
    response = model.invoke(state["messages"])
    return {"messages": response}


# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

# Add memory
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

#visualize the graph
from IPython.display import Image, display

png_image = app.get_graph().draw_mermaid_png()

# Save the PNG file
with open("langgraph_workflow.png", "wb") as f:
    f.write(png_image)

print("Graph saved as langgraph_workflow.png")


config = {"configurable": {"thread_id": "abc123"}}

query = "Hi! I'm Bob."

input_messages = [HumanMessage(query)]
output = app.invoke({"messages": input_messages}, config)
#restarts every run, need backend to load in memory
output["messages"][-1].pretty_print()  # output contains all messages in state