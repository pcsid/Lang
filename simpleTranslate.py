# Simple translate tutorial from langchain

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model("gpt-4o-mini", model_provider="openai")


#Part 1 - manual prompt
input_text = input("Enter the text to translate: ")

messages = [
    SystemMessage(content="Your persona is an expert English to French translator"),
    SystemMessage(content="Your job is to translate the text from English to French - no other text is needed"),
    HumanMessage(content=input_text)
]

#response = model.invoke(messages)
#print(response)
#part 2 - streaming
for token in model.stream(messages):
    print(token.content, end="", flush=True)

