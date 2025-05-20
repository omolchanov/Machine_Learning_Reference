import os
import warnings
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool

warnings.filterwarnings('ignore')

# Load OpenRouter parameters
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_URL = "https://openrouter.ai/api/v1"
MODEL = "meta-llama/llama-3.3-8b-instruct:free"

# Initialize the model
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base=API_URL,
    model_name=MODEL,
    temperature=0.7
)


# Tool function that accepts an input (returns time)
def get_time(_input: str) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# Define the Tool
time_tool = Tool(
    name="get_time",
    func=get_time,
    description="Use this tool to get the current time."
)


# Tool function that accepts an input (returns joke)
def get_joke(_input: str) -> str:
    return 'Here is a very cool joke'


# Define the Tool
joke_tool = Tool(
    name="get_joke",
    func=get_joke,
    description="Use this tool to get a random joke."
)

# Memory
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Initialize the agent
agent = initialize_agent(
    tools=[time_tool, joke_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    verbose=False,
    memory=memory
)

# Try it
print(agent.run("My name is Alex."))
print(agent.run("What is my name?"))
print(agent.run("What time is it?"))
print(agent.run("Tell me some joke"))
print(agent.run("Repeat the last joke you sent to me"))
