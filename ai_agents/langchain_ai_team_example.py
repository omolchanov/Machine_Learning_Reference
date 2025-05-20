import os
import warnings
from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool


warnings.filterwarnings('ignore')

# Load OpenRouter parameters
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
API_URL = "https://api.together.xyz/v1"
MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

# Shared LLM
llm = ChatOpenAI(
    openai_api_key=TOGETHER_API_KEY,
    openai_api_base=API_URL,
    model_name=MODEL,
    temperature=0.7
)


# Time Tool
def get_time(_input: str) -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


time_tool = Tool(
    name="get_time",
    func=get_time,
    description="Use this tool to get the current time."
)


# Joke Tool
def get_joke(_input: str) -> str:
    return "Why don't scientists trust atoms? Because they make up everything!"


joke_tool = Tool(
    name="get_joke",
    func=get_joke,
    description="Use this tool to get a random joke."
)

# Memories for each agent
time_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
joke_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Agent 1: TimeBot
time_bot = initialize_agent(
    tools=[time_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=time_memory,
    verbose=False
)

# Agent 2: JokeBot
joke_bot = initialize_agent(
    tools=[joke_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=joke_memory,
    verbose=False
)

# --- Example interaction loop ---
print("TimeBot:", time_bot.run("What time is it?"))
print("JokeBot:", joke_bot.run("Tell me a joke"))

# Let them "talk" to each other
msg_from_timebot = time_bot.run("Please tell JokeBot the current time.")
msg_from_jokebot = joke_bot.run(f"TimeBot said: '{msg_from_timebot}'. What do you say?")
print("TimeBot -> JokeBot:", msg_from_timebot)
print("JokeBot -> TimeBot:", msg_from_jokebot)
