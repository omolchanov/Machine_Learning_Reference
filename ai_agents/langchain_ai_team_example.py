import os
import warnings
import requests
import json
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
    print("\n===Requesting API for a joke===")

    url = "https://official-joke-api.appspot.com/random_joke"
    response = requests.get(url)

    dec_response = json.loads(response.content)
    joke = dec_response["setup"] + " " + dec_response["punchline"]

    print("Here is a raw joke: ", joke, "\n")

    return joke


# Word count tool
def get_word_count(_input: str) -> str:
    word_count = len(_input.split())
    return f"The text contains {word_count} words."


joke_tool = Tool(
    name="get_joke",
    func=get_joke,
    description="Use this tool to get a random joke."
)

get_word_count_tool = Tool(
    name="get_word_count",
    func=get_word_count,
    description="Use this tool to count words in the input"
)


# Memories for each agent
analysis_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
joke_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# Agent 1: TimeBot
analysis_bot = initialize_agent(
    tools=[time_tool],
    llm=llm,
    agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
    memory=analysis_memory,
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

print("AnalysisBot -> JokeBot: ", "Tell me a joke and I will analyse it")
answer_1 = joke_bot.run("tell a joke")

print("JokeBot -> AnalysisBot: ", "Sure here is it!", answer_1)

answer_2 = analysis_bot.run("rate the joke from the this message " + answer_1)
print("AnalysisBot -> JokeBot: ", "Thanks, here is my analysis", answer_2)

print("AnalysisBot -> JokeBot: ", analysis_bot.run("Please count words in " + answer_2))


