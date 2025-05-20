import os
import warnings

from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

warnings.filterwarnings('ignore')


# Load OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize the model using OpenRouter's endpoint
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="meta-llama/llama-3.3-8b-instruct:free",
    temperature=0.7
)


# Memory
memory = ConversationBufferMemory()

# Prebuilt chain
conversation = ConversationChain(
    llm=llm,
    memory=memory,
    verbose=False
)

# Try it
print(conversation.run("My name is Alex"))
print(conversation.run("What is my name?"))
