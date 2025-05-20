import os

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI


# Load OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Initialize the model using OpenRouter's endpoint
llm = ChatOpenAI(
    openai_api_key=OPENROUTER_API_KEY,
    openai_api_base="https://openrouter.ai/api/v1",
    model_name="meta-llama/llama-3.3-8b-instruct:free",
    temperature=0.7
)

# Create a prompt template
prompt = PromptTemplate.from_template("""
You are an AI assistant with expertise in data analysis and automation.
Answer the following question:
Question: {question}
""")

# Chain using the new Runnable interface
# Build the chain using pipe syntax
chain = prompt | llm | StrOutputParser()


# Example query
query = "hi"
response = chain.invoke({"question": query})
print(f"Agent Response: {response}")


